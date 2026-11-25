"""Tests for src.gate: promote, and reject on each criterion separately.

Most tests use lookup models on a synthetic frozen eval set: every row has unique feature values,
and a model predicts dga exactly for the rows it was given, so each test controls which rows each
model gets right and fails one criterion while the others pass. One end-to-end test trains two
small real models on data/sample.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src import config
from src.features import FEATURE_NAMES, extract_features
from src.gate import committed_sha256, decide, write_decision
from src.prepare_retrain import content_hash

GROUP_SIZES = {"benign": 200, "fam_a": 200, "matsnu": 100, "nymaim": 100}
NEW, PRIOR = ["matsnu"], ["nymaim"]
CRITERIA = [
    "frozen_eval_hash",
    "new_family_recall",
    "seen_f1",
    "false_positive_rate",
    "prior_family_recall",
    "sanity_domains",
]


def features_key(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


SANITY_KEYS = {
    domain: (features_key(extract_features(domain)), label)
    for domain, label in config.SANITY_DOMAINS
}


class LookupModel:
    """P(dga) = 1 for the feature rows in dga_keys, else 0 (scikit-learn predict_proba shape)."""

    def __init__(self, dga_keys: set[tuple[float, ...]]) -> None:
        self.dga_keys = dga_keys

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.array([1.0 if features_key(row) in self.dga_keys else 0.0 for row in x.to_numpy()])
        return np.column_stack([1 - p, p])


def synthetic_eval() -> pd.DataFrame:
    """Frozen-eval-like frame; every row has unique feature values (row id + column offset)."""
    rows, i = [], 0
    for family, n in GROUP_SIZES.items():
        for _ in range(n):
            label = "benign" if family == "benign" else "dga"
            features = [1_000_000 + i * 100 + c for c in range(len(FEATURE_NAMES))]
            rows.append([f"row{i}.com", f"row{i}", family, label, *features])
            i += 1
    return pd.DataFrame(rows, columns=["domain", "sld", "family", "label", *FEATURE_NAMES])


EVAL = synthetic_eval()
EVAL_HASH = content_hash(EVAL)


def model(
    benign_fp: int = 10,
    seen_tp: int = 180,
    new_tp: int = 10,
    prior_tp: int = 80,
    sanity_ok: bool = True,
) -> LookupModel:
    """A model flagging the first N rows of each group as dga (counts out of GROUP_SIZES)."""
    keys: set[tuple[float, ...]] = set()
    for family, n in (
        ("benign", benign_fp),
        ("fam_a", seen_tp),
        ("matsnu", new_tp),
        ("nymaim", prior_tp),
    ):
        rows = EVAL[EVAL["family"] == family].head(n)
        keys |= {features_key(r) for r in rows[FEATURE_NAMES].to_numpy()}
    for domain, (key, label) in SANITY_KEYS.items():
        wrong = not sanity_ok and domain == "google.com"
        if (label == "dga") != wrong:
            keys.add(key)
    return LookupModel(keys)


CHAMPION = model()


def run(challenger: LookupModel, prior: list[str] | None = None, **kwargs) -> dict:
    return decide(
        CHAMPION,
        challenger,
        EVAL,
        NEW,
        PRIOR if prior is None else prior,
        expected_sha256=kwargs.get("expected_sha256", EVAL_HASH),
    )


def reason(decision: dict, name: str) -> dict:
    return next(r for r in decision["reasons"] if r["name"] == name)


# ---------------------------------------------------------------------------
# Promote, and reject on each criterion separately
# ---------------------------------------------------------------------------


def test_promotes_a_better_challenger() -> None:
    decision = run(model(new_tp=50))
    assert decision["promote"] is True
    assert decision["failed"] == []
    assert [r["name"] for r in decision["reasons"]] == CRITERIA
    new = reason(decision, "new_family_recall")
    assert (new["champion"], new["challenger"], new["delta"]) == (0.1, 0.5, 0.4)


@pytest.mark.parametrize(
    ("criterion", "challenger", "expected_sha256"),
    [
        ("frozen_eval_hash", model(new_tp=50), "0" * 64),
        ("new_family_recall", model(new_tp=11), None),  # +0.01 < +0.02
        ("seen_f1", model(new_tp=50, seen_tp=170), None),  # F1 0.9231 -> 0.8947
        ("false_positive_rate", model(new_tp=50, benign_fp=12), None),  # 0.05 -> 0.06
        ("prior_family_recall", model(new_tp=50, prior_tp=77), None),  # 0.80 -> 0.77
        ("sanity_domains", model(new_tp=50, sanity_ok=False), None),  # google.com flagged
    ],
)
def test_rejects_on_each_criterion_alone(
    criterion: str, challenger: LookupModel, expected_sha256: str | None
) -> None:
    decision = run(challenger, expected_sha256=expected_sha256 or EVAL_HASH)
    assert decision["promote"] is False
    assert decision["failed"] == [criterion]
    assert reason(decision, criterion)["message"]


def test_rejection_lists_every_failed_criterion() -> None:
    decision = run(model(new_tp=10, benign_fp=40, sanity_ok=False))
    assert decision["failed"] == [
        "new_family_recall",
        "seen_f1",
        "false_positive_rate",
        "sanity_domains",
    ]
    sanity = reason(decision, "sanity_domains")
    assert "google.com" in sanity["message"]
    assert (sanity["champion"], sanity["challenger"]) == ("6/6", "5/6")


# ---------------------------------------------------------------------------
# Boundaries (inclusive)
# ---------------------------------------------------------------------------


def test_new_family_recall_gain_of_exactly_the_threshold_passes() -> None:
    assert reason(run(model(new_tp=12)), "new_family_recall")["passed"]  # 0.10 -> 0.12
    assert not reason(run(model(new_tp=11)), "new_family_recall")["passed"]


def test_fpr_rise_of_exactly_the_threshold_passes() -> None:
    assert reason(run(model(new_tp=50, benign_fp=11)), "false_positive_rate")["passed"]  # +0.005
    assert not reason(run(model(new_tp=50, benign_fp=12)), "false_positive_rate")["passed"]


def test_prior_recall_drop_of_exactly_the_threshold_passes() -> None:
    assert reason(run(model(new_tp=50, prior_tp=78)), "prior_family_recall")["passed"]  # -0.02
    assert not reason(run(model(new_tp=50, prior_tp=77)), "prior_family_recall")["passed"]


def test_seen_f1_small_drop_passes() -> None:
    within = reason(run(model(new_tp=50, seen_tp=177)), "seen_f1")  # 0.9231 -> 0.9147
    assert within["passed"] and -config.GATE_MAX_SEEN_F1_DROP <= within["delta"] < 0
    assert not reason(run(model(new_tp=50, seen_tp=175)), "seen_f1")["passed"]


def test_thresholds_from_config() -> None:
    assert config.GATE_MIN_NEW_FAMILY_RECALL_GAIN == 0.02
    assert config.GATE_MAX_SEEN_F1_DROP == 0.01
    assert config.GATE_MAX_FPR_RISE == 0.005
    assert config.GATE_MAX_PRIOR_RECALL_DROP == 0.02
    thresholds = run(model(new_tp=50))["thresholds"]
    assert thresholds["max_prior_family_recall_drop"] == config.GATE_MAX_PRIOR_RECALL_DROP


# ---------------------------------------------------------------------------
# Prior families (the sixth criterion)
# ---------------------------------------------------------------------------


def test_prior_family_criterion_passes_when_there_are_none() -> None:
    # Round 1: nothing was phased in earlier. Even a collapse on nymaim does not count then.
    decision = run(model(new_tp=50, prior_tp=0), prior=[])
    prior = reason(decision, "prior_family_recall")
    assert prior["passed"] and prior["champion"] is None and prior["challenger"] is None
    assert "no families phased in" in prior["message"]
    assert decision["promote"] is True


def test_prior_family_collapse_is_invisible_to_seen_f1() -> None:
    # Why the sixth criterion exists: earlier families are held out, so not in the seen-F1 rows.
    # Losing one completely leaves seen F1 unchanged; only the sixth criterion catches it.
    decision = run(model(new_tp=50, prior_tp=0))
    assert decision["failed"] == ["prior_family_recall"]
    assert reason(decision, "seen_f1")["delta"] == 0.0
    overall = decision["metrics"]
    assert overall["challenger"]["overall"]["f1"] < overall["champion"]["overall"]["f1"]


# ---------------------------------------------------------------------------
# Scored on the rows, never on stored numbers; configuration errors
# ---------------------------------------------------------------------------


def test_stored_metrics_are_ignored() -> None:
    challenger = model(new_tp=10)
    challenger.f1_seen = 1.0  # what a stored-number comparison would read
    challenger.recall_future = 1.0
    assert run(challenger)["failed"] == ["new_family_recall"]


def test_default_expected_hash_is_the_committed_one() -> None:
    committed = json.loads(config.EVAL_FROZEN_REPORT_PATH.read_text(encoding="utf-8"))
    assert committed_sha256() == committed["sha256"]
    decision = decide(CHAMPION, model(new_tp=50), EVAL, NEW, PRIOR)  # synthetic set: other hash
    assert decision["failed"] == ["frozen_eval_hash"]
    assert decision["eval_frozen"]["expected_sha256"] == committed["sha256"]


@pytest.mark.parametrize(
    ("new", "prior", "message"),
    [
        ([], [], "new_families is empty"),
        (["simda"], [], "no DGA rows"),
        (["matsnu"], ["simda"], "no DGA rows"),
        (["matsnu"], ["matsnu"], "both new and phased in earlier"),
    ],
)
def test_configuration_errors_raise(new: list[str], prior: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decide(CHAMPION, CHAMPION, EVAL, new, prior, expected_sha256=EVAL_HASH)


def test_decision_json_is_deterministic(tmp_path: Path) -> None:
    first, second = run(model(new_tp=50)), run(model(new_tp=50))
    assert first == second
    path = tmp_path / "gate" / "decision.json"
    write_decision(first, path)
    assert json.loads(path.read_text(encoding="utf-8")) == first
    metrics = first["metrics"]
    assert set(metrics) == {"champion", "challenger"}
    assert set(metrics["champion"]["per_family_recall"]) == set(GROUP_SIZES)
    assert set(metrics["champion"]["overall"]) == {"precision", "recall", "f1"}


# ---------------------------------------------------------------------------
# End to end: two small real models on data/sample
# ---------------------------------------------------------------------------


def featurized(domains: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame([extract_features(d) for d in domains["domain"]], columns=FEATURE_NAMES)
    return pd.concat([domains.reset_index(drop=True), x], axis=1)


def test_end_to_end_with_real_models() -> None:
    benign = pd.read_csv(config.TRANCO_SAMPLE_PATH).sample(300, random_state=config.SEED)
    dga = pd.read_csv(config.UMUDGA_SAMPLE_PATH)
    seen = dga[dga["family"].isin(["chinad", "locky", "ramnit"])].sample(300, random_state=1)
    matsnu = dga[dga["family"] == "matsnu"].sample(300, random_state=2)
    rows = pd.concat(
        [
            benign[["domain"]].assign(family="benign", label="benign"),
            seen[["domain", "family"]].assign(label="dga"),
            matsnu[["domain", "family"]].assign(label="dga"),
        ],
        ignore_index=True,
    )
    data = featurized(rows).assign(sld=lambda d: d["domain"].str.split(".").str[0])
    # Halves: even rows train, odd rows evaluate. matsnu training rows only go to the challenger.
    train, eval_frozen = data.iloc[::2], data.iloc[1::2].reset_index(drop=True)
    old = train[train["family"] != "matsnu"]
    params = {"n_estimators": 20, "max_depth": 2, "random_state": config.SEED}
    champion = GradientBoostingClassifier(**params).fit(old[FEATURE_NAMES], old["label"] == "dga")
    challenger = GradientBoostingClassifier(**params).fit(
        train[FEATURE_NAMES], train["label"] == "dga"
    )

    decision = decide(
        champion, challenger, eval_frozen, ["matsnu"], [], expected_sha256=content_hash(eval_frozen)
    )
    new = reason(decision, "new_family_recall")
    assert new["challenger"] > new["champion"]
    assert decision["eval_frozen"]["rows"] == len(eval_frozen)
    assert [r["name"] for r in decision["reasons"]] == CRITERIA
    # The gate's number is the challenger's own predictions on these rows.
    matsnu_rows = eval_frozen.loc[eval_frozen["family"] == "matsnu", FEATURE_NAMES]
    proba = challenger.predict_proba(matsnu_rows)[:, 1]
    assert new["challenger"] == round(float((proba >= config.DGA_THRESHOLD).mean()), 6)
