"""Promotion gate: does a retrained challenger replace the champion?

Both models are scored on the same rows of the frozen eval set (src.prepare_retrain); stored
metrics are never compared. The challenger is promoted only if all six criteria hold (thresholds in
src.config, boundaries inclusive):

1. frozen_eval_hash: the frozen eval set's content hash matches reports/eval_frozen.json;
2. new_family_recall: pooled recall on the newly added families rises by at least 0.02;
3. seen_f1: F1 on the test_seen rows (training families + benign) drops by at most 0.01;
4. false_positive_rate: the share of benign rows flagged as dga rises by at most 0.005;
5. prior_family_recall: pooled recall on the families phased in during earlier rounds drops by at
   most 0.02 (passes when there are none, as in round 1). Added beyond the slice's five criteria:
   those families are held out, so their rows are not in the seen-F1 rows, and a collapse on one
   would otherwise go unnoticed;
6. sanity_domains: the challenger classifies every config.SANITY_DOMAINS entry correctly.

ROC-AUC and PR-AUC (src.metrics) are reported for both models, overall and for the new and the
earlier families against all benign rows, but they are not criteria: they show whether a model
ranks a family correctly independently of the threshold.

All criteria are always evaluated, so a rejection lists every reason. Retraining (S11) promotes
through decide(); src.train.promote_if_better stays for reproducing Milestone 1.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config
from src.features import FEATURE_NAMES, extract_features
from src.metrics import against_benign, dga_probability, threshold_free
from src.prepare_retrain import content_hash
from src.train import predict

# Float tolerance at the boundaries, so a difference of exactly a threshold passes.
TOLERANCE = 1e-9
DIGITS = 6  # rounding of the reported values (decisions use the unrounded ones)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _share(mask: pd.Series) -> float:
    """Mean of a boolean series (0.0 when empty)."""
    return float(mask.mean()) if len(mask) else 0.0


def sanity_labels(model: Any) -> dict[str, str]:
    """The label the model gives each sanity domain."""
    domains = [domain for domain, _ in config.SANITY_DOMAINS]
    x = pd.DataFrame([extract_features(d) for d in domains], columns=FEATURE_NAMES)
    predicted = predict(model, x)
    return {d: "dga" if p == 1 else "benign" for d, p in zip(domains, predicted, strict=True)}


def score(
    model: Any, eval_frozen: pd.DataFrame, new_families: list[str], prior_families: list[str]
) -> dict[str, Any]:
    """Every number the gate needs for one model, computed from its predictions on eval_frozen."""
    y = (eval_frozen["label"] == "dga").astype(int)
    pred = predict(model, eval_frozen)
    is_dga = y == 1
    seen = ~eval_frozen["family"].isin(config.HELD_OUT_FAMILIES)
    new = is_dga & eval_frozen["family"].isin(new_families)
    prior = is_dga & eval_frozen["family"].isin(prior_families)
    correct = (pred == y).groupby(eval_frozen["family"]).mean().sort_index()
    p = dga_probability(model, eval_frozen)
    return {
        "overall": {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        },
        "seen_f1": float(f1_score(y[seen], pred[seen], zero_division=0)),
        "new_family_recall": _share(pred[new] == 1),
        "prior_family_recall": _share(pred[prior] == 1) if prior.any() else None,
        "false_positive_rate": _share(pred[~is_dga] == 1),
        "per_family_recall": {family: float(r) for family, r in correct.items()},
        "sanity": sanity_labels(model),
        # Reported only, never a criterion.
        "threshold_free": {
            "overall": threshold_free(y, p),
            "new_families": _auc_only(against_benign(eval_frozen, p, new_families)),
            "prior_families": _auc_only(against_benign(eval_frozen, p, prior_families))
            if prior.any()
            else None,
        },
    }


def _auc_only(scores: dict[str, Any]) -> dict[str, float | None]:
    return {"roc_auc": scores["roc_auc"], "pr_auc": scores["pr_auc"]}


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def _criterion(
    name: str, passed: bool, champion: Any, challenger: Any, threshold: Any, message: str
) -> dict[str, Any]:
    delta = None
    if isinstance(champion, float) and isinstance(challenger, float):
        delta = _round(challenger - champion)
        champion, challenger = _round(champion), _round(challenger)
    return {
        "name": name,
        "passed": bool(passed),
        "champion": champion,
        "challenger": challenger,
        "delta": delta,
        "threshold": threshold,
        "message": message,
    }


def criteria(
    champion: dict[str, Any], challenger: dict[str, Any], actual_sha256: str, expected_sha256: str
) -> list[dict[str, Any]]:
    """The six criteria, in a fixed order, each with its values and a one-line message."""
    results = []

    same_set = actual_sha256 == expected_sha256
    results.append(
        _criterion(
            "frozen_eval_hash",
            same_set,
            None,
            None,
            None,
            "frozen eval set hash matches reports/eval_frozen.json"
            if same_set
            else f"frozen eval set hash {actual_sha256[:12]} != expected {expected_sha256[:12]}",
        )
    )

    gain = challenger["new_family_recall"] - champion["new_family_recall"]
    threshold = config.GATE_MIN_NEW_FAMILY_RECALL_GAIN
    results.append(
        _criterion(
            "new_family_recall",
            gain >= threshold - TOLERANCE,
            champion["new_family_recall"],
            challenger["new_family_recall"],
            threshold,
            f"recall on the new families changes by {gain:+.4f} (needs at least +{threshold})",
        )
    )

    drop = champion["seen_f1"] - challenger["seen_f1"]
    threshold = config.GATE_MAX_SEEN_F1_DROP
    results.append(
        _criterion(
            "seen_f1",
            drop <= threshold + TOLERANCE,
            champion["seen_f1"],
            challenger["seen_f1"],
            threshold,
            f"F1 on the seen families changes by {-drop:+.4f} (may drop by at most {threshold})",
        )
    )

    rise = challenger["false_positive_rate"] - champion["false_positive_rate"]
    threshold = config.GATE_MAX_FPR_RISE
    results.append(
        _criterion(
            "false_positive_rate",
            rise <= threshold + TOLERANCE,
            champion["false_positive_rate"],
            challenger["false_positive_rate"],
            threshold,
            f"false-positive rate changes by {rise:+.4f} (may rise by at most {threshold})",
        )
    )

    threshold = config.GATE_MAX_PRIOR_RECALL_DROP
    if champion["prior_family_recall"] is None:
        results.append(
            _criterion(
                "prior_family_recall",
                True,
                None,
                None,
                threshold,
                "no families phased in during earlier rounds: passes",
            )
        )
    else:
        drop = champion["prior_family_recall"] - challenger["prior_family_recall"]
        results.append(
            _criterion(
                "prior_family_recall",
                drop <= threshold + TOLERANCE,
                champion["prior_family_recall"],
                challenger["prior_family_recall"],
                threshold,
                f"recall on the earlier phased-in families changes by {-drop:+.4f} "
                f"(may drop by at most {threshold})",
            )
        )

    expected = dict(config.SANITY_DOMAINS)
    wrong = sorted(d for d, label in challenger["sanity"].items() if label != expected[d])
    champion_correct = sum(champion["sanity"][d] == label for d, label in expected.items())
    results.append(
        _criterion(
            "sanity_domains",
            not wrong,
            f"{champion_correct}/{len(expected)}",
            f"{len(expected) - len(wrong)}/{len(expected)}",
            None,
            "every sanity domain classified correctly"
            if not wrong
            else f"misclassified sanity domains: {', '.join(wrong)}",
        )
    )
    return results


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def committed_sha256() -> str:
    """The frozen eval set hash committed in reports/eval_frozen.json."""
    report = json.loads(config.EVAL_FROZEN_REPORT_PATH.read_text(encoding="utf-8"))
    return report["sha256"]


def _check_families(
    eval_frozen: pd.DataFrame, new_families: list[str], prior_families: list[str]
) -> None:
    """Configuration errors raise instead of turning into a rejection."""
    if not new_families:
        raise ValueError("new_families is empty: there is nothing for the gate to measure")
    overlap = set(new_families) & set(prior_families)
    if overlap:
        raise ValueError(f"families both new and phased in earlier: {sorted(overlap)}")
    present = set(eval_frozen.loc[eval_frozen["label"] == "dga", "family"])
    missing = sorted((set(new_families) | set(prior_families)) - present)
    if missing:
        raise ValueError(f"families with no DGA rows in the frozen eval set: {missing}")


def _rounded_scores(scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {k: _round(v) for k, v in scores["overall"].items()},
        "seen_f1": _round(scores["seen_f1"]),
        "new_family_recall": _round(scores["new_family_recall"]),
        "prior_family_recall": _round(scores["prior_family_recall"]),
        "false_positive_rate": _round(scores["false_positive_rate"]),
        "per_family_recall": {k: _round(v) for k, v in scores["per_family_recall"].items()},
        "sanity": scores["sanity"],
        "threshold_free": {
            part: None if values is None else {k: _round(v) for k, v in values.items()}
            for part, values in scores["threshold_free"].items()
        },
    }


def decide(
    champion: Any,
    challenger: Any,
    eval_frozen: pd.DataFrame,
    new_families: list[str],
    prior_families: list[str] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Score both models on eval_frozen and decide whether the challenger is promoted.

    new_families: the families added to training in this round. prior_families: the families
    phased in during earlier rounds (S11 passes phased_in_families from the production
    latest.json). expected_sha256 defaults to the hash in reports/eval_frozen.json.
    """
    prior_families = list(prior_families or [])
    _check_families(eval_frozen, new_families, prior_families)
    expected = expected_sha256 or committed_sha256()
    actual = content_hash(eval_frozen)

    champion_scores = score(champion, eval_frozen, new_families, prior_families)
    challenger_scores = score(challenger, eval_frozen, new_families, prior_families)
    results = criteria(champion_scores, challenger_scores, actual, expected)
    return {
        "promote": all(r["passed"] for r in results),
        "failed": [r["name"] for r in results if not r["passed"]],
        "reasons": results,
        "new_families": sorted(new_families),
        "prior_families": sorted(prior_families),
        "eval_frozen": {"sha256": actual, "expected_sha256": expected, "rows": len(eval_frozen)},
        "thresholds": {
            "min_new_family_recall_gain": config.GATE_MIN_NEW_FAMILY_RECALL_GAIN,
            "max_seen_f1_drop": config.GATE_MAX_SEEN_F1_DROP,
            "max_false_positive_rate_rise": config.GATE_MAX_FPR_RISE,
            "max_prior_family_recall_drop": config.GATE_MAX_PRIOR_RECALL_DROP,
        },
        "metrics": {
            "champion": _rounded_scores(champion_scores),
            "challenger": _rounded_scores(challenger_scores),
        },
    }


def write_decision(decision: dict[str, Any], path: Path) -> None:
    """Write the decision as JSON (deterministic: no times or paths)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
