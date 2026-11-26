"""Tests for src.metrics and the M1 AUC report (python -m src.report --auc).

Hand-computed examples for the metrics, small models on synthetic frames for the report; the
committed reports/m1_auc.json is checked against other committed files only (CI has no
data/processed/ and no M1 logreg/rf files).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.features import FEATURE_NAMES
from src.metrics import against_benign, per_family, threshold_free
from src.model_loader import file_sha256, read_pointer
from src.report import (
    CHECKED_METRICS,
    build_auc_report,
    m1_model_files,
    parse_args,
    threshold_metrics,
)

# ---------------------------------------------------------------------------
# Metrics on hand-computed examples
# ---------------------------------------------------------------------------


def test_threshold_free_on_a_hand_computed_example() -> None:
    # Positives 0.35 and 0.8, negatives 0.1 and 0.4: 3 of the 4 pairs are ranked right.
    # Precision at each positive, highest score first: 1/1 at 0.8, 2/3 at 0.35.
    result = threshold_free(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)


def test_perfect_and_inverted_rankings() -> None:
    y = np.array([0, 0, 1, 1])
    assert threshold_free(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert threshold_free(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0


def test_single_class_gives_none() -> None:
    assert threshold_free(np.array([1, 1]), np.array([0.2, 0.9])) == {
        "roc_auc": None,
        "pr_auc": None,
    }


def frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["family", "label"])


# benign 0.1, 0.3, 0.6; family a 0.5, 0.9; family b 0.05 (would lower a's AUC if it were counted)
FAMILIES = frame(
    [
        ("benign", "benign"),
        ("benign", "benign"),
        ("benign", "benign"),
        ("a", "dga"),
        ("a", "dga"),
        ("b", "dga"),
    ]
)
P = np.array([0.1, 0.3, 0.6, 0.5, 0.9, 0.05])


def test_against_benign_uses_only_the_given_families_and_all_benign_rows() -> None:
    a = against_benign(FAMILIES, P, ["a"])
    # a: 0.5 beats 0.1 and 0.3, loses to 0.6; 0.9 beats all three: 5 of 6 pairs.
    assert a["roc_auc"] == pytest.approx(5 / 6)
    assert (a["positives"], a["benign"]) == (2, 3)
    pooled = against_benign(FAMILIES, P, ["a", "b"])
    assert pooled["roc_auc"] == pytest.approx(5 / 9)
    assert pooled["positives"] == 3


def test_per_family_reports_baseline_and_recall() -> None:
    result = per_family(FAMILIES, P, ["a", "b"])
    assert result["a"]["pr_auc_baseline"] == pytest.approx(2 / 5)
    assert result["a"]["recall"] == 1.0  # 0.5 and 0.9 are both >= the 0.5 threshold
    assert result["b"]["roc_auc"] == 0.0  # 0.05 is below every benign score
    assert result["b"]["recall"] == 0.0
    assert result["b"]["positives"] == 1


# ---------------------------------------------------------------------------
# The M1 AUC report
# ---------------------------------------------------------------------------


def synthetic_split(seed: int, families: list[str]) -> pd.DataFrame:
    """Benign and DGA rows whose features overlap partly, so the AUCs are between 0 and 1."""
    rng = np.random.default_rng(seed)
    parts = [pd.DataFrame({"family": "benign", "label": "benign"}, index=range(60))]
    parts += [pd.DataFrame({"family": f, "label": "dga"}, index=range(30)) for f in families]
    df = pd.concat(parts, ignore_index=True)
    shift = (df["label"] == "dga").to_numpy()[:, None] * 0.8
    features = rng.normal(size=(len(df), len(FEATURE_NAMES))) + shift
    return pd.concat([df, pd.DataFrame(features, columns=FEATURE_NAMES)], axis=1)


SEEN = synthetic_split(1, ["fam_a"])
FUTURE = synthetic_split(2, ["held_x", "held_y"])


def fitted(c: float) -> LogisticRegression:
    train = synthetic_split(3, ["fam_a"])
    model = LogisticRegression(C=c, max_iter=1000)
    return model.fit(train[FEATURE_NAMES], (train["label"] == "dga").astype(int))


MODELS = {name: fitted(c) for name, c in (("logreg", 0.01), ("rf", 0.1), ("gb", 1.0))}


def m1_results_for(models: dict) -> dict:
    """m1_results.json-like numbers that agree with the given models."""
    out = {}
    for name, model in models.items():
        p_seen = model.predict_proba(SEEN[FEATURE_NAMES])[:, 1]
        p_future = model.predict_proba(FUTURE[FEATURE_NAMES])[:, 1]
        out[name] = threshold_metrics(p_seen, p_future, SEEN, FUTURE)
    return {"models": out}


def report_for(models: dict, m1: dict | None = None) -> dict:
    entries = {name: (m, f"models/{name}.joblib", "0" * 64) for name, m in models.items()}
    return build_auc_report(SEEN, FUTURE, entries, m1 or m1_results_for(models))


def test_auc_report_has_every_model_test_set_and_held_out_family() -> None:
    report = report_for(MODELS)
    assert list(report["models"]) == ["logreg", "rf", "gb"]
    for entry in report["models"].values():
        assert entry["threshold_metrics_match_m1"] is True
        for test_set in ("test_seen", "test_future"):
            assert 0 < entry[test_set]["roc_auc"] < 1
        assert list(entry["held_out_families"]) == ["held_x", "held_y"]
        family = entry["held_out_families"]["held_x"]
        assert family["pr_auc_baseline"] == round(30 / 90, 4)
        assert family["positives"] == 30
    assert report["rows"] == {"test_seen": len(SEEN), "test_future": len(FUTURE)}


def test_auc_report_values_are_rounded_and_deterministic(tmp_path: Path) -> None:
    first = json.dumps(report_for(MODELS), indent=2)
    second = json.dumps(
        report_for({n: fitted(c) for n, c in (("logreg", 0.01), ("rf", 0.1), ("gb", 1.0))}),
        indent=2,
    )
    assert first == second
    gb = report_for(MODELS)["models"]["gb"]["test_seen"]["roc_auc"]
    assert gb == round(gb, 4)
    for leak in ("run_id", "time", str(config.ROOT_DIR), "\\\\"):
        assert leak not in first, leak


def test_auc_report_refuses_a_model_that_is_not_the_m1_model() -> None:
    m1 = m1_results_for(MODELS)
    m1["models"]["rf"]["f1_seen"] = round(m1["models"]["rf"]["f1_seen"] + 0.0001, 4)
    with pytest.raises(ValueError, match="is not M1's rf"):
        report_for(MODELS, m1)


def write_pointer(folder: Path, model_file: Path, **overrides) -> Path:
    pointer = {
        "path": str(model_file),
        "version": "v_gb",
        "model_type": "gb",
        "sha256": file_sha256(model_file),
        "feature_names": FEATURE_NAMES,
        **overrides,
    }
    path = folder / "latest.json"
    path.write_text(json.dumps(pointer), encoding="utf-8")
    return path


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, dict]:
    paths = {}
    for name, model in MODELS.items():
        paths[name] = tmp_path / f"{name}.joblib"
        joblib.dump(model, paths[name])
    m1 = {
        "models": {
            "gb": {"params": {"version": "v_gb"}},
            "logreg": {"params": {"model_path": str(paths["logreg"])}},
            "rf": {"params": {"model_path": str(paths["rf"])}},
        }
    }
    return tmp_path, {"paths": paths, "m1": m1}


def test_model_files_come_from_the_pointer_and_m1s_model_paths(model_files) -> None:
    folder, info = model_files
    pointer = write_pointer(folder, info["paths"]["gb"])
    files = m1_model_files(info["m1"], pointer)
    assert files == {
        "logreg": info["paths"]["logreg"],
        "rf": info["paths"]["rf"],
        "gb": info["paths"]["gb"],
    }


def test_model_files_refuse_a_retrained_production_model(model_files) -> None:
    folder, info = model_files
    pointer = write_pointer(folder, info["paths"]["gb"], model_type="hgb", version="r1")
    with pytest.raises(ValueError, match="not M1's gb"):
        m1_model_files(info["m1"], pointer)


def test_model_files_refuse_a_tampered_gb(model_files) -> None:
    folder, info = model_files
    pointer = write_pointer(folder, info["paths"]["gb"], sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match the SHA-256"):
        m1_model_files(info["m1"], pointer)


def test_model_files_explain_a_missing_m1_file(model_files) -> None:
    folder, info = model_files
    pointer = write_pointer(folder, info["paths"]["gb"])
    info["paths"]["rf"].unlink()
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        m1_model_files(info["m1"], pointer)


def test_cli_flag() -> None:
    assert parse_args(["--auc"]).auc is True
    assert parse_args([]).auc is False


# ---------------------------------------------------------------------------
# The committed reports/m1_auc.json, against other committed files
# ---------------------------------------------------------------------------


def test_committed_auc_report_matches_the_committed_m1_records() -> None:
    report = json.loads(config.M1_AUC_REPORT_PATH.read_text(encoding="utf-8"))
    m1 = json.loads(config.RESULTS_REPORT_PATH.read_text(encoding="utf-8"))
    per_family_m1 = json.loads(config.PER_FAMILY_REPORT_PATH.read_text(encoding="utf-8"))
    pointer = read_pointer(config.PRODUCTION_MODEL_POINTER)

    assert list(report["models"]) == config.MODEL_TYPES
    assert report["rows"]["test_seen"] == m1["class_balance"]["test_seen"]["rows"]
    assert report["rows"]["test_future"] == m1["class_balance"]["test_future"]["rows"]
    for model_type, entry in report["models"].items():
        assert entry["threshold_metrics_match_m1"] is True
        assert sorted(entry["held_out_families"]) == sorted(config.HELD_OUT_FAMILIES)
        for family, values in entry["held_out_families"].items():
            # recall at the threshold must equal M1's per-family recall for the same model
            assert values["recall"] == per_family_m1["test_future"][model_type][family], family
    if pointer.get("version") == m1["models"]["gb"]["params"]["version"]:
        assert report["models"]["gb"]["sha256"] == pointer["sha256"]
    assert set(CHECKED_METRICS) <= set(m1["models"]["gb"])
