"""Tests for src.train on tiny synthetic splits in tmp_path (no data/raw, no repo MLflow store)."""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_TYPES, PER_FAMILY_FILE
from src.features import FEATURE_NAMES
from src.report import write_reports
from src.train import (
    BASIC_FEATURES,
    comparison_table,
    parse_args,
    promote_if_better,
    run_training,
    stratified_subsample,
    write_latest,
)

TINY_GRID = {"n_estimators": [5, 10], "max_depth": [2], "learning_rate": [0.1]}
LATEST_FIELDS = {
    "path",
    "version",
    "model_type",
    "f1_seen",
    "f1_future",
    "recall_future",
    "mlflow_run_id",
    "trained_at",
}


def make_rows(family: str, label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n rows with features drawn around 0 (benign) or 2 (dga), each with its own sld."""
    center = 2.0 if label == "dga" else 0.0
    features = rng.normal(center, 1.0, size=(n, len(FEATURE_NAMES)))
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df.insert(0, "label", label)
    df.insert(0, "family", family)
    df.insert(0, "sld", [f"{family}{i}" for i in range(n)])
    df.insert(0, "domain", df["sld"] + ".com")
    return df


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One temporary folder shared by the module, so training runs only once."""
    return tmp_path_factory.mktemp("train")


@pytest.fixture(scope="module")
def processed_dir(tmp_dir: Path) -> Path:
    """Write train / test_seen / test_future parquet files shaped like src.prepare_data's output."""
    rng = np.random.default_rng(0)
    train = pd.concat(
        [make_rows("benign", "benign", 40, rng)]
        + [make_rows(f"fam_{c}", "dga", 20, rng) for c in "ab"]
    )
    seen_benign = make_rows("benign_seen", "benign", 10, rng).assign(family="benign")
    test_seen = pd.concat(
        [seen_benign] + [make_rows(f"fam_{c}_seen", "dga", 5, rng) for c in "ab"]
    ).assign(family=lambda d: d["family"].str.removesuffix("_seen"))
    test_future = pd.concat([make_rows("fam_held", "dga", 10, rng), seen_benign])

    out = tmp_dir / "processed"
    out.mkdir()
    for name, frame in [("train", train), ("test_seen", test_seen), ("test_future", test_future)]:
        frame.reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
    return out


@pytest.fixture(scope="module")
def trained(processed_dir: Path, tmp_dir: Path) -> tuple[list[dict], Path, str]:
    """Run all three model types into tmp_path; restore the global MLflow tracking URI after."""
    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_dir / 'mlflow.db').as_posix()}"
    pointer = tmp_dir / "models" / "latest.json"
    results = run_training(
        MODEL_TYPES,
        processed_dir=processed_dir,
        models_dir=tmp_dir / "models",
        pointer=pointer,
        tracking_uri=tracking_uri,
        artifact_dir=tmp_dir / "mlruns",
        experiment_name="test-train",
        gb_grid=TINY_GRID,
        gb_max_rows=60,  # smaller than train (80 rows): exercises the stratified subsample
    )
    yield results, pointer, tracking_uri
    mlflow.set_tracking_uri(previous_uri)


def test_latest_json_is_valid_and_points_at_best_model(trained) -> None:
    results, pointer, _ = trained
    latest = json.loads(pointer.read_text(encoding="utf-8"))

    assert set(latest) == LATEST_FIELDS
    best = max(results, key=lambda r: r["metrics"]["f1_seen"])
    assert latest["model_type"] == best["model_type"]
    assert latest["mlflow_run_id"] == best["mlflow_run_id"]
    assert latest["f1_seen"] == pytest.approx(best["metrics"]["f1_seen"])
    for metric in ("f1_seen", "f1_future", "recall_future"):
        assert 0.0 <= latest[metric] <= 1.0
    assert Path(latest["path"]).name == f"model_{latest['version']}.joblib"


def test_every_saved_model_loads_and_predicts(trained, processed_dir: Path) -> None:
    results, _, _ = trained
    test_seen = pd.read_parquet(processed_dir / "test_seen.parquet")

    assert len({r["version"] for r in results}) == len(results)  # one file per run
    for r in results:
        model = joblib.load(r["path"])
        proba = model.predict_proba(test_seen[FEATURE_NAMES])
        assert proba.shape == (len(test_seen), 2)
        # The two clusters are far apart, so every model should separate them well.
        assert r["metrics"]["f1_seen"] > 0.8


def test_logreg_uses_only_basic_features(trained, processed_dir: Path) -> None:
    results, _, _ = trained
    logreg = joblib.load(next(r["path"] for r in results if r["model_type"] == "logreg"))
    x = pd.read_parquet(processed_dir / "test_seen.parquet")[FEATURE_NAMES]

    changed = x.copy()
    extra = [f for f in FEATURE_NAMES if f not in BASIC_FEATURES]
    changed[extra] = 999.0  # dictionary / n-gram features must not affect logreg

    np.testing.assert_allclose(logreg.predict_proba(x), logreg.predict_proba(changed))


def test_mlflow_runs_metrics_and_per_family_artifact(trained) -> None:
    results, _, tracking_uri = trained
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-train")
    runs = client.search_runs([experiment.experiment_id])

    parents = [r for r in runs if r.data.tags.get("model_type") in MODEL_TYPES]
    children = [r for r in runs if r.data.tags.get("model_type") == "gb_grid_point"]
    assert sorted(r.data.tags["model_type"] for r in parents) == sorted(MODEL_TYPES)
    assert len(children) == 2  # one nested run per TINY_GRID point

    for run in parents:
        assert set(comparison_table(results).columns) <= set(run.data.metrics)
        path = client.download_artifacts(run.info.run_id, PER_FAMILY_FILE)
        families = json.loads(Path(path).read_text(encoding="utf-8"))
        assert set(families["test_seen"]) == {"benign", "fam_a", "fam_b"}
        assert set(families["test_future"]) == {"benign", "fam_held"}


def test_stratified_subsample_keeps_family_shares() -> None:
    rng = np.random.default_rng(1)
    df = pd.concat(
        [make_rows("benign", "benign", 200, rng), make_rows("fam_a", "dga", 200, rng)],
        ignore_index=True,
    )
    sample = stratified_subsample(df, 100)

    assert len(sample) == 100
    assert sample["family"].value_counts().to_dict() == {"benign": 50, "fam_a": 50}
    assert stratified_subsample(df, 1000) is df  # small enough: returned unchanged


# ---------------------------------------------------------------------------
# latest.json promotion: only a strictly better f1_seen (or --force) replaces the current model
# ---------------------------------------------------------------------------


def candidate(model_type: str, version: str, f1_seen: float) -> dict:
    """A training result shaped like train_one()'s return value."""
    return {
        "model_type": model_type,
        "version": version,
        "path": f"models/model_{version}.joblib",
        "mlflow_run_id": f"run-{version}",
        "trained_at": "2026-11-05T00:00:00+00:00",
        "metrics": {"f1_seen": f1_seen, "f1_future": 0.5, "recall_future": 0.5},
    }


@pytest.fixture
def pointer(tmp_path: Path) -> Path:
    """latest.json already pointing at a model with f1_seen 0.80."""
    path = tmp_path / "models" / "latest.json"
    write_latest(candidate("rf", "20260101_000000", 0.80), path)
    return path


def current_version(pointer: Path) -> str:
    return json.loads(pointer.read_text(encoding="utf-8"))["version"]


def test_promote_with_no_existing_pointer(tmp_path: Path, capsys) -> None:
    path = tmp_path / "models" / "latest.json"

    assert promote_if_better(candidate("gb", "20260202_000000", 0.10), path)
    assert current_version(path) == "20260202_000000"
    assert "no current model" in capsys.readouterr().out


def test_promote_better_candidate_replaces_pointer(pointer: Path, capsys) -> None:
    assert promote_if_better(candidate("gb", "20260202_000000", 0.85), pointer)
    assert current_version(pointer) == "20260202_000000"
    assert "updated" in capsys.readouterr().out


@pytest.mark.parametrize("f1_seen", [0.70, 0.80])  # worse, and equal (not strictly better)
def test_promote_worse_or_equal_candidate_keeps_pointer(
    pointer: Path, f1_seen: float, capsys
) -> None:
    assert not promote_if_better(candidate("gb", "20260202_000000", f1_seen), pointer)
    assert current_version(pointer) == "20260101_000000"
    assert "kept" in capsys.readouterr().out


def test_promote_worse_candidate_with_force_replaces_pointer(pointer: Path, capsys) -> None:
    assert promote_if_better(candidate("logreg", "20260202_000000", 0.10), pointer, force=True)
    assert current_version(pointer) == "20260202_000000"
    assert "--force" in capsys.readouterr().out


def test_cli_force_flag() -> None:
    assert parse_args(["--model", "rf", "--force"]).force is True
    assert parse_args([]).force is False
    assert parse_args([]).model == "all"


# ---------------------------------------------------------------------------
# reports/ built from the MLflow store of the trained fixture (no retraining)
# ---------------------------------------------------------------------------


def test_write_reports_from_mlflow(trained, tmp_dir: Path) -> None:
    results, _, tracking_uri = trained
    split_report = tmp_dir / "split_report.json"
    split_report.write_text(
        json.dumps(
            {
                "totals": {
                    "test_seen": {"rows": 20, "dga": 10},
                    "test_future": {"rows": 20, "dga": 15},
                }
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_dir / "reports" / "m1_results.json"
    per_family_path = tmp_dir / "reports" / "per_family_recall.json"

    write_reports(
        results_path,
        per_family_path,
        tracking_uri=tracking_uri,
        experiment_name="test-train",
        split_report_path=split_report,
    )
    report = json.loads(results_path.read_text(encoding="utf-8"))
    per_family = json.loads(per_family_path.read_text(encoding="utf-8"))

    assert set(report["models"]) == set(MODEL_TYPES)
    for r in results:
        logged = report["models"][r["model_type"]]
        assert logged["mlflow_run_id"] == r["mlflow_run_id"]
        assert logged["f1_seen"] == pytest.approx(r["metrics"]["f1_seen"], abs=1e-4)
        assert logged["params"]["model_type"] == r["model_type"]
    assert report["class_balance"]["test_seen"]["dga_share"] == 0.5
    assert report["class_balance"]["test_future"]["dga_share"] == 0.75
    best_grid = report["gb_best_grid_point"]
    assert best_grid["grid_points"] == 2
    assert best_grid["params"]["n_estimators"] in TINY_GRID["n_estimators"]
    assert set(per_family) == {"test_seen", "test_future"}
    assert set(per_family["test_future"]["gb"]) == {"benign", "fam_held"}
