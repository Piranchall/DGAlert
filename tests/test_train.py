"""Tests for src.train on tiny synthetic splits in tmp_path (no data/raw, no repo MLflow store)."""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES
from src.train import (
    BASIC_FEATURES,
    MODEL_TYPES,
    PER_FAMILY_FILE,
    comparison_table,
    run_training,
    stratified_subsample,
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
