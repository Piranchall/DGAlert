"""Tests for src.retrain: one round on a small synthetic setup built from data/sample.

A temporary production folder (a small GB champion + latest.json), temporary processed parts and a
temporary MLflow store; the real models/production/ and mlflow stores are never touched. The gate
itself is tested in tests/test_gate.py, so promotion and rejection are forced here by overriding
the gate's verdict; one test runs the real gate unchanged.
"""

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src import config, retrain
from src import gate as gate_module
from src.explain import build_explainer
from src.features import FEATURE_NAMES, extract_features
from src.model_loader import file_sha256, load_model
from src.prepare_retrain import content_hash
from src.retrain import (
    next_batch,
    parse_args,
    run_round,
    summary_markdown,
    training_rows,
    write_github_outputs,
)

SMALL_HGB = {"max_iter": 20, "early_stopping": False, "random_state": config.SEED}
CHAMPION_VERSION = "champion_v0"


def featurized(rows: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame([extract_features(d) for d in rows["domain"]], columns=FEATURE_NAMES)
    out = pd.concat([rows.reset_index(drop=True), x], axis=1)
    return out.assign(sld=out["domain"].str.split(".").str[0])


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """train_fit, phase_in_pool and eval_frozen parquet files from data/sample rows."""
    benign = pd.read_csv(config.TRANCO_SAMPLE_PATH).sample(400, random_state=1)
    dga = pd.read_csv(config.UMUDGA_SAMPLE_PATH)

    def family(name: str, n: int, seed: int) -> pd.DataFrame:
        rows = dga[dga["family"] == name].sample(n, random_state=seed)
        return rows[["domain", "family"]].assign(label="dga")

    rows = featurized(
        pd.concat(
            [
                benign[["domain"]].assign(family="benign", label="benign"),
                family("chinad", 200, 2),
                family("locky", 200, 3),
                family("matsnu", 200, 4),
                family("nymaim", 100, 5),
                family("simda", 100, 6),
            ],
            ignore_index=True,
        )
    )
    held = rows["family"].isin(["matsnu", "nymaim", "simda"])
    evaluation = rows.index % 4 == 0
    out = tmp_path_factory.mktemp("processed")
    rows[~held & ~evaluation].to_parquet(out / config.TRAIN_FIT_PATH.name, index=False)
    rows[held & ~evaluation].to_parquet(out / config.PHASE_IN_POOL_PATH.name, index=False)
    rows[evaluation].reset_index(drop=True).to_parquet(out / config.EVAL_FROZEN_PATH.name)
    return out


@pytest.fixture(scope="module")
def eval_hash(processed: Path) -> str:
    return content_hash(pd.read_parquet(processed / config.EVAL_FROZEN_PATH.name))


@pytest.fixture(scope="module")
def champion_file(processed: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small GB trained on train_fit only (it has never seen a held-out family)."""
    train = pd.read_parquet(processed / config.TRAIN_FIT_PATH.name)
    model = GradientBoostingClassifier(n_estimators=20, random_state=config.SEED)
    model.fit(train[FEATURE_NAMES], (train["label"] == "dga").astype(int))
    path = tmp_path_factory.mktemp("champion") / f"model_{CHAMPION_VERSION}.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture
def production(tmp_path: Path, champion_file: Path) -> Path:
    """A production folder like models/production/: the champion model and its latest.json."""
    folder = tmp_path / "production"
    folder.mkdir()
    model_path = folder / champion_file.name
    shutil.copy(champion_file, model_path)
    latest = {
        "path": model_path.as_posix(),
        "version": CHAMPION_VERSION,
        "model_type": "gb",
        "sha256": file_sha256(model_path),
        "mlflow_run_id": "none",
        "phased_in_families": [],
    }
    (folder / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> dict:
    folder = tmp_path_factory.mktemp("mlflow")
    return {"tracking_uri": config.sqlite_uri(folder / "mlflow.db"), "artifact_dir": folder / "art"}


def folder_hashes(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def force_verdict(monkeypatch: pytest.MonkeyPatch, promote: bool) -> None:
    """Run the real gate, then override its verdict (the gate's own tests are in test_gate.py)."""

    def forced(*args, **kwargs) -> dict:
        decision = gate_module.decide(*args, **kwargs)
        return {**decision, "promote": promote}

    monkeypatch.setattr(retrain, "decide", forced)


def round_in(
    tmp_path: Path, processed: Path, production: Path, store: dict, eval_hash: str, **kwargs
) -> dict | None:
    return run_round(
        processed_dir=processed,
        production_dir=production,
        reports_dir=tmp_path / "reports" / "retrain",
        production_report_path=tmp_path / "reports" / "production_model.json",
        tracking_uri=store["tracking_uri"],
        artifact_dir=store["artifact_dir"],
        hgb_params=SMALL_HGB,
        expected_sha256=eval_hash,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Round planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phased_in", "expected"),
    [
        ([], ["matsnu"]),
        (["matsnu"], ["nymaim", "simda"]),
        (["matsnu", "nymaim", "simda"], ["dyre", "shiotob"]),
        (["matsnu", "nymaim", "simda", "dyre", "shiotob"], None),
    ],
)
def test_next_batch(phased_in: list[str], expected: list[str] | None) -> None:
    assert next_batch(phased_in) == expected


def test_partly_phased_in_batch_raises() -> None:
    with pytest.raises(ValueError, match="only partly phased in"):
        next_batch(["matsnu", "nymaim"])


def test_training_rows_are_train_fit_plus_the_phased_in_pool_families(processed: Path) -> None:
    train_fit = pd.read_parquet(processed / config.TRAIN_FIT_PATH.name)
    pool = pd.read_parquet(processed / config.PHASE_IN_POOL_PATH.name)
    rows = training_rows(train_fit, pool, ["matsnu"])
    assert len(rows) == len(train_fit) + (pool["family"] == "matsnu").sum()
    assert set(rows["family"]) == set(train_fit["family"]) | {"matsnu"}


def test_hgb_params_are_fixed_and_deterministic() -> None:
    assert config.HGB_PARAMS["early_stopping"] is False
    assert config.HGB_PARAMS["random_state"] == config.SEED
    search = json.loads(
        (config.RETRAIN_REPORTS_DIR / "hgb_param_search.json").read_text(encoding="utf-8")
    )
    assert search["chosen"] == config.HGB_PARAMS
    assert search["chosen_validation_f1"] >= search["best_validation_f1"] - search["tie_margin"]


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------


def test_promotion_replaces_the_production_model(
    tmp_path: Path,
    processed: Path,
    production: Path,
    store: dict,
    eval_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)  # CI sets it: the store label differs
    force_verdict(monkeypatch, promote=True)
    report = round_in(tmp_path, processed, production, store, eval_hash)

    assert report["round"] == 1 and report["applied"] is True
    latest = json.loads((production / "latest.json").read_text(encoding="utf-8"))
    assert latest["model_type"] == "hgb"
    assert latest["round"] == 1
    assert latest["phased_in_families"] == ["matsnu"]
    assert latest["eval_frozen_sha256"] == eval_hash
    assert "f1_seen" not in latest and "f1_future" not in latest
    # Old model deleted, new one next to latest.json; the loader accepts it (sha256 checked).
    assert sorted(p.name for p in production.iterdir()) == [
        "latest.json",
        f"model_{latest['version']}.joblib",
    ]
    model, metadata = load_model(production / "latest.json")
    assert metadata["version"] == latest["version"] == report["model"]["version"]
    assert build_explainer(model, "hgb").output_units == "log_odds"

    production_report = json.loads(
        (tmp_path / "reports" / "production_model.json").read_text(encoding="utf-8")
    )
    assert production_report["version"] == latest["version"]
    assert production_report["mlflow_run_id"] == latest["mlflow_run_id"] == report["mlflow_run_id"]
    assert production_report["experiment"] == config.MLFLOW_KAIZEN_EXPERIMENT_NAME
    assert production_report["mlflow_store"] == "local"
    assert production_report["phased_in_families"] == ["matsnu"]

    summary = (tmp_path / "reports" / "retrain" / "round_1.md").read_text(encoding="utf-8")
    assert "PROMOTE" in summary and latest["version"] in summary


def test_rejection_leaves_production_unchanged(
    tmp_path: Path,
    processed: Path,
    production: Path,
    store: dict,
    eval_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = folder_hashes(production)
    force_verdict(monkeypatch, promote=False)
    report = round_in(tmp_path, processed, production, store, eval_hash)
    assert report["applied"] is False and report["promote"] is False
    assert folder_hashes(production) == before
    assert not (tmp_path / "reports" / "production_model.json").exists()
    saved = json.loads((tmp_path / "reports" / "retrain" / "round_1.json").read_text("utf-8"))
    assert saved["decision"]["reasons"] == report["decision"]["reasons"]
    assert "REJECT" in (tmp_path / "reports" / "retrain" / "round_1.md").read_text("utf-8")


def test_dry_run_never_changes_production(
    tmp_path: Path,
    processed: Path,
    production: Path,
    store: dict,
    eval_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = folder_hashes(production)
    force_verdict(monkeypatch, promote=True)
    report = round_in(tmp_path, processed, production, store, eval_hash, dry_run=True)
    assert report["promote"] is True and report["applied"] is False
    assert folder_hashes(production) == before
    assert "Dry run" in (tmp_path / "reports" / "retrain" / "round_1.md").read_text("utf-8")


def test_second_round_phases_in_the_next_batch_with_prior_families(
    tmp_path: Path,
    processed: Path,
    production: Path,
    store: dict,
    eval_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    force_verdict(monkeypatch, promote=True)
    round_in(tmp_path, processed, production, store, eval_hash)
    report = round_in(tmp_path, processed, production, store, eval_hash)
    assert report["round"] == 2
    assert report["new_families"] == ["nymaim", "simda"]
    assert report["prior_families"] == ["matsnu"]
    assert report["champion"]["model_type"] == "hgb"  # round 1's model is now the champion
    latest = json.loads((production / "latest.json").read_text(encoding="utf-8"))
    assert latest["phased_in_families"] == ["matsnu", "nymaim", "simda"]
    assert latest["round"] == 2


def test_nothing_left_to_phase_in(
    tmp_path: Path, processed: Path, production: Path, store: dict, eval_hash: str
) -> None:
    latest = json.loads((production / "latest.json").read_text(encoding="utf-8"))
    latest["phased_in_families"] = [f for batch in config.PHASE_IN_BATCHES for f in batch]
    (production / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    assert round_in(tmp_path, processed, production, store, eval_hash) is None
    assert not (tmp_path / "reports").exists()


def test_real_gate_reasons_and_mlflow_run(
    tmp_path: Path, processed: Path, production: Path, store: dict, eval_hash: str
) -> None:
    report = round_in(tmp_path, processed, production, store, eval_hash, dry_run=True)
    names = [r["name"] for r in report["decision"]["reasons"]]
    assert names == [
        "frozen_eval_hash",
        "new_family_recall",
        "seen_f1",
        "false_positive_rate",
        "prior_family_recall",
        "sanity_domains",
    ]
    run = mlflow.MlflowClient(store["tracking_uri"]).get_run(report["mlflow_run_id"])
    assert run.data.tags["model_type"] == "hgb"
    assert run.data.params["new_families"] == '["matsnu"]'
    assert run.data.metrics["promoted"] == float(report["promote"])
    assert "challenger_new_family_recall" in run.data.metrics


def test_frozen_hash_mismatch_rejects(
    tmp_path: Path, processed: Path, production: Path, store: dict
) -> None:
    report = round_in(tmp_path, processed, production, store, "0" * 64)
    assert "frozen_eval_hash" in report["decision"]["failed"]
    assert report["applied"] is False


# ---------------------------------------------------------------------------
# Command line and workflow outputs
# ---------------------------------------------------------------------------


def test_store_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert retrain.store_label() == "github_actions_artifact"
    monkeypatch.delenv("GITHUB_ACTIONS")
    assert retrain.store_label() == "local"


def test_parse_args() -> None:
    assert parse_args([]).dry_run is False
    assert parse_args(["--dry-run"]).dry_run is True


def test_github_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    write_github_outputs({"applied": True, "round": 3}, tmp_path)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == ["promoted=true", "round=3", f"summary={(tmp_path / 'round_3.md').as_posix()}"]
    output.unlink()
    write_github_outputs(None, tmp_path)
    assert output.read_text(encoding="utf-8").splitlines() == [
        "promoted=false",
        "round=",
        "summary=",
    ]


def test_summary_markdown_lists_every_criterion(
    tmp_path: Path, processed: Path, production: Path, store: dict, eval_hash: str
) -> None:
    report = round_in(tmp_path, processed, production, store, eval_hash, dry_run=True)
    summary = summary_markdown(report)
    for reason in report["decision"]["reasons"]:
        assert f"| {reason['name']} |" in summary
    assert "| matsnu |" in summary
