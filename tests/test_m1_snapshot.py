"""Tests for src.m1_snapshot and the reports built from the committed M1 MLflow snapshot.

These read the real committed mlflow.db and mlruns/ but only ever write to tmp_path.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src import config
from src.m1_snapshot import SnapshotChangedError, main, prepare_m1_snapshot, verify_snapshot
from src.model_loader import file_sha256
from src.report import build_production_report, build_reports

M1_RUN_IDS = {
    "logreg": "da37618448e64d4e8a8d72405ee86ffb",
    "rf": "5aa355e272f9403d935e68045593b943",
    "gb": "9c15d284722c46fc9355f98e134b44d1",
}


def tree_hashes(folder: Path) -> dict[str, str]:
    """SHA-256 of every file under folder, keyed by its relative path."""
    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def artifact_location(db_path: Path) -> str:
    """The dgalert-m1 experiment's artifact location, read without writing (read-only URI)."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT artifact_location FROM experiments WHERE name = ?",
            (config.MLFLOW_EXPERIMENT_NAME,),
        ).fetchone()[0]
    finally:
        conn.close()


def test_committed_snapshot_matches_recorded_hash() -> None:
    assert file_sha256(config.M1_SNAPSHOT_DB) == config.M1_SNAPSHOT_DB_SHA256


def test_viewer_command_leaves_the_committed_store_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    db_before = file_sha256(config.M1_SNAPSHOT_DB)
    artifacts_before = tree_hashes(config.M1_SNAPSHOT_ARTIFACT_DIR)
    location_before = artifact_location(config.M1_SNAPSHOT_DB)

    dest = tmp_path / "m1"
    main(["--dest", str(dest)])

    assert file_sha256(config.M1_SNAPSHOT_DB) == db_before == config.M1_SNAPSHOT_DB_SHA256
    assert tree_hashes(config.M1_SNAPSHOT_ARTIFACT_DIR) == artifacts_before
    assert artifact_location(config.M1_SNAPSHOT_DB) == location_before
    out = capsys.readouterr().out
    assert f"mlflow ui --backend-store-uri {config.sqlite_uri(dest / 'mlflow.db')}" in out


def test_copy_is_relocated_and_complete(tmp_path: Path) -> None:
    dest = tmp_path / "m1"
    tracking_uri = prepare_m1_snapshot(dest)

    assert tracking_uri == config.sqlite_uri(dest / "mlflow.db")
    assert artifact_location(dest / "mlflow.db") == (dest / "mlruns").resolve().as_uri()
    assert tree_hashes(dest / "mlruns") == tree_hashes(config.M1_SNAPSHOT_ARTIFACT_DIR)


def test_copy_starts_fresh_each_time(tmp_path: Path) -> None:
    dest = tmp_path / "m1"
    prepare_m1_snapshot(dest)
    (dest / "leftover.txt").write_text("from an earlier session", encoding="utf-8")
    prepare_m1_snapshot(dest)
    assert not (dest / "leftover.txt").exists()


def test_changed_snapshot_is_refused(tmp_path: Path) -> None:
    changed = tmp_path / "mlflow.db"
    changed.write_bytes(config.M1_SNAPSHOT_DB.read_bytes() + b"x")
    with pytest.raises(SnapshotChangedError, match="git restore"):
        verify_snapshot(changed)
    with pytest.raises(SnapshotChangedError):
        prepare_m1_snapshot(tmp_path / "m1", db_path=changed)
    assert not (tmp_path / "m1").exists()


# ---------------------------------------------------------------------------
# Regression: reports rebuilt from the snapshot equal the committed reports/
# Compared as parsed JSON, so CRLF (Windows checkout) and LF (Linux CI) both pass.
# ---------------------------------------------------------------------------


def committed_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snapshot_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    return prepare_m1_snapshot(tmp_path_factory.mktemp("m1") / "store")


@pytest.fixture(scope="module")
def split_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A split_report.json with the committed class balance, so data/processed/ is not needed."""
    balance = committed_report(config.RESULTS_REPORT_PATH)["class_balance"]
    totals = {name: {"rows": v["rows"], "dga": v["dga"]} for name, v in balance.items()}
    path = tmp_path_factory.mktemp("split") / "split_report.json"
    path.write_text(json.dumps({"totals": totals}), encoding="utf-8")
    return path


def test_m1_reports_rebuilt_from_snapshot_match_committed(
    snapshot_uri: str, split_report: Path
) -> None:
    results, per_family = build_reports(snapshot_uri, split_report_path=split_report)

    assert results == committed_report(config.RESULTS_REPORT_PATH)
    assert per_family == committed_report(config.PER_FAMILY_REPORT_PATH)
    assert {m: results["models"][m]["mlflow_run_id"] for m in M1_RUN_IDS} == M1_RUN_IDS


def test_production_report_matches_committed(snapshot_uri: str, tmp_path: Path) -> None:
    absent_local = config.sqlite_uri(tmp_path / "no_local_store" / "mlflow.db")
    stores = [("local", absent_local), ("m1_snapshot", snapshot_uri)]

    report = build_production_report(config.PRODUCTION_MODEL_POINTER, stores)

    assert report == committed_report(config.PRODUCTION_REPORT_PATH)
    assert report["mlflow_store"] == "m1_snapshot"
    assert report["mlflow_run_id"] == M1_RUN_IDS["gb"]
