"""Tests for src.mlflow_store on a small MLflow SQLite store in tmp_path."""

import json
import shutil
import sqlite3
from pathlib import Path

import mlflow
import pytest

from src.mlflow_store import relocate_mlflow_store, sqlite_db_path

EXPERIMENT = "relocate-test"
ARTIFACT = "per_family_recall.json"
CONTENT = {"test_seen": {"benign": 0.9}}


def stored_locations(db_path: Path) -> tuple[str, list[str]]:
    """(experiment artifact_location, artifact_uri of each of its runs), read from SQLite."""
    with sqlite3.connect(db_path) as conn:
        exp_id, location = conn.execute(
            "SELECT experiment_id, artifact_location FROM experiments WHERE name = ?",
            (EXPERIMENT,),
        ).fetchone()
        uris = [
            r[0]
            for r in conn.execute(
                "SELECT artifact_uri FROM runs WHERE experiment_id = ?", (exp_id,)
            )
        ]
    conn.close()
    return location, uris


@pytest.fixture
def store(tmp_path: Path):
    """An MLflow store whose experiment points at folder A, with one run and one artifact.

    Returns (db path, tracking URI, folder A, run id).
    """
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    folder_a = tmp_path / "A"
    client = mlflow.MlflowClient(tracking_uri)
    exp_id = client.create_experiment(EXPERIMENT, artifact_location=folder_a.resolve().as_uri())
    run = client.create_run(exp_id)
    client.log_dict(run.info.run_id, CONTENT, ARTIFACT)
    client.set_terminated(run.info.run_id)
    assert (folder_a / run.info.run_id / "artifacts" / ARTIFACT).is_file()
    return db_path, tracking_uri, folder_a, run.info.run_id


def test_artifact_downloads_from_new_folder_after_move(store, tmp_path: Path) -> None:
    db_path, tracking_uri, folder_a, run_id = store
    folder_b = tmp_path / "B"
    shutil.move(folder_a, folder_b)  # as if the repo was cloned somewhere else

    assert relocate_mlflow_store(db_path, folder_b, EXPERIMENT) is True

    location, uris = stored_locations(db_path)
    assert location == folder_b.resolve().as_uri()
    assert uris == [f"{folder_b.resolve().as_uri()}/{run_id}/artifacts"]
    path = mlflow.MlflowClient(tracking_uri).download_artifacts(run_id, ARTIFACT, str(tmp_path))
    assert json.loads(Path(path).read_text(encoding="utf-8")) == CONTENT


def test_relocating_twice_is_harmless(store, tmp_path: Path) -> None:
    db_path, _, _, _ = store
    folder_b = tmp_path / "B"
    assert relocate_mlflow_store(db_path, folder_b, EXPERIMENT) is True
    after_first = stored_locations(db_path)

    assert relocate_mlflow_store(db_path, folder_b, EXPERIMENT) is False
    assert stored_locations(db_path) == after_first


def test_already_matching_location_is_left_alone(store) -> None:
    db_path, _, folder_a, _ = store
    before = stored_locations(db_path)
    assert relocate_mlflow_store(db_path, folder_a, EXPERIMENT) is False
    assert stored_locations(db_path) == before


def test_missing_db_does_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    assert relocate_mlflow_store(missing, tmp_path / "B", EXPERIMENT) is False
    assert not missing.exists()


def test_unknown_experiment_does_nothing(store, tmp_path: Path) -> None:
    db_path, _, _, _ = store
    before = stored_locations(db_path)
    assert relocate_mlflow_store(db_path, tmp_path / "B", "no-such-experiment") is False
    assert stored_locations(db_path) == before


def test_sqlite_db_path() -> None:
    assert sqlite_db_path("sqlite:///C:/repo/mlflow.db") == Path("C:/repo/mlflow.db")
    assert sqlite_db_path("sqlite:////home/u/repo/mlflow.db") == Path("/home/u/repo/mlflow.db")
    assert sqlite_db_path("http://localhost:5000") is None
