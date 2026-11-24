"""Work with a copy of the committed Milestone 1 MLflow store, never with the committed files.

The committed mlflow.db and mlruns/ are a read-only snapshot of the M1 runs. MLflow stores artifact
locations as absolute paths of the machine that trained, so the store has to be relocated
(src.mlflow_store) before its artifacts can be read, and relocating writes to the database. This
module therefore checks the committed database against config.M1_SNAPSHOT_DB_SHA256, copies it and
mlruns/ into a gitignored working folder (mlflow_m1/ by default), and relocates the copy.

Run with: python -m src.m1_snapshot   (prints the `mlflow ui` command for the copy)
"""

import argparse
import shutil
from pathlib import Path

from src import config
from src.mlflow_store import relocate_mlflow_store
from src.model_loader import file_sha256

DB_NAME = "mlflow.db"
ARTIFACT_DIR_NAME = "mlruns"


class SnapshotChangedError(Exception):
    """The committed mlflow.db no longer matches config.M1_SNAPSHOT_DB_SHA256."""


def verify_snapshot(
    db_path: Path = config.M1_SNAPSHOT_DB, expected: str = config.M1_SNAPSHOT_DB_SHA256
) -> None:
    """Refuse to work from a committed store that changed (restore it with git restore)."""
    actual = file_sha256(db_path)
    if actual != expected:
        raise SnapshotChangedError(
            f"{db_path} has SHA-256 {actual}, expected {expected}. The M1 snapshot must not "
            f"change; restore it with: git restore {db_path.name}"
        )


def prepare_m1_snapshot(
    dest: Path = config.M1_SNAPSHOT_WORKDIR,
    db_path: Path = config.M1_SNAPSHOT_DB,
    artifact_dir: Path = config.M1_SNAPSHOT_ARTIFACT_DIR,
    expected_sha256: str = config.M1_SNAPSHOT_DB_SHA256,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> str:
    """Make a fresh, relocated copy of the M1 store in dest. Returns its tracking URI.

    dest is emptied first, so the copy always starts from the committed snapshot. Only the copy
    is relocated; the committed files are only read.
    """
    verify_snapshot(db_path, expected_sha256)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copy_db = dest / DB_NAME
    copy_artifacts = dest / ARTIFACT_DIR_NAME
    shutil.copyfile(db_path, copy_db)
    shutil.copytree(artifact_dir, copy_artifacts)
    relocate_mlflow_store(copy_db, copy_artifacts, experiment_name)
    return config.sqlite_uri(copy_db)


def mlflow_ui_command(tracking_uri: str, port: int = 5000) -> str:
    return f"mlflow ui --backend-store-uri {tracking_uri} --port {port}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Copy the committed M1 MLflow store to a working folder for viewing."
    )
    parser.add_argument("--dest", type=Path, default=config.M1_SNAPSHOT_WORKDIR)
    args = parser.parse_args(argv)
    tracking_uri = prepare_m1_snapshot(args.dest)
    print(f"M1 snapshot copied to {args.dest.resolve()}")
    print("View it with (venv activated):")
    print(f"  {mlflow_ui_command(tracking_uri)}")


if __name__ == "__main__":
    main()
