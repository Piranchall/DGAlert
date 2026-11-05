"""Keep the committed MLflow store usable on any machine.

MLflow saves the experiment's artifact location, and every run's artifact URI, as absolute
file:/// paths. The committed mlflow.db therefore points at the folder of the machine that first
trained (e.g. file:///C:/Users/<someone>/.../mlruns). On another machine, training would then
write outside the repo (or fail with a permission error), and reading old runs' artifacts fails.

relocate_mlflow_store() rewrites those paths in the SQLite database so they point at this
machine's artifact folder. It uses only the standard-library sqlite3 module.
"""

import sqlite3
from pathlib import Path

SQLITE_PREFIX = "sqlite:///"


def sqlite_db_path(tracking_uri: str) -> Path | None:
    """The database file of a sqlite:/// tracking URI, or None for any other kind of store."""
    if not tracking_uri.startswith(SQLITE_PREFIX):
        return None
    return Path(tracking_uri.removeprefix(SQLITE_PREFIX))


def relocate_mlflow_store(db_path: Path, artifact_dir: Path, experiment_name: str) -> bool:
    """Point an experiment and its runs at artifact_dir. Returns True if anything changed.

    Sets experiments.artifact_location to artifact_dir.resolve().as_uri(), and in runs.artifact_uri
    of that experiment's runs replaces the old location prefix with the new one. Does nothing if
    the database or the experiment does not exist, or if the location already matches.
    """
    if not db_path.is_file():
        return False
    new_location = artifact_dir.resolve().as_uri()
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT experiment_id, artifact_location FROM experiments WHERE name = ?",
            (experiment_name,),
        ).fetchone()
        if row is None or row[1] == new_location:
            return False
        experiment_id, old_location = row
        with conn:  # one transaction: both updates are saved together, or neither is
            conn.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (new_location, experiment_id),
            )
            # Only URIs that start with the old location exactly (followed by "/") are
            # rewritten. substr() is 1-based, so the text after the old prefix starts at
            # len(old) + 1.
            conn.execute(
                "UPDATE runs SET artifact_uri = ? || substr(artifact_uri, ?) "
                "WHERE experiment_id = ? AND substr(artifact_uri, 1, ?) = ?",
                (
                    new_location,
                    len(old_location) + 1,
                    experiment_id,
                    len(old_location) + 1,
                    old_location + "/",
                ),
            )
        return True
    finally:
        conn.close()  # `with conn` only commits; it does not close the connection
