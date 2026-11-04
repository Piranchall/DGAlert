"""Project-wide settings: paths, random seed, data identifiers and MLflow tracking.

Every other module imports its paths and constants from here, so nothing is hard-coded twice.
All paths are built with pathlib from the repo root, so they work on Windows and Linux.
"""

from pathlib import Path

# Repo root = the folder that contains src/.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"  # gitignored: UMUDGA lists + Tranco top-1M
SAMPLE_DIR: Path = DATA_DIR / "sample"  # committed: training pool
REFERENCE_DIR: Path = DATA_DIR / "reference"  # committed: English word list
PROCESSED_DIR: Path = DATA_DIR / "processed"  # gitignored: parquet splits + split report

# English reference data (built by scripts/build_wordlist.py from SCOWL, committed)
WORDLIST_PATH: Path = REFERENCE_DIR / "english_words.txt"
WORDLIST_LICENSE_PATH: Path = REFERENCE_DIR / "SCOWL-LICENSE.txt"

# Model artifacts
MODELS_DIR: Path = ROOT_DIR / "models"
LATEST_MODEL_POINTER: Path = MODELS_DIR / "latest.json"  # MODEL_POINTER env var can override

# Reproducibility
SEED: int = 42

# Tranco list ID used for benign domains (fill in when the list is downloaded).
TRANCO_LIST_ID: str = "TODO"

# UMUDGA families kept out of training to simulate future, unseen malware (~30%, chosen by team).
HELD_OUT_FAMILIES: list[str] = []

# MLflow: MLflow 3.x recommends a SQLite backend for local tracking (the ./mlruns file store is
# legacy). Both the database and the artifact folder live inside the repo and are committed.
MLFLOW_DB_PATH: Path = ROOT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_DIR: Path = ROOT_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME: str = "dgalert-m1"
