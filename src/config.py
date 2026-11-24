"""Project-wide settings: paths, random seed, data identifiers and MLflow tracking.

Every other module imports its paths and constants from here, so nothing is hard-coded twice.
All paths are built with pathlib from the repo root, so they work on Windows and Linux.
"""

import os
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

# Raw inputs (gitignored, downloaded manually)
UMUDGA_DIR: Path = RAW_DIR / "umudga"  # one folder per family, each with list/10000.txt
TRANCO_PATH: Path = RAW_DIR / "tranco" / "top-1m.csv"  # rank,domain; no header; CRLF

# Committed sample (built by scripts/make_sample.py) = the training pool
UMUDGA_SAMPLE_PATH: Path = SAMPLE_DIR / "umudga_sample.csv.gz"  # domain, family
TRANCO_SAMPLE_PATH: Path = SAMPLE_DIR / "tranco_sample.csv.gz"  # domain, tranco_rank
SAMPLE_INFO_PATH: Path = SAMPLE_DIR / "SAMPLE_INFO.md"
SAMPLE_PER_FAMILY: int = 3000  # max domains sampled per UMUDGA family

# Processed splits (built by src/prepare_data.py)
TRAIN_PATH: Path = PROCESSED_DIR / "train.parquet"
TEST_SEEN_PATH: Path = PROCESSED_DIR / "test_seen.parquet"
TEST_FUTURE_PATH: Path = PROCESSED_DIR / "test_future.parquet"
SPLIT_REPORT_PATH: Path = PROCESSED_DIR / "split_report.json"

# Model artifacts
MODELS_DIR: Path = ROOT_DIR / "models"
# Written by src.train for a locally trained candidate (gitignored). Serve it with
# MODEL_POINTER=models/latest.json.
LATEST_MODEL_POINTER: Path = MODELS_DIR / "latest.json"
# The committed production model the API serves by default (MODEL_POINTER env var can override).
PRODUCTION_DIR: Path = MODELS_DIR / "production"
PRODUCTION_MODEL_POINTER: Path = PRODUCTION_DIR / "latest.json"

# Reproducibility
SEED: int = 42

# A domain is labelled "dga" when P(dga) is at least this value. Fixed in advance, not tuned on the
# test sets, so test scores stay unbiased; the API and the evaluation use the same cut-off.
DGA_THRESHOLD: float = 0.5

# Tranco list used for benign domains: https://tranco-list.eu/list/746WX/1000000
TRANCO_LIST_ID: str = "746WX"

# UMUDGA families never used anywhere. corebot, symmi, kraken_v1 and kraken_v2 put the generated
# string in a dynamic-DNS subdomain (e.g. x.ddns.net), which a second-level-label classifier cannot
# see. legit is UMUDGA's benign list; it overlaps Tranco, our benign source.
EXCLUDED_FAMILIES: list[str] = ["corebot", "symmi", "kraken_v1", "kraken_v2", "legit"]

# UMUDGA families kept out of training to simulate future, unseen malware (~30% of the 46 usable
# families, chosen by the team). Their rows go only to test_future.
HELD_OUT_FAMILIES: list[str] = [
    "matsnu",
    "nymaim",
    "banjori",
    "sisron",
    "simda",
    "dyre",
    "padcrypt",
    "shiotob",
    "necurs",
    "qakbot",
    "tinba",
    "murofet_v1",
    "murofet_v2",
    "murofet_v3",
]

# MLflow: MLflow 3.x recommends a SQLite backend for local tracking (the ./mlruns file store is
# legacy).
#
# New runs go to a local, gitignored store. The env vars below override it (read when training
# starts, see mlflow_tracking_uri() / mlflow_artifact_dir()). DGALERT_ names, not MLflow's own
# MLFLOW_TRACKING_URI, so a global MLflow setting for another project cannot redirect our runs.
MLFLOW_LOCAL_DIR: Path = ROOT_DIR / "mlflow_local"
MLFLOW_LOCAL_DB: Path = MLFLOW_LOCAL_DIR / "mlflow.db"
MLFLOW_LOCAL_ARTIFACT_DIR: Path = MLFLOW_LOCAL_DIR / "mlruns"
TRACKING_URI_ENV_VAR = "DGALERT_MLFLOW_TRACKING_URI"
ARTIFACT_DIR_ENV_VAR = "DGALERT_MLFLOW_ARTIFACT_DIR"

# The committed Milestone 1 store is a read-only snapshot: nothing writes to these files. Tools
# that need it (src.m1_snapshot, src.report) work on a copy in M1_SNAPSHOT_WORKDIR (gitignored).
# The hash is of the committed blob (git cat-file blob HEAD:mlflow.db), checked before copying.
M1_SNAPSHOT_DB: Path = ROOT_DIR / "mlflow.db"
M1_SNAPSHOT_ARTIFACT_DIR: Path = ROOT_DIR / "mlruns"
M1_SNAPSHOT_DB_SHA256: str = "fadc3edc61edd90dde3ce0031fa9d698402d3d88a22c7534cf860a53427d8bbd"
M1_SNAPSHOT_WORKDIR: Path = ROOT_DIR / "mlflow_m1"

MLFLOW_EXPERIMENT_NAME: str = "dgalert-m1"
MODEL_TYPES: list[str] = ["logreg", "rf", "gb"]  # one MLflow parent run each (tag model_type)
PER_FAMILY_FILE: str = "per_family_recall.json"  # MLflow artifact name in every parent run

# Results copied out of MLflow into the repo (MLflow artifact paths are machine-specific).
# Built by src/report.py from the M1 snapshot (latest finished parent run of each model type) and
# from the run of the production model; committed.
REPORTS_DIR: Path = ROOT_DIR / "reports"
RESULTS_REPORT_PATH: Path = REPORTS_DIR / "m1_results.json"
PER_FAMILY_REPORT_PATH: Path = REPORTS_DIR / "per_family_recall.json"
PRODUCTION_REPORT_PATH: Path = REPORTS_DIR / "production_model.json"


def sqlite_uri(db_path: Path) -> str:
    """sqlite:/// tracking URI for a database file."""
    return f"sqlite:///{Path(db_path).as_posix()}"


def mlflow_tracking_uri() -> str:
    """Tracking URI for new runs: $DGALERT_MLFLOW_TRACKING_URI, else the local store."""
    return os.environ.get(TRACKING_URI_ENV_VAR) or sqlite_uri(MLFLOW_LOCAL_DB)


def mlflow_artifact_dir() -> Path:
    """Artifact folder for new runs: $DGALERT_MLFLOW_ARTIFACT_DIR, else the local store's."""
    value = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    if not value:
        return MLFLOW_LOCAL_ARTIFACT_DIR
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path
