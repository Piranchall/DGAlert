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
LATEST_MODEL_POINTER: Path = MODELS_DIR / "latest.json"  # MODEL_POINTER env var can override

# Reproducibility
SEED: int = 42

# A domain is labelled "dga" when the model's P(dga) is at least this value (CLAUDE.md decision 11).
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
# legacy). Both the database and the artifact folder live inside the repo and are committed.
MLFLOW_DB_PATH: Path = ROOT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_DIR: Path = ROOT_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME: str = "dgalert-m1"
