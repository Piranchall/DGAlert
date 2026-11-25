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

# ---------------------------------------------------------------------------
# Retraining data (Milestone 2), built by src/prepare_retrain.py from the three M1 splits, which it
# does not change.
# ---------------------------------------------------------------------------
# Rows are assigned by a hash of their second-level label (sld): an sld is selected when
# sha256("<salt>:<sld>"), read as a fraction of 2**64 from its first 8 bytes, is below the fraction.
# The result depends only on the sld string, so it stays the same when other rows or families
# change, and all rows of one sld always land on the same side.
FROZEN_EVAL_SALT: str = "frozen-eval"
FROZEN_EVAL_FRACTION: float = 0.2  # of each held-out family: the frozen eval slice
VALIDATION_SALT: str = "validation"
VALIDATION_FRACTION: float = 0.1  # of train: the validation split for model selection

# Outputs (gitignored, like the M1 splits)
TRAIN_FIT_PATH: Path = PROCESSED_DIR / "train_fit.parquet"  # train minus validation
VALIDATION_PATH: Path = PROCESSED_DIR / "validation.parquet"
PHASE_IN_POOL_PATH: Path = PROCESSED_DIR / "phase_in_pool.parquet"  # held-out rows to phase in
EVAL_FROZEN_PATH: Path = PROCESSED_DIR / "eval_frozen.parquet"  # test_seen + frozen eval slices
RETRAIN_SPLIT_REPORT_PATH: Path = PROCESSED_DIR / "retrain_split_report.json"
# Committed: the frozen eval set's content hash and row counts per family.
EVAL_FROZEN_REPORT_PATH: Path = REPORTS_DIR / "eval_frozen.json"

# Held-out families are added to training one batch per retraining round, in this order. Batch
# order follows the M1 gb recall on test_future (reports/per_family_recall.json), worst first:
#   1. matsnu 0.049 (word-based labels)
#   2. nymaim 0.211, simda 0.432
#   3. dyre 0.646 (hex-character labels, unlike any training family), shiotob 0.740
# The other held-out families are never phased in (PHASE_IN_CONTROL_FAMILIES): their recall is
# 0.98 to 1.00, so a batch of them could not improve recall by the gate's margin, and keeping them
# unseen still measures generalization to families never trained on.
PHASE_IN_BATCHES: list[list[str]] = [["matsnu"], ["nymaim", "simda"], ["dyre", "shiotob"]]
PHASE_IN_CONTROL_FAMILIES: list[str] = [
    family for family in HELD_OUT_FAMILIES if not any(family in batch for batch in PHASE_IN_BATCHES)
]

# ---------------------------------------------------------------------------
# Promotion gate (src/gate.py): a retrained challenger replaces the champion only if all hold,
# with both models scored on the same frozen eval set rows. Boundaries are inclusive.
# ---------------------------------------------------------------------------
GATE_MIN_NEW_FAMILY_RECALL_GAIN: float = 0.02  # pooled recall on the newly added families
GATE_MAX_SEEN_F1_DROP: float = 0.01  # F1 on the test_seen rows (training families + benign)
GATE_MAX_FPR_RISE: float = 0.005  # false-positive rate on the benign rows
# Pooled recall on the families phased in during earlier rounds (a sixth criterion, beyond the
# slice's five). Those families are held-out families, so their rows are not in the seen-F1 rows at
# all: without this criterion a collapse on a ~600-row earlier family would go unnoticed (it would
# move F1 over all ~46.8k frozen rows only slightly, and that F1 is not gated).
GATE_MAX_PRIOR_RECALL_DROP: float = 0.02

# ---------------------------------------------------------------------------
# Retraining loop (src/retrain.py)
# ---------------------------------------------------------------------------
MLFLOW_KAIZEN_EXPERIMENT_NAME: str = "dgalert-m2-kaizen"
RETRAIN_REPORTS_DIR: Path = REPORTS_DIR / "retrain"  # round_<n>.json (committed with a promotion)
RETRAIN_MODEL_TYPE: str = "hgb"
# Fixed HistGradientBoostingClassifier parameters, chosen once by scripts/select_hgb_params.py on
# the validation split (fit on train_fit, F1 on validation; results in
# reports/retrain/hgb_param_search.json). Best F1 0.8902 (lr 0.05, 400 iterations, 63 leaves);
# three points were within 0.001 of it and the cheapest of those was chosen: F1 0.8894 with half
# the iterations. Fixed parameters keep a round deterministic and as fast as one fit (about 10 s).
HGB_PARAMS: dict = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "early_stopping": False,
    "random_state": SEED,
}

# Domains every served model must classify correctly (the gate's sanity criterion and the sanity
# tests on the production model).
SANITY_DOMAINS: list[tuple[str, str]] = [
    ("google.com", "benign"),
    ("wikipedia.org", "benign"),
    ("github.com", "benign"),
    ("teco.com.tw", "benign"),
    ("moiqidfvjjhjmndqp.in", "dga"),
    ("qxxsoxcwrcplo.net", "dga"),
]


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
