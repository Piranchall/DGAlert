"""Smoke tests for src.config: paths stay inside the repo and constants have the expected values."""

from pathlib import Path

from src import config

PATHS_IN_REPO = [
    config.DATA_DIR,
    config.RAW_DIR,
    config.SAMPLE_DIR,
    config.REFERENCE_DIR,
    config.PROCESSED_DIR,
    config.MODELS_DIR,
    config.LATEST_MODEL_POINTER,
    config.MLFLOW_DB_PATH,
    config.MLFLOW_ARTIFACT_DIR,
]


def test_root_dir_is_repo_root() -> None:
    """ROOT_DIR must be the folder holding src/config.py and requirements.txt."""
    assert (config.ROOT_DIR / "src" / "config.py").is_file()
    assert (config.ROOT_DIR / "requirements.txt").is_file()


def test_all_paths_are_absolute_and_inside_repo() -> None:
    for path in PATHS_IN_REPO:
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert path.is_relative_to(config.ROOT_DIR), f"{path} is outside the repo"


def test_committed_folders_exist() -> None:
    """Folders tracked in git (via .gitkeep) must exist in a clean clone."""
    for folder in (config.SAMPLE_DIR, config.REFERENCE_DIR, config.MODELS_DIR):
        assert folder.is_dir(), f"{folder} is missing"


def test_mlflow_tracking_uri_points_to_repo_sqlite_db() -> None:
    assert config.MLFLOW_TRACKING_URI.startswith("sqlite:///")
    db_path = Path(config.MLFLOW_TRACKING_URI.removeprefix("sqlite:///"))
    assert db_path == config.MLFLOW_DB_PATH
    assert config.MLFLOW_EXPERIMENT_NAME == "dgalert-m1"


def test_reproducibility_constants() -> None:
    assert config.SEED == 42
    assert isinstance(config.HELD_OUT_FAMILIES, list)
