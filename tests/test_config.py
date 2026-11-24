"""Smoke tests for src.config: paths stay inside the repo and constants have the expected values."""

from pathlib import Path

import pytest

from src import config

PATHS_IN_REPO = [
    config.DATA_DIR,
    config.RAW_DIR,
    config.SAMPLE_DIR,
    config.REFERENCE_DIR,
    config.PROCESSED_DIR,
    config.WORDLIST_PATH,
    config.WORDLIST_LICENSE_PATH,
    config.UMUDGA_DIR,
    config.TRANCO_PATH,
    config.UMUDGA_SAMPLE_PATH,
    config.TRANCO_SAMPLE_PATH,
    config.SAMPLE_INFO_PATH,
    config.TRAIN_PATH,
    config.TEST_SEEN_PATH,
    config.TEST_FUTURE_PATH,
    config.SPLIT_REPORT_PATH,
    config.MODELS_DIR,
    config.LATEST_MODEL_POINTER,
    config.MLFLOW_LOCAL_DB,
    config.MLFLOW_LOCAL_ARTIFACT_DIR,
    config.M1_SNAPSHOT_DB,
    config.M1_SNAPSHOT_ARTIFACT_DIR,
    config.M1_SNAPSHOT_WORKDIR,
    config.PRODUCTION_REPORT_PATH,
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


def test_new_runs_default_to_the_local_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.TRACKING_URI_ENV_VAR, raising=False)
    monkeypatch.delenv(config.ARTIFACT_DIR_ENV_VAR, raising=False)
    uri = config.mlflow_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert Path(uri.removeprefix("sqlite:///")) == config.MLFLOW_LOCAL_DB
    assert config.mlflow_artifact_dir() == config.MLFLOW_LOCAL_ARTIFACT_DIR
    assert config.MLFLOW_LOCAL_DB != config.M1_SNAPSHOT_DB
    assert config.MLFLOW_EXPERIMENT_NAME == "dgalert-m1"


def test_mlflow_env_vars_override_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.TRACKING_URI_ENV_VAR, "sqlite:///elsewhere/mlflow.db")
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV_VAR, str(tmp_path / "artifacts"))
    assert config.mlflow_tracking_uri() == "sqlite:///elsewhere/mlflow.db"
    assert config.mlflow_artifact_dir() == tmp_path / "artifacts"

    monkeypatch.setenv(config.ARTIFACT_DIR_ENV_VAR, "relative/artifacts")
    assert config.mlflow_artifact_dir() == config.ROOT_DIR / "relative/artifacts"


def test_mlflow_working_folders_are_gitignored() -> None:
    ignored = (config.ROOT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "mlflow_local/" in ignored
    assert "mlflow_m1/" in ignored


def test_reproducibility_constants() -> None:
    assert config.SEED == 42
    assert isinstance(config.HELD_OUT_FAMILIES, list)


def test_family_lists_are_disjoint_and_unique() -> None:
    held, excluded = config.HELD_OUT_FAMILIES, config.EXCLUDED_FAMILIES
    assert len(set(held)) == len(held)
    assert len(set(excluded)) == len(excluded)
    assert not set(held) & set(excluded), "a family cannot be both held out and excluded"
