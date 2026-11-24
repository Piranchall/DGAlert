"""Tests for src.model_loader: the tiny fixture model (see conftest.py) and the production model."""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest
import sklearn

from src import config
from src.features import FEATURE_NAMES, extract_features
from src.model_loader import (
    POINTER_ENV_VAR,
    REQUIRED_FIELDS,
    ModelIntegrityError,
    ModelNotFoundError,
    file_sha256,
    load_model,
    resolve_model_path,
    resolve_pointer,
)


def test_load_from_explicit_pointer(model_pointer: Path) -> None:
    model, metadata = load_model(model_pointer)
    assert metadata["version"] == "test_fixture"
    assert set(REQUIRED_FIELDS) <= metadata.keys()
    x = pd.DataFrame([extract_features("google.com")], columns=FEATURE_NAMES)
    proba = model.predict_proba(x)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_env_var_override(model_pointer: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POINTER_ENV_VAR, str(model_pointer))
    assert resolve_pointer() == model_pointer
    _, metadata = load_model()
    assert metadata["version"] == "test_fixture"


def test_default_pointer_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POINTER_ENV_VAR, raising=False)
    assert resolve_pointer() == config.PRODUCTION_MODEL_POINTER


def test_relative_env_pointer_resolves_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POINTER_ENV_VAR, "models/latest.json")
    assert resolve_pointer() == config.LATEST_MODEL_POINTER


def test_relative_model_path_resolves_from_repo_root() -> None:
    assert resolve_model_path("models/model_x.joblib") == config.ROOT_DIR / "models/model_x.joblib"


def test_missing_pointer_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="not found"):
        load_model(tmp_path / "latest.json")


def test_missing_model_file_raises(tmp_path: Path) -> None:
    pointer = tmp_path / "latest.json"
    missing = (tmp_path / "model_gone.joblib").as_posix()
    pointer.write_text(json.dumps({"path": missing, "version": "v", "model_type": "gb"}))
    with pytest.raises(ModelNotFoundError, match="Model file"):
        load_model(pointer)


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[1, 2, 3]",
        json.dumps({"version": "v", "model_type": "gb"}),  # no "path"
    ],
    ids=["invalid-json", "not-an-object", "missing-path"],
)
def test_malformed_pointer_raises(tmp_path: Path, content: str) -> None:
    pointer = tmp_path / "latest.json"
    pointer.write_text(content, encoding="utf-8")
    with pytest.raises(ModelNotFoundError):
        load_model(pointer)


# ---------------------------------------------------------------------------
# SHA-256 check and version warnings
# ---------------------------------------------------------------------------


def copy_fixture(model_pointer: Path, folder: Path, **changes: object) -> Path:
    """Copy the fixture model and its pointer into folder, apply changes to the pointer fields."""
    metadata = json.loads(model_pointer.read_text(encoding="utf-8"))
    model_path = folder / Path(metadata["path"]).name
    shutil.copyfile(metadata["path"], model_path)
    metadata |= {"path": model_path.as_posix(), **changes}
    metadata = {key: value for key, value in metadata.items() if value is not None}
    pointer = folder / "latest.json"
    pointer.write_text(json.dumps(metadata), encoding="utf-8")
    return pointer


def test_matching_sha256_loads(model_pointer: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.model_loader"):
        _, metadata = load_model(model_pointer)
    assert len(metadata["sha256"]) == 64
    assert caplog.records == []


def test_tampered_model_file_is_refused(model_pointer: Path, tmp_path: Path) -> None:
    pointer = copy_fixture(model_pointer, tmp_path)
    model_path = Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
    data = bytearray(model_path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # flip one byte in the middle of the pickle
    model_path.write_bytes(bytes(data))

    with pytest.raises(ModelIntegrityError, match="refusing to load"):
        load_model(pointer)


def test_sha256_mismatch_is_a_model_not_found_error(model_pointer: Path, tmp_path: Path) -> None:
    pointer = copy_fixture(model_pointer, tmp_path, sha256="0" * 64)
    with pytest.raises(ModelNotFoundError, match="0{64}"):
        load_model(pointer)


def test_pointer_without_sha256_loads_with_warning(
    model_pointer: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pointer = copy_fixture(model_pointer, tmp_path, sha256=None)
    with caplog.at_level(logging.WARNING, logger="src.model_loader"):
        _, metadata = load_model(pointer)
    assert "sha256" not in metadata
    assert "has no sha256" in caplog.text


def test_sklearn_version_mismatch_warns(
    model_pointer: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pointer = copy_fixture(model_pointer, tmp_path, sklearn_version="0.0.1")
    with caplog.at_level(logging.WARNING, logger="src.model_loader"):
        load_model(pointer)
    assert f"scikit-learn 0.0.1 but {sklearn.__version__} is installed" in caplog.text


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"dgalert")
    assert file_sha256(path) == hashlib.sha256(b"dgalert").hexdigest()


# ---------------------------------------------------------------------------
# The committed production model
# ---------------------------------------------------------------------------

PRODUCTION_FIELDS = {
    "path",
    "version",
    "model_type",
    "sha256",
    "mlflow_run_id",
    "sklearn_version",
    "python_version",
    "feature_names",
    "phased_in_families",
}


def production_metadata() -> dict:
    return json.loads(config.PRODUCTION_MODEL_POINTER.read_text(encoding="utf-8"))


def test_production_pointer_has_all_fields() -> None:
    metadata = production_metadata()
    assert metadata.keys() >= PRODUCTION_FIELDS
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["phased_in_families"] == []
    assert not Path(metadata["path"]).is_absolute()


def test_production_model_file_matches_recorded_sha256() -> None:
    metadata = production_metadata()
    assert file_sha256(resolve_model_path(metadata["path"])) == metadata["sha256"]


def test_production_model_loads_from_another_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths resolve against the repo root from config, not the current working directory."""
    monkeypatch.delenv(POINTER_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    model, metadata = load_model()
    assert metadata["version"] == production_metadata()["version"]
    assert list(model.feature_names_in_) == list(FEATURE_NAMES)
