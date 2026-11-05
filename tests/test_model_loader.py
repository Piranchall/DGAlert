"""Tests for src.model_loader using the tiny fixture model (see conftest.py)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.features import FEATURE_NAMES, extract_features
from src.model_loader import (
    POINTER_ENV_VAR,
    REQUIRED_FIELDS,
    ModelNotFoundError,
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
