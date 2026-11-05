"""Load the current model via models/latest.json (overridable with the MODEL_POINTER env var).

latest.json is written by src.train. Its "path" is relative to the repo root when the model is
inside the repo (so the pointer works on any machine), or absolute otherwise (e.g. in tests).
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib

from src import config

POINTER_ENV_VAR = "MODEL_POINTER"

# Fields the loader and the API need. src.train also writes f1_seen, f1_future, etc.
REQUIRED_FIELDS: tuple[str, ...] = ("path", "version", "model_type")


class ModelNotFoundError(Exception):
    """latest.json or the model file it points to is missing, unreadable or incomplete."""


def resolve_pointer(pointer: Path | None = None) -> Path:
    """The latest.json to use: the argument, else $MODEL_POINTER, else models/latest.json."""
    if pointer is not None:
        return Path(pointer)
    env_value = os.environ.get(POINTER_ENV_VAR)
    if env_value:
        return Path(env_value)
    return config.LATEST_MODEL_POINTER


def resolve_model_path(path_value: str) -> Path:
    """Absolute paths are used as they are; relative paths are relative to the repo root."""
    path = Path(path_value)
    return path if path.is_absolute() else config.ROOT_DIR / path


def read_pointer(pointer: Path) -> dict[str, Any]:
    """Read latest.json and check it is a JSON object with all REQUIRED_FIELDS."""
    if not pointer.is_file():
        raise ModelNotFoundError(
            f"Model pointer {pointer} not found: train a model first (python -m src.train)"
        )
    try:
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelNotFoundError(f"Model pointer {pointer} is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ModelNotFoundError(f"Model pointer {pointer} must contain a JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise ModelNotFoundError(f"Model pointer {pointer} is missing fields: {missing}")
    return metadata


def load_model(pointer_path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the model named in latest.json. Returns (fitted estimator, latest.json contents).

    Raises ModelNotFoundError if the pointer or the model file is missing, or the pointer is
    malformed.
    """
    pointer = resolve_pointer(pointer_path)
    metadata = read_pointer(pointer)
    model_path = resolve_model_path(metadata["path"])
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model file {model_path} (from {pointer}) not found")
    model = joblib.load(model_path)
    return model, metadata
