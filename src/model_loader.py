"""Load the served model via a latest.json pointer.

The default pointer is the committed production model, models/production/latest.json. The
MODEL_POINTER env var overrides it, e.g. MODEL_POINTER=models/latest.json serves the candidate
that src.train wrote locally.

A pointer's "path" is relative to the repo root when the model is inside the repo (so the pointer
works on any machine and from any working directory), or absolute otherwise (e.g. in tests).
A relative MODEL_POINTER is also resolved against the repo root.

When the pointer records "sha256", the model file's hash is checked before it is unpickled and a
mismatch is refused (ModelIntegrityError). A pointer without "sha256" still loads, with a warning.
A warning is also logged when the running scikit-learn differs from the recorded "sklearn_version".
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import sklearn

from src import config

logger = logging.getLogger(__name__)

POINTER_ENV_VAR = "MODEL_POINTER"

# Fields the loader and the API need. Pointers also carry metrics, sha256, versions, etc.
REQUIRED_FIELDS: tuple[str, ...] = ("path", "version", "model_type")

HASH_CHUNK_BYTES = 1 << 20


class ModelNotFoundError(Exception):
    """latest.json or the model file it points to is missing, unreadable or incomplete."""


class ModelIntegrityError(ModelNotFoundError):
    """The model file's SHA-256 differs from the one recorded in latest.json."""


def resolve_pointer(pointer: Path | None = None) -> Path:
    """The pointer to use: the argument, else $MODEL_POINTER, else the production pointer.

    A relative path is taken relative to the repo root, not the current working directory.
    """
    if pointer is None:
        env_value = os.environ.get(POINTER_ENV_VAR)
        if not env_value:
            return config.PRODUCTION_MODEL_POINTER
        pointer = Path(env_value)
    pointer = Path(pointer)
    return pointer if pointer.is_absolute() else config.ROOT_DIR / pointer


def resolve_model_path(path_value: str) -> Path:
    """Absolute paths are used as they are; relative paths are relative to the repo root."""
    path = Path(path_value)
    return path if path.is_absolute() else config.ROOT_DIR / path


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_sha256(model_path: Path, metadata: dict[str, Any], pointer: Path) -> None:
    """Refuse a model file whose hash differs from the pointer's sha256; warn if there is none."""
    expected = metadata.get("sha256")
    if not expected:
        logger.warning(
            "Model pointer %s has no sha256: %s is loaded without an integrity check",
            pointer,
            model_path,
        )
        return
    actual = file_sha256(model_path)
    if actual != str(expected).lower():
        raise ModelIntegrityError(
            f"Model file {model_path} has SHA-256 {actual}, but {pointer} expects {expected}: "
            "refusing to load it"
        )


def warn_on_sklearn_mismatch(metadata: dict[str, Any]) -> None:
    """A model pickled with another scikit-learn version may load but behave differently."""
    recorded = metadata.get("sklearn_version")
    if recorded and recorded != sklearn.__version__:
        logger.warning(
            "Model %s was saved with scikit-learn %s but %s is installed",
            metadata["version"],
            recorded,
            sklearn.__version__,
        )


def load_model(pointer_path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the model named in latest.json. Returns (fitted estimator, latest.json contents).

    Raises ModelNotFoundError if the pointer or the model file is missing, or the pointer is
    malformed, and ModelIntegrityError (a subclass) if the file fails the SHA-256 check.
    """
    pointer = resolve_pointer(pointer_path)
    metadata = read_pointer(pointer)
    model_path = resolve_model_path(metadata["path"])
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model file {model_path} (from {pointer}) not found")
    verify_sha256(model_path, metadata, pointer)
    warn_on_sklearn_mismatch(metadata)
    model = joblib.load(model_path)
    return model, metadata
