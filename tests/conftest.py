"""Shared fixtures: a tiny real model trained on data/sample, and a TestClient for the API.

No MLflow, no models/ folder and no data/raw are used, so this runs in CI on a clean clone.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingClassifier

from src import config
from src.app import app
from src.features import FEATURE_NAMES, extract_features
from src.model_loader import POINTER_ENV_VAR

ROWS_PER_CLASS = 200
FIXTURE_VERSION = "test_fixture"


def sample_domains() -> pd.DataFrame:
    """ROWS_PER_CLASS benign (Tranco) + ROWS_PER_CLASS dga (training families) from data/sample."""
    benign = pd.read_csv(config.TRANCO_SAMPLE_PATH).sample(ROWS_PER_CLASS, random_state=config.SEED)
    dga = pd.read_csv(config.UMUDGA_SAMPLE_PATH)
    excluded = set(config.HELD_OUT_FAMILIES) | set(config.EXCLUDED_FAMILIES)
    dga = dga[~dga["family"].isin(excluded)].sample(ROWS_PER_CLASS, random_state=config.SEED)
    return pd.concat(
        [benign[["domain"]].assign(y=0), dga[["domain"]].assign(y=1)], ignore_index=True
    )


@pytest.fixture(scope="session")
def sample_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Features (FEATURE_NAMES columns) and target (1 = dga) for the sample_domains() rows."""
    data = sample_domains()
    x = pd.DataFrame([extract_features(d) for d in data["domain"]], columns=FEATURE_NAMES)
    return x, data["y"]


@pytest.fixture(scope="session")
def model_pointer(
    tmp_path_factory: pytest.TempPathFactory, sample_xy: tuple[pd.DataFrame, pd.Series]
) -> Iterator[Path]:
    """Train a 20-tree GB model, save it + latest.json in a temp folder, set MODEL_POINTER."""
    folder = tmp_path_factory.mktemp("model")
    x, y = sample_xy
    model = GradientBoostingClassifier(n_estimators=20, random_state=config.SEED)
    model.fit(x, y)

    model_path = folder / f"model_{FIXTURE_VERSION}.joblib"
    joblib.dump(model, model_path)
    pointer = folder / "latest.json"
    latest = {
        "path": model_path.as_posix(),  # absolute: the model is outside the repo
        "version": FIXTURE_VERSION,
        "model_type": "gb",
        "f1_seen": 0.0,
        "f1_future": 0.0,
        "mlflow_run_id": "none",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    pointer.write_text(json.dumps(latest), encoding="utf-8")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(POINTER_ENV_VAR, str(pointer))
        yield pointer


@pytest.fixture
def client(model_pointer: Path) -> Iterator[TestClient]:
    """TestClient whose lifespan loaded the fixture model.

    Function-scoped on purpose: each `with TestClient(app)` re-runs the lifespan, so a test that
    starts the app with another MODEL_POINTER cannot leave a shared client without a model.
    """
    with TestClient(app) as test_client:
        yield test_client
