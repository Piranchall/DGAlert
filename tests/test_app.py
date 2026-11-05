"""Tests for the FastAPI app (src.app) with the tiny fixture model (see conftest.py)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from src.app import app
from src.model_loader import POINTER_ENV_VAR

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_with_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "test_fixture",
    }


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App started with MODEL_POINTER at a file that does not exist."""
    monkeypatch.setenv(POINTER_ENV_VAR, str(tmp_path / "missing.json"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "model_loaded": False}


def test_predict_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.post("/predict", json={"domain": "google.com"})
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_valid(client: TestClient) -> None:
    response = client.post("/predict", json={"domain": "HTTP://Example.COM/path"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"domain", "label", "probability", "model_version"}
    assert body["domain"] == "example.com"
    assert body["label"] in {"benign", "dga"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["probability"] == round(body["probability"], 4)
    assert body["model_version"] == "test_fixture"


def test_label_matches_threshold(client: TestClient) -> None:
    for domain in ["google.com", "pidfksipycmyg.com", "mail.google.co.uk"]:
        body = client.post("/predict", json={"domain": domain}).json()
        expected = "dga" if body["probability"] >= config.DGA_THRESHOLD else "benign"
        assert body["label"] == expected


def test_punycode_domain_is_accepted(client: TestClient) -> None:
    response = client.post("/predict", json={"domain": "xn--bcher-kva.de"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"domain": ""}, "empty"),
        ({"domain": "   "}, "empty"),
        ({"domain": "localhost"}, "at least one dot"),
        ({"domain": "exa_mple.com"}, "letters, digits and hyphens"),
        ({"domain": "exa mple.com"}, "letters, digits and hyphens"),
        ({"domain": "a" * 64 + ".com"}, "1-63 characters"),
        ({"domain": "example..com"}, "1-63 characters"),
        ({"domain": ".".join(["a" * 60] * 5)}, "longer than 253"),
        ({"domain": "-example.com"}, "hyphen"),
        ({"domain": "example-.com"}, "hyphen"),
        ({"domain": "bücher.de"}, "ASCII"),
        ({"domain": "co.uk"}, "public suffix"),
        ({"domain": "pidfksipycmyg.local"}, "no known public suffix"),
        ({"domain": "abc.internal"}, "no known public suffix"),
        ({"domain": "8.8.8.8"}, "no known public suffix"),
        ({}, "Field required"),
        ({"domain": 123}, "valid string"),
        ({"domain": None}, "valid string"),
    ],
    ids=[
        "empty",
        "whitespace",
        "no-dot",
        "underscore",
        "space",
        "label-over-63",
        "empty-label",
        "total-over-253",
        "leading-hyphen",
        "trailing-hyphen",
        "non-ascii",
        "bare-suffix",
        "local-suffix",
        "internal-suffix",
        "ip-address",
        "missing-field",
        "wrong-type-int",
        "wrong-type-null",
    ],
)
def test_predict_invalid_input(client: TestClient, payload: dict, message: str) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert message in response.json()["detail"][0]["msg"]


def test_openapi_has_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "DGAlert"
    assert schema["components"]["schemas"]["PredictRequest"]["examples"]
    assert schema["components"]["schemas"]["PredictResponse"]["examples"]


# ---------------------------------------------------------------------------
# Sanity check on the real trained model (skipped when models/latest.json is absent, e.g. in CI)
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.skipif(
    not config.LATEST_MODEL_POINTER.exists(), reason="no trained model in models/latest.json"
)
@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("google.com", "benign"),
        ("wikipedia.org", "benign"),
        ("github.com", "benign"),
        ("teco.com.tw", "benign"),
        ("moiqidfvjjhjmndqp.in", "dga"),
        ("qxxsoxcwrcplo.net", "dga"),
    ],
)
def test_real_model_sanity(monkeypatch: pytest.MonkeyPatch, domain: str, expected: str) -> None:
    monkeypatch.setenv(POINTER_ENV_VAR, str(config.LATEST_MODEL_POINTER))
    with TestClient(app) as real_client:
        response = real_client.post("/predict", json={"domain": domain})
    assert response.status_code == 200
    assert response.json()["label"] == expected
