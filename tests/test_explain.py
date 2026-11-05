"""Tests for src.explain and the /explain endpoint, using the tiny fixture models (conftest.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import EXPLAIN_TOP_K, app
from src.explain import DomainExplainer, build_explainer
from src.features import FEATURE_NAMES, feature_frame
from src.model_loader import POINTER_ENV_VAR, load_model
from src.train import BASIC_FEATURES, build_logreg, build_rf

DOMAINS = ["google.com", "pidfksipycmyg.com", "mail.wikipedia.org", "x7k2-q9z.net"]


@pytest.fixture(scope="module")
def gb_model(model_pointer: Path):
    model, _ = load_model(model_pointer)
    return model


# ---------------------------------------------------------------------------
# /explain endpoint
# ---------------------------------------------------------------------------


def test_explain_response_shape(client: TestClient) -> None:
    response = client.post("/explain", json={"domain": "pidfksipycmyg.com"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "domain",
        "label",
        "probability",
        "model_version",
        "base_value",
        "top_contributing_features",
    }
    items = body["top_contributing_features"]
    assert len(items) == EXPLAIN_TOP_K
    for item in items:
        assert set(item) == {"feature", "value", "contribution"}
        assert item["feature"] in FEATURE_NAMES
    sizes = [abs(item["contribution"]) for item in items]
    assert sizes == sorted(sizes, reverse=True)


@pytest.mark.parametrize("domain", ["HTTP://Example.COM/path", *DOMAINS])
def test_explain_matches_predict(client: TestClient, domain: str) -> None:
    predicted = client.post("/predict", json={"domain": domain}).json()
    explained = client.post("/explain", json={"domain": domain}).json()
    for key in ("domain", "label", "probability", "model_version"):
        assert explained[key] == predicted[key]


@pytest.mark.parametrize("domain", ["", "localhost", "-bad.com", "co.uk", "bücher.de"])
def test_explain_invalid_domain(client: TestClient, domain: str) -> None:
    assert client.post("/explain", json={"domain": domain}).status_code == 422


def test_explain_missing_field(client: TestClient) -> None:
    assert client.post("/explain", json={}).status_code == 422


def test_explain_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POINTER_ENV_VAR, str(tmp_path / "missing.json"))
    with TestClient(app) as no_model_client:
        response = no_model_client.post("/explain", json={"domain": "google.com"})
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# explain() directly
# ---------------------------------------------------------------------------


def all_contributions(explainer: DomainExplainer, domain: str) -> tuple[float, float]:
    """(base_value, sum of all 15 contributions) from explain(top_k=None)."""
    explanation = explainer.explain(domain, top_k=None)
    items = explanation.top_contributing_features
    assert sorted(item["feature"] for item in items) == sorted(FEATURE_NAMES)
    return explanation.base_value, sum(item["contribution"] for item in items)


@pytest.mark.parametrize("domain", DOMAINS)
def test_gb_contributions_sum_to_log_odds(gb_model, domain: str) -> None:
    explainer = build_explainer(gb_model, "gb")
    base, total = all_contributions(explainer, domain)
    log_odds = gb_model.decision_function(feature_frame(domain))[0]
    assert base + total == pytest.approx(log_odds, abs=1e-6)


def test_feature_values_are_the_extracted_features(gb_model) -> None:
    explanation = build_explainer(gb_model, "gb").explain("pidfksipycmyg.com", top_k=None)
    expected = feature_frame("pidfksipycmyg.com").iloc[0]
    for item in explanation.top_contributing_features:
        assert item["value"] == pytest.approx(expected[item["feature"]])


def test_top_k_limits_items(gb_model) -> None:
    explanation = build_explainer(gb_model, "gb").explain("google.com", top_k=3)
    assert len(explanation.top_contributing_features) == 3


@pytest.mark.parametrize("domain", DOMAINS)
def test_rf_explainer(sample_xy: tuple[pd.DataFrame, pd.Series], domain: str) -> None:
    x, y = sample_xy
    model = build_rf().set_params(n_estimators=10, n_jobs=1).fit(x, y)
    explainer = build_explainer(model, "rf")
    top = explainer.explain(domain).top_contributing_features
    assert len(top) == 5
    assert all(item["feature"] in FEATURE_NAMES for item in top)
    # RF SHAP values are in probability units: base + all contributions = P(dga)
    base, total = all_contributions(explainer, domain)
    assert base + total == pytest.approx(model.predict_proba(feature_frame(domain))[0, 1], abs=1e-6)


@pytest.mark.parametrize("domain", DOMAINS)
def test_logreg_explainer(sample_xy: tuple[pd.DataFrame, pd.Series], domain: str) -> None:
    x, y = sample_xy
    model = build_logreg().fit(x, y)
    explainer = build_explainer(model, "logreg")
    top = explainer.explain(domain).top_contributing_features
    assert len(top) == 5
    assert all(item["feature"] in BASIC_FEATURES for item in top if item["contribution"] != 0)
    # Features the pipeline does not use contribute exactly 0
    every = explainer.explain(domain, top_k=None).top_contributing_features
    unused = [i for i in every if i["feature"] not in BASIC_FEATURES]
    assert unused and all(i["contribution"] == 0.0 for i in unused)
    # coefficient x standardized value + intercept = the pipeline's log-odds
    base, total = all_contributions(explainer, domain)
    assert base + total == pytest.approx(
        model.decision_function(feature_frame(domain))[0], abs=1e-6
    )


def test_unknown_model_type_rejected(gb_model) -> None:
    with pytest.raises(ValueError, match="No explainer"):
        build_explainer(gb_model, "svm")


def test_explain_matches_predict_proba(gb_model) -> None:
    """The /explain probability comes from predict_proba, not from the SHAP sum."""
    x = feature_frame("pidfksipycmyg.com")
    base, total = all_contributions(build_explainer(gb_model, "gb"), "pidfksipycmyg.com")
    assert 1 / (1 + np.exp(-(base + total))) == pytest.approx(gb_model.predict_proba(x)[0, 1])
