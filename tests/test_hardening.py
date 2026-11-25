"""Tests for the S4 API hardening items C1 to C6 (src.scope, src.app, src.explain)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.app import MAX_RAW_INPUT_LENGTH, app
from src.explain import OUTPUT_UNITS, build_explainer
from src.scope import check_scope, has_punycode, private_suffix
from src.train import build_logreg, build_rf

BLOGSPOT = "abcxyzqwrtplk.blogspot.com"

# ---------------------------------------------------------------------------
# C1, C2: scope check (src.scope)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "suffix"),
    [
        (BLOGSPOT, "blogspot.com"),
        ("x.duckdns.org", "duckdns.org"),
        ("a.b.github.io", "github.io"),
        ("blogspot.com", None),  # the provider's own site has no label below the private suffix
        ("google.com", None),
        ("mail.google.co.uk", None),
    ],
)
def test_private_suffix(domain: str, suffix: str | None) -> None:
    assert private_suffix(domain) == suffix


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("xn--bcher-kva.de", True),
        ("example.xn--p1ai", True),  # punycode TLD
        ("mail.xn--bcher-kva.de", True),
        ("google.com", False),
        ("axn--b.com", False),  # "xn--" inside a label is not a punycode prefix
    ],
)
def test_has_punycode(domain: str, expected: bool) -> None:
    assert has_punycode(domain) is expected


def test_in_scope_domain_has_no_warning() -> None:
    assert check_scope("google.com") == (True, None)
    assert check_scope("teco.com.tw") == (True, None)


def test_private_suffix_warning_names_suffix_and_labels() -> None:
    in_scope, warning = check_scope(BLOGSPOT)
    assert in_scope is False
    assert "private suffix 'blogspot.com'" in warning
    assert "only sees the label 'blogspot'" in warning
    assert "says nothing about 'abcxyzqwrtplk'" in warning


def test_punycode_warning() -> None:
    in_scope, warning = check_scope("xn--bcher-kva.de")
    assert in_scope is False
    assert "punycode (xn--) label" in warning


def test_both_warnings_are_joined() -> None:
    in_scope, warning = check_scope("xn--abc-xyz.blogspot.com")
    assert in_scope is False
    assert "private suffix" in warning
    assert "punycode" in warning


# ---------------------------------------------------------------------------
# C1, C2 through the API: the prediction is still returned
# ---------------------------------------------------------------------------


def test_blogspot_is_out_of_scope_with_warning(client: TestClient) -> None:
    response = client.post("/predict", json={"domain": BLOGSPOT})
    assert response.status_code == 200
    body = response.json()
    assert body["in_scope"] is False
    assert "blogspot.com" in body["warning"]
    assert body["label"] in {"benign", "dga"}
    assert 0.0 <= body["probability"] <= 1.0


@pytest.mark.parametrize("domain", [BLOGSPOT, "x.duckdns.org", "xn--bcher-kva.de"])
def test_explain_reports_scope_like_predict(client: TestClient, domain: str) -> None:
    predicted = client.post("/predict", json={"domain": domain}).json()
    explained = client.post("/explain", json={"domain": domain}).json()
    assert explained["in_scope"] is False
    assert explained["warning"] == predicted["warning"]
    assert explained["probability"] == predicted["probability"]


@pytest.mark.parametrize("domain", ["google.com", "blogspot.com", "moiqidfvjjhjmndqp.in"])
def test_ordinary_domains_are_in_scope(client: TestClient, domain: str) -> None:
    body = client.post("/predict", json={"domain": domain}).json()
    assert body["in_scope"] is True
    assert body["warning"] is None


def test_stats_counts_out_of_scope_predictions(client: TestClient) -> None:
    for domain in ["google.com", BLOGSPOT, "xn--bcher-kva.de", "x.duckdns.org"]:
        client.post("/predict", json={"domain": domain})
    client.post("/explain", json={"domain": BLOGSPOT})  # /explain is not a prediction in /stats

    predictions = client.get("/stats").json()["predictions"]
    assert predictions["count"] == 4
    assert predictions["out_of_scope"] == 3


# ---------------------------------------------------------------------------
# C3: /health reports the explainer separately
# ---------------------------------------------------------------------------


def test_health_degraded_when_explainer_fails(
    model_pointer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_explainer(model, model_type):
        raise RuntimeError("SHAP could not be initialised")

    monkeypatch.setattr(app_module, "build_explainer", broken_explainer)
    with TestClient(app) as degraded:
        health = degraded.get("/health")
        explain = degraded.post("/explain", json={"domain": "google.com"})
        predict = degraded.post("/predict", json={"domain": "google.com"})

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "model_loaded": True,
        "explainer_loaded": False,
        "git_sha": "unknown",
        "model_version": "test_fixture",
    }
    assert explain.status_code == 503
    assert predict.status_code == 200


def test_health_healthy_reports_explainer(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["explainer_loaded"] is True


# ---------------------------------------------------------------------------
# C4: raw input length limit, checked before normalisation
# ---------------------------------------------------------------------------


def url_of_length(length: int) -> str:
    """A valid URL of exactly `length` characters whose domain is google.com."""
    prefix = "https://google.com/"
    return prefix + "a" * (length - len(prefix))


def test_input_at_the_limit_is_accepted(client: TestClient) -> None:
    response = client.post("/predict", json={"domain": url_of_length(MAX_RAW_INPUT_LENGTH)})
    assert response.status_code == 200
    assert response.json()["domain"] == "google.com"


def test_input_over_the_limit_is_rejected_before_normalisation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_normalize = app_module.normalize_domain
    monkeypatch.setattr(
        app_module, "normalize_domain", lambda raw: calls.append(raw) or real_normalize(raw)
    )

    response = client.post("/predict", json={"domain": url_of_length(MAX_RAW_INPUT_LENGTH + 1)})

    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["type"] == "string_too_long"
    assert f"at most {MAX_RAW_INPUT_LENGTH} characters" in detail["msg"]
    assert calls == []  # normalize_domain never ran


def test_explain_has_the_same_length_limit(client: TestClient) -> None:
    response = client.post("/explain", json={"domain": url_of_length(MAX_RAW_INPUT_LENGTH + 1)})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# C5: /explain output units
# ---------------------------------------------------------------------------


def test_explain_output_units_for_gb(client: TestClient) -> None:
    assert client.post("/explain", json={"domain": "google.com"}).json()["output_units"] == (
        "log_odds"
    )


def test_output_units_per_model_type(sample_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = sample_xy
    rf = build_rf().set_params(n_estimators=5, n_jobs=1).fit(x, y)
    logreg = build_logreg().fit(x, y)
    assert build_explainer(rf, "rf").output_units == "probability"
    assert build_explainer(logreg, "logreg").output_units == "log_odds"
    assert OUTPUT_UNITS == {
        "gb": "log_odds",
        "hgb": "log_odds",  # the retraining loop's model type (src.retrain)
        "logreg": "log_odds",
        "rf": "probability",
    }


# ---------------------------------------------------------------------------
# C6: GET / redirects to the docs
# ---------------------------------------------------------------------------


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

    followed = client.get("/")
    assert followed.status_code == 200
    assert "swagger" in followed.text.lower()


def test_root_is_not_in_the_openapi_schema(client: TestClient) -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_openapi_examples_include_scope_fields(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    predict_examples = schemas["PredictResponse"]["examples"]
    assert {e["in_scope"] for e in predict_examples} == {True, False}
    assert schemas["ExplainResponse"]["examples"][0]["output_units"] == "log_odds"
    assert {e["status"] for e in schemas["HealthResponse"]["examples"]} == {"healthy", "degraded"}
