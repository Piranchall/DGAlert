"""FastAPI service exposing /predict, /explain, /health and /stats. Interactive docs at /docs
(GET / redirects there).

Run with: uvicorn src.app:app
"""

import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.routing import Match

from src import config
from src.explain import DomainExplainer, build_explainer
from src.features import feature_frame, normalize_domain, public_suffix, second_level_label
from src.model_loader import load_model
from src.monitoring import (
    UNMATCHED_ENDPOINT,
    StatsCollector,
    access_log_line,
    configure_access_logger,
)
from src.scope import check_scope

logger = logging.getLogger(__name__)
access_logger = configure_access_logger()

# The commit the running image was built from (Docker build arg GIT_SHA); identifies a deployment.
GIT_SHA_ENV_VAR = "GIT_SHA"
UNKNOWN_GIT_SHA = "unknown"

MAX_DOMAIN_LENGTH = 253
# Limit on the raw input, checked by Pydantic before normalize_domain runs (C4). A domain is at
# most 253 characters; the rest leaves room for a pasted URL (scheme, credentials, port, path).
MAX_RAW_INPUT_LENGTH = 1024
MAX_LABEL_LENGTH = 63
LABEL_CHARS = re.compile(r"^[a-z0-9-]+$")


# ---------------------------------------------------------------------------
# Domain validation (DNS rules, applied after normalize_domain)
# ---------------------------------------------------------------------------


def validate_domain(raw: str) -> str:
    """Normalize a domain and check DNS syntax. Returns the normalized domain.

    Raises ValueError with a readable message for the first rule that fails.
    """
    domain = normalize_domain(raw)
    if not domain:
        raise ValueError("domain is empty after normalization")
    if not domain.isascii():
        raise ValueError("domain must be ASCII (use punycode, e.g. xn--bcher-kva.de)")
    if len(domain) > MAX_DOMAIN_LENGTH:
        raise ValueError(f"domain is longer than {MAX_DOMAIN_LENGTH} characters")
    if "." not in domain:
        raise ValueError("domain must contain at least one dot, e.g. example.com")
    for label in domain.split("."):
        if not 1 <= len(label) <= MAX_LABEL_LENGTH:
            raise ValueError(f"each label must be 1-{MAX_LABEL_LENGTH} characters long")
        if not LABEL_CHARS.match(label):
            raise ValueError(f"label '{label}' may only contain letters, digits and hyphens")
        if label.startswith("-") or label.endswith("-"):
            raise ValueError(f"label '{label}' must not start or end with a hyphen")
    if not public_suffix(domain):
        raise ValueError(
            f"'{domain}' has no known public suffix; "
            "IP addresses and internal names are not supported"
        )
    if not second_level_label(domain):
        raise ValueError(f"'{domain}' has no second-level label (it is a bare public suffix)")
    return domain


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    domain: str = Field(
        max_length=MAX_RAW_INPUT_LENGTH,
        description=(
            "Domain name or URL; it is normalized before prediction. "
            f"At most {MAX_RAW_INPUT_LENGTH} characters."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"domain": "moiqidfvjjhjmndqp.in"},
                {"domain": "teco.com.tw"},
                {"domain": "HTTP://Example.COM/path"},
                {"domain": "abcxyzqwrtplk.blogspot.com"},
            ]
        }
    )

    @field_validator("domain")
    @classmethod
    def check_domain(cls, value: str) -> str:
        return validate_domain(value)


OUT_OF_SCOPE_EXAMPLE: dict[str, Any] = {  # real response of model 20261105_062411
    "domain": "abcxyzqwrtplk.blogspot.com",
    "label": "benign",
    "probability": 0.0451,
    "model_version": "20261105_062411",
    "in_scope": False,
    "warning": (
        "'abcxyzqwrtplk.blogspot.com' is under the private suffix 'blogspot.com' (a hosting or "
        "dynamic-DNS provider); the model only sees the label 'blogspot', so the prediction says "
        "nothing about 'abcxyzqwrtplk'."
    ),
}


class PredictResponse(BaseModel):
    domain: str = Field(description="The normalized domain that was classified.")
    label: Literal["benign", "dga"]
    probability: float = Field(ge=0, le=1, description="P(dga), rounded to 4 decimals.")
    model_version: str
    in_scope: bool = Field(
        description=(
            "false when the domain is under a private suffix (e.g. x.blogspot.com) or contains "
            "a punycode (xn--) label: the prediction is returned but is not meaningful."
        )
    )
    warning: str | None = Field(description="Why the domain is out of scope; null when in scope.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "domain": "moiqidfvjjhjmndqp.in",
                    "label": "dga",
                    "probability": 0.996,
                    "model_version": "20261105_062411",
                    "in_scope": True,
                    "warning": None,
                },
                OUT_OF_SCOPE_EXAMPLE,
            ]
        }
    )


class FeatureContribution(BaseModel):
    feature: str = Field(description="Feature name (one of src.features.FEATURE_NAMES).")
    value: float = Field(description="The feature's value for this domain.")
    contribution: float = Field(description="Push toward dga (positive) or benign (negative).")


class ExplainResponse(PredictResponse):
    base_value: float = Field(
        description="Model output before any feature is considered (the explainer's baseline)."
    )
    top_contributing_features: list[FeatureContribution]
    output_units: Literal["log_odds", "probability"] = Field(
        description=(
            "Units of base_value and the contributions: log_odds for gb and logreg, "
            "probability (P(dga)) for rf."
        )
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {  # real response of model 20261105_062411 (rounded)
                    "domain": "moiqidfvjjhjmndqp.in",
                    "label": "dga",
                    "probability": 0.996,
                    "model_version": "20261105_062411",
                    "in_scope": True,
                    "warning": None,
                    "output_units": "log_odds",
                    "base_value": -0.0369,
                    "top_contributing_features": [
                        {"feature": "length", "value": 17.0, "contribution": 1.4049},
                        {
                            "feature": "trigram_mean_logfreq",
                            "value": -5.5008,
                            "contribution": 1.175,
                        },
                        {
                            "feature": "longest_dict_word_ratio",
                            "value": 0.0,
                            "contribution": 1.0429,
                        },
                        {"feature": "longest_consonant_run", "value": 12.0, "contribution": 0.6608},
                        {"feature": "digit_ratio", "value": 0.0, "contribution": 0.4129},
                    ],
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description=(
            "healthy: model and explainer loaded. degraded (still HTTP 200): the model serves "
            "/predict but the explainer failed, so /explain returns 503. unhealthy (HTTP 503): "
            "no model."
        )
    )
    model_loaded: bool
    explainer_loaded: bool
    git_sha: str = Field(
        description='Commit the image was built from (GIT_SHA env var); "unknown" when not set.'
    )
    model_version: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "explainer_loaded": True,
                    "git_sha": "67fc13b",
                    "model_version": "20261105_062411",
                },
                {
                    "status": "degraded",
                    "model_loaded": True,
                    "explainer_loaded": False,
                    "git_sha": "67fc13b",
                    "model_version": "20261105_062411",
                },
            ]
        }
    )


class EndpointStats(BaseModel):
    count: int
    by_status_class: dict[str, int] = Field(description='Counts per status class, e.g. "2xx".')
    avg_latency_ms: float | None = Field(description="Mean latency; null before any request.")


class RequestStats(BaseModel):
    total: int = Field(description="All requests, including /health checks and /stats itself.")
    by_status_class: dict[str, int]
    avg_latency_ms: float | None
    by_endpoint: dict[str, EndpointStats] = Field(
        description='Keyed by "METHOD /route"; requests that match no route are "unmatched".'
    )


class PredictionStats(BaseModel):
    count: int = Field(description="Successful /predict calls (/explain is not counted).")
    by_label: dict[str, int]
    mean_probability_dga: float | None = Field(
        description="Mean P(dga) over those predictions; null before the first one."
    )
    out_of_scope: int = Field(description="Of those predictions, how many had in_scope false.")


class StatsResponse(BaseModel):
    model_version: str | None = Field(description="null when no model is loaded.")
    started_at: str = Field(description="UTC start time of this process; counters start here.")
    uptime_seconds: float
    requests: RequestStats
    predictions: PredictionStats

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_version": "20261105_062411",
                    "started_at": "2026-11-25T09:00:00.000+00:00",
                    "uptime_seconds": 123.456,
                    "requests": {
                        "total": 6,
                        "by_status_class": {"2xx": 5, "4xx": 1},
                        "avg_latency_ms": 7.512,
                        "by_endpoint": {
                            "GET /health": {
                                "count": 2,
                                "by_status_class": {"2xx": 2},
                                "avg_latency_ms": 0.811,
                            },
                            "POST /predict": {
                                "count": 4,
                                "by_status_class": {"2xx": 3, "4xx": 1},
                                "avg_latency_ms": 10.863,
                            },
                        },
                    },
                    "predictions": {
                        "count": 3,
                        "by_label": {"benign": 2, "dga": 1},
                        "mean_probability_dga": 0.3349,
                        "out_of_scope": 1,
                    },
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup. If loading fails, log it and run without a model."""
    app.state.model = None
    app.state.metadata = {}
    app.state.explainer = None
    app.state.stats = StatsCollector()  # in memory: counters reset on every start
    app.state.git_sha = os.environ.get(GIT_SHA_ENV_VAR) or UNKNOWN_GIT_SHA
    try:
        app.state.model, app.state.metadata = load_model()
        logger.info("Loaded model version %s", app.state.metadata["version"])
    except Exception:
        logger.exception("Could not load the model; /predict and /explain will return 503")
    if app.state.model is not None:
        try:
            app.state.explainer = build_explainer(app.state.model, app.state.metadata["model_type"])
        except Exception:
            logger.exception("Could not build the explainer; /explain will return 503")
    yield


DESCRIPTION = """
DGAlert classifies a domain name as **benign** or **dga** (generated by a malware Domain
Generation Algorithm) from the domain string alone. Features are computed on the second-level
label (`mail.google.co.uk` -> `google`). A domain is labelled `dga` when P(dga) >= 0.5.
"""

app = FastAPI(title="DGAlert", description=DESCRIPTION, version="1.0.0", lifespan=lifespan)


def endpoint_key(request: Request) -> str:
    """ "METHOD /route/template" for the route that served the request, else "unmatched".

    Using the template (not the raw path) keeps the number of keys bounded, e.g. under 404 scans.
    FastAPI's own /docs and /openapi.json routes do not set scope["route"], so match them here.
    """
    route = request.scope.get("route")
    if route is None:
        route = next(
            (r for r in request.app.router.routes if r.matches(request.scope)[0] == Match.FULL),
            None,
        )
    if route is None:
        return UNMATCHED_ENDPOINT
    return f"{request.method} {route.path}"


@app.middleware("http")
async def record_stats(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Time every request, update /stats and write one JSON access log line."""
    start = time.perf_counter()
    status_code = 500  # if the handler raises, the client gets a 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        prediction = getattr(request.state, "prediction", None)
        stats: StatsCollector | None = getattr(request.app.state, "stats", None)
        if stats is not None:
            stats.record_request(endpoint_key(request), status_code, latency_ms)
            if prediction is not None:
                stats.record_prediction(
                    prediction["label"], prediction["probability"], prediction["in_scope"]
                )
        access_logger.info(
            access_log_line(request.method, request.url.path, status_code, latency_ms, prediction)
        )


NOT_LOADED = "Model not loaded; the server log gives the reason (e.g. a failed SHA-256 check)."
EXPLAIN_TOP_K = 5
NO_EXPLAINER = "Explainer not available for the loaded model; see the server log."

EXPLAIN_DESCRIPTION = """Classify one domain and show which features drove the decision.

Each contribution is in the model's own output units (log-odds for gb and logreg, P(dga) for
rf): **positive values push toward "dga", negative values toward "benign"**. base_value plus the
contributions of all 15 features equals the model output; only the top 5 by absolute size are
returned."""


def get_model(request: Request) -> Any:
    """The loaded model, or 503 if startup could not load one."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=NOT_LOADED)
    return model


def predict_proba(model: Any, domain: str) -> float:
    """P(dga) for one normalized domain; features passed as a named DataFrame, as in training."""
    return float(model.predict_proba(feature_frame(domain))[0, 1])


def classify(model: Any, domain: str, request: Request) -> PredictResponse:
    """Shared by /predict and /explain: P(dga), label, model version and the scope check."""
    proba = predict_proba(model, domain)
    in_scope, warning = check_scope(domain)
    return PredictResponse(
        domain=domain,
        label="dga" if proba >= config.DGA_THRESHOLD else "benign",
        probability=round(proba, 4),
        model_version=str(request.app.state.metadata["version"]),
        in_scope=in_scope,
        warning=warning,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"description": "No model is loaded"}},
)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Classify one domain as benign or dga."""
    response = classify(get_model(request), body.domain, request)
    # Read by the stats middleware: counted in /stats and logged (never the domain itself).
    request.state.prediction = {
        "label": response.label,
        "probability": response.probability,
        "in_scope": response.in_scope,
    }
    return response


@app.post(
    "/explain",
    response_model=ExplainResponse,
    description=EXPLAIN_DESCRIPTION,
    responses={503: {"description": "No model (or no explainer) is loaded"}},
)
def explain(body: PredictRequest, request: Request) -> ExplainResponse:
    model = get_model(request)
    explainer: DomainExplainer | None = request.app.state.explainer
    if explainer is None:
        raise HTTPException(status_code=503, detail=NO_EXPLAINER)
    prediction = classify(model, body.domain, request)
    explanation = explainer.explain(body.domain, top_k=EXPLAIN_TOP_K)
    return ExplainResponse(
        **prediction.model_dump(),
        output_units=explainer.output_units,
        base_value=explanation.base_value,
        top_contributing_features=explanation.top_contributing_features,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={503: {"model": HealthResponse, "description": "No model is loaded"}},
)
def health(request: Request) -> HealthResponse | JSONResponse:
    """Model and explainer status (used by the container and Render health checks).

    200 "healthy" with both loaded; 200 "degraded" when the explainer failed (only /explain is
    down, so the service should not be restarted for it); 503 "unhealthy" without a model.
    """
    explainer_loaded = request.app.state.explainer is not None
    if request.app.state.model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "explainer_loaded": explainer_loaded,
                "git_sha": request.app.state.git_sha,
            },
        )
    return HealthResponse(
        status="healthy" if explainer_loaded else "degraded",
        model_loaded=True,
        explainer_loaded=explainer_loaded,
        git_sha=request.app.state.git_sha,
        model_version=str(request.app.state.metadata["version"]),
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The bare URL has nothing to show: send visitors to the interactive docs (C6)."""
    return RedirectResponse(url="/docs")


@app.get("/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    """Request counts, latency and /predict label statistics since this process started.

    Counters are kept in memory and reset when the service restarts. Health checks call /health
    regularly, so requests.total rises on its own; read "POST /predict" under by_endpoint.
    """
    snapshot = request.app.state.stats.snapshot()
    version = request.app.state.metadata.get("version") if request.app.state.model else None
    return StatsResponse(model_version=None if version is None else str(version), **snapshot)
