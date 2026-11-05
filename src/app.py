"""FastAPI service exposing /predict, /explain and /health. Interactive docs at /docs.

Run with: uvicorn src.app:app
"""

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config
from src.explain import DomainExplainer, build_explainer
from src.features import feature_frame, normalize_domain, second_level_label
from src.model_loader import load_model

logger = logging.getLogger(__name__)

MAX_DOMAIN_LENGTH = 253
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
    if not second_level_label(domain):
        raise ValueError(f"'{domain}' has no second-level label (it is a bare public suffix)")
    return domain


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    domain: str = Field(description="Domain name or URL; it is normalized before prediction.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"domain": "google.com"}, {"domain": "HTTP://Example.COM/path"}]
        }
    )

    @field_validator("domain")
    @classmethod
    def check_domain(cls, value: str) -> str:
        return validate_domain(value)


class PredictResponse(BaseModel):
    domain: str = Field(description="The normalized domain that was classified.")
    label: Literal["benign", "dga"]
    probability: float = Field(ge=0, le=1, description="P(dga), rounded to 4 decimals.")
    model_version: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "domain": "pidfksipycmyg.com",
                    "label": "dga",
                    "probability": 0.9938,
                    "model_version": "20261105_062411",
                }
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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {  # real response of model 20261105_062411 (rounded)
                    "domain": "pidfksipycmyg.com",
                    "label": "dga",
                    "probability": 0.9938,
                    "model_version": "20261105_062411",
                    "base_value": -0.0369,
                    "top_contributing_features": [
                        {
                            "feature": "trigram_mean_logfreq",
                            "value": -5.1719,
                            "contribution": 1.3637,
                        },
                        {
                            "feature": "longest_dict_word_ratio",
                            "value": 0.2308,
                            "contribution": 1.0481,
                        },
                        {"feature": "length", "value": 13.0, "contribution": 1.0319},
                        {"feature": "longest_consonant_run", "value": 6.0, "contribution": 0.527},
                        {"feature": "dict_coverage", "value": 0.2308, "contribution": 0.3532},
                    ],
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup. If loading fails, log it and run without a model."""
    app.state.model = None
    app.state.metadata = {}
    app.state.explainer = None
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

NOT_LOADED = "Model not loaded: train one (python -m src.train) and restart the service."
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
    """Shared by /predict and /explain: P(dga), thresholded label and model version."""
    proba = predict_proba(model, domain)
    return PredictResponse(
        domain=domain,
        label="dga" if proba >= config.DGA_THRESHOLD else "benign",
        probability=round(proba, 4),
        model_version=str(request.app.state.metadata["version"]),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"description": "No model is loaded"}},
)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Classify one domain as benign or dga."""
    return classify(get_model(request), body.domain, request)


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
    """200 when a model is loaded, 503 otherwise (used by the container healthcheck)."""
    if request.app.state.model is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model_loaded": False})
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=str(request.app.state.metadata["version"]),
    )
