"""Tests for the container setup: GIT_SHA in /health, and static checks of Dockerfile and
.dockerignore (no Docker needed, so these run in CI before the image is built)."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from src.app import GIT_SHA_ENV_VAR, UNKNOWN_GIT_SHA, app

BASE_IMAGE = "python:3.13.18-slim-trixie"
DOCKERFILE = (config.ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (config.ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines()


def instructions(keyword: str) -> list[str]:
    """Arguments of every Dockerfile instruction with this keyword (continuation lines joined)."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    return [
        line.split(None, 1)[1].strip()
        for line in joined.splitlines()
        if line.strip().upper().startswith(keyword + " ")
    ]


def runtime_stage() -> str:
    return DOCKERFILE.split(" AS runtime", 1)[1]


# ---------------------------------------------------------------------------
# GIT_SHA in /health
# ---------------------------------------------------------------------------


def test_health_reports_git_sha(model_pointer: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GIT_SHA_ENV_VAR, "abc1234")
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["git_sha"] == "abc1234"


def test_health_git_sha_defaults_to_unknown(client: TestClient) -> None:
    assert client.get("/health").json()["git_sha"] == UNKNOWN_GIT_SHA == "unknown"


def test_unhealthy_response_also_reports_git_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_POINTER", str(tmp_path / "missing.json"))
    monkeypatch.setenv(GIT_SHA_ENV_VAR, "abc1234")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["git_sha"] == "abc1234"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def test_both_stages_use_the_pinned_base_image() -> None:
    assert instructions("FROM") == [f"{BASE_IMAGE} AS builder", f"{BASE_IMAGE} AS runtime"]


def test_only_api_requirements_are_installed() -> None:
    copies = " ".join(instructions("COPY"))
    assert "requirements-api.txt" in copies
    assert not re.search(r"(?<!-)requirements\.txt", copies)
    assert any("pip install -r /tmp/requirements-api.txt" in run for run in instructions("RUN"))


def test_runtime_copies_what_the_api_reads() -> None:
    runtime = runtime_stage()
    for source in ("src/", "data/reference/", "models/production/"):
        assert f"COPY {source}" in runtime
    assert "COPY --from=builder /opt/venv /opt/venv" in runtime
    assert "COPY tests" not in runtime


def test_runs_as_non_root_user() -> None:
    users = instructions("USER")
    assert users and users[-1] not in {"root", "0"}
    assert f"--uid 10001 --shell /usr/sbin/nologin {users[-1]}" in runtime_stage()


def test_environment() -> None:
    env = " ".join(instructions("ENV"))
    assert "GIT_SHA=$GIT_SHA" in env
    assert "NUMBA_CACHE_DIR=/tmp/numba_cache" in env
    assert "PYTHONUNBUFFERED=1" in env
    assert "ARG GIT_SHA=unknown" in runtime_stage()


def test_port_and_command() -> None:
    assert instructions("EXPOSE") == ["8000"]
    (cmd,) = instructions("CMD")
    assert "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log" in cmd


def test_healthcheck_calls_health_with_urllib() -> None:
    (healthcheck,) = instructions("HEALTHCHECK")
    assert "urllib.request.urlopen" in healthcheck
    assert "'/health'" in healthcheck
    assert "os.environ.get('PORT', '8000')" in healthcheck
    assert "curl" not in healthcheck  # slim images have no curl


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "venv/",
        ".git/",
        "data/raw/",
        "data/processed/",
        "mlruns/",
        "mlflow.db",
        "mlflow_local/",
        "mlflow_m1/",
        "tests/",
        "**/__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".env",
    ],
)
def test_dockerignore_excludes(entry: str) -> None:
    assert entry in DOCKERIGNORE


def test_dockerignore_keeps_only_the_production_model() -> None:
    assert DOCKERIGNORE.index("models/*") < DOCKERIGNORE.index("!models/production/")
    for needed in ("src/", "data/reference/", "requirements-api.txt"):
        assert needed not in DOCKERIGNORE
