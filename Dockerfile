# DGAlert API image: multi-stage build on Python 3.13 slim (Debian 13 "trixie").
#
# Build:  docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t dgalert:local .
# Run:    docker run --rm -p 8000:8000 dgalert:local      (Render sets PORT; default 8000)
#
# The base image is pinned to an exact Python patch and Debian release, so rebuilds get the same
# interpreter. Bump both stages together.

# ---------------------------------------------------------------------------
# builder: install the API's runtime requirements into a venv
# ---------------------------------------------------------------------------
FROM python:3.13.18-slim-trixie AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Only the runtime requirements (no mlflow, pytest or ruff). Copied on their own so this layer stays
# cached when only the code changes.
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt

# ---------------------------------------------------------------------------
# runtime: the venv, the code, the word list and the production model; runs as non-root
# ---------------------------------------------------------------------------
FROM python:3.13.18-slim-trixie AS runtime

ARG GIT_SHA=unknown

# NUMBA_CACHE_DIR: shap uses numba, whose cache would otherwise go next to the venv packages, which
# the dgalert user cannot write. GIT_SHA identifies the deployment in /health.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    GIT_SHA=$GIT_SHA \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dgalert

WORKDIR /app

# Owned by root, so the app user can read but not modify them.
COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/
COPY data/reference/ /app/data/reference/
COPY models/production/ /app/models/production/

USER dgalert

EXPOSE 8000

# urlopen raises on HTTP 503, so "unhealthy" fails the check; "degraded" (200) passes it.
# --start-interval checks every 2 s while starting, so (healthy) shows soon after the model loads.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"]

# exec: uvicorn becomes PID 1 and receives SIGTERM from docker stop. --no-access-log: the app's
# JSON line (logger dgalert.access) is the access log.
CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
