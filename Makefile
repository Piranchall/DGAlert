# DGAlert task runner (Linux/macOS/CI). Run every command from the repo root.
#
# Windows has no `make`. Equivalent commands in PowerShell:
#   setup:  py -3.13 -m venv venv ; venv\Scripts\python -m pip install -r requirements.txt
#   data:   venv\Scripts\python -m src.prepare_data
#   retrain-data: venv\Scripts\python -m src.prepare_retrain
#   train:  venv\Scripts\python -m src.train
#   report: venv\Scripts\python -m src.report
#   mlflow-m1: venv\Scripts\python -m src.m1_snapshot
#   test:   venv\Scripts\python -m pytest -q
#   lint:   venv\Scripts\ruff check src tests scripts ; venv\Scripts\ruff format --check src tests scripts
#   serve:  venv\Scripts\python -m uvicorn src.app:app --reload
#   docker-build: docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t dgalert:local .
#   docker-run:   docker run --rm -p 8000:8000 --name dgalert dgalert:local

PY := venv/bin/python

.PHONY: setup data retrain-data train report mlflow-m1 test lint serve docker-build docker-run

setup:
	test -d venv || python3.13 -m venv venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.prepare_data

# Frozen eval set, phase-in pool and validation split (needs `make data` first)
retrain-data:
	$(PY) -m src.prepare_retrain

train:
	$(PY) -m src.train

report:
	$(PY) -m src.report

# Copy the read-only M1 MLflow store to mlflow_m1/ and print the `mlflow ui` command
mlflow-m1:
	$(PY) -m src.m1_snapshot

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

serve:
	$(PY) -m uvicorn src.app:app --reload

# Build the API image, tagged with the current commit (shown as git_sha in /health)
docker-build:
	docker build --build-arg GIT_SHA=$$(git rev-parse --short HEAD) -t dgalert:local .

# Run it on http://127.0.0.1:8000 (Ctrl+C stops and removes the container)
docker-run:
	docker run --rm -p 8000:8000 --name dgalert dgalert:local
