# DGAlert task runner (Linux/macOS/CI). Run every command from the repo root.
#
# Windows has no `make`. Equivalent commands in PowerShell:
#   setup:  py -3.11 -m venv venv ; venv\Scripts\python -m pip install -r requirements.txt
#   data:   venv\Scripts\python -m src.prepare_data
#   train:  venv\Scripts\python -m src.train
#   test:   venv\Scripts\python -m pytest -q
#   lint:   venv\Scripts\ruff check src tests scripts ; venv\Scripts\ruff format --check src tests scripts
#   serve:  venv\Scripts\python -m uvicorn src.app:app --reload

PY := venv/bin/python

.PHONY: setup data train test lint serve

setup:
	test -d venv || python3.11 -m venv venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.prepare_data

train:
	$(PY) -m src.train

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

serve:
	$(PY) -m uvicorn src.app:app --reload
