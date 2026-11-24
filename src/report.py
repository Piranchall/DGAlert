"""Copy results out of MLflow into reports/ (committed, readable on any machine without MLflow).

The M1 reports are built from a fresh copy of the committed M1 snapshot (src.m1_snapshot), never
from the committed mlflow.db itself, from the latest finished parent run of each model type:

- reports/m1_results.json: params + metrics + run id per model, the best GB grid point and its
  CV F1, and the DGA share of each test set (F1 depends on the class balance, so f1_seen and
  f1_future are not directly comparable).
- reports/per_family_recall.json: per-family recall on test_seen and test_future, per model.

reports/production_model.json describes the model the API serves: the run named by
mlflow_run_id in models/production/latest.json, looked up in the local store (new runs) and in the
M1 snapshot copy. It holds no timestamps or paths, so rerunning this leaves git status clean.

No training happens here. Needs data/processed/split_report.json (python -m src.prepare_data).
Run with: python -m src.report
"""

import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from src import config
from src.m1_snapshot import prepare_m1_snapshot
from src.mlflow_store import sqlite_db_path

METRICS = [
    "f1_seen",
    "precision_seen",
    "recall_seen",
    "f1_future",
    "recall_future",
    "train_time_s",
]
TEST_SETS = ["test_seen", "test_future"]
# Where training saved the file (may be an absolute path); the pointer says where it is served from.
MACHINE_SPECIFIC_PARAMS = {"model_path"}


def parse_param(value: str) -> object:
    """MLflow stores params as strings; turn numbers back into numbers ("0.05" -> 0.05)."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def latest_parent_run(client: mlflow.MlflowClient, experiment_id: str, model_type: str) -> Run:
    """The most recent finished parent run tagged with this model_type."""
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.model_type = '{model_type}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"No finished MLflow run with model_type={model_type!r}")
    return runs[0]


def best_grid_point(client: mlflow.MlflowClient, experiment_id: str, parent: Run) -> dict:
    """The nested grid-search child run of a GB parent run with the highest mean CV F1."""
    children = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    if not children:
        raise LookupError(f"GB run {parent.info.run_id} has no grid-search child runs")
    best = max(children, key=lambda r: r.data.metrics["cv_f1_mean"])
    grid_keys = ["n_estimators", "max_depth", "learning_rate"]
    return {
        "params": {k: parse_param(best.data.params[k]) for k in grid_keys},
        "cv_f1_mean": round(best.data.metrics["cv_f1_mean"], 4),
        "cv_f1_std": round(best.data.metrics["cv_f1_std"], 4),
        "cv_rows": parse_param(best.data.params["cv_rows"]),
        "cv_folds": parse_param(best.data.params["cv_folds"]),
        "grid_points": len(children),
        "mlflow_run_id": best.info.run_id,
    }


def class_balance(split_report_path: Path) -> dict:
    """Rows, DGA rows and DGA share of each test set, from prepare_data's split_report.json."""
    totals = json.loads(split_report_path.read_text(encoding="utf-8"))["totals"]
    return {
        name: {
            "rows": totals[name]["rows"],
            "dga": totals[name]["dga"],
            "dga_share": round(totals[name]["dga"] / totals[name]["rows"], 4),
        }
        for name in TEST_SETS
    }


def read_per_family(client: mlflow.MlflowClient, run_id: str) -> dict:
    """The per-family recall artifact of a run, downloaded to a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        path = client.download_artifacts(run_id, config.PER_FAMILY_FILE, tmp)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def build_reports(
    tracking_uri: str,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    split_report_path: Path = config.SPLIT_REPORT_PATH,
) -> tuple[dict, dict]:
    """Return (m1_results, per_family_recall) built from an MLflow store.

    The store must already point at this machine's artifact folder; for the M1 snapshot use the
    copy made by src.m1_snapshot.prepare_m1_snapshot().
    """
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"MLflow experiment {experiment_name!r} not found at {tracking_uri}")

    models: dict[str, dict] = {}
    per_family: dict[str, dict] = {}
    gb_best = None
    for model_type in config.MODEL_TYPES:
        run = latest_parent_run(client, experiment.experiment_id, model_type)
        models[model_type] = {
            "mlflow_run_id": run.info.run_id,
            "params": {k: parse_param(v) for k, v in sorted(run.data.params.items())},
            **{m: round(run.data.metrics[m], 4) for m in METRICS},
        }
        per_family[model_type] = read_per_family(client, run.info.run_id)
        if model_type == "gb":
            gb_best = best_grid_point(client, experiment.experiment_id, run)

    results = {
        "experiment": experiment_name,
        "positive_class": "dga",
        "threshold": config.DGA_THRESHOLD,
        "best_model_by_f1_seen": max(models, key=lambda m: models[m]["f1_seen"]),
        "class_balance": class_balance(split_report_path),
        "models": models,
        "gb_best_grid_point": gb_best,
    }
    # Reshape to {test set: {model: {family: recall}}} so models sit side by side per family.
    by_test_set = {
        test_set: {model: per_family[model][test_set] for model in per_family}
        for test_set in TEST_SETS
    }
    return results, by_test_set


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def write_reports(
    results_path: Path = config.RESULTS_REPORT_PATH,
    per_family_path: Path = config.PER_FAMILY_REPORT_PATH,
    **build_kwargs,
) -> dict:
    """Build the M1 reports from an MLflow store and write them as JSON; return m1_results."""
    results, per_family = build_reports(**build_kwargs)
    write_json(results_path, results)
    write_json(per_family_path, per_family)
    return results


def find_run(stores: list[tuple[str, str]], run_id: str) -> tuple[str, mlflow.MlflowClient, Run]:
    """(store label, client, run) for the first store holding run_id.

    A sqlite store whose database file does not exist is skipped (opening it would create it).
    """
    for label, tracking_uri in stores:
        db_path = sqlite_db_path(tracking_uri)
        if db_path is not None and not db_path.is_file():
            continue
        client = mlflow.MlflowClient(tracking_uri)
        try:
            return label, client, client.get_run(run_id)
        except MlflowException:
            continue
    labels = ", ".join(label for label, _ in stores)
    raise LookupError(f"MLflow run {run_id} not found in any store ({labels})")


def build_production_report(pointer_path: Path, stores: list[tuple[str, str]]) -> dict:
    """Describe the model a pointer names, from its MLflow run.

    Deterministic: no timestamps and no paths; the store is recorded by its label ("local" or
    "m1_snapshot").
    """
    pointer = json.loads(Path(pointer_path).read_text(encoding="utf-8"))
    label, client, run = find_run(stores, pointer["mlflow_run_id"])
    experiment = client.get_experiment(run.info.experiment_id)
    return {
        "version": pointer["version"],
        "model_type": pointer["model_type"],
        "sha256": pointer.get("sha256"),
        "mlflow_run_id": run.info.run_id,
        "mlflow_store": label,
        "experiment": experiment.name,
        "phased_in_families": pointer.get("phased_in_families", []),
        "params": {
            k: parse_param(v)
            for k, v in sorted(run.data.params.items())
            if k not in MACHINE_SPECIFIC_PARAMS
        },
        "metrics": {k: round(v, 4) for k, v in sorted(run.data.metrics.items())},
        "per_family_recall": read_per_family(client, run.info.run_id),
    }


def main() -> None:
    m1_uri = prepare_m1_snapshot()
    write_reports(tracking_uri=m1_uri)
    stores = [("local", config.mlflow_tracking_uri()), ("m1_snapshot", m1_uri)]
    report = build_production_report(config.PRODUCTION_MODEL_POINTER, stores)
    write_json(config.PRODUCTION_REPORT_PATH, report)


if __name__ == "__main__":
    main()
