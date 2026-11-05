"""Copy the Milestone 1 results out of MLflow into reports/ (committed, readable on any machine).

MLflow stores artifact locations as absolute paths, so the per-family artifacts in the committed
store only open on the machine that trained. This module reads the latest finished parent run of
each model type in the experiment and writes two plain JSON files:

- reports/m1_results.json: params + metrics + run id per model, the best GB grid point and its
  CV F1, and the DGA share of each test set (F1 depends on the class balance, so f1_seen and
  f1_future are not directly comparable).
- reports/per_family_recall.json: per-family recall on test_seen and test_future, per model.

No training happens here. Run with: python -m src.report
"""

import json
from pathlib import Path

import mlflow
from mlflow.entities import Run

from src import config

METRICS = [
    "f1_seen",
    "precision_seen",
    "recall_seen",
    "f1_future",
    "recall_future",
    "train_time_s",
]
TEST_SETS = ["test_seen", "test_future"]


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


def build_reports(
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    split_report_path: Path = config.SPLIT_REPORT_PATH,
) -> tuple[dict, dict]:
    """Return (m1_results, per_family_recall) built from the MLflow store."""
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
        path = client.download_artifacts(run.info.run_id, config.PER_FAMILY_FILE)
        per_family[model_type] = json.loads(Path(path).read_text(encoding="utf-8"))
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


def write_reports(
    results_path: Path = config.RESULTS_REPORT_PATH,
    per_family_path: Path = config.PER_FAMILY_REPORT_PATH,
    **build_kwargs,
) -> dict:
    """Build both reports from MLflow and write them as JSON; return m1_results."""
    results, per_family = build_reports(**build_kwargs)
    for path, content in [(results_path, results), (per_family_path, per_family)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return results


if __name__ == "__main__":
    write_reports()
