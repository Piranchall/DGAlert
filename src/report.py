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

`python -m src.report --auc` instead writes reports/m1_auc.json: ROC-AUC and PR-AUC of the three
M1 models on test_seen and test_future, and per held-out family. The M1 runs never logged these, and
the snapshot is read-only, so they are computed from the model files on the processed splits:
gb from models/production/ (committed, SHA-256 checked), logreg and rf from the M1 files in models/
(gitignored, on the machine that trained them). Each model is first checked against M1: its
threshold metrics, recomputed, must equal reports/m1_results.json. This runs locally only (it needs
data/processed/ and the M1 files), never in CI; m1_results.json and per_family_recall.json are not
touched.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config, metrics
from src.m1_snapshot import prepare_m1_snapshot
from src.mlflow_store import sqlite_db_path
from src.model_loader import file_sha256, read_pointer, resolve_model_path

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


# ---------------------------------------------------------------------------
# Threshold-free metrics of the M1 models (--auc)
# ---------------------------------------------------------------------------

AUC_DIGITS = 4
# The threshold metrics recomputed to check that a model file is the M1 model.
CHECKED_METRICS = ["f1_seen", "precision_seen", "recall_seen", "f1_future", "recall_future"]


def m1_model_files(m1_results: dict, pointer_path: Path) -> dict[str, Path]:
    """Where each M1 model is: gb from the production pointer, logreg and rf from M1's model_path.

    The production model must be M1's gb (same version); its SHA-256 is checked against the pointer.
    """
    pointer = read_pointer(pointer_path)
    gb_version = m1_results["models"]["gb"]["params"]["version"]
    if pointer.get("model_type") != "gb" or pointer.get("version") != gb_version:
        raise ValueError(
            f"production is {pointer.get('model_type')} {pointer.get('version')}, not M1's gb "
            f"{gb_version}: the M1 AUC report needs M1's gb"
        )
    gb_path = resolve_model_path(pointer["path"])
    if file_sha256(gb_path) != pointer["sha256"]:
        raise ValueError(f"{gb_path} does not match the SHA-256 in {pointer_path}")
    files = {"gb": gb_path}
    for model_type in ("logreg", "rf"):
        path = resolve_model_path(m1_results["models"][model_type]["params"]["model_path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} (M1's {model_type}) is missing: it is gitignored and only exists where M1 "
                "was trained; python -m src.train --model all retrains it"
            )
        files[model_type] = path
    return {m: files[m] for m in config.MODEL_TYPES}


def threshold_metrics(p_seen, p_future, seen: pd.DataFrame, future: pd.DataFrame) -> dict:
    """The M1 threshold metrics recomputed from P(dga), rounded like m1_results.json."""
    y_seen = (seen["label"] == "dga").astype(int)
    y_future = (future["label"] == "dga").astype(int)
    pred_seen = p_seen >= config.DGA_THRESHOLD
    pred_future = p_future >= config.DGA_THRESHOLD
    values = {
        "f1_seen": f1_score(y_seen, pred_seen),
        "precision_seen": precision_score(y_seen, pred_seen),
        "recall_seen": recall_score(y_seen, pred_seen),
        "f1_future": f1_score(y_future, pred_future),
        "recall_future": recall_score(y_future, pred_future),
    }
    return {k: round(float(v), 4) for k, v in values.items()}


def _rounded(values: Any) -> Any:
    if isinstance(values, dict):
        return {k: _rounded(v) for k, v in values.items()}
    if isinstance(values, float):
        return round(values, AUC_DIGITS)
    return values


def build_auc_report(
    seen: pd.DataFrame, future: pd.DataFrame, models: dict[str, Any], m1_results: dict
) -> dict:
    """ROC-AUC and PR-AUC per model on both test sets and per held-out family.

    models: {model_type: (fitted model, repo-relative file name, sha256)}. Raises if a model's
    recomputed threshold metrics differ from m1_results.json (then it is not the M1 model).
    """
    held_out = sorted(set(future.loc[future["label"] == "dga", "family"]))
    out = {}
    for model_type, (model, file_name, sha256) in models.items():
        p_seen = metrics.dga_probability(model, seen)
        p_future = metrics.dga_probability(model, future)
        recomputed = threshold_metrics(p_seen, p_future, seen, future)
        expected = {m: m1_results["models"][model_type][m] for m in CHECKED_METRICS}
        if recomputed != expected:
            raise ValueError(
                f"{file_name} is not M1's {model_type}: recomputed {recomputed}, M1 {expected}"
            )
        out[model_type] = {
            "file": file_name,
            "sha256": sha256,
            "threshold_metrics_match_m1": True,
            "test_seen": metrics.threshold_free((seen["label"] == "dga").astype(int), p_seen),
            "test_future": metrics.threshold_free((future["label"] == "dga").astype(int), p_future),
            "held_out_families": metrics.per_family(future, p_future, held_out),
        }
    return _rounded(
        {
            "positive_class": "dga",
            "threshold": config.DGA_THRESHOLD,
            "metrics": "roc_auc and pr_auc (average precision) from P(dga); recall at threshold",
            "per_family_rows": "the family's DGA rows against all benign rows of test_future",
            "rows": {"test_seen": len(seen), "test_future": len(future)},
            "models": out,
            "notes": [
                "Regenerated locally with python -m src.report --auc: it needs data/processed/ and "
                "the M1 model files, so CI does not rebuild it.",
                "Each model's threshold metrics were recomputed and equal reports/m1_results.json "
                "before its AUCs were computed.",
                "pr_auc_baseline is the family's share of the rows scored (the PR-AUC of a random "
                "ranking); per-family PR-AUC is not comparable with the overall one.",
            ],
        }
    )


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.ROOT_DIR).as_posix()
    except ValueError:
        return path.name


def write_auc_report(
    processed_dir: Path = config.PROCESSED_DIR,
    pointer_path: Path = config.PRODUCTION_MODEL_POINTER,
    output: Path = config.M1_AUC_REPORT_PATH,
) -> dict:
    m1_results = json.loads(config.RESULTS_REPORT_PATH.read_text(encoding="utf-8"))
    seen = pd.read_parquet(processed_dir / config.TEST_SEEN_PATH.name)
    future = pd.read_parquet(processed_dir / config.TEST_FUTURE_PATH.name)
    models = {
        model_type: (joblib.load(path), portable(path), file_sha256(path))
        for model_type, path in m1_model_files(m1_results, pointer_path).items()
    }
    report = build_auc_report(seen, future, models, m1_results)
    write_json(output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild reports/ (see the module docstring).")
    parser.add_argument(
        "--auc",
        action="store_true",
        help="write reports/m1_auc.json from the M1 model files instead (local only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if parse_args(argv).auc:
        write_auc_report()
        return
    m1_uri = prepare_m1_snapshot()
    write_reports(tracking_uri=m1_uri)
    stores = [("local", config.mlflow_tracking_uri()), ("m1_snapshot", m1_uri)]
    report = build_production_report(config.PRODUCTION_MODEL_POINTER, stores)
    write_json(config.PRODUCTION_REPORT_PATH, report)


if __name__ == "__main__":
    main()
