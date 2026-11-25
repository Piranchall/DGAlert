"""Feature-set ablation: separate the model family from the feature set (M1 review, point A2).

M1 trained logreg on the 8 basic features and the trees on all 15, so "logreg against trees" mixed
two effects. This script fits the full 2 x 2 grid on the M1 splits:

                 basic (8 features)          all (15 features)
    logreg       M1's logreg (reproduced)    new
    gb           new                         M1's gb (reproduced)

- logreg: StandardScaler + LogisticRegression with M1's LOGREG_PARAMS.
- gb: GradientBoostingClassifier with M1's best grid point, read from reports/m1_results.json
  (no new search, so only the feature set changes).

The two cells M1 already trained are fitted again and compared with reports/m1_results.json and
reports/per_family_recall.json; any mismatch is listed in the report, never hidden. Each cell is
one MLflow run in experiment config.MLFLOW_ABLATION_EXPERIMENT_NAME (local store). The results go
to reports/ablation.json: rounded metrics and per-family recall, no run ids, times or paths, so a
rerun on the same machine writes the same bytes.

Needs data/processed/ (python -m src.prepare_data). Takes about 3 minutes, mostly the two gb fits.
Run with: python -m scripts.ablation
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURE_NAMES
from src.train import (
    BASIC_FEATURES,
    LOGREG_PARAMS,
    build_gb,
    evaluate,
    load_splits,
    per_family_recall,
    setup_mlflow,
    xy,
)

FEATURE_SETS: dict[str, list[str]] = {"basic": BASIC_FEATURES, "all": FEATURE_NAMES}
MODELS: list[str] = ["logreg", "gb"]
CELLS: list[tuple[str, str]] = [(m, f) for m in MODELS for f in FEATURE_SETS]
# The cells M1 trained, and the M1 model each one must reproduce.
M1_CELLS: dict[str, str] = {"logreg_basic": "logreg", "gb_all": "gb"}
METRICS: list[str] = ["f1_seen", "precision_seen", "recall_seen", "f1_future", "recall_future"]
# Families the review's claims are about (held out, so only in test_future).
FOCUS_FAMILIES: list[str] = ["dyre", "matsnu"]
DIGITS = 4

# The M1 review's numbers, measured with other settings (HistGradientBoosting with defaults, maybe
# other logreg parameters): rough expectations, not targets.
REVIEW_EXPECTATIONS: dict[str, str] = {
    "logreg_all": "f1_seen about 0.829",
    "trees_basic": "dyre recall about 0.89, against about 0.61 for trees on all 15 features",
}
NOTES: list[str] = [
    "All four cells are trained on the M1 train split and scored on test_seen and test_future "
    "with threshold config.DGA_THRESHOLD; no new parameter search.",
    "Reruns on the same machine write a byte-identical report. On other machines (other CPUs, "
    "BLAS or platforms) gb trees can differ slightly, so reruns match to the report's rounding, "
    "not necessarily byte for byte.",
    "The review's numbers were measured with different settings (HistGradientBoosting with "
    "default parameters, possibly other logreg parameters); they are rough expectations, and this "
    "report states what this run finds.",
]


def cell_name(model: str, feature_set: str) -> str:
    return f"{model}_{feature_set}"


def build(model: str, feature_set: str, gb_params: dict) -> Pipeline:
    """A pipeline that takes all FEATURE_NAMES columns and uses only the cell's feature set.

    logreg on "basic" is the same pipeline as src.train.build_logreg.
    """
    columns = FEATURE_SETS[feature_set]
    if model == "logreg":
        select = ColumnTransformer([("scale", StandardScaler(), columns)])
        clf = LogisticRegression(random_state=config.SEED, **LOGREG_PARAMS)
    elif model == "gb":
        select = ColumnTransformer([("keep", "passthrough", columns)])
        clf = build_gb(**gb_params)
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline([("features", select), ("clf", clf)])


def gb_params_from_m1(m1_results: dict) -> dict:
    """M1's best gb grid point (n_estimators, max_depth, learning_rate)."""
    return dict(m1_results["gb_best_grid_point"]["params"])


def run_cell(
    model: str, feature_set: str, splits: dict[str, pd.DataFrame], gb_params: dict
) -> dict[str, Any]:
    """Fit one cell, log it as one MLflow run, return its rounded metrics and per-family recall."""
    name = cell_name(model, feature_set)
    columns = FEATURE_SETS[feature_set]
    train = splits["train"]
    print(f"[{name}] {len(columns)} features, {len(train)} training rows")

    pipeline = build(model, feature_set, gb_params)
    params = LOGREG_PARAMS if model == "logreg" else gb_params
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"model_type": model, "feature_set": feature_set})
        start = time.perf_counter()
        pipeline.fit(*xy(train))
        train_time = time.perf_counter() - start

        seen = evaluate(pipeline, splits["test_seen"])
        future = evaluate(pipeline, splits["test_future"])
        metrics = {
            "f1_seen": seen["f1"],
            "precision_seen": seen["precision"],
            "recall_seen": seen["recall"],
            "f1_future": future["f1"],
            "recall_future": future["recall"],
        }
        families = {
            "test_seen": per_family_recall(pipeline, splits["test_seen"]),
            "test_future": per_family_recall(pipeline, splits["test_future"]),
        }
        mlflow.log_params(
            {
                **params,
                "model_type": model,
                "feature_set": feature_set,
                "n_features": len(columns),
                "features": json.dumps(columns),
                "seed": config.SEED,
                "train_rows": len(train),
            }
        )
        mlflow.log_metrics({**metrics, "train_time_s": train_time})
        mlflow.log_dict(families, config.PER_FAMILY_FILE)
    print(f"  f1_seen {metrics['f1_seen']:.4f}  recall_future {metrics['recall_future']:.4f}")

    return {
        "model": model,
        "feature_set": feature_set,
        "n_features": len(columns),
        "features": columns,
        "params": dict(params),
        "metrics": {k: round(v, DIGITS) for k, v in metrics.items()},
        "per_family_recall": families,
    }


def reproduction(cells: dict[str, dict], m1_results: dict, m1_families: dict) -> dict[str, Any]:
    """Compare the two cells M1 trained with M1's committed numbers, to the report's rounding."""
    out = {}
    for cell, m1_model in M1_CELLS.items():
        mismatches = []
        for metric in METRICS:
            expected = m1_results["models"][m1_model][metric]
            got = cells[cell]["metrics"][metric]
            if got != expected:
                mismatches.append({"what": metric, "m1": expected, "ablation": got})
        for split, by_model in m1_families.items():
            expected_families = by_model[m1_model]
            got_families = cells[cell]["per_family_recall"][split]
            for family in sorted(set(expected_families) | set(got_families)):
                expected = expected_families.get(family)
                got = got_families.get(family)
                if got != expected:
                    what = f"{split}/{family}"
                    mismatches.append({"what": what, "m1": expected, "ablation": got})
        out[cell] = {
            "m1_model": m1_model,
            "m1_f1_seen": m1_results["models"][m1_model]["f1_seen"],
            "ablation_f1_seen": cells[cell]["metrics"]["f1_seen"],
            "matches_m1": not mismatches,
            "mismatches": mismatches,
        }
    return out


def _focus(cell: dict) -> dict[str, float | None]:
    """The headline numbers of one cell: f1_seen, recall_future and the focus families' recall."""
    future = cell["per_family_recall"]["test_future"]
    return {
        "f1_seen": cell["metrics"]["f1_seen"],
        "recall_future": cell["metrics"]["recall_future"],
        **{f"{family}_recall": future.get(family) for family in FOCUS_FAMILIES},
    }


def _difference(after: dict, before: dict) -> dict[str, float | None]:
    return {
        key: None
        if after[key] is None or before[key] is None
        else round(after[key] - before[key], DIGITS)
        for key in after
    }


def effects(cells: dict[str, dict]) -> dict[str, Any]:
    """Each factor's effect with the other held fixed (positive = the second level is higher)."""
    focus = {name: _focus(cell) for name, cell in cells.items()}
    return {
        "headline": focus,
        "feature_set_effect_all_minus_basic": {
            model: _difference(focus[cell_name(model, "all")], focus[cell_name(model, "basic")])
            for model in MODELS
        },
        "model_effect_gb_minus_logreg": {
            fs: _difference(focus[cell_name("gb", fs)], focus[cell_name("logreg", fs)])
            for fs in FEATURE_SETS
        },
    }


def run_ablation(
    splits: dict[str, pd.DataFrame],
    m1_results: dict,
    m1_families: dict,
    gb_params: dict | None = None,
) -> dict[str, Any]:
    """All four cells (in the active MLflow experiment) and the report; gb defaults to M1's."""
    gb_params = gb_params_from_m1(m1_results) if gb_params is None else gb_params
    cells = {cell_name(m, f): run_cell(m, f, splits, gb_params) for m, f in CELLS}
    return {
        "question": "Is the logreg/trees gap the model family or the feature set? (M1 review, A2)",
        "positive_class": "dga",
        "threshold": config.DGA_THRESHOLD,
        "seed": config.SEED,
        "rows": {name: len(df) for name, df in splits.items()},
        "feature_sets": FEATURE_SETS,
        "params": {"logreg": dict(LOGREG_PARAMS), "gb": dict(gb_params)},
        "cells": cells,
        "effects": effects(cells),
        "m1_reproduction": reproduction(cells, m1_results, m1_families),
        "review_expectations": REVIEW_EXPECTATIONS,
        "notes": NOTES,
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=config.ABLATION_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    splits = load_splits(args.processed_dir)
    m1_results = json.loads(config.RESULTS_REPORT_PATH.read_text(encoding="utf-8"))
    m1_families = json.loads(config.PER_FAMILY_REPORT_PATH.read_text(encoding="utf-8"))
    setup_mlflow(
        config.mlflow_tracking_uri(),
        config.mlflow_artifact_dir(),
        config.MLFLOW_ABLATION_EXPERIMENT_NAME,
    )
    report = run_ablation(splits, m1_results, m1_families)
    write_report(report, args.output)

    print(f"\nwrote {args.output}")
    for cell, check in report["m1_reproduction"].items():
        status = "matches M1" if check["matches_m1"] else f"{len(check['mismatches'])} MISMATCHES"
        print(f"  {cell}: {status}")
        for m in check["mismatches"]:
            print(f"    {m['what']}: M1 {m['m1']}, ablation {m['ablation']}")


if __name__ == "__main__":
    main()
