"""Train models, evaluate on test_seen and test_future, log to MLflow, save versioned artifacts.

Three model types (one MLflow parent run each, experiment config.MLFLOW_EXPERIMENT_NAME):
- logreg: basic features only (no dictionary / n-gram features), StandardScaler + LogisticRegression
- rf:     all features, RandomForestClassifier
- gb:     all features, GradientBoostingClassifier tuned with GridSearchCV (GroupKFold on sld,
          on a stratified subsample of train); every grid point is a nested child run, and the
          best parameters are refit on the full train set.

Every saved model is a scikit-learn estimator that takes a DataFrame with all FEATURE_NAMES
columns (logreg selects its own subset inside its pipeline), so callers never need to know
which features a model uses. models/latest.json points at the run with the best f1_seen.

Run with: python -m src.train --model all   (or logreg / rf / gb)
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURE_NAMES

MODEL_TYPES = ["logreg", "rf", "gb"]

# Character-level features only: no dictionary and no n-gram features.
BASIC_FEATURES: list[str] = [
    "length",
    "digit_ratio",
    "vowel_ratio",
    "hyphen_count",
    "entropy",
    "unique_char_ratio",
    "longest_consonant_run",
    "digit_letter_transitions",
]

LOGREG_PARAMS: dict = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
RF_PARAMS: dict = {"n_estimators": 100, "class_weight": "balanced", "n_jobs": -1}

GB_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}
GB_CV_MAX_ROWS = 40_000  # the grid search runs on a stratified subsample of at most this many rows
GB_CV_FOLDS = 3

PER_FAMILY_FILE = "per_family_recall.json"
TABLE_COLUMNS = [
    "f1_seen",
    "precision_seen",
    "recall_seen",
    "f1_future",
    "recall_future",
    "train_time_s",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the train / test_seen / test_future parquet files written by src.prepare_data."""
    splits = {}
    for name in ("train", "test_seen", "test_future"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run python -m src.prepare_data first")
        splits[name] = pd.read_parquet(path)
    return splits


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix (all FEATURE_NAMES, in order) and target (1 = dga, 0 = benign)."""
    return df[FEATURE_NAMES], (df["label"] == "dga").astype(int)


def stratified_subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """At most max_rows rows, keeping each family's share (so benign/dga stays balanced)."""
    if len(df) <= max_rows:
        return df
    sample, _ = train_test_split(
        df, train_size=max_rows, stratify=df["family"], random_state=config.SEED
    )
    return sample


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_logreg() -> Pipeline:
    """Keep only BASIC_FEATURES, standardize them, then logistic regression."""
    select_and_scale = ColumnTransformer([("scale", StandardScaler(), BASIC_FEATURES)])
    clf = LogisticRegression(random_state=config.SEED, **LOGREG_PARAMS)
    return Pipeline([("features", select_and_scale), ("clf", clf)])


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=config.SEED, **RF_PARAMS)


def build_gb(**params) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=config.SEED, **params)


def tune_gb(train: pd.DataFrame, grid: dict[str, list], max_rows: int) -> tuple[dict, float]:
    """Grid-search GB on a stratified subsample of train, GroupKFold grouped by sld.

    Every grid point is logged as a nested MLflow child run of the active (parent) run.
    Returns (best params, best mean CV f1).
    """
    sample = stratified_subsample(train, max_rows)
    x, y = xy(sample)
    search = GridSearchCV(
        build_gb(),
        grid,
        scoring="f1",
        cv=GroupKFold(n_splits=GB_CV_FOLDS),
        n_jobs=-1,
        refit=False,  # the best params are refit below on the full train set, not the subsample
    )
    print(
        f"  grid search: {len(ParameterGrid(grid))} points x {GB_CV_FOLDS} folds "
        f"on {len(sample)} rows"
    )
    search.fit(x, y, groups=sample["sld"])

    results = search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"gb-grid-{i:02d}", nested=True):
            mlflow.set_tag("model_type", "gb_grid_point")
            mlflow.log_params({**params, "cv_rows": len(sample), "cv_folds": GB_CV_FOLDS})
            mlflow.log_metrics(
                {
                    "cv_f1_mean": float(results["mean_test_score"][i]),
                    "cv_f1_std": float(results["std_test_score"][i]),
                    "cv_rank": int(results["rank_test_score"][i]),
                    "cv_fit_time_mean_s": float(results["mean_fit_time"][i]),
                }
            )
    return search.best_params_, float(search.best_score_)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def predict(model, df: pd.DataFrame) -> pd.Series:
    """1 = dga where P(dga) >= config.DGA_THRESHOLD, else 0."""
    proba = model.predict_proba(df[FEATURE_NAMES])[:, 1]
    return pd.Series((proba >= config.DGA_THRESHOLD).astype(int), index=df.index)


def evaluate(model, df: pd.DataFrame) -> dict[str, float]:
    """Precision, recall and F1 with dga as the positive class."""
    _, y = xy(df)
    y_pred = predict(model, df)
    return {
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }


def per_family_recall(model, df: pd.DataFrame) -> dict[str, float]:
    """Share of each family's rows predicted with their true label.

    For a DGA family this is its DGA recall; for "benign" it is the share predicted benign.
    """
    _, y = xy(df)
    correct = predict(model, df) == y
    by_family = correct.groupby(df["family"]).mean().sort_index()
    return {family: round(float(r), 4) for family, r in by_family.items()}


# ---------------------------------------------------------------------------
# One MLflow run per model type
# ---------------------------------------------------------------------------


def new_version(models_dir: Path) -> str:
    """UTC timestamp YYYYmmdd_HHMMSS; waits a moment if that model file already exists."""
    while True:
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        if not (models_dir / f"model_{version}.joblib").exists():
            return version
        time.sleep(0.2)


def portable_path(path: Path) -> str:
    """Path relative to the repo root when inside it (works on any machine), else absolute."""
    try:
        return path.resolve().relative_to(config.ROOT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def train_one(
    model_type: str,
    splits: dict[str, pd.DataFrame],
    models_dir: Path,
    gb_grid: dict[str, list],
    gb_max_rows: int,
) -> dict:
    """Train, evaluate, log and save one model inside its own MLflow parent run."""
    train = splits["train"]
    x_train, y_train = xy(train)
    print(f"\n[{model_type}] training on {len(train)} rows")

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.set_tag("model_type", model_type)
        params: dict = {"model_type": model_type, "seed": config.SEED, "train_rows": len(train)}

        if model_type == "logreg":
            model = build_logreg()
            params |= {**LOGREG_PARAMS, "features": "basic", "n_features": len(BASIC_FEATURES)}
        elif model_type == "rf":
            model = build_rf()
            params |= {**RF_PARAMS, "features": "all", "n_features": len(FEATURE_NAMES)}
        elif model_type == "gb":
            best_params, cv_f1 = tune_gb(train, gb_grid, gb_max_rows)
            print(f"  best grid point: {best_params} (cv f1 {cv_f1:.4f})")
            mlflow.log_metric("cv_f1_best", cv_f1)
            model = build_gb(**best_params)
            params |= {**best_params, "features": "all", "n_features": len(FEATURE_NAMES)}
            params |= {"cv_folds": GB_CV_FOLDS, "cv_max_rows": gb_max_rows}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - start

        seen = evaluate(model, splits["test_seen"])
        future = evaluate(model, splits["test_future"])
        metrics = {
            "f1_seen": seen["f1"],
            "precision_seen": seen["precision"],
            "recall_seen": seen["recall"],
            "f1_future": future["f1"],
            "recall_future": future["recall"],
            "train_time_s": train_time,
        }
        families = {
            "test_seen": per_family_recall(model, splits["test_seen"]),
            "test_future": per_family_recall(model, splits["test_future"]),
        }

        models_dir.mkdir(parents=True, exist_ok=True)
        version = new_version(models_dir)
        model_path = models_dir / f"model_{version}.joblib"
        joblib.dump(model, model_path, compress=3)
        params |= {"version": version, "model_path": portable_path(model_path)}

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(families, PER_FAMILY_FILE)
        print(f"  f1_seen {seen['f1']:.4f}  f1_future {future['f1']:.4f}  -> {model_path.name}")

        return {
            "model_type": model_type,
            "version": version,
            "path": portable_path(model_path),
            "mlflow_run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "metrics": metrics,
            "per_family_recall": families,
        }


def write_latest(result: dict, pointer: Path) -> dict:
    """Write models/latest.json for one training result (fields per CLAUDE.md + recall_future)."""
    latest = {
        "path": result["path"],
        "version": result["version"],
        "model_type": result["model_type"],
        "f1_seen": result["metrics"]["f1_seen"],
        "f1_future": result["metrics"]["f1_future"],
        "recall_future": result["metrics"]["recall_future"],
        "mlflow_run_id": result["mlflow_run_id"],
        "trained_at": result["trained_at"],
    }
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    return latest


def setup_mlflow(tracking_uri: str, artifact_dir: Path, experiment_name: str) -> None:
    """Point MLflow at the given store and create the experiment if it does not exist yet."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment_name)


def run_training(
    model_types: list[str],
    processed_dir: Path = config.PROCESSED_DIR,
    models_dir: Path = config.MODELS_DIR,
    pointer: Path = config.LATEST_MODEL_POINTER,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    artifact_dir: Path = config.MLFLOW_ARTIFACT_DIR,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    gb_grid: dict[str, list] = GB_GRID,
    gb_max_rows: int = GB_CV_MAX_ROWS,
) -> list[dict]:
    """Train each model type, then point latest.json at the result with the best f1_seen."""
    splits = load_splits(processed_dir)
    setup_mlflow(tracking_uri, artifact_dir, experiment_name)
    results = [
        train_one(model_type, splits, models_dir, gb_grid, gb_max_rows)
        for model_type in model_types
    ]
    best = max(results, key=lambda r: r["metrics"]["f1_seen"])
    write_latest(best, pointer)
    print(f"\nlatest.json -> {best['model_type']} ({best['path']})")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """One row per model with the metrics in TABLE_COLUMNS."""
    rows = {r["model_type"]: r["metrics"] for r in results}
    return pd.DataFrame.from_dict(rows, orient="index")[TABLE_COLUMNS].round(4)


def held_out_recall_table(results: list[dict]) -> pd.DataFrame:
    """Recall per held-out family (rows) for each model (columns)."""
    columns = {}
    for r in results:
        future = r["per_family_recall"]["test_future"]
        columns[r["model_type"]] = {f: v for f, v in future.items() if f != "benign"}
    return pd.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DGAlert models and log them to MLflow.")
    parser.add_argument("--model", choices=[*MODEL_TYPES, "all"], default="all")
    args = parser.parse_args()
    model_types = MODEL_TYPES if args.model == "all" else [args.model]

    results = run_training(model_types)

    print("\nComparison (test_seen / test_future; positive class = dga):")
    print(comparison_table(results).to_string())

    best = max(results, key=lambda r: r["metrics"]["f1_seen"])["model_type"]
    held_out = held_out_recall_table(results).sort_values(best)
    print(f"\n5 held-out families with the lowest recall (sorted by {best}, the best model):")
    print(held_out.head(5).to_string())


if __name__ == "__main__":
    main()
