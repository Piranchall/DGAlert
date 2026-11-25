"""Tests for scripts/ablation.py on small splits built from data/sample.

gb uses a few small trees here; the real run uses M1's best grid point. MLflow goes to a temporary
store; the real stores and reports/ablation.json are never touched.
"""

import copy
import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from scripts.ablation import (
    CELLS,
    FEATURE_SETS,
    M1_CELLS,
    METRICS,
    build,
    gb_params_from_m1,
    parse_args,
    reproduction,
    run_ablation,
    write_report,
)
from src import config
from src.features import FEATURE_NAMES, extract_features
from src.train import BASIC_FEATURES, build_logreg, predict, setup_mlflow

SMALL_GB = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
CELL_NAMES = ["logreg_basic", "logreg_all", "gb_basic", "gb_all"]


def featurized(rows: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame([extract_features(d) for d in rows["domain"]], columns=FEATURE_NAMES)
    return pd.concat([rows.reset_index(drop=True), x], axis=1)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    """train / test_seen from seen families and benign, test_future from dyre and matsnu."""
    benign = pd.read_csv(config.TRANCO_SAMPLE_PATH).sample(600, random_state=1)[["domain"]]
    dga = pd.read_csv(config.UMUDGA_SAMPLE_PATH)

    def family(name: str, n: int, seed: int) -> pd.DataFrame:
        rows = dga[dga["family"] == name].sample(n, random_state=seed)
        return rows[["domain", "family"]].assign(label="dga")

    seen = featurized(
        pd.concat(
            [
                benign.iloc[:400].assign(family="benign", label="benign"),
                family("chinad", 150, 2),
                family("locky", 150, 3),
                family("suppobox_1", 100, 4),
            ],
            ignore_index=True,
        )
    )
    future = featurized(
        pd.concat(
            [
                benign.iloc[400:].assign(family="benign", label="benign"),
                family("dyre", 100, 5),
                family("matsnu", 100, 6),
            ],
            ignore_index=True,
        )
    )
    test = seen.index % 4 == 0
    return {
        "train": seen[~test].reset_index(drop=True),
        "test_seen": seen[test].reset_index(drop=True),
        "test_future": future,
    }


def m1_from(report: dict) -> tuple[dict, dict]:
    """M1-style results and per-family files that agree exactly with a report's M1 cells."""
    models = {m1: dict(report["cells"][cell]["metrics"]) for cell, m1 in M1_CELLS.items()}
    families = {
        split: {
            m1: dict(report["cells"][cell]["per_family_recall"][split])
            for cell, m1 in M1_CELLS.items()
        }
        for split in ("test_seen", "test_future")
    }
    results = {"models": models, "gb_best_grid_point": {"params": dict(SMALL_GB)}}
    return results, families


PLACEHOLDER_M1 = (
    {
        "models": {m: dict.fromkeys(METRICS, 0.0) for m in ("logreg", "gb")},
        "gb_best_grid_point": {"params": dict(SMALL_GB)},
    },
    {"test_seen": {"logreg": {}, "gb": {}}, "test_future": {"logreg": {}, "gb": {}}},
)


def run_in_store(store: Path, splits: dict, m1: tuple[dict, dict] = PLACEHOLDER_M1) -> dict:
    setup_mlflow(
        config.sqlite_uri(store / "mlflow.db"),
        store / "art",
        config.MLFLOW_ABLATION_EXPERIMENT_NAME,
    )
    return run_ablation(splits, *m1)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("mlflow")


@pytest.fixture(scope="module")
def report(store: Path, splits: dict) -> dict:
    return run_in_store(store, splits)


# ---------------------------------------------------------------------------
# The grid and its pipelines
# ---------------------------------------------------------------------------


def test_the_grid_crosses_two_models_with_two_feature_sets() -> None:
    assert ["_".join(c) for c in CELLS] == CELL_NAMES
    assert FEATURE_SETS == {"basic": BASIC_FEATURES, "all": FEATURE_NAMES}
    assert len(BASIC_FEATURES) == 8 and len(FEATURE_NAMES) == 15
    assert M1_CELLS == {"logreg_basic": "logreg", "gb_all": "gb"}


@pytest.mark.parametrize(("model", "feature_set"), CELLS)
def test_each_pipeline_uses_exactly_its_feature_set(
    model: str, feature_set: str, splits: dict
) -> None:
    pipeline = build(model, feature_set, SMALL_GB).fit(*_xy(splits["train"]))
    assert pipeline.named_steps["clf"].n_features_in_ == len(FEATURE_SETS[feature_set])

    # Scrambling every column outside the feature set must not change a single prediction.
    unused = [c for c in FEATURE_NAMES if c not in FEATURE_SETS[feature_set]]
    assert len(unused) == (7 if feature_set == "basic" else 0)
    test = splits["test_future"]
    scrambled = test.copy()
    scrambled[unused] = test[unused].sample(frac=1, random_state=0).to_numpy() + 1000
    assert predict(pipeline, test).equals(predict(pipeline, scrambled))


def test_logreg_basic_is_m1s_logreg_pipeline(splits: dict) -> None:
    ours = build("logreg", "basic", SMALL_GB).fit(*_xy(splits["train"]))
    m1 = build_logreg().fit(*_xy(splits["train"]))
    test = splits["test_seen"]
    assert (
        ours.predict_proba(test[FEATURE_NAMES]).tolist()
        == m1.predict_proba(test[FEATURE_NAMES]).tolist()
    )


def test_gb_params_come_from_m1s_best_grid_point() -> None:
    m1 = json.loads(config.RESULTS_REPORT_PATH.read_text(encoding="utf-8"))
    assert gb_params_from_m1(m1) == {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05}


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_NAMES], (df["label"] == "dga").astype(int)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_has_every_cell_with_overall_and_per_family_numbers(
    report: dict, splits: dict
) -> None:
    assert list(report["cells"]) == CELL_NAMES
    for name, cell in report["cells"].items():
        assert cell["n_features"] == len(cell["features"]) == len(FEATURE_SETS[cell["feature_set"]])
        assert set(cell["metrics"]) == set(METRICS)
        assert all(0 <= v <= 1 and round(v, 4) == v for v in cell["metrics"].values()), name
        assert set(cell["per_family_recall"]["test_seen"]) == set(splits["test_seen"]["family"])
        assert set(cell["per_family_recall"]["test_future"]) == {"benign", "dyre", "matsnu"}
    assert report["params"]["gb"] == SMALL_GB
    assert report["rows"] == {name: len(df) for name, df in splits.items()}


def test_effects_are_differences_between_cells(report: dict) -> None:
    cells, fx = report["cells"], report["effects"]
    dyre = {n: c["per_family_recall"]["test_future"]["dyre"] for n, c in cells.items()}
    f1 = {n: c["metrics"]["f1_seen"] for n, c in cells.items()}
    assert fx["headline"]["gb_basic"]["dyre_recall"] == dyre["gb_basic"]
    assert fx["feature_set_effect_all_minus_basic"]["gb"]["dyre_recall"] == round(
        dyre["gb_all"] - dyre["gb_basic"], 4
    )
    assert fx["model_effect_gb_minus_logreg"]["all"]["f1_seen"] == round(
        f1["gb_all"] - f1["logreg_all"], 4
    )


def test_report_is_deterministic_and_has_no_run_ids_times_or_paths(
    report: dict, splits: dict, tmp_path: Path
) -> None:
    again = run_in_store(tmp_path, splits)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_report(report, first)
    write_report(again, second)
    assert first.read_bytes() == second.read_bytes()

    text = first.read_text(encoding="utf-8")
    for leak in ("run_id", "train_time", "mlflow.db", str(config.ROOT_DIR), "\\\\"):
        assert leak not in text, leak
    assert "match to the report's rounding" in " ".join(report["notes"])


def test_one_mlflow_run_per_cell(report: dict, store: Path) -> None:
    client = mlflow.MlflowClient(config.sqlite_uri(store / "mlflow.db"))
    experiment = client.get_experiment_by_name(config.MLFLOW_ABLATION_EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])
    assert sorted(r.info.run_name for r in runs) == sorted(CELL_NAMES)
    for run in runs:
        cell = report["cells"][run.info.run_name]
        assert run.data.tags["feature_set"] == cell["feature_set"]
        assert run.data.params["n_features"] == str(cell["n_features"])
        assert round(run.data.metrics["f1_seen"], 4) == cell["metrics"]["f1_seen"]
        assert "train_time_s" in run.data.metrics


# ---------------------------------------------------------------------------
# Reproduction of M1's two cells
# ---------------------------------------------------------------------------


def test_reproduction_passes_when_the_numbers_agree(report: dict) -> None:
    check = reproduction(report["cells"], *m1_from(report))
    assert all(c["matches_m1"] and c["mismatches"] == [] for c in check.values())
    assert check["gb_all"]["m1_model"] == "gb"


def test_reproduction_lists_every_mismatch(report: dict) -> None:
    results, families = m1_from(report)
    results = copy.deepcopy(results)
    results["models"]["gb"]["f1_seen"] = round(results["models"]["gb"]["f1_seen"] + 0.0001, 4)
    families["test_future"]["logreg"]["dyre"] = -1.0
    check = reproduction(report["cells"], results, families)

    assert not check["gb_all"]["matches_m1"]
    assert [m["what"] for m in check["gb_all"]["mismatches"]] == ["f1_seen"]
    assert check["logreg_basic"]["mismatches"] == [
        {
            "what": "test_future/dyre",
            "m1": -1.0,
            "ablation": report["cells"]["logreg_basic"]["per_family_recall"]["test_future"]["dyre"],
        }
    ]


def test_placeholder_m1_numbers_are_reported_as_mismatches(report: dict) -> None:
    assert not report["m1_reproduction"]["logreg_basic"]["matches_m1"]


def test_cli_defaults() -> None:
    args = parse_args([])
    assert args.processed_dir == config.PROCESSED_DIR
    assert args.output == config.ABLATION_REPORT_PATH == config.REPORTS_DIR / "ablation.json"
