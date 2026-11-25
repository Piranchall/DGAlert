"""Choose the retraining model's fixed parameters once, on the validation split (run once).

Fits HistGradientBoostingClassifier on train_fit for every point of GRID, scores F1 on validation
(dga = P(dga) >= config.DGA_THRESHOLD), and writes every result to reports/retrain/
hgb_param_search.json. The winner is the best validation F1; points within TIE_MARGIN of it count
as tied and the cheapest of those wins (fewest iterations, then fewest leaves, then the larger
min_samples_leaf). The chosen values are copied by hand into config.HGB_PARAMS.

Validation holds only training families, so the choice does not look at any held-out family or at
the frozen eval set.

Needs data/processed/ (python -m src.prepare_data, then python -m src.prepare_retrain).
Run with: python -m scripts.select_hgb_params
"""

import itertools
import json
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from src import config
from src.features import FEATURE_NAMES

GRID: dict[str, list] = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 400],
    "max_leaf_nodes": [31, 63],
    "min_samples_leaf": [20, 50],
}
FIXED: dict = {"early_stopping": False, "random_state": config.SEED}
TIE_MARGIN = 0.001
OUTPUT = config.REPORTS_DIR / "retrain" / "hgb_param_search.json"


def cost(params: dict) -> tuple:
    """Smaller is cheaper: fewer iterations, fewer leaves, then more rows per leaf (smoother)."""
    return (params["max_iter"], params["max_leaf_nodes"], -params["min_samples_leaf"])


def main() -> None:
    train = pd.read_parquet(config.TRAIN_FIT_PATH)
    validation = pd.read_parquet(config.VALIDATION_PATH)
    y_train = (train["label"] == "dga").astype(int)
    y_val = (validation["label"] == "dga").astype(int)

    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values, strict=True))
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(**params, **FIXED)
        model.fit(train[FEATURE_NAMES], y_train)
        fit_s = time.perf_counter() - start
        proba = model.predict_proba(validation[FEATURE_NAMES])[:, 1]
        f1 = float(f1_score(y_val, proba >= config.DGA_THRESHOLD))
        results.append({"params": params, "validation_f1": round(f1, 4), "fit_s": round(fit_s, 1)})
        print(f"{params}  validation F1 {f1:.4f}  ({fit_s:.1f} s)")

    best_f1 = max(r["validation_f1"] for r in results)
    tied = [r for r in results if r["validation_f1"] >= best_f1 - TIE_MARGIN]
    chosen = min(tied, key=lambda r: cost(r["params"]))
    report = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "metric": "F1 on validation, dga = P(dga) >= DGA_THRESHOLD",
        "fixed": FIXED,
        "grid": GRID,
        "tie_margin": TIE_MARGIN,
        "best_validation_f1": best_f1,
        "chosen": {**chosen["params"], **FIXED},
        "chosen_validation_f1": chosen["validation_f1"],
        "points_within_tie_margin": len(tied),
        "results": sorted(results, key=lambda r: -r["validation_f1"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nchosen: {report['chosen']} (validation F1 {chosen['validation_f1']}, best {best_f1})")
    print(f"{len(tied)} of {len(results)} points within {TIE_MARGIN} of the best")


if __name__ == "__main__":
    main()
