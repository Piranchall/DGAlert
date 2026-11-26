"""Threshold-free metrics: ROC-AUC and PR-AUC (average precision) from P(dga).

Recall, precision and F1 depend on the decision threshold (config.DGA_THRESHOLD), which was fixed
on the seen families. ROC-AUC and PR-AUC score how well P(dga) ranks DGA domains above benign ones,
whatever the threshold, so they separate "ranked wrongly" from "ranked right, cut at the wrong
place" on families the model never saw.

Per family, a family's DGA rows are scored against all benign rows of the same set. PR-AUC depends
on the share of positives, so each per-family entry records that share as `pr_auc_baseline` (the
PR-AUC of a model that ranks at random).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.features import FEATURE_NAMES


def dga_probability(model: Any, df: pd.DataFrame) -> np.ndarray:
    """P(dga) for every row (the model takes all FEATURE_NAMES columns)."""
    return model.predict_proba(df[FEATURE_NAMES])[:, 1]


def threshold_free(y: pd.Series | np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """ROC-AUC and PR-AUC (dga = positive); both None when the rows hold only one class."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def against_benign(df: pd.DataFrame, p: np.ndarray, families: list[str]) -> dict[str, Any]:
    """Threshold-free metrics of the given families' DGA rows (pooled) against all benign rows."""
    is_benign = (df["label"] == "benign").to_numpy()
    positive = ((df["label"] == "dga") & df["family"].isin(families)).to_numpy()
    rows = is_benign | positive
    return {
        **threshold_free(positive[rows].astype(int), p[rows]),
        "positives": int(positive.sum()),
        "benign": int(is_benign.sum()),
    }


def per_family(df: pd.DataFrame, p: np.ndarray, families: list[str]) -> dict[str, dict]:
    """For each family: ROC-AUC, PR-AUC, its PR-AUC baseline and recall at the threshold."""
    out = {}
    for family in families:
        scores = against_benign(df, p, [family])
        positives = (df["family"] == family).to_numpy() & (df["label"] == "dga").to_numpy()
        share = scores["positives"] / (scores["positives"] + scores["benign"])
        out[family] = {
            "roc_auc": scores["roc_auc"],
            "pr_auc": scores["pr_auc"],
            "pr_auc_baseline": share,
            "recall": float((p[positives] >= config.DGA_THRESHOLD).mean())
            if scores["positives"]
            else None,
            "positives": scores["positives"],
        }
    return out
