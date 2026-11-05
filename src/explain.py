"""Per-feature explanations of single predictions, toward the "dga" class.

build_explainer() is called once (in the API's lifespan handler) and returns a DomainExplainer:
- gb, rf:  shap.TreeExplainer. GB explains its log-odds; RF explains P(dga) (SHAP's "raw" output
           for a random forest is the averaged leaf probability).
- logreg:  no SHAP. The pipeline is StandardScaler + LogisticRegression on its own feature subset,
           so each selected feature contributes coefficient x standardized value to the log-odds
           (base value = intercept). Features the pipeline does not use contribute 0.

In every case base_value + sum(contributions over all features) = the model's output for that
domain (log-odds for gb/logreg, P(dga) for rf), and a positive contribution pushes toward "dga".
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features import FEATURE_NAMES, feature_frame

DGA_CLASS = 1  # train.py encodes dga as 1, benign as 0
TREE_MODEL_TYPES = ("gb", "rf")


@dataclass
class Explanation:
    """Result of DomainExplainer.explain(): base value and the top contributing features."""

    base_value: float
    top_contributing_features: list[dict[str, float | str]]


def dga_index(model: Any) -> int:
    """Column of the dga class in predict_proba / per-class SHAP output."""
    return list(model.classes_).index(DGA_CLASS)


def selected_features(column_transformer: ColumnTransformer) -> list[str]:
    """Names of the input columns a fitted ColumnTransformer actually uses (in output order)."""
    names: list[str] = []
    for _, transformer, columns in column_transformer.transformers_:
        if transformer != "drop":
            names.extend(columns)
    return names


class DomainExplainer:
    """Explains one model's predictions; build it once and reuse it for every request."""

    def __init__(self, model: Any, model_type: str) -> None:
        self.model = model
        self.model_type = model_type
        self.dga_column = dga_index(model)
        if model_type in TREE_MODEL_TYPES:
            self.tree_explainer = shap.TreeExplainer(model)
        elif model_type == "logreg":
            if not isinstance(model, Pipeline):
                raise TypeError("logreg model must be a scikit-learn Pipeline")
            self.tree_explainer = None
        else:
            raise ValueError(f"No explainer for model type {model_type!r}")

    def contributions(self, domain: str) -> tuple[float, np.ndarray, np.ndarray]:
        """(base value, contribution per FEATURE_NAMES, feature values) for one domain."""
        x = feature_frame(domain)
        values = x.to_numpy()[0]
        if self.tree_explainer is not None:
            base, contrib = self._tree_contributions(x)
        else:
            base, contrib = self._logreg_contributions(x)
        return base, contrib, values

    def _tree_contributions(self, x) -> tuple[float, np.ndarray]:
        shap_values = self.tree_explainer.shap_values(x)
        expected = np.atleast_1d(self.tree_explainer.expected_value)
        if isinstance(shap_values, list):  # older SHAP: one array per class
            shap_values = shap_values[self.dga_column]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:  # per-class output (RF): (rows, features, classes)
            shap_values = shap_values[:, :, self.dga_column]
        # Binary GB has a single output (log-odds of the dga class); per-class models have one each.
        base = expected[self.dga_column] if len(expected) > 1 else expected[0]
        return float(base), shap_values[0]

    def _logreg_contributions(self, x) -> tuple[float, np.ndarray]:
        preprocess, clf = self.model[:-1], self.model[-1]
        standardized = preprocess.transform(x)[0]
        coefs = clf.coef_[0]  # binary: coefficients for classes_[1]
        sign = 1.0 if self.dga_column == 1 else -1.0
        contrib = np.zeros(len(FEATURE_NAMES))
        for name, coef, z in zip(
            selected_features(preprocess[-1]), coefs, standardized, strict=True
        ):
            contrib[FEATURE_NAMES.index(name)] = sign * coef * z
        return float(sign * clf.intercept_[0]), contrib

    def explain(self, domain: str, top_k: int | None = 5) -> Explanation:
        """Top features by absolute contribution toward "dga" (top_k=None returns all)."""
        base, contrib, values = self.contributions(domain)
        order = np.argsort(-np.abs(contrib), kind="stable")
        if top_k is not None:
            order = order[:top_k]
        top = [
            {
                "feature": FEATURE_NAMES[i],
                "value": float(values[i]),
                "contribution": float(contrib[i]),
            }
            for i in order
        ]
        return Explanation(base_value=base, top_contributing_features=top)


def build_explainer(model: Any, model_type: str) -> DomainExplainer:
    """Build the explainer for a loaded model (model_type comes from latest.json)."""
    return DomainExplainer(model, model_type)
