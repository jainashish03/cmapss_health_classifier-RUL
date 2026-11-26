"""Model factory."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("random_forest", "hist_gradient_boosting", "logistic_regression", "xgboost")


def build_model(name: str, seed: int = 42):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=seed,
        )
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install xgboost to use model 'xgboost'") from e
        return XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
