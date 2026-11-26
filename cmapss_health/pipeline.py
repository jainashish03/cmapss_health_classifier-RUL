"""End-to-end health-stage pipeline: scaling -> features -> classifier (single picklable object)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import OP_COLS, SENSOR_COLS
from .features import FeatureBuilder
from .modeling import build_model
from .preprocessing import ConditionScaler, detect_n_conditions


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


class HealthPipeline:
    def __init__(self, cfg: dict, model_name: str | None = None):
        self.cfg = cfg
        self.model_name = model_name or cfg["model"]["name"]
        self.stage_names = list(cfg["labeling"]["stage_names"])

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "HealthPipeline":
        """df: raw columns + 'stage' (int labels). Fitted on training engines only."""
        df = _sort(df)
        seed = self.cfg.get("seed", 42)
        n_cond = self.cfg["preprocessing"]["n_conditions"]
        if n_cond == "auto":
            n_cond = detect_n_conditions(df)
        self.n_conditions_ = int(n_cond)
        self.scaler_ = ConditionScaler(self.n_conditions_, random_state=seed).fit(df)
        fcfg = self.cfg["features"]
        self.builder_ = FeatureBuilder(
            self.scaler_.keep_, fcfg["windows"], fcfg.get("include_cycle", True)
        )
        X = self.builder_.transform(self.scaler_.transform(df))
        self.feature_names_ = list(X.columns)
        self.model_ = build_model(self.model_name, seed)
        self.model_.fit(X, df["stage"].to_numpy())
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    # ------------------------------------------------------------- inference
    def _features(self, df: pd.DataFrame):
        df = _sort(df)
        X = self.builder_.transform(self.scaler_.transform(df))
        return df, X[self.feature_names_]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        _, X = self._features(df)
        p = self.model_.predict_proba(X)
        full = np.zeros((len(X), len(self.stage_names)))
        full[:, self.classes_.astype(int)] = p
        return full

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-cycle predictions (rows sorted by unit, cycle). Keeps RUL/stage if present."""
        df, X = self._features(df)
        p = self.model_.predict_proba(X)
        full = np.zeros((len(X), len(self.stage_names)))
        full[:, self.classes_.astype(int)] = p
        pred = full.argmax(axis=1)
        keep = [c for c in ["unit", "cycle", "RUL", "stage"] if c in df.columns]
        out = df[keep].copy()
        out = out.rename(columns={"stage": "true_stage_id"})
        out["pred_stage_id"] = pred
        out["pred_stage"] = [self.stage_names[i] for i in pred]
        for i, n in enumerate(self.stage_names):
            out[f"p_{n}"] = full[:, i]
        return out

    def predict_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """Current health stage of every engine (its most recent cycle)."""
        return self.predict_frame(df).groupby("unit").tail(1).reset_index(drop=True)

    # ------------------------------------------------------------ importance
    def feature_importance(self, df: pd.DataFrame | None = None, n: int = 3000, seed: int = 42):
        m = self.model_
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "steps") and hasattr(m.steps[-1][1], "coef_"):
            imp = np.abs(m.steps[-1][1].coef_).mean(axis=0)
        elif df is not None:  # e.g. HistGradientBoosting -> permutation importance
            d, X = self._features(df)
            idx = np.random.RandomState(seed).choice(len(X), min(n, len(X)), replace=False)
            r = permutation_importance(
                m, X.iloc[idx], d["stage"].to_numpy()[idx], n_repeats=3,
                random_state=seed, scoring="f1_macro", n_jobs=-1,
            )
            imp = r.importances_mean
        else:
            return None
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)

    # ------------------------------------------------------------------- io
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "HealthPipeline":
        return joblib.load(path)
