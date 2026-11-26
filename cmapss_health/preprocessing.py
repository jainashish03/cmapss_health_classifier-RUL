"""Operating-condition aware sensor normalisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import OP_COLS, SENSOR_COLS


def detect_n_conditions(df: pd.DataFrame, max_conditions: int = 12) -> int:
    """FD001/FD003 have 1 operating regime, FD002/FD004 have 6."""
    rounded = df[OP_COLS].round({"op1": 0, "op2": 2, "op3": 0}).to_numpy()
    n = len(np.unique(rounded, axis=0))
    return n if n <= max_conditions else 6


class ConditionScaler:
    """Standardise each sensor *within* its operating regime.

    In multi-condition data (FD002/FD004) the operating regime shifts sensor values far more
    than degradation does, so a global z-score would hide the degradation trend. Regimes are
    found with k-means on the three operational settings. Sensors that are constant inside
    every regime are dropped. Everything is fitted on training data only.
    """

    def __init__(self, n_conditions: int = 1, min_std: float = 1e-6, random_state: int = 42):
        self.n_conditions = n_conditions
        self.min_std = min_std
        self.random_state = random_state

    def _cluster(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions <= 1:
            return np.zeros(len(df), dtype=int)
        ops = self.op_scaler_.transform(df[OP_COLS].to_numpy())
        return self.kmeans_.predict(ops)

    def fit(self, df: pd.DataFrame) -> "ConditionScaler":
        if self.n_conditions > 1:
            self.op_scaler_ = StandardScaler().fit(df[OP_COLS].to_numpy())
            self.kmeans_ = KMeans(
                n_clusters=self.n_conditions, n_init=10, random_state=self.random_state
            ).fit(self.op_scaler_.transform(df[OP_COLS].to_numpy()))
        cl = pd.Series(self._cluster(df), index=df.index)
        sens = df[SENSOR_COLS]
        self.mean_ = sens.groupby(cl).mean()
        std = sens.groupby(cl).std(ddof=0)
        self.keep_ = [s for s in SENSOR_COLS if (std[s] > self.min_std).any()]
        self.std_ = std.where(std > self.min_std, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cl = self._cluster(df)
        mean = self.mean_.loc[cl, self.keep_].to_numpy()
        std = self.std_.loc[cl, self.keep_].to_numpy()
        out = df[["unit", "cycle"]].copy()
        out[self.keep_] = (df[self.keep_].to_numpy() - mean) / std
        out["condition"] = cl
        return out
