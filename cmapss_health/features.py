"""Time-series feature engineering (per engine, strictly causal: only past cycles are used)."""
from __future__ import annotations

import pandas as pd


class FeatureBuilder:
    def __init__(self, sensors: list[str], windows=(5, 15), include_cycle: bool = True):
        self.sensors = list(sensors)
        self.windows = tuple(windows)
        self.include_cycle = include_cycle

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """df must be sorted by (unit, cycle) with a clean RangeIndex."""
        parts = [df[self.sensors]]
        g = df.groupby("unit")[self.sensors]
        for w in self.windows:
            roll = g.rolling(w, min_periods=1)
            mean = roll.mean().reset_index(level=0, drop=True).sort_index()
            std = roll.std().reset_index(level=0, drop=True).sort_index().fillna(0.0)
            delta = g.diff(periods=max(w - 1, 1)).fillna(0.0)  # trend over the window
            parts.append(mean.add_suffix(f"_mean{w}"))
            parts.append(std.add_suffix(f"_std{w}"))
            parts.append(delta.add_suffix(f"_delta{w}"))
        X = pd.concat(parts, axis=1)
        if self.include_cycle:
            X["cycle"] = df["cycle"].to_numpy()
        return X
