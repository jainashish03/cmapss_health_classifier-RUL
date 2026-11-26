"""Turn run-to-failure data into health-stage labels.

C-MAPSS has no health label, only run-to-failure trajectories. We derive:
  * RUL (Remaining Useful Life) for every row
  * a discrete health stage from RUL (or from the fraction of life consumed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """Training engines run to failure: RUL = last_cycle - cycle."""
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def add_rul_test(df: pd.DataFrame, rul_last: pd.Series) -> pd.DataFrame:
    """Test engines are cut before failure; RUL_FD00X.txt gives RUL at the last cycle."""
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["unit"].map(rul_last) + last_cycle - df["cycle"]
    return df


def assign_stage(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Integer stage id: 0 = healthiest ... K = most critical."""
    lab = cfg["labeling"]
    names = lab["stage_names"]
    if lab["method"] == "rul":
        bins = np.sort(np.asarray(lab["rul_thresholds"], dtype=float))
        # digitize(right=True): rul<=b0 ->0 ... rul>b_last -> K ; invert so that high RUL = 0
        stage = len(bins) - np.digitize(df["RUL"].to_numpy(), bins, right=True)
    elif lab["method"] == "life_fraction":
        bins = np.sort(np.asarray(lab["life_fraction_thresholds"], dtype=float))
        frac = df["cycle"].to_numpy() / (df["cycle"].to_numpy() + df["RUL"].to_numpy())
        stage = np.digitize(frac, bins, right=True)
    else:
        raise ValueError("labeling.method must be 'rul' or 'life_fraction'")
    if len(names) != len(bins) + 1:
        raise ValueError("stage_names must have len(thresholds)+1 entries")
    return pd.Series(stage.astype(int), index=df.index, name="stage")


def label_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["stage"] = assign_stage(df, cfg)
    return df
