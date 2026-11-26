"""Loading raw C-MAPSS text files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INDEX_COLS = ["unit", "cycle"]
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + OP_COLS + SENSOR_COLS
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _find_file(data_dir: str | Path, name: str) -> Path:
    d = Path(data_dir)
    direct = d / name
    if direct.exists():
        return direct
    matches = sorted(d.rglob(name)) if d.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find '{name}' under '{d.resolve()}'.\n"
        "Download the NASA C-MAPSS turbofan dataset from Kaggle, unzip it, and copy "
        "train_FD00X.txt, test_FD00X.txt and RUL_FD00X.txt into data/raw/ "
        "(or run with --synthetic for a quick smoke test)."
    )


def read_engine_file(path: str | Path) -> pd.DataFrame:
    """Read a train_/test_ file (whitespace separated, 26 columns, no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]  # some copies have trailing empty columns
    df.columns = COLUMNS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df.sort_values(INDEX_COLS).reset_index(drop=True)


def read_rul_file(path: str | Path) -> pd.Series:
    """RUL_FD00X.txt: one value per test engine (RUL at its last observed cycle)."""
    s = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(int)
    s.index = range(1, len(s) + 1)  # engine ids are 1-based
    s.name = "rul_last"
    return s


def load_subset(data_dir: str | Path, subset: str = "FD001"):
    """Return (train_df, test_df, rul_last) for one C-MAPSS sub-dataset."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset}")
    train = read_engine_file(_find_file(data_dir, f"train_{subset}.txt"))
    test = read_engine_file(_find_file(data_dir, f"test_{subset}.txt"))
    rul = read_rul_file(_find_file(data_dir, f"RUL_{subset}.txt"))
    return train, test, rul
