"""Synthetic generator producing files in the exact C-MAPSS format (for tests / smoke runs only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_CONDS = [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0), (20.0, 0.7, 100.0),
          (25.0, 0.62, 60.0), (35.0, 0.84, 100.0), (42.0, 0.84, 40.0)]


def _engine(rng, unit, life, cut, multi, base, amp, active, cond_off):
    t = np.arange(1, life + 1)
    d = (t / life) ** 2.5
    rows = []
    cond = rng.integers(0, 6, size=life) if multi else np.zeros(life, dtype=int)
    for i, tt in enumerate(t):
        op = np.array(_CONDS[cond[i]]) if multi else np.array([0.0, 0.0, 100.0])
        if not multi:
            op = op + rng.normal(0, [0.002, 0.0003, 0.0])
        sens = base + cond_off[cond[i]]
        noise = np.zeros(21); noise[active] = rng.normal(0, 1, len(active)) * np.abs(amp[active]) * 0.15
        sens = sens + amp * d[i] + noise
        rows.append([unit, tt, *op, *sens])
    rows = np.array(rows)
    return rows[:cut] if cut else rows


def generate_subset(out_dir, subset="FD001", n_train=60, n_test=40, seed=0):
    rng = np.random.default_rng(seed)
    multi = subset in ("FD002", "FD004")
    base = rng.uniform(20, 600, 21)
    amp = base * rng.uniform(0.005, 0.03, 21) * rng.choice([-1, 1], 21)
    active = rng.choice(21, 14, replace=False)
    mask = np.zeros(21); mask[active] = 1; amp = amp * mask
    cond_off = rng.uniform(-0.2, 0.2, (6, 21)) * base if multi else np.zeros((6, 21))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fmt = ["%d", "%d"] + ["%.4f"] * 24

    tr = [_engine(rng, u, int(np.clip(rng.normal(205, 40), 130, 360)), None, multi, base, amp, active, cond_off)
          for u in range(1, n_train + 1)]
    np.savetxt(out / f"train_{subset}.txt", np.vstack(tr), fmt=fmt)

    te, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(np.clip(rng.normal(205, 40), 130, 360))
        cut = int(rng.integers(30, life - 1))
        te.append(_engine(rng, u, life, cut, multi, base, amp, active, cond_off))
        ruls.append(life - cut)
    np.savetxt(out / f"test_{subset}.txt", np.vstack(te), fmt=fmt)
    np.savetxt(out / f"RUL_{subset}.txt", np.array(ruls), fmt="%d")
    return out
