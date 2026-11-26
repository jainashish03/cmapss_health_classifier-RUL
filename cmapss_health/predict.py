"""Predict current health stage for engines in a raw C-MAPSS-format file.

    python -m cmapss_health.predict --input data/raw/test_FD001.txt --output reports/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .data import read_engine_file
from .pipeline import HealthPipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="26-column whitespace-separated file")
    ap.add_argument("--subset", default="FD001", help="which trained model to use (models/health_model_<subset>.joblib)")
    ap.add_argument("--model", default=None, help="explicit model path (overrides --subset)")
    ap.add_argument("--output", default=None, help="CSV path (default: print only)")
    ap.add_argument("--all-cycles", action="store_true", help="predict every cycle, not just the last")
    a = ap.parse_args(argv)
    pipe = HealthPipeline.load(a.model or PROJECT_ROOT / "models" / f"health_model_{a.subset}.joblib")
    df = read_engine_file(a.input)
    res = pipe.predict_frame(df) if a.all_cycles else pipe.predict_last(df)
    res = res.drop(columns=[c for c in ["true_stage_id", "RUL"] if c in res.columns])
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.output, index=False)
        print(f"saved {len(res)} rows -> {a.output}")
    print(res.head(15).round(3).to_string(index=False))
    if not a.all_cycles:
        print("\nFleet summary:\n", res["pred_stage"].value_counts().to_string())


if __name__ == "__main__":
    main()
