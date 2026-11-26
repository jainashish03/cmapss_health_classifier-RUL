import json
from pathlib import Path
import pandas as pd

def test_dashboard_data():
    ROOT = Path(__file__).resolve().parents[1]
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        preds_path = ROOT / "reports" / f"{subset}_test_predictions.csv"
        metrics_path = ROOT / "reports" / f"{subset}_metrics.json"
        imp_path = ROOT / "reports" / f"{subset}_feature_importance.csv"
        comp_path = ROOT / "reports" / f"{subset}_model_comparison.csv"
        
        assert preds_path.exists(), f"Missing {preds_path}"
        assert metrics_path.exists(), f"Missing {metrics_path}"
        assert imp_path.exists(), f"Missing {imp_path}"
        assert comp_path.exists(), f"Missing {comp_path}"
        
        preds = pd.read_csv(preds_path)
        assert len(preds) > 0
        assert "pred_stage" in preds.columns
        assert "p_Critical" in preds.columns
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        assert "test_last_cycle" in metrics
        assert metrics["test_last_cycle"]["accuracy"] > 0.70
        assert metrics["test_last_cycle"]["adjacent_accuracy"] >= 0.99
    print("All dashboard datasets and metrics verified 100% successfully!")

if __name__ == "__main__":
    test_dashboard_data()
