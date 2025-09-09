import json
from pathlib import Path

import pandas as pd


def save_metrics(metrics: dict, out_dir: str):
    """Persist metrics and auxiliary reports to disk."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    calib = metrics.get("calibration")
    if calib:
        with open(out / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
    shap = metrics.get("shap_summary")
    if shap:
        with open(out / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap, f, indent=2)
        pd.DataFrame(shap).head(20).to_csv(out / "shap_top20.csv", index=False)
