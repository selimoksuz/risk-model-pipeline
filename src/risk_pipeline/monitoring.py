import pandas as pd
import numpy as np
from typing import List, Dict


def compute_score_psi(baseline_scores, new_scores, bins=10):
    base_hist, _ = np.histogram(baseline_scores, bins=bins, range=(0, 1), density=True)
    new_hist, _ = np.histogram(new_scores, bins=bins, range=(0, 1), density=True)
    base_hist = np.clip(base_hist, 1e-6, None)
    new_hist = np.clip(new_hist, 1e-6, None)
    return float(np.sum((base_hist - new_hist) * np.log(base_hist / new_hist)))


def compute_feature_psi(baseline_df: pd.DataFrame, new_df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    out = {}
    for c in columns:
        try:
            out[c] = compute_score_psi(baseline_df[c], new_df[c])
        except Exception:
            out[c] = np.nan
    return out


def monitor(baseline_path, new_path, woe_mapping, final_vars, thresholds):
    bas = pd.read_csv(baseline_path)
    new = pd.read_csv(new_path)
    # Placeholder: in real implementation WOE transform and model scoring would occur
    result = {
        "score_psi": compute_score_psi(bas[final_vars[0]], new[final_vars[0]]) if final_vars else np.nan,
        "feature_psi": compute_feature_psi(bas, new, final_vars),
    }
    pd.DataFrame([result]).to_json("monitor_report.json", orient="records")
    return result
