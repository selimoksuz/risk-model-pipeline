import pandas as pd
import numpy as np
from typing import List, Dict
import json
import joblib
import pickle


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


def _apply_woe(df_in: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = {}
    for v, info in mapping.get("variables", {}).items():
        if v not in df_in.columns:
            continue
        if info.get("type") == "numeric":
            s = df_in[v]
            w = pd.Series(index=s.index, dtype="float32")
            miss = s.isna()
            miss_woe = 0.0
            for b in info.get("bins", []):
                left = b.get("left"); right = b.get("right"); woe = b.get("woe", 0.0)
                if left is None or right is None or (pd.isna(left) and pd.isna(right)):
                    miss_woe = float(woe)
                    continue
                m = (~miss) & (s >= left) & (s <= right)
                w.loc[m] = float(woe)
            w.loc[miss] = float(miss_woe)
            out[v] = w.values
        else:
            s = df_in[v].astype(object)
            w = pd.Series(index=s.index, dtype="float32")
            miss = s.isna()
            assigned = miss.copy()
            miss_woe = 0.0
            other_woe = 0.0
            for g in info.get("groups", []):
                lab = g.get("label"); woe = float(g.get("woe", 0.0))
                if lab == "MISSING":
                    miss_woe = woe
                    continue
                if lab == "OTHER":
                    other_woe = woe
                    continue
                members = set(map(str, g.get("members", [])))
                m = (~miss) & (s.astype(str).isin(members))
                w.loc[m] = woe
                assigned |= m
            w.loc[miss] = float(miss_woe)
            w.loc[~assigned] = float(other_woe)
            out[v] = w.values
    return pd.DataFrame(out, index=df_in.index)


def monitor_scores(baseline_path, new_path, woe_mapping, final_vars, model_path, calibrator_path=None, bins=10):
    bas = pd.read_csv(baseline_path)
    new = pd.read_csv(new_path)
    if isinstance(woe_mapping, str):
        with open(woe_mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = woe_mapping
    X_bas = _apply_woe(bas, mapping)
    X_new = _apply_woe(new, mapping)
    cols = [c for c in final_vars if c in X_bas.columns]
    try:
        mdl = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            mdl = pickle.load(f)
    def score(m, Xdf):
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(Xdf)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:,1]
            if p.ndim == 1:
                return p
            return p[:,0]
        return m.predict(Xdf)
    bas_scores = score(mdl, X_bas[cols])
    new_scores = score(mdl, X_new[cols])
    if calibrator_path:
        try:
            with open(calibrator_path, "rb") as f:
                calib = pickle.load(f)
            from .model.calibrate import apply_calibrator
            bas_scores = apply_calibrator(calib, bas_scores)
            new_scores = apply_calibrator(calib, new_scores)
        except Exception:
            pass
    result = {
        "score_psi": compute_score_psi(bas_scores, new_scores, bins=bins),
        "feature_psi": compute_feature_psi(X_bas[cols], X_new[cols], cols),
    }
    pd.DataFrame([result]).to_json("monitor_report.json", orient="records")
    return result
