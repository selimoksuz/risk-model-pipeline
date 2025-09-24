import json
import pickle
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from .utils.metrics import calculate_metrics

ArrayLike = Union[pd.Series, np.ndarray, List[float]]


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
                left = b.get("left")
                right = b.get("right")
                woe = b.get("woe", 0.0)

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
                lab = g.get("label")
                woe = float(g.get("woe", 0.0))
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


def _ensure_series(values: Optional[ArrayLike], name: str) -> Optional[pd.Series]:
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    arr = np.asarray(values)
    return pd.Series(arr, name=name)


def _build_lift_table(y_true: ArrayLike, y_pred: ArrayLike, bands: int = 10) -> pd.DataFrame:
    y_true_series = _ensure_series(y_true, "actual")
    y_pred_series = _ensure_series(y_pred, "score")

    if y_true_series is None or y_pred_series is None:
        raise ValueError("y_true and y_pred must be provided to build lift table")

    df = pd.DataFrame({
        "actual": y_true_series.astype(float),
        "score": y_pred_series.astype(float),
    })

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n_groups = int(max(1, min(bands, len(df))))
    group_size = int(np.ceil(len(df) / n_groups))
    df["band"] = (df.index // group_size) + 1
    df.loc[df["band"] > n_groups, "band"] = n_groups

    grouped = df.groupby("band", sort=True)
    total_events = df["actual"].sum()
    total_non_events = len(df) - total_events
    base_event_rate = total_events / len(df) if len(df) else np.nan

    summary = grouped.agg(
        records=("actual", "size"),
        events=("actual", "sum"),
        mean_score=("score", "mean"),
    )
    summary["non_events"] = summary["records"] - summary["events"]
    summary["event_rate"] = summary["events"] / summary["records"]
    summary["lift"] = summary["event_rate"] / base_event_rate if base_event_rate else np.nan

    summary["cum_events"] = summary["events"].cumsum()
    summary["cum_non_events"] = summary["non_events"].cumsum()
    summary["cum_event_rate"] = summary["cum_events"] / total_events if total_events else 0.0
    summary["cum_non_event_rate"] = summary["cum_non_events"] / total_non_events if total_non_events else 0.0
    summary["ks"] = (summary["cum_event_rate"] - summary["cum_non_event_rate"]).abs()
    summary["capture_rate"] = summary["cum_events"] / total_events if total_events else 0.0
    summary["population_pct"] = summary["records"].cumsum() / len(df)

    return summary.reset_index()


def monitor_scores(
    baseline_path: Optional[str] = None,
    new_path: Optional[str] = None,
    woe_mapping: Optional[Union[dict, str]] = None,
    final_vars: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    bins: int = 10,
    *,
    actuals: Optional[ArrayLike] = None,
    predictions: Optional[ArrayLike] = None,
    baseline_actuals: Optional[ArrayLike] = None,
    baseline_predictions: Optional[ArrayLike] = None,
    bands: int = 10,
    baseline: Optional[Any] = None,
) -> Dict[str, Any]:
    """Monitor score stability with optional in-memory inputs."""

    if actuals is not None and predictions is not None:
        y_true = _ensure_series(actuals, "actual")
        y_pred = _ensure_series(predictions, "score")

        metrics = calculate_metrics(y_true.values, y_pred.values)
        lift_table = _build_lift_table(y_true, y_pred, bands=bands)

        result: Dict[str, Any] = {
            "ks": float(metrics.get("ks_statistic", np.nan)),
            "auc": float(metrics.get("auc", np.nan)),
            "gini": float(metrics.get("gini", np.nan)),
            "metrics": metrics,
            "lift_table": lift_table,
        }

        baseline_scores = None
        if baseline_predictions is not None:
            baseline_scores = _ensure_series(baseline_predictions, "baseline_score")
        elif isinstance(baseline, (pd.Series, list, np.ndarray)):
            baseline_scores = _ensure_series(baseline, "baseline_score")

        if baseline_scores is not None:
            result["score_psi"] = compute_score_psi(baseline_scores.values, y_pred.values, bins=bins)

        baseline_actual_series = _ensure_series(baseline_actuals, "baseline_actual")
        if baseline_actual_series is not None:
            baseline_pred_values = (
                baseline_scores.values if baseline_scores is not None else baseline_actual_series.values
            )
            result["baseline_metrics"] = calculate_metrics(baseline_actual_series.values, baseline_pred_values)

        return result

    bas = pd.read_csv(baseline_path)
    new = pd.read_csv(new_path)
    if isinstance(woe_mapping, str):
        with open(woe_mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = woe_mapping
    X_bas = _apply_woe(bas, mapping)
    X_new = _apply_woe(new, mapping)
    cols = [c for c in (final_vars or []) if c in X_bas.columns]
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
                return p[:, 1]
            if p.ndim == 1:
                return p
            return p[:, 0]
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
