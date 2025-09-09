from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

from .pipeline import RiskModelPipeline
from .core.config import Config
from .model.calibrate import apply_calibrator


def run_pipeline(df: pd.DataFrame, config: Optional[Config] = None, **config_kwargs) -> RiskModelPipeline:
    """Run the risk model pipeline on an in-memory DataFrame.

    Returns the fitted pipeline object. Reports are written to Excel only
    CSV exports are
    disabled unless explicitly enabled via config (write_csv = True).
    """
    cfg = config or Config(**config_kwargs)
    pipe = RiskModelPipeline(cfg)
    pipe.run(df)
    return pipe


def score_df(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    final_vars: Tuple[str, ...] | list[str],
    model,
    *,
    id_col: str = "app_id",
    calibrator: Any | None = None,
) -> pd.DataFrame:
    """Score an in-memory DataFrame and return a combined DataFrame containing:
    id, optional target, raw variables, WOE-transformed variables, proba, predict and
    optionally calibrated proba. Does not write any CSV by default.
    """

    def apply_woe(df_in: pd.DataFrame, mapping: dict) -> pd.DataFrame:
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
                groups = info.get("groups", [])
                for g in groups:
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

    X = apply_woe(df, mapping)
    cols = [c for c in final_vars if c in X.columns]
    if not cols:
        raise ValueError("No overlap between final_vars and available columns after WOE transform.")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[cols])
        proba = np.asarray(proba)
        if proba.ndim == 1:
            score = proba
        elif proba.shape[1] >= 2:
            score = proba[:, 1]
        else:
            score = proba[:, 0]
    else:
        score = model.predict(X[cols])

    idx = pd.Index(range(len(df)))
    id_series = df[id_col] if id_col in df.columns else pd.Series(idx, name=id_col)
    raw_vars = [v for v in mapping.get("variables", {}).keys() if v in df.columns]
    raw_df = df[raw_vars].copy() if raw_vars else pd.DataFrame(index=idx)
    woe_df = X[raw_vars].copy() if raw_vars else pd.DataFrame(index=idx)
    woe_df.columns = [f"{c}_woe" for c in woe_df.columns]

    out = pd.DataFrame({id_col: id_series, "proba": score})
    if "target" in df.columns:
        out["target"] = df["target"].values
    try:
        out["predict"] = (out["proba"] >= 0.5).astype(int)
    except Exception:
        pass
    if calibrator is not None:
        try:
            out["proba_calibrated"] = apply_calibrator(calibrator, out["proba"].values)
        except Exception:
            pass

    combined = pd.concat([
        out[[id_col] + (["target"] if "target" in out.columns else [])].reset_index(drop=True),
        raw_df.reset_index(drop=True),
        woe_df.reset_index(drop=True),
        out.drop(columns=[id_col] + (["target"] if "target" in out.columns else [])).reset_index(drop=True),
    ], axis=1)
    return combined
