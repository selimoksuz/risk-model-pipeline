from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .woe import apply_woe


def build_scored_frame(
    df: pd.DataFrame,
    *,
    mapping: Dict[str, Any],
    final_vars: List[str],
    model: Any,
    id_col: str = "app_id",
    calibrator: Any | None = None,
) -> pd.DataFrame:
    """Build combined scored DataFrame using mapping + model without disk I/O."""
    import numpy as np

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
            from .calibration import apply_calibrator as _apply_cal

            out["proba_calibrated"] = _apply_cal(calibrator, out["proba"].values)
        except Exception:
            pass

    combined = pd.concat(
        [
            out[[id_col] + (["target"] if "target" in out.columns else [])].reset_index(drop=True),
            raw_df.reset_index(drop=True),
            woe_df.reset_index(drop=True),
            out.drop(columns=[id_col] + (["target"] if "target" in out.columns else [])).reset_index(drop=True),
        ],
        axis=1,
    )
    return combined
