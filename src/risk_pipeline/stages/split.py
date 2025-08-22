from __future__ import annotations

import pandas as pd
from typing import Tuple


def month_floor(ts) -> pd.Timestamp:
    try:
        if ts is None or pd.isna(ts):
            return pd.NaT
        if getattr(ts, "tzinfo", None):
            ts = ts.tz_localize(None)
        return pd.Timestamp(ts).to_period("M").to_timestamp()
    except Exception:
        ts2 = pd.to_datetime(ts, errors="coerce")
        return ts2.to_period("M").to_timestamp()


def time_based_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    use_test_split: bool,
    oot_window_months: int,
    test_size_row_frac: float = 0.2,
) -> Tuple[pd.Index, pd.Index | None, pd.Index]:
    """Split into TRAIN/TEST (optional)/OOT by months.

    - OOT is the last `oot_window_months` months.
    - TRAIN is the remaining earlier months; optional TEST is a row-fraction split from TRAIN.
    Returns (train_idx, test_idx_or_none, oot_idx).
    """
    dfx = df.copy()
    dfx["__month"] = dfx[time_col].apply(month_floor)
    months = dfx["__month"].dropna().sort_values().unique()
    if len(months) == 0:
        # fall back: last 20% rows as OOT
        n = len(dfx)
        k = max(1, int(n * 0.2))
        oot_idx = dfx.index[-k:]
        rem = dfx.index[:-k]
    else:
        anchor = months.max()
        cutoff = (pd.Period(anchor, freq="M") - oot_window_months + 1).to_timestamp()
        oot_idx = dfx.index[dfx["__month"] >= cutoff]
        rem = dfx.index[dfx["__month"] < cutoff]
    test_idx = None
    if use_test_split and len(rem) > 0:
        k = max(1, int(len(rem) * test_size_row_frac))
        test_idx = rem[:k]
        train_idx = rem[k:]
    else:
        train_idx = rem
    return pd.Index(train_idx), (pd.Index(test_idx) if test_idx is not None else None), pd.Index(oot_idx)

