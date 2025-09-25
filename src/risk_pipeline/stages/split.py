from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


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


def _sort_index_by_time(df: pd.DataFrame, indices: List, time_col: str) -> pd.Index:
    if not indices:
        return pd.Index([])
    if time_col not in df.columns:
        return pd.Index(indices)
    ordered = df.loc[indices].sort_values(time_col)
    return pd.Index(ordered.index)



def _monthly_stratified_split(
    df: pd.DataFrame,
    candidate_idx: pd.Index,
    time_col: str,
    target_col: str,
    test_ratio: float,
    stratify_flag: bool,
    seed: int | None = 42,
) -> Tuple[pd.Index, pd.Index]:
    """Split remaining records into train/test preserving event rate in each calendar month."""
    if test_ratio <= 0 or len(candidate_idx) == 0:
        return pd.Index(candidate_idx), pd.Index([])

    work_df = df.loc[candidate_idx].copy()

    if time_col in work_df.columns:
        work_df["__split_month"] = work_df[time_col].apply(month_floor)
        valid = work_df.dropna(subset=["__split_month"])
        month_groups = list(valid.groupby("__split_month", sort=True))
        missing_month = work_df[work_df["__split_month"].isna()]
        if not missing_month.empty:
            month_groups.append((None, missing_month))
    else:
        month_groups = [(None, work_df)]

    train_parts: List[pd.Index] = []
    test_parts: List[pd.Index] = []

    for offset, (_, month_df) in enumerate(month_groups):
        if time_col in month_df.columns:
            month_df = month_df.sort_values(time_col)
        month_idx = month_df.index

        if len(month_idx) < 2 or target_col not in month_df.columns:
            train_parts.append(pd.Index(month_idx))
            continue

        same_class = month_df[target_col].nunique() < 2

        if stratify_flag and not same_class:
            try:
                train_ids, test_ids = train_test_split(
                    month_idx,
                    test_size=test_ratio,
                    stratify=month_df[target_col],
                    random_state=None if seed is None else seed + offset,
                )
                train_parts.append(pd.Index(train_ids))
                if len(test_ids) > 0:
                    test_parts.append(pd.Index(test_ids))
                continue
            except ValueError:
                same_class = True

        k = max(1, int(len(month_idx) * test_ratio)) if len(month_idx) > 1 else 0
        if k == 0:
            train_parts.append(pd.Index(month_idx))
        else:
            train_parts.append(pd.Index(month_idx[:-k]))
            test_parts.append(pd.Index(month_idx[-k:]))

    train_idx = _sort_index_by_time(df, [idx for part in train_parts for idx in part], time_col)
    test_idx = (
        _sort_index_by_time(df, [idx for part in test_parts for idx in part], time_col)
        if test_parts
        else pd.Index([])
    )

    return train_idx, test_idx



def time_based_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    use_test_split: bool,
    oot_window_months: int | None,
    test_size_row_frac: float = 0.2,
) -> Tuple[pd.Index, pd.Index | None, pd.Index]:
    """Split into TRAIN/TEST (optional)/OOT by months with month-level stratification."""
    dfx = df.copy()
    dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    dfx["__month"] = dfx[time_col].apply(month_floor)

    months = dfx["__month"].dropna().sort_values().unique()
    oot_months = oot_window_months or 0

    if oot_months <= 0:
        oot_idx = pd.Index([])
        remainder_idx = pd.Index(dfx.index)
    elif len(months) == 0:
        n_rows = len(dfx)
        k = max(1, int(n_rows * 0.2))
        oot_idx = pd.Index(dfx.index[-k:])
        remainder_idx = pd.Index(dfx.index[:-k])
    else:
        anchor = months[-1]
        earliest_oot_period = pd.Period(anchor, freq="M") - (oot_months - 1)
        cutoff_ts = earliest_oot_period.to_timestamp()
        oot_idx = pd.Index(dfx.index[dfx["__month"] >= cutoff_ts])
        remainder_idx = pd.Index(dfx.index.difference(oot_idx))

    stratify_flag = (
        target_col in dfx.columns
        and dfx.loc[remainder_idx, target_col].nunique() > 1
    )

    if use_test_split and len(remainder_idx) > 0:
        train_idx, monthly_test_idx = _monthly_stratified_split(
            dfx,
            remainder_idx,
            time_col,
            target_col,
            test_size_row_frac,
            stratify_flag,
        )
        test_idx = monthly_test_idx if len(monthly_test_idx) else None
    else:
        train_idx = _sort_index_by_time(dfx, list(remainder_idx), time_col)
        test_idx = None

    return train_idx, test_idx, oot_idx



