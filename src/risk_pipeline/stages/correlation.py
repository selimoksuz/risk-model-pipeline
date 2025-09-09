from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def drop_correlated(df: pd.DataFrame, *, threshold: float = 0.8) -> Tuple[List[str], pd.DataFrame]:
    """Return kept variables and a DataFrame of dropped pairs based on absolute Spearman correlation.
    Simple greedy approach: iterate sorted pairs and drop the second variable if both are still kept.
    """
    if df.shape[1] <= 1:
        return list(df.columns), pd.DataFrame(columns=["var1", "var2", "rho"])
    corr = df.corr(method="spearman").abs()
    pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    kept = set(cols)
    dropped_rows = []
    for a, b, r in pairs:
        if r >= threshold and a in kept and b in kept:
            kept.remove(b)
            dropped_rows.append({"var1": a, "var2": b, "rho": r})
    dropped_df = pd.DataFrame(dropped_rows)
    return list(kept), dropped_df
