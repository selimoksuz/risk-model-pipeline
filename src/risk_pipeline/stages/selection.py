from __future__ import annotations

import pandas as pd
from typing import List


def iv_rank_select(iv_df: pd.DataFrame, *, min_iv: float = 0.02, max_features: int | None = None) -> List[str]:
    """Select variables by IV threshold and rank.
    Expects columns ['variable', 'iv'] in iv_df.
    """
    if iv_df is None or iv_df.empty:
        return []
    df = iv_df.copy()
    df = df[df["iv"] >= min_iv].sort_values("iv", ascending=False)
    if max_features is not None:
        df = df.head(max_features)
    return [str(v) for v in df["variable"].tolist()]
