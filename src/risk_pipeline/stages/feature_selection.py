from __future__ import annotations

from typing import List

import pandas as pd


def select_features(
    candidates: List[str], *, iv_scores: pd.DataFrame | None = None, max_features: int | None = None
) -> List[str]:
    """Simple feature selection placeholder.
    If iv_scores provided (with columns ['variable', 'iv']), pick by descending IV up to max_features.
    Otherwise, return candidates unchanged.
    """
    if iv_scores is not None and not iv_scores.empty:
        df = iv_scores.copy()
        df = df.sort_values("iv", ascending=False)
        if max_features is not None:
            df = df.head(max_features)
        return [str(v) for v in df["variable"].tolist()]
    if max_features is not None:
        return candidates[:max_features]
    return candidates
