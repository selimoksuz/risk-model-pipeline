from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict

from ..features.psi import psi as psi_scalar


def feature_psi(train_woe: pd.DataFrame, other_woe: pd.DataFrame, *
    , sample: int | None = None, bins: int = 10) -> Dict[str, float]:
    """Compute PSI per-column between two WOE-transformed frames (same columns).
    Optionally subsample rows for speed. Returns a dict {variable: psi_value}.
    """
    if sample is not None and sample > 0:
        i1 = np.random.RandomState(0).choice(train_woe.index, size=min(sample, len(train_woe)), replace=False)
        i2 = np.random.RandomState(1).choice(other_woe.index, size=min(sample, len(other_woe)), replace=False)
        a = train_woe.loc[i1]
        b = other_woe.loc[i2]
    else:
        a, b = train_woe, other_woe
    out: Dict[str, float] = {}
    for c in a.columns:
        if c in b.columns:
            out[c] = psi_scalar(a[c], b[c], bins=bins)
    return out
