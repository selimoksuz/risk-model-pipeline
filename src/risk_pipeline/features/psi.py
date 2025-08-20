import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.qcut(expected.rank(method="first"), q=bins, duplicates="drop")
    a = pd.qcut(actual.rank(method="first"), q=bins, duplicates="drop")
    e_dist = e.value_counts(normalize=True).sort_index()
    a_dist = a.value_counts(normalize=True).sort_index()
    # align indexes
    idx = sorted(set(e_dist.index) | set(a_dist.index))
    e_dist = e_dist.reindex(idx, fill_value=1e-6)
    a_dist = a_dist.reindex(idx, fill_value=1e-6)
    diff = a_dist - e_dist
    ln = np.log(a_dist / e_dist)
    return float((diff * ln).sum())
