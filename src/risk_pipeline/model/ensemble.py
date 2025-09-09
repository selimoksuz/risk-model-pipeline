import numpy as np
from typing import List, Optional

from .calibrate import apply_calibrator


def soft_voting_ensemble(models: List,
                         calibrators: Optional[List] = None,
                         weights: Optional[List[float]] = None,
                         X=None):
    """Return weighted average of model probabilities."""
    probs = []
    for i, m in enumerate(models):
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[:, 1]
        else:
            p = m.predict(X)
        if calibrators and len(calibrators) > i and calibrators[i] is not None:
            p = apply_calibrator(calibrators[i], p)
        probs.append(p)
    probs = np.array(probs)
    if weights is None:
        weights = np.ones(probs.shape[0])
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(probs, axis=0, weights=weights)
