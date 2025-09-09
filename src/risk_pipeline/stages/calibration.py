from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from ..model.calibrate import fit_calibrator as _fit, apply_calibrator as _apply


def fit_calibrator(y_true: np.ndarray, proba: np.ndarray, method: str = "isotonic") -> Any:
    return _fit(y_true, proba, method=method)


def apply_calibrator(model: Any, proba: np.ndarray) -> np.ndarray:
    return _apply(model, proba)
