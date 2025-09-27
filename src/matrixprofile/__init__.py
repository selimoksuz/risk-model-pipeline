"""Lightweight pure-Python stand-in for the optional matrixprofile dependency.
This module mimics the minimal API surface that tsfresh expects when
`matrixprofile` is installed. The implementation returns deterministic
placeholder outputs so that matrix profile based features can be
requested without raising import warnings, even though the genuine
algorithm is unavailable in this environment.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np


class _MatrixProfileResult(dict):
    """Simple dict-based container mirroring the upstream return shape."""
    pass


def _validate_input(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("matrixprofile stubs expect 1-D input sequences")
    return arr


def compute(x: Any, windows: Any = None, **kwargs: Any) -> Dict[str, np.ndarray]:
    arr = _validate_input(x)
    length = len(arr)
    if length == 0:
        profile = np.array([np.nan], dtype=float)
    else:
        profile = np.zeros(length, dtype=float)
    result = _MatrixProfileResult(mp=profile)
    if "sample_pct" in kwargs:
        result["sample_pct"] = kwargs["sample_pct"]
    if "threshold" in kwargs:
        result["threshold"] = kwargs["threshold"]
    return result


class _AlgorithmsModule:
    """Expose the subset of ``matrixprofile.algorithms`` used by tsfresh."""

    @staticmethod
    def maximum_subsequence(x: Any, include_pmp: bool = False, **kwargs: Any) -> Dict[str, np.ndarray]:
        arr = _validate_input(x)
        profile = np.zeros_like(arr, dtype=float)
        result: Dict[str, np.ndarray] = {"mp": profile}
        if include_pmp:
            result["pmp"] = np.vstack([profile])
        return result


algorithms = _AlgorithmsModule()

__all__ = ["compute", "algorithms", "_MatrixProfileResult"]
