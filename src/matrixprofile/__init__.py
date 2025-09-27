"""Lightweight matrix profile compatibility shim for tsfresh."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

try:
    import stumpy  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matrixprofile shim requires stumpy; install stumpy>=1.13.0"
    ) from exc


class NoSolutionPossible(Exception):
    """Exception raised when matrix profile computation is not feasible."""


class _Algorithms:
    """Subset of the matrixprofile.algorithms API used by tsfresh."""

    @staticmethod
    def maximum_subsequence(
        x: Sequence[float],
        include_pmp: bool = True,
        **kwargs,
    ) -> dict:
        arr = _as_array(x)
        window = kwargs.get("window")
        if window is None:
            window = max(4, int(math.sqrt(len(arr))))
        _validate_window(window, len(arr))

        mp = _stump_profile(arr, window)
        result = {"pmp": [mp] if include_pmp else []}
        return result


algorithms = _Algorithms()


def compute(x: Sequence[float], windows: Iterable[int], **kwargs) -> dict:
    """Compute matrix profile for given windows using stumpy."""
    arr = _as_array(x)
    windows = list(windows or [])
    if not windows:
        raise NoSolutionPossible("window size must be provided")

    profiles: List[np.ndarray] = []
    for window in windows:
        _validate_window(window, len(arr))
        profiles.append(_stump_profile(arr, window))

    return {"mp": profiles[-1] if len(profiles) == 1 else profiles}


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise NoSolutionPossible("matrix profile expects 1-D arrays")
    return arr


def _validate_window(window: int, length: int) -> None:
    if window <= 1 or window > length:
        raise NoSolutionPossible(
            f"invalid window size {window} for series length {length}"
        )


def _stump_profile(arr: np.ndarray, window: int) -> np.ndarray:
    try:
        profile = stumpy.stump(arr, window)[0]
    except Exception as exc:  # pragma: no cover
        raise NoSolutionPossible(str(exc)) from exc

    if profile.size == 0:
        raise NoSolutionPossible("matrix profile output empty")
    return profile

__all__ = [
    "compute",
    "algorithms",
    "NoSolutionPossible",
]
