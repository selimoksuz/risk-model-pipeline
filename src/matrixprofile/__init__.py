"""Lightweight stub for the optional matrixprofile dependency used by tsfresh."""
from __future__ import annotations
from typing import Any, Dict

class _MatrixProfileResult(dict):
    """Placeholder dict returned by compute() for compatibility."""
    pass

class _AlgorithmsModule:
    def maximum_subsequence(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "matrixprofile algorithms are not supported in this environment."
        )

algorithms = _AlgorithmsModule()


def compute(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(
        "matrixprofile compute is not supported in this environment."
    )
