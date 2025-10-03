"""Runtime environment checks and feature gating.

This module inspects Python version and optional dependencies to gracefully
disable unsupported features and log a concise summary for users.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List


def _has(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def apply_runtime_feature_gates(config: Any) -> Dict[str, Any]:
    """Inspect environment and toggle config features safely.

    Returns a dict with diagnostics about disabled features and environment.
    """

    diag: Dict[str, Any] = {
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'disabled_algorithms': [],
        'disabled_features': [],
        'notes': [],
    }

    # Python version policy: prefer >=3.8
    if sys.version_info < (3, 8):
        diag['notes'].append('Python < 3.8 detected; disabling advanced algorithms.')
        # Disable heavier algos when Python too old
        heavy = {'gam', 'catboost', 'lightgbm', 'xgboost', 'xbooster'}
        if getattr(config, 'algorithms', None):
            keep = [a for a in config.algorithms if a not in heavy]
            dropped = [a for a in config.algorithms if a in heavy]
            config.algorithms = keep
            diag['disabled_algorithms'].extend(dropped)

    # Optional ML libraries
    algos: List[str] = list(getattr(config, 'algorithms', []) or [])
    if 'gam' in algos and not _has('pygam'):
        algos.remove('gam')
        diag['disabled_algorithms'].append('gam (pygam not installed)')
    if 'catboost' in algos and not _has('catboost'):
        algos.remove('catboost')
        diag['disabled_algorithms'].append('catboost (catboost not installed)')
    if 'lightgbm' in algos and not _has('lightgbm'):
        algos.remove('lightgbm')
        diag['disabled_algorithms'].append('lightgbm (lightgbm not installed)')
    # xgboost / xbooster rely on xgboost
    if ('xgboost' in algos or 'xbooster' in algos) and not _has('xgboost'):
        if 'xgboost' in algos:
            algos.remove('xgboost')
            diag['disabled_algorithms'].append('xgboost (xgboost not installed)')
        if 'xbooster' in algos:
            algos.remove('xbooster')
            diag['disabled_algorithms'].append('xbooster (xgboost not installed)')
    config.algorithms = algos

    # SHAP
    if getattr(config, 'calculate_shap', False) and not _has('shap'):
        config.calculate_shap = False
        diag['disabled_features'].append('calculate_shap (shap not installed)')

    # TSFresh
    if getattr(config, 'enable_tsfresh_features', False) and not _has('tsfresh'):
        config.enable_tsfresh_features = False
        diag['disabled_features'].append('enable_tsfresh_features (tsfresh not installed)')

    return diag

