import os
import warnings
import numpy as np

_SHAP_DISABLED = os.getenv("RISK_PIPELINE_DISABLE_SHAP") == "1"
_SHAP_IMPORT_ERROR = None
shap = None

if not _SHAP_DISABLED:
    try:
        from llvmlite.binding import ffi as _ffi  # type: ignore
        _ = _ffi.lib  # force DLL load on Windows before importing shap
    except Exception as exc:
        _SHAP_IMPORT_ERROR = exc
    else:
        try:
            import shap as _shap  # type: ignore
        except Exception as exc:
            _SHAP_IMPORT_ERROR = exc  # Optional dependency could not be loaded
        else:
            shap = _shap
            _SHAP_IMPORT_ERROR = None
else:
    _SHAP_IMPORT_ERROR = RuntimeError("SHAP disabled via RISK_PIPELINE_DISABLE_SHAP")


def compute_shap_values(model, X, shap_sample=25000, random_state=42):
    if shap is None:
        if _SHAP_IMPORT_ERROR is not None:
            warnings.warn(
                f"SHAP import skipped ({_SHAP_IMPORT_ERROR}); set RISK_PIPELINE_DISABLE_SHAP=1 to silence.",
                RuntimeWarning,
            )
        return None
    if shap_sample and len(X) > shap_sample:
        Xs = X.sample(shap_sample, random_state=random_state)
    else:
        Xs = X
    explainer = shap.Explainer(model, Xs)
    return explainer(Xs)


def summarize_shap(shap_values, feature_names):
    if shap_values is None:
        return {}
    vals = np.abs(shap_values.values).mean(axis=0)
    return {name: float(val) for name, val in zip(feature_names, vals)}
