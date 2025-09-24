"""
Risk Model Pipeline - Production-ready risk modeling with WOE transformation
"""

from ._version import __version__, __version_info__
from .api import run_pipeline, score_df
from .core.config import Config
from .pipeline import RiskModelPipeline

import warnings

warnings.filterwarnings(
    "ignore", message="is_sparse is deprecated", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message="is_categorical_dtype is deprecated", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message="Downcasting object dtype arrays on .fillna", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"sklearn\..*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"pandas\..*"
)
from .data.sample import (
    CreditRiskSample,
    load_credit_risk_sample,
    copy_credit_risk_sample,
)

__all__ = [
    "__version__",
    "__version_info__",
    "run_pipeline",
    "score_df",
    "RiskModelPipeline",
    "Config",
    "CreditRiskSample",
    "load_credit_risk_sample",
    "copy_credit_risk_sample",
]

# Package metadata
__author__ = "Selim Oksuz and contributors"
__email__ = "selimoksuz@users.noreply.github.com"
__license__ = "MIT"
