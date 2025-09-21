"""
Risk Model Pipeline - Production-ready risk modeling with WOE transformation
"""

from ._version import __version__, __version_info__
from .api import run_pipeline, score_df
from .core.config import Config
from .pipeline import RiskModelPipeline

__all__ = [
    "__version__",
    "__version_info__",
    "run_pipeline",
    "score_df",
    "RiskModelPipeline",
    "Config",
]

# Package metadata
__author__ = "Risk Model Pipeline Contributors"
__email__ = "your.email@example.com"  # TODO: Update
__license__ = "MIT"
