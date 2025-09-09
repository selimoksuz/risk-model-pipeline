"""
Risk Model Pipeline - Production-ready risk modeling with WOE transformation
"""

from ._version import __version__, __version_info__
from .api import run_pipeline, score_df
from .core.config import Config
from .pipeline import DualRiskModelPipeline, RiskModelPipeline
from .complete_pipeline import CompletePipeline
from .advanced_pipeline import AdvancedPipeline

__all__ = [
    "__version__", 
    "__version_info__", 
    "run_pipeline", 
    "score_df", 
    "RiskModelPipeline", 
    "DualRiskModelPipeline", 
    "CompletePipeline",
    "AdvancedPipeline",
    "Config"
]

# Package metadata
__author__ = "Risk Model Pipeline Contributors"
__email__ = "your.email@example.com"  # TODO: Update
__license__ = "MIT"
