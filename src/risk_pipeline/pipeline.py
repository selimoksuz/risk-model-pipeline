"""Compatibility shims for unified pipeline.

This module exposes legacy names to the new unified pipeline implementation
to keep external imports working while the project consolidates to a single
pipeline class.
"""

from typing import Optional

from .core.config import Config
from .unified_pipeline import UnifiedRiskPipeline


class RiskModelPipeline(UnifiedRiskPipeline):
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)


class DualRiskModelPipeline(UnifiedRiskPipeline):
    def __init__(self, config: Optional[Config] = None):
        cfg = config or Config()
        cfg.enable_dual_pipeline = True
        super().__init__(cfg)


class DualPipeline(DualRiskModelPipeline):
    pass

