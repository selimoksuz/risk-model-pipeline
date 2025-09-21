import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from risk_pipeline.pipeline import RiskModelPipeline


def test_pipeline_fit_produces_calibrated_model(sample_training_data, minimal_config):
    pipeline = RiskModelPipeline(minimal_config)
    results = pipeline.fit(sample_training_data)

    assert pipeline.is_fitted is True
    assert pipeline.best_model in pipeline.models
    assert pipeline.best_model in pipeline.calibration_models
    assert "best_model" in results
