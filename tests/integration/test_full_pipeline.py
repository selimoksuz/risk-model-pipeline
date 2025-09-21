import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from risk_pipeline.pipeline import RiskModelPipeline


def test_pipeline_persistence(tmp_path, sample_training_data, minimal_config):
    pipeline = RiskModelPipeline(minimal_config)
    pipeline.fit(sample_training_data)

    save_path = tmp_path / "trained_pipeline.pkl"
    pipeline.save_pipeline(str(save_path))

    assert save_path.exists()

    loaded_pipeline = RiskModelPipeline.load_pipeline(str(save_path))
    assert loaded_pipeline.is_fitted is True
    assert loaded_pipeline.best_model == pipeline.best_model
    assert loaded_pipeline.selected_features[loaded_pipeline.best_model] == pipeline.selected_features[pipeline.best_model]
