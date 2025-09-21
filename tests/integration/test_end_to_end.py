import sys
from pathlib import Path

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from risk_pipeline.pipeline import RiskModelPipeline


def test_pipeline_scoring_flow(sample_training_data, minimal_config):
    minimal_config.enable_scoring = True
    pipeline = RiskModelPipeline(minimal_config)
    pipeline.fit(sample_training_data)

    scores = pipeline.score(sample_training_data.head(10), return_calibrated=False)

    assert len(scores) == 10
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)
