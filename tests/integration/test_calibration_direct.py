import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from risk_pipeline.utils.calibration import Calibrator


def test_calibrator_stage1_produces_finite_output(minimal_config):
    calibrator = Calibrator(minimal_config)

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.linspace(0.1, 0.9, num=len(y_true))

    calibrator.fit_stage1(y_true, y_pred)
    calibrated = calibrator.transform(y_pred)

    assert np.all(np.isfinite(calibrated))
    assert calibrated.min() >= 0
    assert calibrated.max() <= 1
