from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


import pytest

from risk_pipeline.core.config import Config


def test_config_accepts_ratio_inputs():
    cfg = Config(
        target_column='target',
        id_column='customer_id',
        use_test_split=True,
        train_ratio=0.6,
        test_ratio=0.2,
        oot_ratio=0.2,
    )

    assert cfg.test_size == pytest.approx(0.2)
    assert cfg.oot_size == pytest.approx(0.2)
    assert cfg.train_ratio == pytest.approx(0.6)
    assert cfg.use_test_split is True
    assert cfg.create_test_split is True


def test_config_ratio_sum_validation():
    with pytest.raises(ValueError, match='cannot exceed 1.0'):
        Config(
            target_column='target',
            id_column='customer_id',
            use_test_split=True,
            test_ratio=0.7,
            oot_ratio=0.4,
        )
