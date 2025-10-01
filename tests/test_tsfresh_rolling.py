import pandas as pd
import pytest

from risk_pipeline.core.config import Config
from risk_pipeline.core.data_processor import DataProcessor


def test_generate_rolling_tsfresh_features_produces_coverage():
    data = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "B"],
        "snapshot_month": [
            "2023-01-01", "2023-02-01", "2023-03-01",
            "2023-01-01", "2023-02-01", "2023-03-01",
        ],
        "balance": [100, 110, 120, 200, 210, 220],
        "utilization": [0.1, 0.2, 0.25, 0.5, 0.55, 0.6],
        "target": [0, 0, 1, 0, 1, 0],
    })
    data["snapshot_month"] = pd.to_datetime(data["snapshot_month"])

    cfg = Config(
        id_column="customer_id",
        time_column="snapshot_month",
        target_column="target",
        enable_tsfresh_features=True,
        enable_tsfresh_rolling=True,
        tsfresh_window_months=2,
        tsfresh_min_events=0,
        tsfresh_min_unique_months=0,
        tsfresh_min_coverage_ratio=0.0,
    )

    processor = DataProcessor(cfg)
    df_valid = processor.validate_and_freeze(data)
    features = processor.generate_tsfresh_features(df_valid)

    assert processor.tsfresh_merge_mode == "row"
    assert isinstance(features, pd.DataFrame)
    assert "tsfresh_events_count" in features.columns
    assert "tsfresh_window_ready" in features.columns
    assert len(features) == len(data)

    coverage = getattr(processor, "tsfresh_coverage_", None)
    assert isinstance(coverage, pd.DataFrame)
    assert len(coverage) == len(data)
    assert "tsfresh_coverage_ratio" in coverage.columns
    assert coverage["tsfresh_coverage_ratio"].between(0.0, 1.0).all()
    assert set(coverage["tsfresh_window_ready"].unique()).issubset({0, 1})


def test_config_requires_id_and_time_for_rolling():
    with pytest.raises(ValueError):
        Config(enable_tsfresh_features=True, enable_tsfresh_rolling=True)


def test_config_auto_enables_tsfresh_when_rolling_enabled():
    cfg = Config(
        id_column="customer_id",
        time_column="snapshot_month",
        enable_tsfresh_rolling=True,
    )
    assert cfg.enable_tsfresh_features is True
    assert cfg.tsfresh_window_months >= 1
