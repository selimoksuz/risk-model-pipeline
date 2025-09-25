
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


import types
from types import SimpleNamespace
import sys
import pandas as pd
import pytest

from risk_pipeline.core.data_processor import DataProcessor


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            'customer_id': ['a', 'a', 'b', 'b'],
            'snapshot_time': [1, 2, 1, 2],
            'target': [0, 1, 0, 1],
            'balance': [10.0, 12.0, 8.0, 9.5],
        }
    )


def _make_config(**overrides):
    base = {
        'enable_tsfresh_features': True,
        'id_col': 'customer_id',
        'time_col': 'snapshot_time',
        'target_col': 'target',
        'random_state': 42,
        'tsfresh_window': None,
        'tsfresh_max_ids': None,
        'tsfresh_n_jobs': 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_tsfresh_features_fallback(monkeypatch, sample_dataframe):
    original_import = __import__

    def _raise_for_tsfresh(name, *args, **kwargs):
        if name.startswith('tsfresh'):
            raise ImportError('tsfresh unavailable in test context')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(sys.modules['builtins'], '__import__', _raise_for_tsfresh)

    processor = DataProcessor(_make_config())
    features = processor.generate_tsfresh_features(sample_dataframe)

    assert not features.empty
    assert 'balance_mean_tsfresh' in features.columns
    assert 'snapshot_time_mean_tsfresh' not in features.columns
    assert set(processor.tsfresh_metadata_['generator']) == {'tsfresh_simple'}


def test_generate_tsfresh_features_with_stub(monkeypatch, sample_dataframe):
    captured = {}

    def stub_extract_features(df, column_id, column_sort, column_kind, column_value, default_fc_parameters, **kwargs):
        captured['fc_parameters'] = default_fc_parameters
        unique_ids = pd.Index(df[column_id].drop_duplicates(), name=column_id)
        kinds = sorted(df[column_kind].unique())
        cols = {}
        for kind in kinds:
            cols[f'{kind}__mean'] = [1.0] * len(unique_ids)
            cols[f'{kind}__variance__lag_1'] = [0.5] * len(unique_ids)
        return pd.DataFrame(cols, index=unique_ids)

    feature_extraction_mod = types.ModuleType('tsfresh.feature_extraction')
    feature_extraction_mod.MinimalFCParameters = lambda: {'mean': {}}
    feature_extraction_mod.EfficientFCParameters = lambda: {'mean': {}, 'variance__lag_1': {}}
    feature_extraction_mod.ComprehensiveFCParameters = lambda: {'mean': {}, 'variance__lag_1': {}}

    tsfresh_mod = types.ModuleType('tsfresh')
    tsfresh_mod.extract_features = stub_extract_features

    monkeypatch.setitem(sys.modules, 'tsfresh', tsfresh_mod)
    monkeypatch.setitem(sys.modules, 'tsfresh.feature_extraction', feature_extraction_mod)

    processor = DataProcessor(
        _make_config(tsfresh_custom_fc_parameters=['mean', 'variance__lag_1'])
    )

    features = processor.generate_tsfresh_features(sample_dataframe)

    assert set(features.columns) == {
        'balance__mean_tsfresh',
        'balance__variance__lag_1_tsfresh',
    }
    assert captured['fc_parameters'] == {'mean': {}, 'variance__lag_1': {}}
    assert set(processor.tsfresh_metadata_['generator']) == {'tsfresh_custom'}
    assert 'lag_1' in set(processor.tsfresh_metadata_['parameters'])
