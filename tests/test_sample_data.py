import pandas as pd

from risk_pipeline.data.sample import CreditRiskSample, copy_credit_risk_sample, load_credit_risk_sample


def test_sample_loader_returns_complete_bundle(tmp_path):
    sample = load_credit_risk_sample()

    assert isinstance(sample, CreditRiskSample)
    datasets = sample.as_dict()
    assert set(datasets.keys()) == {
        'development',
        'calibration_longrun',
        'calibration_recent',
        'scoring_future',
        'data_dictionary',
    }
    for frame in datasets.values():
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty

    destination = tmp_path / 'credit_risk_copy'
    target_dir = copy_credit_risk_sample(destination)
    copied_files = {p.name for p in target_dir.iterdir() if p.is_file()}
    assert {'development.csv', 'calibration_longrun.csv'}.issubset(copied_files)
