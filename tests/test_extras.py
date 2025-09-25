import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("RISK_PIPELINE_DISABLE_SHAP", "1")

from risk_pipeline.reporting import shap_utils
from risk_pipeline.monitoring import monitor_scores

if shap_utils.shap is None:
    pytest.skip("SHAP desteklenmediği için test atlandı", allow_module_level=True)


def test_shap_and_monitor(tmp_path):
    X = pd.DataFrame({'x': [-1.0, 1.0, -2.0, 2.0]})
    y = pd.Series([0, 1, 0, 1])
    mdl = LogisticRegression().fit(X, y)

    shap_values = shap_utils.compute_shap_values(mdl, X, shap_sample=2, random_state=0)
    summary = shap_utils.summarize_shap(shap_values, ['x'])
    assert 'x' in summary

    monitor = monitor_scores(
        actuals=y,
        predictions=pd.Series([0.1, 0.8, 0.2, 0.7]),
        bands=5,
        baseline=None,
    )

    assert 'ks' in monitor
    assert 'lift_table' in monitor
