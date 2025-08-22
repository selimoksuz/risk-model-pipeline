import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from risk_pipeline.reporting.shap_utils import compute_shap_values, summarize_shap, analyze_shap
from risk_pipeline.monitoring import monitor_scores


def test_shap_and_monitor(tmp_path):
    X = pd.DataFrame({'x': [-1.0, 1.0, -2.0, 2.0]})
    y = pd.Series([0, 1, 0, 1])
    mdl = LogisticRegression().fit(X, y)

    sv, Xs = compute_shap_values(mdl, X, shap_sample=2, random_state=0)
    summary = summarize_shap(sv, ['x'])
    details, stability = analyze_shap(sv, Xs)
    assert 'x' in summary and not details.empty

    baseline = pd.DataFrame({'x': [-1, -0.5, 0.5, 1], 'target': [0, 0, 1, 1]})
    new = pd.DataFrame({'x': [-1, 0, 1, 2], 'target': [0, 0, 1, 1]})
    base_path = tmp_path / 'base.csv'
    new_path = tmp_path / 'new.csv'
    baseline.to_csv(base_path, index=False)
    new.to_csv(new_path, index=False)

    mapping = {
        'variables': {
            'x': {
                'type': 'numeric',
                'bins': [
                    {'left': -np.inf, 'right': 0, 'woe': -0.5},
                    {'left': 0, 'right': np.inf, 'woe': 0.5},
                ],
            }
        }
    }

    model_path = tmp_path / 'model.joblib'
    joblib.dump(mdl, model_path)

    res = monitor_scores(str(base_path), str(new_path), mapping, ['x'], str(model_path), expected_model_type='LogisticRegression')
    assert 'score_psi' in res and 'feature_psi' in res
