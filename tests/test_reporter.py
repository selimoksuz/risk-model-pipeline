from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from risk_pipeline.core.reporter import EnhancedReporter

class DummyConfig:
    pass


def _build_model_results():
    return {
        'selected_features': ['AGE_woe', 'income'],
        'feature_importance': {
            'LogisticRegression': pd.DataFrame(
                {
                    'feature': ['AGE_woe', 'income'],
                    'importance': [0.7, 0.3],
                }
            )
        },
        'shap_importance': pd.DataFrame(
            {
                'feature': ['AGE_woe', 'income'],
                'shap_importance': [0.5, 0.2],
            }
        ),
        'scores': {'LogisticRegression': {'test_auc': 0.82, 'oot_auc': 0.80}},
        'best_model_name': 'LogisticRegression',
    }


def _build_woe_results():
    return {
        'woe_values': {
            'AGE': {
                'iv': 0.32,
                'woe_map': {'bin1': -0.12, 'bin2': 0.21},
                'type': 'numeric',
                'bins': [float('-inf'), 0.0, float('inf')],
                'stats': [
                    {
                        'bin_index': 1,
                        'bin': '(-inf, 0.00]',
                        'bin_left': float('-inf'),
                        'bin_right': 0.0,
                        'score_min': float('-inf'),
                        'score_max': 0.0,
                        'event_count': 10,
                        'nonevent_count': 90,
                        'total_count': 100,
                        'event_rate': 0.10,
                        'woe': -0.12,
                        'iv_contrib': 0.12,
                        'total_iv': 0.32,
                    },
                    {
                        'bin_index': 2,
                        'bin': '(0.00, inf)',
                        'bin_left': 0.0,
                        'bin_right': float('inf'),
                        'score_min': 0.0,
                        'score_max': float('inf'),
                        'event_count': 30,
                        'nonevent_count': 70,
                        'total_count': 100,
                        'event_rate': 0.30,
                        'woe': 0.21,
                        'iv_contrib': 0.20,
                        'total_iv': 0.32,
                    },
                ],
            },
            'income': {
                'iv': 0.18,
                'woe_map': {'bin1': -0.05, 'bin2': 0.08},
                'type': 'numeric',
                'bins': [float('-inf'), 1.0, float('inf')],
                'stats': [
                    {
                        'bin_index': 1,
                        'bin': '(-inf, 1.00]',
                        'bin_left': float('-inf'),
                        'bin_right': 1.0,
                        'score_min': float('-inf'),
                        'score_max': 1.0,
                        'event_count': 5,
                        'nonevent_count': 95,
                        'total_count': 100,
                        'event_rate': 0.05,
                        'woe': -0.05,
                        'iv_contrib': 0.05,
                        'total_iv': 0.18,
                    },
                    {
                        'bin_index': 2,
                        'bin': '(1.00, inf)',
                        'bin_left': 1.0,
                        'bin_right': float('inf'),
                        'score_min': 1.0,
                        'score_max': float('inf'),
                        'event_count': 20,
                        'nonevent_count': 80,
                        'total_count': 100,
                        'event_rate': 0.20,
                        'woe': 0.08,
                        'iv_contrib': 0.13,
                        'total_iv': 0.18,
                    },
                ],
            },
        },
        'univariate_gini': {
            'AGE': {'gini_raw': 0.27, 'gini_woe': 0.31, 'gini_drop': -0.04},
            'income': {'gini_raw': 0.14, 'gini_woe': 0.17, 'gini_drop': -0.03},
        },
    }


def test_feature_report_includes_dictionary_details():
    reporter = EnhancedReporter(DummyConfig())
    model_results = _build_model_results()
    woe_results = _build_woe_results()
    data_dictionary = pd.DataFrame(
        {
            'Variable': ['AGE', 'income'],
            'Desc_Text': ['Age of applicant', 'Monthly income'],
            'Category': ['Demographic', 'Financial'],
        }
    )

    features_df = reporter.generate_feature_report(
        model_results,
        woe_results,
        data_dictionary,
    )

    assert {'feature', 'raw_feature', 'description', 'category'} <= set(features_df.columns)

    age_row = features_df[features_df['feature'] == 'AGE_woe'].iloc[0]
    assert age_row['raw_feature'].lower() == 'age'
    assert age_row['description'] == 'Age of applicant'
    assert age_row['category'] == 'Demographic'
    assert age_row['iv'] == pytest.approx(0.32)

    income_row = features_df[features_df['feature'] == 'income'].iloc[0]
    assert income_row['raw_feature'] == 'income'
    assert income_row['description'] == 'Monthly income'
    assert income_row['category'] == 'Financial'


def test_model_report_uses_selected_features_count():
    reporter = EnhancedReporter(DummyConfig())
    data_dictionary = pd.DataFrame(
        {
            'variable': ['age'],
            'description': ['Age of applicant'],
            'category': ['Demographic'],
        }
    )
    model_results = {
        'scores': {'LR': {'test_auc': 0.81}},
        'best_model_name': 'LR',
        'selected_features': ['age'],
    }

    report = reporter.generate_model_report(model_results, data_dictionary)

    assert report['best_model'] == 'LR'
    assert report['feature_count'] == 1
    assert report['model_scores']['LR']['test_auc'] == 0.81




def test_generate_best_model_reports_creates_bins():
    reporter = EnhancedReporter(DummyConfig())
    model_results = _build_model_results()
    woe_results = _build_woe_results()
    data_dictionary = pd.DataFrame(
        {
            'Variable': ['AGE', 'income'],
            'Description': ['Age of applicant', 'Monthly income'],
            'Category': ['Demographic', 'Financial'],
        }
    )

    reporter.generate_feature_report(model_results, woe_results, data_dictionary)
    best_reports = reporter.generate_best_model_reports(model_results, woe_results, data_dictionary)

    assert 'best_model_bins' in reporter.reports_
    bins_df = reporter.reports_['best_model_bins']
    assert not bins_df.empty
    assert 'event_count' in bins_df.columns
    age_bins = bins_df[bins_df['raw_feature'] == 'AGE']
    assert not age_bins.empty
    assert age_bins.iloc[0]['event_count'] == 10

    final_vars = best_reports.get('final_variables')
    assert final_vars is not None
    assert set(final_vars['raw_feature']) == {'AGE', 'income'}
