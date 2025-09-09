"""
Scoring utilities for the risk model pipeline
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
import json
import joblib


def load_model_artifacts(output_folder: str, run_id: str) -> Tuple[object, list, dict, Optional[object]]:
    """Load trained model, features, WOE mapping, and calibrator"""

    # Load best model
    model_path = f"{output_folder}/best_model_{run_id}.joblib"
    model = joblib.load(model_path)

    # Load final features
    features_path = f"{output_folder}/final_vars_{run_id}.json"
    with open(features_path, 'r') as f:
        final_features = json.load(f)

    # Load WOE mapping
    woe_path = f"{output_folder}/woe_mapping_{run_id}.json"
    with open(woe_path, 'r') as f:
        woe_mapping = json.load(f)

    # Load calibrator if available
    calibrator = None
    calibrator_path = f"{output_folder}/calibrator_{run_id}.joblib"
    try:
        if os.path.exists(calibrator_path):
            calibrator = joblib.load(calibrator_path)
            print(f"✅ Calibrator loaded from: {calibrator_path}")
        else:
            print(f"⚠️  No calibrator found at: {calibrator_path}")
    except Exception as e:
        print(f"⚠️  Could not load calibrator: {e}")

    return model, final_features, woe_mapping, calibrator


def apply_woe_transform(df: pd.DataFrame, woe_mapping: dict) -> pd.DataFrame:
    """Apply WOE transformation to scoring data"""
    from ..stages.woe import apply_woe

    # Check if woe_mapping needs format conversion
    if isinstance(woe_mapping, dict) and "variables" not in woe_mapping:
        # Convert old format to new format for apply_woe
        new_mapping = {"variables": {}}

        for var_name, var_info in woe_mapping.items():
            # Get the original variable name
            if hasattr(var_info, 'var'):
                original_var = var_info.var
            else:
                # Try to extract from the transformed name (e.g., num1_corr95 -> num1)
                original_var = var_name.split('_')[0] if '_' in var_name else var_name

            if original_var not in df.columns:
                continue

            new_mapping["variables"][original_var] = {
                "type": "numeric",
                "bins": []
            }

            # Convert bins
            if hasattr(var_info, 'bins'):
                for bin_info in var_info.bins:
                    bin_range = bin_info.get('range', [-float('inf'), float('inf')])
                    new_mapping["variables"][original_var]["bins"].append({
                        "left": bin_range[0] if bin_range[0] != -float('inf') else None,
                        "right": bin_range[1] if bin_range[1] != float('inf') else None,
                        "woe": bin_info.get('woe', 0)
                    })
            elif hasattr(var_info, 'special'):
                # Handle categorical
                new_mapping["variables"][original_var] = {
                    "type": "categorical",
                    "groups": []
                }
                for label, members in var_info.special.items():
                    new_mapping["variables"][original_var]["groups"].append({
                        "label": label,
                        "members": members,
                        "woe": 0  # You'd need to get the actual WOE value
                    })

        woe_mapping = new_mapping

    # Apply WOE transformation
    df_woe = apply_woe(df, woe_mapping)

    # Add transformed column names to match final_features format
    # e.g., if final_features has num1_corr95 but WOE creates num1, we need to handle this
    transformed_cols = {}
    for col in df_woe.columns:
        # Keep original column
        transformed_cols[col] = df_woe[col]
        # Also create potential transformed names
        transformed_cols[f"{col}_corr95"] = df_woe[col]
        transformed_cols[f"{col}_corr92"] = df_woe[col]
        transformed_cols[f"{col}_woe"] = df_woe[col]

    df_woe_extended = pd.DataFrame(transformed_cols, index=df_woe.index)

    return df_woe_extended


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index"""

    # Convert to float arrays to avoid boolean issues
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Create bins based on expected distribution
    bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf  # Include all values
    bin_edges[-1] = np.inf

    # Bin both distributions
    expected_binned = pd.cut(expected, bins=bin_edges, duplicates='drop')
    actual_binned = pd.cut(actual, bins=bin_edges, duplicates='drop')

    # Calculate distributions (manual normalization for pandas compatibility)
    expected_counts = expected_binned.value_counts()
    actual_counts = actual_binned.value_counts()

    expected_dist = expected_counts / expected_counts.sum()
    actual_dist = actual_counts / actual_counts.sum()

    # Align indices and fill missing values
    all_bins = expected_dist.index.union(actual_dist.index)
    expected_dist = expected_dist.reindex(all_bins, fill_value=0.001)
    actual_dist = actual_dist.reindex(all_bins, fill_value=0.001)

    # Calculate PSI
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))

    return float(psi)


def calculate_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate Gini coefficient"""
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    return float(gini)


def calculate_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    return float(ks)


def score_data(scoring_df: pd.DataFrame,
               model: object,
               final_features: list,
               woe_mapping: dict,
               calibrator: Optional[object] = None,
               training_scores: Optional[np.ndarray] = None,
               feature_mapping: Optional[Dict[str, str]] = None) -> Dict:
    """
    Score new data and calculate metrics

    Args:
        scoring_df: DataFrame to score
        model: Trained model
        final_features: List of features to use
        woe_mapping: WOE transformation mapping
        training_scores: Training scores for PSI calculation (optional)

    Returns:
        Dictionary with scores and metrics
    """

    # Apply WOE transformation
    df_woe = apply_woe_transform(scoring_df, woe_mapping)

    # Handle feature mapping if provided
    if feature_mapping:
        # Map the original final_features to available WOE features
        mapped_features = [feature_mapping.get(f, f) for f in final_features]
        X_score = df_woe[mapped_features]
        # Rename columns back to what model expects
        X_score.columns = final_features
    else:
        # Select features directly
        X_score = df_woe[final_features]

    # Clean any NaN or Inf values before scoring
    X_score = X_score.replace([np.inf, -np.inf], np.nan)
    X_score = X_score.fillna(0)

    # Generate raw predictions
    try:
        pred_result = model.predict_proba(X_score)
        if pred_result.ndim == 2 and pred_result.shape[1] >= 2:
            raw_scores = pred_result[:, 1]  # Use probability of class 1
        else:
            raw_scores = pred_result.ravel()  # Flatten if 1D
    except (AttributeError, IndexError):
        # Fallback for models without predict_proba or with 1D output
        raw_scores = model.predict(X_score)

    # Apply calibration if available
    if calibrator is not None:
        try:
            # For sklearn calibrators that expect 2D input
            if hasattr(calibrator, 'predict_proba'):
                scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            elif hasattr(calibrator, 'predict'):
                scores = calibrator.predict(raw_scores.reshape(-1, 1))
            else:
                # For isotonic regression
                scores = calibrator.transform(raw_scores)
            print(f"✅ Calibration applied to {len(scores)} predictions")
        except Exception as e:
            print(f"⚠️  Calibration failed, using raw scores: {e}")
            scores = raw_scores
    else:
        print(f"⚠️  No calibrator available, using raw model scores")
        scores = raw_scores

    # Calculate PSI if training scores provided
    psi_score = None
    if training_scores is not None:
        try:
            psi_score = calculate_psi(training_scores, scores)
            if np.isinf(psi_score) or np.isnan(psi_score):
                psi_score = None
        except Exception as e:
            print(f"⚠️  PSI calculation failed: {e}")
            psi_score = None

    # Separate records with/without targets
    has_target = ~scoring_df['target'].isna()

    results = {
        'scores': scores,
        'raw_scores': raw_scores,
        'has_target_mask': has_target,
        'n_total': len(scoring_df),
        'n_with_target': has_target.sum(),
        'n_without_target': (~has_target).sum(),
        'psi_score': psi_score,
        'calibration_applied': calibrator is not None
    }

    # Calculate metrics for records WITH targets
    if has_target.sum() > 0:
        y_true = scoring_df.loc[has_target, 'target'].values
        y_scores_with_target = scores[has_target]

        results.update({
            'with_target': {
                'n_records': has_target.sum(),
                'default_rate': float(y_true.mean()),
                'auc': roc_auc_score(y_true, y_scores_with_target),
                'gini': calculate_gini(y_true, y_scores_with_target),
                'ks': calculate_ks_statistic(y_true, y_scores_with_target),
                'score_stats': {
                    'mean': float(y_scores_with_target.mean()),
                    'std': float(y_scores_with_target.std()),
                    'min': float(y_scores_with_target.min()),
                    'max': float(y_scores_with_target.max()),
                    'q25': float(np.percentile(y_scores_with_target, 25)),
                    'q50': float(np.percentile(y_scores_with_target, 50)),
                    'q75': float(np.percentile(y_scores_with_target, 75))
                }
            }
        })

    # Calculate metrics for records WITHOUT targets
    if (~has_target).sum() > 0:
        y_scores_without_target = scores[~has_target]

        results.update({
            'without_target': {
                'n_records': (~has_target).sum(),
                'score_stats': {
                    'mean': float(y_scores_without_target.mean()),
                    'std': float(y_scores_without_target.std()),
                    'min': float(y_scores_without_target.min()),
                    'max': float(y_scores_without_target.max()),
                    'q25': float(np.percentile(y_scores_without_target, 25)),
                    'q50': float(np.percentile(y_scores_without_target, 50)),
                    'q75': float(np.percentile(y_scores_without_target, 75))
                }
            }
        })

    return results


def create_scoring_report(results: Dict) -> Dict[str, pd.DataFrame]:
    """Create comprehensive scoring reports"""

    reports = {}

    # Overall summary
    summary_data = {
        'Metric': [
            'Total Records',
            'Records with Target',
            'Records without Target',
            'Percentage with Target',
            'Calibration Applied'
        ],
        'Value': [
            f"{results['n_total']:, }",
            f"{results['n_with_target']:, }",
            f"{results['n_without_target']:, }",
            f"{results['n_with_target'] / results['n_total'] * 100:.1f}%",
            'Yes' if results.get('calibration_applied', False) else 'No'
        ]
    }

    # Add PSI if available
    if results.get('psi_score') is not None and results['psi_score'] != float('inf'):
        summary_data['Metric'].append('PSI (Population Stability Index)')
        summary_data['Value'].append(f"{results['psi_score']:.4f}")

    reports['summary'] = pd.DataFrame(summary_data)

    # Report for records WITH targets
    if 'with_target' in results:
        wt = results['with_target']
        with_target_data = {
            'Metric': [
                'Records with Target',
                'Default Rate',
                'AUC (Area Under Curve)',
                'Gini Coefficient',
                'KS Statistic',
                'Score Mean',
                'Score Std',
                'Score Min',
                'Score 25%',
                'Score 50% (Median)',
                'Score 75%',
                'Score Max'
            ],
            'Value': [
                f"{wt['n_records']:, }",
                f"{wt['default_rate']:.3f}",
                f"{wt['auc']:.4f}",
                f"{wt['gini']:.4f}",
                f"{wt['ks']:.4f}",
                f"{wt['score_stats']['mean']:.4f}",
                f"{wt['score_stats']['std']:.4f}",
                f"{wt['score_stats']['min']:.4f}",
                f"{wt['score_stats']['q25']:.4f}",
                f"{wt['score_stats']['q50']:.4f}",
                f"{wt['score_stats']['q75']:.4f}",
                f"{wt['score_stats']['max']:.4f}"
            ]
        }
        reports['with_target'] = pd.DataFrame(with_target_data)

    # Report for records WITHOUT targets
    if 'without_target' in results:
        wot = results['without_target']
        without_target_data = {
            'Metric': [
                'Records without Target',
                'Score Mean',
                'Score Std',
                'Score Min',
                'Score 25%',
                'Score 50% (Median)',
                'Score 75%',
                'Score Max'
            ],
            'Value': [
                f"{wot['n_records']:, }",
                f"{wot['score_stats']['mean']:.4f}",
                f"{wot['score_stats']['std']:.4f}",
                f"{wot['score_stats']['min']:.4f}",
                f"{wot['score_stats']['q25']:.4f}",
                f"{wot['score_stats']['q50']:.4f}",
                f"{wot['score_stats']['q75']:.4f}",
                f"{wot['score_stats']['max']:.4f}"
            ]
        }
        reports['without_target'] = pd.DataFrame(without_target_data)

    return reports
