"""
Scoring utilities for the risk model pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
import json
import joblib

def load_model_artifacts(output_folder: str, run_id: str) -> Tuple[object, list, dict]:
    """Load trained model, features, and WOE mapping"""
    
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
    
    return model, final_features, woe_mapping

def apply_woe_transform(df: pd.DataFrame, woe_mapping: dict) -> pd.DataFrame:
    """Apply WOE transformation to scoring data"""
    from ..stages import apply_woe
    
    # Apply WOE transformation
    df_woe = apply_woe(df, woe_mapping)
    
    return df_woe

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index"""
    
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
    
    # Generate predictions
    try:
        scores = model.predict_proba(X_score)[:, 1]
    except AttributeError:
        # Fallback for models without predict_proba
        scores = model.predict(X_score)
    
    # Calculate PSI if training scores provided
    psi_score = None
    if training_scores is not None:
        psi_score = calculate_psi(training_scores, scores)
    
    # Separate records with/without targets
    has_target = ~scoring_df['target'].isna()
    
    results = {
        'scores': scores,
        'has_target_mask': has_target,
        'n_total': len(scoring_df),
        'n_with_target': has_target.sum(),
        'n_without_target': (~has_target).sum(),
        'psi_score': psi_score
    }
    
    # Calculate metrics for records with targets
    if has_target.sum() > 0:
        y_true = scoring_df.loc[has_target, 'target'].values
        y_scores = scores[has_target]
        
        results.update({
            'auc': roc_auc_score(y_true, y_scores),
            'gini': calculate_gini(y_true, y_scores),
            'ks': calculate_ks_statistic(y_true, y_scores),
            'default_rate': float(y_true.mean())
        })
    
    return results

def create_scoring_report(results: Dict) -> pd.DataFrame:
    """Create a summary report of scoring results"""
    
    report_data = {
        'Metric': [],
        'Value': []
    }
    
    # Basic statistics
    report_data['Metric'].extend([
        'Total Records',
        'Records with Target',
        'Records without Target',
        'Percentage with Target'
    ])
    
    report_data['Value'].extend([
        f"{results['n_total']:,}",
        f"{results['n_with_target']:,}",
        f"{results['n_without_target']:,}",
        f"{results['n_with_target'] / results['n_total'] * 100:.1f}%"
    ])
    
    # PSI
    if results.get('psi_score') is not None:
        report_data['Metric'].append('PSI (Population Stability Index)')
        report_data['Value'].append(f"{results['psi_score']:.4f}")
    
    # Performance metrics for records with targets
    if 'auc' in results:
        report_data['Metric'].extend([
            'AUC (Area Under Curve)',
            'Gini Coefficient', 
            'KS Statistic',
            'Default Rate'
        ])
        
        report_data['Value'].extend([
            f"{results['auc']:.4f}",
            f"{results['gini']:.4f}",
            f"{results['ks']:.4f}",
            f"{results['default_rate']:.3f}"
        ])
    
    return pd.DataFrame(report_data)