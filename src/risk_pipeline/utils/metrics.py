"""
Metrics calculation utilities for model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
    matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for binary classification
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    y_pred_binary : np.ndarray, optional
        Binary predictions. If None, will be calculated using threshold
    threshold : float
        Threshold for binary classification
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all metrics
    """
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Calculate binary predictions if not provided
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # Threshold-independent metrics
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['auc'] = np.nan
    
    try:
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
    except:
        metrics['average_precision'] = np.nan
    
    try:
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    except:
        metrics['brier_score'] = np.nan
    
    try:
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    except:
        metrics['log_loss'] = np.nan
    
    # Threshold-dependent metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred_binary)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Lift and gain at different percentiles
    metrics.update(calculate_lift_gain(y_true, y_pred_proba))
    
    # Gini coefficient
    metrics['gini'] = 2 * metrics['auc'] - 1 if not np.isnan(metrics['auc']) else np.nan
    
    # KS statistic
    metrics['ks_statistic'] = calculate_ks_statistic(y_true, y_pred_proba)
    
    return metrics


def calculate_lift_gain(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    percentiles: list = [10, 20, 30, 40, 50]
) -> Dict[str, float]:
    """
    Calculate lift and gain at different percentiles
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    percentiles : list
        Percentiles to calculate lift and gain
        
    Returns:
    --------
    Dict[str, float]
        Lift and gain metrics
    """
    
    # Create dataframe for easier manipulation
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_proba
    })
    
    # Sort by predicted probability
    df = df.sort_values('y_pred', ascending=False)
    
    # Calculate cumulative metrics
    df['cumulative_positives'] = df['y_true'].cumsum()
    df['cumulative_total'] = range(1, len(df) + 1)
    
    total_positives = df['y_true'].sum()
    total_samples = len(df)
    base_rate = total_positives / total_samples
    
    metrics = {}
    
    for percentile in percentiles:
        n_samples = int(len(df) * (percentile / 100))
        if n_samples > 0:
            # Gain
            gain = df.iloc[n_samples - 1]['cumulative_positives'] / total_positives
            metrics[f'gain_{percentile}'] = gain
            
            # Lift
            observed_rate = df.iloc[:n_samples]['y_true'].mean()
            lift = observed_rate / base_rate if base_rate > 0 else 0
            metrics[f'lift_{percentile}'] = lift
    
    return metrics


def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
        
    Returns:
    --------
    float
        KS statistic
    """
    
    # Separate positive and negative classes
    pos_scores = y_pred_proba[y_true == 1]
    neg_scores = y_pred_proba[y_true == 0]
    
    # Calculate cumulative distributions
    all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))
    
    # Calculate empirical CDFs
    pos_cdf = np.searchsorted(np.sort(pos_scores), all_scores, side='right') / len(pos_scores)
    neg_cdf = np.searchsorted(np.sort(neg_scores), all_scores, side='right') / len(neg_scores)
    
    # KS statistic is the maximum difference
    ks_statistic = np.max(np.abs(pos_cdf - neg_cdf))
    
    return ks_statistic


def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = 'youden'
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate optimal threshold for binary classification
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    method : str
        Method to find optimal threshold ('youden', 'f1', 'precision_recall')
        
    Returns:
    --------
    Tuple[float, Dict[str, Any]]
        Optimal threshold and metrics at that threshold
    """
    
    if method == 'youden':
        # Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif method == 'f1':
        # Maximize F1 score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif method == 'precision_recall':
        # Balance precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate metrics at optimal threshold
    metrics = calculate_metrics(y_true, y_pred_proba, threshold=optimal_threshold)
    
    return optimal_threshold, metrics


def create_performance_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    dataset_name: str = 'Dataset'
) -> pd.DataFrame:
    """
    Create a comprehensive performance report
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    pd.DataFrame
        Performance report
    """
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_proba)
    
    # Create report
    report_data = {
        'Dataset': dataset_name,
        'AUC': f"{metrics['auc']:.4f}",
        'Gini': f"{metrics['gini']:.4f}",
        'KS': f"{metrics['ks_statistic']:.4f}",
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1 Score': f"{metrics['f1']:.4f}",
        'Brier Score': f"{metrics['brier_score']:.4f}",
        'Log Loss': f"{metrics['log_loss']:.4f}",
        'Lift @10%': f"{metrics.get('lift_10', 0):.2f}",
        'Gain @10%': f"{metrics.get('gain_10', 0):.2%}",
    }
    
    return pd.DataFrame([report_data])


def compare_models(
    models_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models
    
    Parameters:
    -----------
    models_results : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary with model names and their predictions
    y_true : np.ndarray
        True labels
        
    Returns:
    --------
    pd.DataFrame
        Comparison report
    """
    
    comparison_data = []
    
    for model_name, y_pred_proba in models_results.items():
        metrics = calculate_metrics(y_true, y_pred_proba)
        
        row = {
            'Model': model_name,
            'AUC': metrics['auc'],
            'Gini': metrics['gini'],
            'KS': metrics['ks_statistic'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'Brier': metrics['brier_score'],
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    
    return comparison_df