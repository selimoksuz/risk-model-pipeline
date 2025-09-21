"""
Calibration utilities for risk model predictions
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')


class Calibrator:
    """
    Two-stage calibration for risk model predictions
    
    Stage 1: Event rate calibration (isotonic or sigmoid)
    Stage 2: Recent predictions calibration with bounds adjustment
    """
    
    def __init__(self, config):
        """
        Initialize calibrator with configuration
        
        Parameters:
        -----------
        config : Config
            Configuration object with calibration settings
        """
        self.config = config
        self.stage1_calibrator = None
        self.stage2_bounds = None
        self.is_fitted = False
        
    def fit_stage1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        method: Optional[str] = None
    ) -> 'Calibrator':
        """
        Fit Stage 1 calibration model
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        method : str, optional
            Calibration method ('isotonic' or 'sigmoid')
            If None, uses config.calibration_method
            
        Returns:
        --------
        self : Calibrator
            Fitted calibrator
        """
        method = method or self.config.calibration_method
        
        if method == 'isotonic':
            self.stage1_calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'sigmoid':
            self.stage1_calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Reshape for sklearn
        y_pred = y_pred.reshape(-1, 1)
        
        # Fit calibrator
        self.stage1_calibrator.fit(y_pred, y_true)
        
        self.is_fitted = True
        return self
    
    def fit_stage2(
        self,
        stage2_predictions: np.ndarray,
        lower_bound_factor: Optional[float] = None,
        upper_bound_factor: Optional[float] = None
    ) -> 'Calibrator':
        """
        Fit Stage 2 calibration bounds
        
        Parameters:
        -----------
        stage2_predictions : np.ndarray
            Recent calibrated predictions for bounds adjustment
        lower_bound_factor : float, optional
            Factor for lower bound (default from config)
        upper_bound_factor : float, optional
            Factor for upper bound (default from config)
            
        Returns:
        --------
        self : Calibrator
            Fitted calibrator with Stage 2 bounds
        """
        if not self.is_fitted:
            raise ValueError("Stage 1 must be fitted before Stage 2")
        
        lower_factor = lower_bound_factor or self.config.stage2_lower_bound
        upper_factor = upper_bound_factor or self.config.stage2_upper_bound
        
        # Calculate bounds from recent predictions
        self.stage2_bounds = {
            'lower': np.min(stage2_predictions) * lower_factor,
            'upper': np.max(stage2_predictions) * upper_factor
        }
        
        return self
    
    def transform(
        self,
        y_pred: np.ndarray,
        apply_stage2: bool = True
    ) -> np.ndarray:
        """
        Apply calibration to predictions
        
        Parameters:
        -----------
        y_pred : np.ndarray
            Uncalibrated predictions
        apply_stage2 : bool
            Whether to apply Stage 2 bounds adjustment
            
        Returns:
        --------
        np.ndarray
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        # Apply Stage 1 calibration
        y_pred = y_pred.reshape(-1, 1)
        calibrated = self.stage1_calibrator.transform(y_pred).ravel()
        
        # Apply Stage 2 bounds if available and requested
        if apply_stage2 and self.stage2_bounds is not None:
            calibrated = np.clip(
                calibrated,
                self.stage2_bounds['lower'],
                self.stage2_bounds['upper']
            )
        
        return calibrated
    
    def fit_transform(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        stage2_predictions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit calibrator and transform predictions
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels for calibration
        y_pred : np.ndarray
            Predictions to calibrate
        stage2_predictions : np.ndarray, optional
            Recent predictions for Stage 2 bounds
            
        Returns:
        --------
        np.ndarray
            Calibrated predictions
        """
        # Fit Stage 1
        self.fit_stage1(y_true, y_pred)
        
        # Fit Stage 2 if data provided
        if stage2_predictions is not None and self.config.enable_stage2_calibration:
            # First calibrate the stage2 predictions
            stage2_cal = self.transform(stage2_predictions, apply_stage2=False)
            self.fit_stage2(stage2_cal)
        
        # Transform
        return self.transform(y_pred)
    
    def calibration_plot_data(
        self,
        y_true: np.ndarray,
        y_pred_uncal: np.ndarray,
        y_pred_cal: np.ndarray,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Generate data for calibration plots
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_uncal : np.ndarray
            Uncalibrated predictions
        y_pred_cal : np.ndarray
            Calibrated predictions
        n_bins : int
            Number of bins for calibration curve
            
        Returns:
        --------
        pd.DataFrame
            Calibration curve data
        """
        calibration_data = []
        
        for pred_type, predictions in [('uncalibrated', y_pred_uncal),
                                      ('calibrated', y_pred_cal)]:
            # Create bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            for i in range(n_bins):
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
                
                if i == n_bins - 1:  # Include right edge for last bin
                    mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
                
                if mask.sum() > 0:
                    mean_predicted = predictions[mask].mean()
                    mean_actual = y_true[mask].mean()
                    count = mask.sum()
                    
                    calibration_data.append({
                        'type': pred_type,
                        'bin': i + 1,
                        'mean_predicted': mean_predicted,
                        'mean_actual': mean_actual,
                        'count': count,
                        'bin_range': f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                    })
        
        return pd.DataFrame(calibration_data)
    
    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> dict:
        """
        Calculate calibration metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins for ECE calculation
            
        Returns:
        --------
        dict
            Calibration metrics including ECE, MCE, etc.
        """
        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        mce = 0  # Maximum Calibration Error
        
        total_samples = len(y_true)
        
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            
            if i == n_bins - 1:
                mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            
            n_bin = mask.sum()
            
            if n_bin > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred[mask].mean()
                
                # ECE component
                ece += (n_bin / total_samples) * abs(bin_accuracy - bin_confidence)
                
                # MCE update
                mce = max(mce, abs(bin_accuracy - bin_confidence))
        
        # Brier Score (calibration component)
        brier_score = np.mean((y_pred - y_true) ** 2)
        
        # Reliability (calibration) and Resolution (discrimination) decomposition
        # Following Murphy decomposition
        base_rate = y_true.mean()
        
        # Reliability (lower is better)
        reliability = 0
        resolution = 0
        
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            
            if i == n_bins - 1:
                mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            
            n_bin = mask.sum()
            
            if n_bin > 0:
                bin_true_rate = y_true[mask].mean()
                bin_pred_rate = y_pred[mask].mean()
                
                reliability += (n_bin / total_samples) * (bin_pred_rate - bin_true_rate) ** 2
                resolution += (n_bin / total_samples) * (bin_true_rate - base_rate) ** 2
        
        # Uncertainty (constant for given dataset)
        uncertainty = base_rate * (1 - base_rate)
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'calibration_error': reliability  # Same as reliability in Murphy decomposition
        }
        
        return metrics
    
    def hosmer_lemeshow_test(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> dict:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins
            
        Returns:
        --------
        dict
            Test statistic, p-value, and details
        """
        from scipy import stats
        
        # Sort by predicted probabilities
        order = np.argsort(y_pred)
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]
        
        # Create equal-sized bins
        n_samples = len(y_true)
        bin_size = n_samples // n_bins
        
        observed_events = []
        expected_events = []
        observed_non_events = []
        expected_non_events = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                # Last bin gets remaining samples
                bin_indices = slice(i * bin_size, None)
            else:
                bin_indices = slice(i * bin_size, (i + 1) * bin_size)
            
            bin_true = y_true_sorted[bin_indices]
            bin_pred = y_pred_sorted[bin_indices]
            
            # Observed
            o_events = bin_true.sum()
            o_non_events = len(bin_true) - o_events
            
            # Expected
            e_events = bin_pred.sum()
            e_non_events = len(bin_pred) - e_events
            
            observed_events.append(o_events)
            expected_events.append(e_events)
            observed_non_events.append(o_non_events)
            expected_non_events.append(e_non_events)
        
        # Calculate Hosmer-Lemeshow statistic
        hl_statistic = 0
        
        for o_e, e_e, o_ne, e_ne in zip(observed_events, expected_events,
                                        observed_non_events, expected_non_events):
            if e_e > 0:
                hl_statistic += ((o_e - e_e) ** 2) / e_e
            if e_ne > 0:
                hl_statistic += ((o_ne - e_ne) ** 2) / e_ne
        
        # Degrees of freedom = n_bins - 2
        df = n_bins - 2
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(hl_statistic, df)
        
        return {
            'statistic': hl_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'n_bins': n_bins,
            'observed_events': observed_events,
            'expected_events': expected_events,
            'calibrated': p_value > 0.05  # Common significance level
        }


def calibrate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'isotonic',
    cv_folds: int = 3
) -> np.ndarray:
    """
    Simple calibration function using cross-validation
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted probabilities
    method : str
        Calibration method ('isotonic' or 'sigmoid')
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    np.ndarray
        Calibrated predictions
    """
    from sklearn.model_selection import StratifiedKFold
    
    calibrated_probs = np.zeros_like(y_pred)
    
    if cv_folds > 1:
        # Use cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(y_pred, y_true):
            # Fit calibrator on training fold
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                calibrator = LogisticRegression()
            
            calibrator.fit(y_pred[train_idx].reshape(-1, 1), y_true[train_idx])
            
            # Calibrate validation fold
            calibrated_probs[val_idx] = calibrator.transform(
                y_pred[val_idx].reshape(-1, 1)
            ).ravel()
    else:
        # No cross-validation
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            calibrator = LogisticRegression()
        
        calibrator.fit(y_pred.reshape(-1, 1), y_true)
        calibrated_probs = calibrator.transform(y_pred.reshape(-1, 1)).ravel()
    
    return calibrated_probs