"""Calibration Engine for Stage 1 and Stage 2 calibration"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


class CalibrationEngine:
    """
    Handles Stage 1 and Stage 2 calibration for risk models.

    Stage 1: Long-run average calibration based on event rates
    Stage 2: Recent period calibration with lower/upper bound adjustment
    """

    def __init__(self, config):
        self.config = config
        self.stage1_model_ = None
        self.stage2_model_ = None
        self.calibration_info_ = {}

    def calibrate_stage1(self, model, X: pd.DataFrame, y: pd.Series,
                        method: str = 'platt') -> 'CalibratedModel':
        """
        Apply Stage 1 calibration using long-run average.

        Parameters:
        -----------
        model : sklearn estimator
            Base model to calibrate
        X : pd.DataFrame
            Features for calibration
        y : pd.Series
            Target variable
        method : str
            Calibration method ('platt' for Platt scaling, 'isotonic' for Isotonic regression)

        Returns:
        --------
        CalibratedModel
            Calibrated model wrapper
        """
        print("    Applying Stage 1 calibration...")

        # Get base predictions
        base_scores = model.predict_proba(X)[:, 1]

        # Calculate long-run average
        long_run_avg = y.mean()
        self.calibration_info_['long_run_average'] = long_run_avg
        print(f"    Long-run average event rate: {long_run_avg:.4f}")

        if method == 'platt':
            # Platt scaling (sigmoid calibration)
            calibrator = LogisticRegression()
            calibrator.fit(base_scores.reshape(-1, 1), y)
            self.stage1_model_ = PlattCalibratedModel(model, calibrator)

        elif method == 'isotonic':
            # Isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(base_scores, y)
            self.stage1_model_ = IsotonicCalibratedModel(model, calibrator)

        else:
            # Sklearn's calibration
            calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
            calibrated.fit(X, y)
            self.stage1_model_ = calibrated

        # Verify calibration
        calibrated_scores = self.stage1_model_.predict_proba(X)[:, 1]
        calibrated_avg = calibrated_scores.mean()
        print(f"    Calibrated average prediction: {calibrated_avg:.4f}")
        print(f"    Calibration difference: {abs(calibrated_avg - long_run_avg):.4f}")

        self.calibration_info_['stage1_method'] = method
        self.calibration_info_['stage1_applied'] = True

        return self.stage1_model_

    def calibrate_stage2(self, stage1_model, X_recent: pd.DataFrame, y_recent: pd.Series,
                        method: str = 'lower_mean') -> 'CalibratedModel':
        """
        Apply Stage 2 calibration using recent predictions.

        Parameters:
        -----------
        stage1_model : CalibratedModel
            Stage 1 calibrated model
        X_recent : pd.DataFrame
            Recent period features
        y_recent : pd.Series
            Recent period target
        method : str
            'lower_mean' or 'upper_bound' calibration

        Returns:
        --------
        CalibratedModel
            Stage 2 calibrated model
        """
        print("    Applying Stage 2 calibration...")

        # Get Stage 1 predictions on recent data
        stage1_scores = stage1_model.predict_proba(X_recent)[:, 1]

        # Calculate recent event rate
        recent_avg = y_recent.mean()
        self.calibration_info_['recent_average'] = recent_avg
        print(f"    Recent period event rate: {recent_avg:.4f}")

        # Calculate adjustment factor based on method
        if method == 'lower_mean':
            # Use lower of long-run and recent average
            target_rate = min(self.calibration_info_['long_run_average'], recent_avg)
            print(f"    Using lower mean: {target_rate:.4f}")

        elif method == 'upper_bound':
            # Use upper bound with confidence interval
            ci_upper = recent_avg + 1.96 * np.sqrt(recent_avg * (1 - recent_avg) / len(y_recent))
            target_rate = min(ci_upper, self.calibration_info_['long_run_average'])
            print(f"    Using upper bound: {target_rate:.4f}")

        else:
            # Weighted average
            weight = self.config.stage2_weight if hasattr(self.config, 'stage2_weight') else 0.3
            target_rate = (weight * recent_avg +
                          (1 - weight) * self.calibration_info_['long_run_average'])
            print(f"    Using weighted average: {target_rate:.4f}")

        # Calculate adjustment factor
        current_mean = stage1_scores.mean()
        adjustment_factor = target_rate / (current_mean + 1e-10)

        # Create Stage 2 calibrated model
        self.stage2_model_ = AdjustedCalibratedModel(
            stage1_model,
            adjustment_factor,
            target_rate
        )

        # Verify calibration
        final_scores = self.stage2_model_.predict_proba(X_recent)[:, 1]
        final_avg = final_scores.mean()
        print(f"    Final calibrated average: {final_avg:.4f}")

        self.calibration_info_['stage2_method'] = method
        self.calibration_info_['stage2_applied'] = True
        self.calibration_info_['adjustment_factor'] = adjustment_factor

        return self.stage2_model_

    def evaluate_calibration(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate calibration quality using multiple metrics.
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        predictions = model.predict_proba(X)[:, 1]

        # Brier Score
        brier = brier_score_loss(y, predictions)

        # Expected Calibration Error (ECE)
        fraction_pos, mean_pred = calibration_curve(y, predictions, n_bins=10)
        ece = np.mean(np.abs(fraction_pos - mean_pred))

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_pred))

        # Hosmer-Lemeshow test
        from scipy import stats
        n_bins = 10
        _, bins = pd.qcut(predictions, n_bins, retbins=True, duplicates='drop')
        binned = pd.cut(predictions, bins, include_lowest=True)

        observed = y.groupby(binned).sum()
        expected = y.groupby(binned).size() * predictions.groupby(binned).mean()

        # Chi-square test
        chi2_stat = np.sum((observed - expected)**2 / (expected + 1e-10))
        p_value = 1 - stats.chi2.cdf(chi2_stat, n_bins - 2)

        return {
            'brier_score': brier,
            'ece': ece,
            'mce': mce,
            'hosmer_lemeshow_stat': chi2_stat,
            'hosmer_lemeshow_pval': p_value,
            'mean_prediction': predictions.mean(),
            'actual_rate': y.mean(),
            'calibration_error': abs(predictions.mean() - y.mean())
        }


class PlattCalibratedModel:
    """Wrapper for Platt scaled model."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        base_scores = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(base_scores.reshape(-1, 1))
        return calibrated

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class IsotonicCalibratedModel:
    """Wrapper for Isotonic regression calibrated model."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        base_scores = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(base_scores)
        # Ensure output is 2D for consistency
        proba = np.zeros((len(calibrated), 2))
        proba[:, 1] = calibrated
        proba[:, 0] = 1 - calibrated
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class AdjustedCalibratedModel:
    """Wrapper for Stage 2 adjusted calibrated model."""

    def __init__(self, stage1_model, adjustment_factor, target_rate):
        self.stage1_model = stage1_model
        self.adjustment_factor = adjustment_factor
        self.target_rate = target_rate

    def predict_proba(self, X):
        # Get Stage 1 predictions
        stage1_proba = self.stage1_model.predict_proba(X)

        # Apply adjustment with bounds
        adjusted = stage1_proba.copy()
        adjusted[:, 1] = np.clip(
            stage1_proba[:, 1] * self.adjustment_factor,
            0.0, 1.0
        )
        adjusted[:, 0] = 1 - adjusted[:, 1]

        return adjusted

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)