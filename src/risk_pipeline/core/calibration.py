"""
Two-Stage Calibration Engine
Stage 1: Long-run average calibration
Stage 2: Recent period adjustment (lower_mean/upper_bound)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


from .utils import predict_positive_proba

class TwoStageCalibrator:
    """
    Two-stage calibration for risk models.

    Stage 1: Calibrate to long-run average default rate
    Stage 2: Adjust for recent period trends
    """

    def __init__(self, config):
        self.config = config
        self.stage1_calibrator_ = None
        self.stage2_calibrator_ = None
        self.calibration_metrics_ = {}

    def calibrate_stage1(self, model, X: pd.DataFrame, y: pd.Series,
                        method: str = 'isotonic') -> Any:
        """
        Stage 1: Calibrate model to long-run average.

        Parameters:
        -----------
        model : sklearn estimator
            Trained model to calibrate
        X : pd.DataFrame
            Calibration features
        y : pd.Series
            Calibration targets
        method : str
            Calibration method ('isotonic', 'sigmoid', 'platt')
        """

        print(f"    Applying Stage 1 calibration (method={method})...")

        # Handle NaN values by filling with 0
        X_filled = X.fillna(0)

        # Get base predictions
        base_scores = predict_positive_proba(model, X_filled)

        # Calculate long-run default rate (Stage-1 anchor)
        long_run_rate = y.mean()
        print(f"      Stage 1 target (long-run mean) default rate: {long_run_rate:.2%}")

        # Apply calibration based on method
        if method == 'isotonic':
            calibrated_model = self._isotonic_calibration(model, X, y)
        elif method in ['sigmoid', 'platt']:
            calibrated_model = self._platt_calibration(model, X, y)
        elif method == 'beta':
            calibrated_model = self._beta_calibration(model, X, y, base_scores)
        else:
            # Simple scaling calibration
            calibrated_model = self._scaling_calibration(model, base_scores, y)

        # Store calibrator
        self.stage1_calibrator_ = calibrated_model

        # Evaluate calibration
        calibrated_scores = predict_positive_proba(calibrated_model, X_filled)
        metrics = self._evaluate_calibration_quality(calibrated_scores, y)
        metrics['long_run_rate'] = float(long_run_rate)
        metrics['base_rate'] = float(base_scores.mean())
        metrics['method'] = method

        print(f"      ECE: {metrics['ece']:.4f}")
        print(f"      MCE: {metrics['mce']:.4f}")
        print(f"      Brier Score: {metrics['brier']:.4f}")

        self.calibration_metrics_['stage1'] = metrics

        return calibrated_model

    def calibrate_stage2(self, stage1_model, X_recent: pd.DataFrame, y_recent: pd.Series,
                        method: str = 'lower_mean') -> Any:
        """Stage 2: Adjust for recent period trends."""

        print(f"    Applying Stage 2 calibration (method={method})...")

        X_recent_filled = X_recent.fillna(0) if hasattr(X_recent, 'fillna') else X_recent
        stage1_scores = predict_positive_proba(stage1_model, X_recent_filled)

        recent_rate = float(y_recent.mean())
        stage1_rate = float(stage1_scores.mean())

        print(f"      Recent default rate: {recent_rate:.2%}")
        print(f"      Stage 1 predicted rate: {stage1_rate:.2%}")

        stage2_info = {
            'method': method,
            'recent_rate': recent_rate,
            'stage1_rate': stage1_rate
        }

        if method == 'lower_mean':
            adjusted_model = self._lower_mean_adjustment(
                stage1_model, stage1_scores, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['adjustment_factor'] = float(getattr(adjusted_model, 'adjustment', 1.0))
        elif method == 'upper_bound':
            adjusted_model = self._upper_bound_adjustment(
                stage1_model, stage1_scores, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = float(getattr(adjusted_model, 'target_rate', recent_rate))
            stage2_info['upper_bound'] = float(getattr(adjusted_model, 'upper_bound', stage1_rate))
        elif method == 'weighted':
            adjusted_model = self._weighted_adjustment(
                stage1_model, X_recent, y_recent, stage1_scores
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['weight'] = float(getattr(adjusted_model, 'weight', 0.5))
        elif method == 'shift':
            adjusted_model = self._shift_adjustment(
                stage1_model, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['adjustment_factor'] = float(getattr(adjusted_model, 'shift', 0.0))
        elif method in ['lower_ci', 'upper_ci', 'mean_ci', 'manual', 'expert']:
            adjusted_model, ci_info = self._confidence_interval_adjustment(
                stage1_model, stage1_scores, y_recent, method
            )
            stage2_info.update(ci_info)
        else:
            adjusted_model = self._shift_adjustment(
                stage1_model, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['adjustment_factor'] = float(getattr(adjusted_model, 'shift', 0.0))

        self.stage2_calibrator_ = adjusted_model

        adjusted_scores = predict_positive_proba(adjusted_model, X_recent_filled)
        metrics = self._evaluate_calibration_quality(adjusted_scores, y_recent)
        achieved_rate = float(adjusted_scores.mean())

        print(f"      ECE: {metrics['ece']:.4f}")
        print(f"      MCE: {metrics['mce']:.4f}")
        print(f"      Adjusted mean prediction: {achieved_rate:.2%}")

        stage2_info['achieved_rate'] = achieved_rate
        metrics['method'] = method
        metrics['recent_rate'] = recent_rate
        metrics['stage1_rate'] = stage1_rate
        metrics['target_rate'] = float(stage2_info.get('target_rate', recent_rate))
        metrics['achieved_rate'] = achieved_rate

        self.calibration_metrics_['stage2'] = metrics
        self.stage2_metadata_ = stage2_info

        return adjusted_model

    def _isotonic_calibration(self, model, X: pd.DataFrame, y: pd.Series):
        """Apply isotonic regression calibration."""

        X_filled = X.fillna(0)
        base_scores = predict_positive_proba(model, X_filled)

        try:
            calibrated = CalibratedClassifierCV(
                model,
                method='isotonic',
                cv='prefit'
            )
            calibrated.fit(X_filled, y)
            return calibrated
        except Exception:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(base_scores, y)

            class _IsotonicWrapper:
                def __init__(self, base_model, iso_model):
                    self.base_model = base_model
                    self.iso_model = iso_model
                    self.classes_ = np.array([0, 1])

                def predict_proba(self, X):
                    probs = predict_positive_proba(self.base_model, X)
                    calibrated_probs = np.clip(self.iso_model.predict(probs), 0, 1)
                    return np.column_stack([1 - calibrated_probs, calibrated_probs])

                def predict(self, X):
                    return (predict_positive_proba(self, X) > 0.5).astype(int)

            return _IsotonicWrapper(model, iso)

    def _platt_calibration(self, model, X: pd.DataFrame, y: pd.Series):
        """Apply Platt scaling (sigmoid) calibration."""

        # Handle NaN values
        X_filled = X.fillna(0)

        calibrated = CalibratedClassifierCV(
            model,
            method='sigmoid',
            cv='prefit'
        )

        calibrated.fit(X_filled, y)
        return calibrated

    def _beta_calibration(self, model, X: pd.DataFrame, y: pd.Series, base_scores: np.ndarray):
        """Apply beta calibration."""

        from scipy.optimize import minimize
        from scipy.stats import beta

        # Fit beta distribution parameters
        def neg_log_likelihood(params):
            a, b = params
            # Transform scores to (0, 1) interval
            eps = 1e-6
            scores_transformed = np.clip(base_scores, eps, 1 - eps)

            # Calculate log likelihood
            ll = np.sum(
                y * np.log(beta.cdf(scores_transformed, a, b) + eps) +
                (1 - y) * np.log(1 - beta.cdf(scores_transformed, a, b) + eps)
            )
            return -ll

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[2, 2],
            bounds=[(0.1, 100), (0.1, 100)]
        )

        a_opt, b_opt = result.x

        # Create calibrated model wrapper
        class BetaCalibratedModel:
            def __init__(self, base_model, a, b):
                self.base_model = base_model
                self.a = a
                self.b = b

            def predict_proba(self, X):
                base_probs = predict_positive_proba(self.base_model, X)
                calibrated = beta.cdf(base_probs, self.a, self.b)
                return np.column_stack([1 - calibrated, calibrated])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return BetaCalibratedModel(model, a_opt, b_opt)

    def _scaling_calibration(self, model, base_scores: np.ndarray, y: pd.Series):
        """Simple scaling to match target rate."""

        target_rate = float(y.mean())
        current_rate = float(base_scores.mean()) if len(base_scores) > 0 else 0.0

        scaled_model, _ = self._scale_scores_to_target(model, target_rate, current_rate)
        return scaled_model

    def _scale_scores_to_target(self, base_model, target_rate: float, current_rate: float) -> Tuple[Any, float]:
        target = float(np.clip(target_rate, 0.0, 1.0))
        current = float(max(current_rate, 0.0))
        if current <= 0:
            scale = 1.0
        else:
            scale = target / current

        scale = float(np.clip(scale, 0.0, 50.0))
        scaled_model = self._build_scaled_model(base_model, scale)
        return scaled_model, scale

    def _build_scaled_model(self, base_model, scale: float):
        class ScaledModel:
            def __init__(self, model, scale_factor):
                self.base_model = model
                self.scale = scale_factor

            def predict_proba(self, X):
                X_filled = X.fillna(0) if hasattr(X, 'fillna') else X
                base_probs = predict_positive_proba(self.base_model, X_filled)
                scaled = np.clip(base_probs * self.scale, 0, 1)
                return np.column_stack([1 - scaled, scaled])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return ScaledModel(base_model, scale)

    def _lower_mean_adjustment(self, stage1_model, stage1_scores: np.ndarray,
                               recent_rate: float, stage1_rate: float):
        """
        Lower mean adjustment: Adjust scores to match recent rate
        while maintaining lower predictions conservative.
        """

        # Calculate adjustment factor
        if stage1_rate > 0:
            adjustment = recent_rate / stage1_rate
        else:
            adjustment = 1.0

        # Apply lower mean adjustment
        # Keep lower scores more conservative
        class LowerMeanAdjustedModel:
            def __init__(self, base_model, adjustment):
                self.base_model = base_model
                self.adjustment = adjustment

            def predict_proba(self, X):
                base_probs = predict_positive_proba(self.base_model, X)

                # Apply non-linear adjustment (more conservative for lower scores)
                adjusted = base_probs ** (1 / self.adjustment)
                adjusted = np.clip(adjusted, 0, 1)

                # Normalize to match target rate
                current_mean = adjusted.mean()
                if current_mean > 0:
                    final_adjustment = recent_rate / current_mean
                    adjusted = np.clip(adjusted * final_adjustment, 0, 1)

                return np.column_stack([1 - adjusted, adjusted])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return LowerMeanAdjustedModel(stage1_model, adjustment)

    def _upper_bound_adjustment(self, stage1_model, stage1_scores: np.ndarray,
                                recent_rate: float, stage1_rate: float):
        """
        Upper bound adjustment: Cap predictions at a certain percentile
        based on recent trends.
        """

        # Calculate upper bound based on recent rate
        upper_percentile = min(95, 100 * (1 + recent_rate))
        upper_bound = np.percentile(stage1_scores, upper_percentile)

        class UpperBoundAdjustedModel:
            def __init__(self, base_model, upper_bound, target_rate):
                self.base_model = base_model
                self.upper_bound = upper_bound
                self.target_rate = target_rate

            def predict_proba(self, X):
                base_probs = predict_positive_proba(self.base_model, X)

                # Cap at upper bound
                adjusted = np.minimum(base_probs, self.upper_bound)

                # Rescale to match target rate
                current_mean = adjusted.mean()
                if current_mean > 0:
                    scale = self.target_rate / current_mean
                    adjusted = np.clip(adjusted * scale, 0, 1)

                return np.column_stack([1 - adjusted, adjusted])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return UpperBoundAdjustedModel(stage1_model, upper_bound, recent_rate)

    def _weighted_adjustment(self, stage1_model, X_recent: pd.DataFrame, y_recent: pd.Series,
                            stage1_scores: np.ndarray):
        """
        Weighted adjustment: Combine Stage 1 predictions with recent period model
        using optimal weights.
        """

        # Train a simple model on recent data
        from sklearn.linear_model import LogisticRegression

        recent_model = LogisticRegression(max_iter=1000)
        recent_model.fit(X_recent, y_recent)
        recent_scores = predict_positive_proba(recent_model, X_recent)

        # Find optimal weight
        best_weight = 0.5
        best_ece = np.inf

        for w in np.arange(0, 1.1, 0.1):
            combined = w * stage1_scores + (1 - w) * recent_scores
            ece = self._calculate_ece(combined, y_recent)

            if ece < best_ece:
                best_ece = ece
                best_weight = w

        class WeightedModel:
            def __init__(self, stage1_model, recent_model, weight):
                self.stage1_model = stage1_model
                self.recent_model = recent_model
                self.weight = weight

            def predict_proba(self, X):
                stage1_probs = predict_positive_proba(self.stage1_model, X)
                recent_probs = predict_positive_proba(self.recent_model, X)

                combined = self.weight * stage1_probs + (1 - self.weight) * recent_probs
                return np.column_stack([1 - combined, combined])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return WeightedModel(stage1_model, recent_model, best_weight)

    def _shift_adjustment(self, stage1_model, recent_rate: float, stage1_rate: float):
        """Simple shift adjustment to match recent rate."""

        shift = recent_rate - stage1_rate

        class ShiftedModel:
            def __init__(self, base_model, shift):
                self.base_model = base_model
                self.shift = shift

            def predict_proba(self, X):
                base_probs = predict_positive_proba(self.base_model, X)
                shifted = np.clip(base_probs + self.shift, 0, 1)
                return np.column_stack([1 - shifted, shifted])

            def predict(self, X):
                return (predict_positive_proba(self, X) > 0.5).astype(int)

        return ShiftedModel(stage1_model, shift)

    def _confidence_interval_adjustment(self, stage1_model, stage1_scores: np.ndarray,
                                        y_recent: pd.Series, method: str) -> Tuple[Any, Dict[str, float]]:
        if len(y_recent) == 0:
            model, scale = self._scale_scores_to_target(stage1_model, stage1_scores.mean(), stage1_scores.mean())
            return model, {'target_rate': float(stage1_scores.mean()), 'adjustment_factor': float(scale)}

        recent_series = pd.Series(y_recent).astype(float)
        mean_rate = float(recent_series.mean())
        if len(recent_series) > 1:
            std = float(recent_series.std(ddof=1))
            se = std / np.sqrt(len(recent_series)) if len(recent_series) > 0 else 0.0
            confidence = float(getattr(self.config, 'stage2_confidence_level', 0.95) or 0.95)
            confidence = np.clip(confidence, 0.0, 0.9999)
            t_value = float(stats.t.ppf((1 + confidence) / 2, df=len(recent_series) - 1))
            lower = max(mean_rate - t_value * se, 0.0)
            upper = min(mean_rate + t_value * se, 1.0)
        else:
            se = 0.0
            t_value = 0.0
            lower = mean_rate
            upper = mean_rate

        if method == 'lower_ci':
            target_rate = lower
        elif method == 'upper_ci':
            target_rate = upper
        elif method in ('manual', 'expert'):
            manual_target = getattr(self.config, 'stage2_manual_target', None)
            if manual_target is None:
                manual_target = mean_rate
            target_rate = float(np.clip(manual_target, 0.0, 1.0))
        else:
            target_rate = mean_rate

        scaled_model, scale = self._scale_scores_to_target(stage1_model, target_rate, stage1_scores.mean())
        info = {
            'target_rate': float(np.clip(target_rate, 0.0, 1.0)),
            'recent_rate': mean_rate,
            'stage1_rate': float(stage1_scores.mean()),
            'lower_ci': float(lower),
            'upper_ci': float(upper),
            'std_error': float(se),
            't_value': float(t_value),
            'confidence_level': float(getattr(self.config, 'stage2_confidence_level', 0.95) or 0.95),
            'adjustment_factor': float(scale)
        }
        return scaled_model, info

    def _evaluate_calibration_quality(self, predictions: np.ndarray, actuals: pd.Series) -> Dict:
        """Evaluate calibration quality with multiple metrics."""

        metrics = {}

        # Expected Calibration Error (ECE)
        metrics['ece'] = self._calculate_ece(predictions, actuals)

        # Maximum Calibration Error (MCE)
        metrics['mce'] = self._calculate_mce(predictions, actuals)

        # Brier Score
        metrics['brier'] = brier_score_loss(actuals, predictions)

        # Log Loss
        try:
            metrics['log_loss'] = log_loss(actuals, predictions)
        except:
            metrics['log_loss'] = np.nan

        # Mean predicted vs actual
        metrics['mean_predicted'] = predictions.mean()
        metrics['mean_actual'] = actuals.mean()
        metrics['calibration_gap'] = abs(metrics['mean_predicted'] - metrics['mean_actual'])

        return metrics

    def _calculate_ece(self, predictions: np.ndarray, actuals: pd.Series, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_mce(self, predictions: np.ndarray, actuals: pd.Series, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return mce

    def evaluate_calibration(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Public method to evaluate calibration of a model."""

        # Handle NaN values
        X_filled = X.fillna(0)
        predictions = predict_positive_proba(model, X_filled)
        return self._evaluate_calibration_quality(predictions, y)

