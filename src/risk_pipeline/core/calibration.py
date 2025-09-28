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


    def calibrate_stage2(self, stage1_model, X_recent: pd.DataFrame, y_recent: Optional[pd.Series] = None,
                        method: str = 'lower_mean') -> Any:
        """Stage 2: Adjust for recent period trends."""

        print(f"    Applying Stage 2 calibration (method={method})...")

        X_recent_filled = X_recent.fillna(0) if hasattr(X_recent, 'fillna') else X_recent
        stage1_scores = predict_positive_proba(stage1_model, X_recent_filled)
        stage1_rate = float(stage1_scores.mean()) if stage1_scores.size else 0.0

        y_series: Optional[pd.Series] = None
        if y_recent is not None:
            y_series = pd.Series(y_recent).astype(float)
            y_series = y_series.dropna()
            if y_series.empty:
                y_series = None

        has_recent_actuals = y_series is not None
        configured_target = getattr(self.config, 'stage2_target_rate', None)
        manual_target = getattr(self.config, 'stage2_manual_target', None)
        fallback_target = configured_target if configured_target is not None else manual_target

        if has_recent_actuals:
            recent_rate = float(y_series.mean())
            n_recent = int(len(y_series))
        else:
            n_recent = 0
            if fallback_target is not None:
                recent_rate = float(np.clip(fallback_target, 0.0, 1.0))
            else:
                recent_rate = stage1_rate

        if has_recent_actuals:
            print(f"      Recent default rate ({n_recent} obs): {recent_rate:.2%}")
        elif fallback_target is not None:
            print(f"      No observed recent targets; using configured target rate: {recent_rate:.2%}")
        else:
            print("      No observed recent targets; defaulting to Stage 1 mean rate.")

        print(f"      Stage 1 predicted rate: {stage1_rate:.2%}")

        effective_method = method
        stage2_info = {
            'method_requested': method,
            'method': effective_method,
            'recent_rate': recent_rate if has_recent_actuals else np.nan,
            'stage1_rate': stage1_rate,
            'target_source': 'observed' if has_recent_actuals else ('configured' if fallback_target is not None else 'stage1'),
            'n_recent_observations': n_recent,
        }

        if not has_recent_actuals and fallback_target is not None:
            stage2_info['configured_target_rate'] = recent_rate

        if effective_method == 'weighted' and not has_recent_actuals:
            warnings.warn(
                "Stage 2 'weighted' method requires observed targets; falling back to 'shift'.",
                RuntimeWarning,
            )
            effective_method = 'shift'
            stage2_info['method'] = effective_method

        if effective_method == 'lower_mean':
            adjusted_model = self._lower_mean_adjustment(
                stage1_model, stage1_scores, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['adjustment_factor'] = float(getattr(adjusted_model, 'adjustment', 1.0))
        elif effective_method == 'upper_bound':
            adjusted_model = self._upper_bound_adjustment(
                stage1_model, stage1_scores, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = float(getattr(adjusted_model, 'target_rate', recent_rate))
            stage2_info['upper_bound'] = float(getattr(adjusted_model, 'upper_bound', stage1_rate))
        elif effective_method == 'weighted':
            adjusted_model = self._weighted_adjustment(
                stage1_model, X_recent, y_series, stage1_scores
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['weight'] = float(getattr(adjusted_model, 'weight', 0.5))
        elif effective_method == 'shift':
            adjusted_model = self._shift_adjustment(
                stage1_model, recent_rate, stage1_rate
            )
            stage2_info['target_rate'] = recent_rate
            stage2_info['adjustment_factor'] = float(getattr(adjusted_model, 'shift', 0.0))
        elif effective_method in ['lower_ci', 'upper_ci', 'mean_ci', 'manual', 'expert']:
            adjusted_model, ci_info = self._confidence_interval_adjustment(
                stage1_model, stage1_scores, y_series, effective_method
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
        metrics = self._evaluate_calibration_quality(adjusted_scores, y_series)
        achieved_rate = float(adjusted_scores.mean())

        print(f"      ECE: {metrics['ece']:.4f}")
        print(f"      MCE: {metrics['mce']:.4f}")
        print(f"      Adjusted mean prediction: {achieved_rate:.2%}")

        stage2_info['achieved_rate'] = achieved_rate
        metrics['method'] = effective_method
        metrics['recent_rate'] = float(stage2_info.get('recent_rate', np.nan))
        metrics['stage1_rate'] = stage1_rate
        metrics['target_rate'] = float(stage2_info.get('target_rate', recent_rate))
        metrics['achieved_rate'] = achieved_rate
        metrics['target_source'] = stage2_info.get('target_source')
        metrics['n_recent_observations'] = stage2_info.get('n_recent_observations', 0)

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
                                        y_recent: Optional[pd.Series], method: str) -> Tuple[Any, Dict[str, float]]:
        stage1_mean = float(stage1_scores.mean()) if stage1_scores.size else 0.0
        confidence = float(getattr(self.config, 'stage2_confidence_level', 0.95) or 0.95)
        confidence = float(np.clip(confidence, 0.0, 0.9999))

        if y_recent is None or len(y_recent) == 0:
            lower_bound = float(np.clip(stage1_mean * getattr(self.config, 'stage2_lower_bound', 0.8), 0.0, 1.0))
            upper_bound = float(np.clip(stage1_mean * getattr(self.config, 'stage2_upper_bound', 1.2), 0.0, 1.0))
            configured_target = getattr(self.config, 'stage2_target_rate', None)
            manual_target = getattr(self.config, 'stage2_manual_target', None)
            fallback_target = configured_target if configured_target is not None else manual_target

            if method == 'lower_ci':
                target_rate = lower_bound
                target_source = 'configured_lower_bound'
            elif method == 'upper_ci':
                target_rate = upper_bound
                target_source = 'configured_upper_bound'
            elif fallback_target is not None:
                target_rate = float(np.clip(fallback_target, 0.0, 1.0))
                target_source = 'configured'
            else:
                target_rate = stage1_mean
                target_source = 'stage1'

            scaled_model, scale = self._scale_scores_to_target(stage1_model, target_rate, stage1_mean)
            info = {
                'target_rate': float(np.clip(target_rate, 0.0, 1.0)),
                'recent_rate': np.nan,
                'stage1_rate': stage1_mean,
                'lower_ci': lower_bound,
                'upper_ci': upper_bound,
                'std_error': np.nan,
                't_value': np.nan,
                'confidence_level': confidence,
                'adjustment_factor': float(scale),
                'target_source': target_source,
            }
            return scaled_model, info

        recent_series = pd.Series(y_recent).astype(float).dropna()
        if recent_series.empty:
            return self._confidence_interval_adjustment(stage1_model, stage1_scores, None, method)

        mean_rate = float(recent_series.mean())
        if len(recent_series) > 1:
            std = float(recent_series.std(ddof=1))
            se = std / np.sqrt(len(recent_series)) if len(recent_series) > 0 else 0.0
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
            target_source = 'observed_ci_lower'
        elif method == 'upper_ci':
            target_rate = upper
            target_source = 'observed_ci_upper'
        elif method in ('manual', 'expert'):
            manual_target = getattr(self.config, 'stage2_manual_target', None)
            if manual_target is None:
                manual_target = mean_rate
            target_rate = float(np.clip(manual_target, 0.0, 1.0))
            target_source = 'configured'
        else:
            target_rate = mean_rate
            target_source = 'observed_ci_mean'

        scaled_model, scale = self._scale_scores_to_target(stage1_model, target_rate, stage1_mean)
        info = {
            'target_rate': float(np.clip(target_rate, 0.0, 1.0)),
            'recent_rate': mean_rate,
            'stage1_rate': stage1_mean,
            'lower_ci': float(lower),
            'upper_ci': float(upper),
            'std_error': float(se),
            't_value': float(t_value),
            'confidence_level': confidence,
            'adjustment_factor': float(scale),
            'target_source': target_source,
        }
        return scaled_model, info

    def _evaluate_calibration_quality(self, predictions: np.ndarray, actuals: Optional[pd.Series]) -> Dict:
        """Evaluate calibration quality with multiple metrics."""

        preds = np.asarray(predictions, dtype=float)
        metrics: Dict[str, Any] = {
            'mean_predicted': float(np.mean(preds)) if preds.size else 0.0
        }

        if actuals is None:
            metrics.update({
                'ece': np.nan,
                'mce': np.nan,
                'brier': np.nan,
                'log_loss': np.nan,
                'mean_actual': np.nan,
                'calibration_gap': np.nan,
            })
            return metrics

        actual_series = pd.Series(actuals).astype(float).dropna()
        if actual_series.empty:
            metrics.update({
                'ece': np.nan,
                'mce': np.nan,
                'brier': np.nan,
                'log_loss': np.nan,
                'mean_actual': np.nan,
                'calibration_gap': np.nan,
            })
            return metrics

        min_len = min(len(actual_series), preds.shape[0])
        if min_len == 0:
            metrics.update({
                'ece': np.nan,
                'mce': np.nan,
                'brier': np.nan,
                'log_loss': np.nan,
                'mean_actual': np.nan,
                'calibration_gap': np.nan,
            })
            return metrics

        actual_aligned = actual_series.iloc[:min_len]
        preds_aligned = preds[:min_len]
        metrics['mean_predicted'] = float(np.mean(preds_aligned))

        metrics['ece'] = self._calculate_ece(preds_aligned, actual_aligned)
        metrics['mce'] = self._calculate_mce(preds_aligned, actual_aligned)
        metrics['brier'] = brier_score_loss(actual_aligned, preds_aligned)

        try:
            metrics['log_loss'] = log_loss(actual_aligned, preds_aligned)
        except Exception:
            metrics['log_loss'] = np.nan

        metrics['mean_actual'] = float(actual_aligned.mean())
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


    def evaluate_calibration(self, model, X: pd.DataFrame, y: Optional[pd.Series]) -> Dict:
        """Public method to evaluate calibration of a model."""

        X_filled = X.fillna(0)
        predictions = predict_positive_proba(model, X_filled)
        return self._evaluate_calibration_quality(predictions, y)

        # Handle NaN values
        X_filled = X.fillna(0)
        predictions = predict_positive_proba(model, X_filled)
        return self._evaluate_calibration_quality(predictions, y)

