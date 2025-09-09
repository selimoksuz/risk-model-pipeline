"""Calibration analysis with risk bands and binomial testing"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class CalibrationAnalyzer:
    """Performs calibration analysis with risk bands and statistical tests"""
    
    def __init__(self):
        # Standard risk rating bands (can be customized)
        self.default_risk_bands = [
            {'name': 'AAA', 'min_score': 0.00, 'max_score': 0.01, 'target_pd': 0.005},
            {'name': 'AA', 'min_score': 0.01, 'max_score': 0.02, 'target_pd': 0.015},
            {'name': 'A', 'min_score': 0.02, 'max_score': 0.05, 'target_pd': 0.035},
            {'name': 'BBB', 'min_score': 0.05, 'max_score': 0.10, 'target_pd': 0.075},
            {'name': 'BB', 'min_score': 0.10, 'max_score': 0.20, 'target_pd': 0.15},
            {'name': 'B', 'min_score': 0.20, 'max_score': 0.35, 'target_pd': 0.275},
            {'name': 'CCC', 'min_score': 0.35, 'max_score': 0.50, 'target_pd': 0.425},
            {'name': 'CC', 'min_score': 0.50, 'max_score': 0.75, 'target_pd': 0.625},
            {'name': 'C', 'min_score': 0.75, 'max_score': 1.00, 'target_pd': 0.875}
        ]
    
    def analyze_calibration(self, y_true: np.ndarray, y_pred: np.ndarray,
                           risk_bands: Optional[List[Dict]] = None,
                           use_deciles: bool = False) -> Dict:
        """
        Perform comprehensive calibration analysis
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels (0/1)
        y_pred : array-like
            Predicted probabilities (0-1)
        risk_bands : list of dict, optional
            Custom risk bands with min_score, max_score, target_pd
        use_deciles : bool
            If True, use 10 equal population deciles instead of risk bands
        """
        
        results = {}
        
        # Choose segmentation method
        if use_deciles:
            segments_df = self._analyze_by_deciles(y_true, y_pred)
        else:
            bands = risk_bands or self.default_risk_bands
            segments_df = self._analyze_by_risk_bands(y_true, y_pred, bands)
        
        results['segments'] = segments_df
        
        # Overall calibration metrics
        results['overall_metrics'] = self._calculate_overall_metrics(y_true, y_pred)
        
        # Hosmer-Lemeshow test
        results['hosmer_lemeshow'] = self._hosmer_lemeshow_test(segments_df)
        
        # Calibration plot data
        results['calibration_curve'] = self._get_calibration_curve(y_true, y_pred)
        
        return results
    
    def _analyze_by_risk_bands(self, y_true: np.ndarray, y_pred: np.ndarray,
                               risk_bands: List[Dict]) -> pd.DataFrame:
        """Analyze calibration by predefined risk bands"""
        
        segments = []
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        for band in risk_bands:
            # Get observations in this band
            mask = (y_pred >= band['min_score']) & (y_pred < band['max_score'])
            
            if not mask.any():
                continue
            
            band_y_true = y_true[mask]
            band_y_pred = y_pred[mask]
            
            # Statistics
            n_obs = len(band_y_true)
            n_defaults = band_y_true.sum()
            observed_pd = n_defaults / n_obs if n_obs > 0 else 0
            predicted_pd = band_y_pred.mean()
            target_pd = band.get('target_pd', predicted_pd)
            
            # Binomial test against target PD
            if n_obs > 0:
                # Test if observed defaults are consistent with target PD
                binom_test_result = stats.binomtest(
                    n_defaults, 
                    n_obs, 
                    target_pd,
                    alternative='two-sided'
                )
                p_value = binom_test_result.pvalue
                
                # 95% confidence interval for observed PD
                ci_result = stats.binomtest(n_defaults, n_obs, observed_pd).proportion_ci(
                    confidence_level=0.95,
                    method='wilson'  # Wilson score interval - better for small samples
                )
                ci_lower, ci_upper = ci_result.low, ci_result.high
                
                # Check if target PD is within CI
                within_ci = ci_lower <= target_pd <= ci_upper
            else:
                p_value = 1.0
                ci_lower, ci_upper = 0.0, 0.0
                within_ci = True
            
            segments.append({
                'rating': band['name'],
                'score_range': f"[{band['min_score']:.3f}, {band['max_score']:.3f})",
                'n_observations': n_obs,
                'n_defaults': n_defaults,
                'target_pd_%': target_pd * 100,
                'predicted_pd_%': predicted_pd * 100,
                'observed_pd_%': observed_pd * 100,
                'difference_%': (observed_pd - target_pd) * 100,
                'ci_lower_%': ci_lower * 100,
                'ci_upper_%': ci_upper * 100,
                'target_in_ci': 'Yes' if within_ci else 'No',
                'binomial_p_value': p_value,
                'reject_h0': 'No' if p_value > 0.05 else 'Yes',
                'calibration_status': self._get_calibration_status(p_value, observed_pd, target_pd)
            })
        
        return pd.DataFrame(segments)
    
    def _analyze_by_deciles(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Analyze calibration by population deciles"""
        
        segments = []
        n_bins = 10
        
        # Sort by predicted probability
        sorted_idx = np.argsort(y_pred)
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        # Create equal population bins
        bin_size = len(y_pred) // n_bins
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
            
            bin_y_true = y_true_sorted[start_idx:end_idx]
            bin_y_pred = y_pred_sorted[start_idx:end_idx]
            
            # Statistics
            n_obs = len(bin_y_true)
            n_defaults = bin_y_true.sum()
            observed_pd = n_defaults / n_obs if n_obs > 0 else 0
            predicted_pd = bin_y_pred.mean()
            
            # Binomial test against predicted PD
            if n_obs > 0:
                binom_test_result = stats.binomtest(
                    n_defaults,
                    n_obs,
                    predicted_pd,
                    alternative='two-sided'
                )
                p_value = binom_test_result.pvalue
                
                # 95% confidence interval
                if n_defaults > 0:
                    ci_result = stats.binomtest(n_defaults, n_obs, observed_pd).proportion_ci(
                        confidence_level=0.95,
                        method='wilson'
                    )
                    ci_lower, ci_upper = ci_result.low, ci_result.high
                else:
                    # Clopper-Pearson for zero events
                    ci_lower = 0.0
                    ci_upper = 1 - (0.05/2)**(1/n_obs) if n_obs > 0 else 0.0
                
                within_ci = ci_lower <= predicted_pd <= ci_upper
            else:
                p_value = 1.0
                ci_lower, ci_upper = 0.0, 0.0
                within_ci = True
            
            segments.append({
                'decile': i + 1,
                'score_range': f"[{bin_y_pred.min():.3f}, {bin_y_pred.max():.3f}]",
                'n_observations': n_obs,
                'n_defaults': n_defaults,
                'predicted_pd_%': predicted_pd * 100,
                'observed_pd_%': observed_pd * 100,
                'difference_%': (observed_pd - predicted_pd) * 100,
                'ci_lower_%': ci_lower * 100,
                'ci_upper_%': ci_upper * 100,
                'predicted_in_ci': 'Yes' if within_ci else 'No',
                'binomial_p_value': p_value,
                'reject_h0': 'No' if p_value > 0.05 else 'Yes',
                'calibration_status': self._get_calibration_status(p_value, observed_pd, predicted_pd)
            })
        
        return pd.DataFrame(segments)
    
    def _get_calibration_status(self, p_value: float, observed_pd: float, expected_pd: float) -> str:
        """Determine calibration status based on statistical test and practical significance"""
        
        # Statistical significance
        if p_value < 0.05:
            # Check direction and magnitude
            relative_diff = abs(observed_pd - expected_pd) / (expected_pd + 0.0001)
            if relative_diff > 0.5:
                return "Poor - Significant deviation"
            else:
                return "Moderate - Statistically significant but small deviation"
        else:
            # Not statistically significant
            relative_diff = abs(observed_pd - expected_pd) / (expected_pd + 0.0001)
            if relative_diff > 0.3:
                return "Fair - Large but not significant (small sample?)"
            else:
                return "Good - Well calibrated"
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate overall calibration metrics"""
        
        # Brier score
        brier_score = brier_score_loss(y_true, y_pred)
        
        # Expected Calibration Error (ECE)
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(fraction_pos - mean_pred))
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_pred))
        
        # Spiegelhalter test (z-test for calibration)
        n = len(y_true)
        z_score = (brier_score - np.mean(y_pred * (1 - y_pred))) / \
                  (np.sqrt(np.sum((y_pred * (1 - y_pred))**2) / n))
        spiegelhalter_p = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'brier_score': brier_score,
            'ece': ece,
            'mce': mce,
            'spiegelhalter_z': z_score,
            'spiegelhalter_p': spiegelhalter_p,
            'calibration_assessment': self._assess_overall_calibration(brier_score, ece, mce)
        }
    
    def _hosmer_lemeshow_test(self, segments_df: pd.DataFrame) -> Dict:
        """Perform Hosmer-Lemeshow goodness-of-fit test"""
        
        # Calculate chi-square statistic
        chi2_stat = 0
        df = len(segments_df) - 2  # degrees of freedom
        
        for _, row in segments_df.iterrows():
            n_obs = row['n_observations']
            n_defaults = row['n_defaults']
            
            # Use predicted PD for expected defaults
            if 'predicted_pd_%' in row:
                expected_pd = row['predicted_pd_%'] / 100
            else:
                expected_pd = row['target_pd_%'] / 100
            
            expected_defaults = n_obs * expected_pd
            
            if expected_defaults > 0:
                # Chi-square contribution
                chi2_stat += (n_defaults - expected_defaults)**2 / expected_defaults
            
            if n_obs - expected_defaults > 0:
                chi2_stat += ((n_obs - n_defaults) - (n_obs - expected_defaults))**2 / \
                            (n_obs - expected_defaults)
        
        # Calculate p-value
        if df > 0:
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        else:
            p_value = 1.0
        
        return {
            'statistic': chi2_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'reject_h0': 'No' if p_value > 0.05 else 'Yes',
            'interpretation': 'Model is well calibrated' if p_value > 0.05 
                           else 'Model calibration is questionable'
        }
    
    def _get_calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Get calibration curve data for plotting"""
        
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
        
        return {
            'fraction_positive': fraction_pos.tolist(),
            'mean_predicted': mean_pred.tolist(),
            'perfect_calibration': [0, 1]  # For reference line
        }
    
    def _assess_overall_calibration(self, brier: float, ece: float, mce: float) -> str:
        """Provide overall calibration assessment"""
        
        if brier < 0.1 and ece < 0.05 and mce < 0.1:
            return "Excellent - Model is very well calibrated"
        elif brier < 0.15 and ece < 0.1 and mce < 0.15:
            return "Good - Model shows good calibration"
        elif brier < 0.2 and ece < 0.15 and mce < 0.2:
            return "Fair - Model calibration is acceptable"
        else:
            return "Poor - Model needs calibration improvement"