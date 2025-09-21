"""
Risk Band Optimization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class RiskBandOptimizer:
    """
    Optimize risk bands for model scores with statistical testing
    """
    
    def __init__(self, config):
        """
        Initialize risk band optimizer
        
        Parameters:
        -----------
        config : Config
            Configuration object with risk band settings
        """
        self.config = config
        
    def optimize_bands(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_bands: Optional[int] = None,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize risk bands using specified method
        
        Parameters:
        -----------
        scores : np.ndarray
            Model scores/probabilities
        y_true : np.ndarray
            True labels
        n_bands : int, optional
            Number of bands (default from config)
        method : str, optional
            Method for creating bands ('quantile', 'equal_width', 'optimal')
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing bands, statistics, and test results
        """
        n_bands = n_bands or self.config.n_risk_bands
        method = method or self.config.risk_band_method
        
        # Create bands based on method
        if method == 'quantile':
            bands = self._create_quantile_bands(scores, n_bands)
        elif method == 'equal_width':
            bands = self._create_equal_width_bands(scores, n_bands)
        elif method == 'optimal':
            bands = self._create_optimal_bands(scores, y_true, n_bands)
        else:
            raise ValueError(f"Unknown band method: {method}")
        
        # Calculate band statistics
        band_stats = self._calculate_band_statistics(scores, y_true, bands)
        
        # Run statistical tests
        test_results = {}
        if 'binomial' in self.config.risk_band_tests:
            test_results['binomial'] = self._binomial_test(band_stats)
        if 'hosmer_lemeshow' in self.config.risk_band_tests:
            test_results['hosmer_lemeshow'] = self._hosmer_lemeshow_test(band_stats)
        if 'herfindahl' in self.config.risk_band_tests:
            test_results['herfindahl'] = self._calculate_herfindahl_index(band_stats)
        
        # Check monotonicity
        monotonicity = self._check_monotonicity(band_stats)
        
        # Calculate PSI for bands
        psi = self._calculate_band_psi(band_stats)
        
        # Assign business risk ratings if configured
        if self.config.business_risk_ratings:
            band_stats = self._assign_risk_ratings(band_stats)
        
        return {
            'bands': bands,
            'band_stats': band_stats,
            'test_results': test_results,
            'monotonicity': monotonicity,
            'psi': psi,
            'n_bands': n_bands,
            'method': method
        }
    
    def _create_quantile_bands(self, scores: np.ndarray, n_bands: int) -> np.ndarray:
        """Create bands using quantiles"""
        
        # Calculate quantiles
        quantiles = np.linspace(0, 100, n_bands + 1)
        bands = np.percentile(scores, quantiles)
        
        # Ensure unique bands
        bands = np.unique(bands)
        
        # If we have fewer unique bands than requested, use unique values
        if len(bands) < n_bands + 1:
            unique_scores = np.unique(scores)
            if len(unique_scores) <= n_bands:
                bands = np.concatenate([[unique_scores[0] - 0.001], 
                                       unique_scores, 
                                       [unique_scores[-1] + 0.001]])
            else:
                # Use quantiles of unique values
                idx = np.linspace(0, len(unique_scores) - 1, n_bands + 1).astype(int)
                bands = unique_scores[idx]
        
        return bands
    
    def _create_equal_width_bands(self, scores: np.ndarray, n_bands: int) -> np.ndarray:
        """Create bands with equal width"""
        
        min_score = scores.min()
        max_score = scores.max()
        
        bands = np.linspace(min_score, max_score, n_bands + 1)
        
        return bands
    
    def _create_optimal_bands(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_bands: int
    ) -> np.ndarray:
        """
        Create optimal bands that maximize discrimination
        Uses decision tree to find optimal splits
        """
        from sklearn.tree import DecisionTreeClassifier
        
        # Use decision tree to find optimal splits
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_bands,
            min_samples_leaf=max(30, len(scores) // (n_bands * 2)),
            random_state=self.config.random_state
        )
        
        tree.fit(scores.reshape(-1, 1), y_true)
        
        # Extract thresholds
        thresholds = []
        
        def extract_thresholds(node_id=0):
            if tree.tree_.feature[node_id] != -2:  # Not a leaf
                threshold = tree.tree_.threshold[node_id]
                thresholds.append(threshold)
                
                # Recurse on children
                extract_thresholds(tree.tree_.children_left[node_id])
                extract_thresholds(tree.tree_.children_right[node_id])
        
        extract_thresholds()
        
        # Create bands from thresholds
        thresholds = sorted(set(thresholds))
        bands = np.array([scores.min()] + thresholds + [scores.max()])
        
        # Ensure we have the right number of bands
        if len(bands) > n_bands + 1:
            # Keep most important splits
            idx = np.linspace(0, len(bands) - 1, n_bands + 1).astype(int)
            bands = bands[idx]
        elif len(bands) < n_bands + 1:
            # Fall back to quantile method
            bands = self._create_quantile_bands(scores, n_bands)
        
        return bands
    
    def _calculate_band_statistics(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        bands: np.ndarray
    ) -> pd.DataFrame:
        """Calculate statistics for each band"""
        
        band_stats = []
        
        for i in range(len(bands) - 1):
            if i == len(bands) - 2:
                # Last band includes upper bound
                mask = (scores >= bands[i]) & (scores <= bands[i + 1])
            else:
                mask = (scores >= bands[i]) & (scores < bands[i + 1])
            
            n_samples = mask.sum()
            
            if n_samples > 0:
                band_scores = scores[mask]
                band_labels = y_true[mask]
                
                stats = {
                    'band': i + 1,
                    'min_score': bands[i],
                    'max_score': bands[i + 1],
                    'mean_score': band_scores.mean(),
                    'n_samples': n_samples,
                    'n_events': band_labels.sum(),
                    'n_non_events': n_samples - band_labels.sum(),
                    'event_rate': band_labels.mean(),
                    'sample_pct': n_samples / len(scores),
                    'event_pct': band_labels.sum() / y_true.sum() if y_true.sum() > 0 else 0,
                    'odds': band_labels.sum() / (n_samples - band_labels.sum()) if (n_samples - band_labels.sum()) > 0 else np.inf,
                    'log_odds': np.log(band_labels.sum() / (n_samples - band_labels.sum())) if (n_samples - band_labels.sum()) > 0 else np.inf
                }
                
                band_stats.append(stats)
        
        return pd.DataFrame(band_stats)
    
    def _binomial_test(self, band_stats: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform binomial test for each band
        Tests if observed event rate differs from expected
        """
        
        test_results = []
        overall_event_rate = band_stats['n_events'].sum() / band_stats['n_samples'].sum()
        
        for _, band in band_stats.iterrows():
            # Binomial test
            result = stats.binomtest(
                k=int(band['n_events']),
                n=int(band['n_samples']),
                p=band['mean_score'],
                alternative='two-sided'
            )
            
            test_results.append({
                'band': band['band'],
                'p_value': result.pvalue,
                'observed_rate': band['event_rate'],
                'expected_rate': band['mean_score'],
                'significant': result.pvalue < 0.05
            })
        
        return {
            'band_tests': test_results,
            'n_significant': sum(1 for r in test_results if r['significant']),
            'overall_calibrated': sum(1 for r in test_results if r['significant']) <= len(test_results) * 0.1
        }
    
    def _hosmer_lemeshow_test(self, band_stats: pd.DataFrame) -> Dict[str, float]:
        """
        Perform Hosmer-Lemeshow test on bands
        """
        
        # Calculate test statistic
        hl_statistic = 0
        
        for _, band in band_stats.iterrows():
            expected_events = band['n_samples'] * band['mean_score']
            expected_non_events = band['n_samples'] * (1 - band['mean_score'])
            
            if expected_events > 0:
                hl_statistic += ((band['n_events'] - expected_events) ** 2) / expected_events
            if expected_non_events > 0:
                hl_statistic += ((band['n_non_events'] - expected_non_events) ** 2) / expected_non_events
        
        # Degrees of freedom
        df = len(band_stats) - 2
        
        # P-value
        p_value = 1 - stats.chi2.cdf(hl_statistic, df)
        
        return {
            'statistic': hl_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'calibrated': p_value > 0.05
        }
    
    def _calculate_herfindahl_index(self, band_stats: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Herfindahl-Hirschman Index for concentration
        Lower values indicate better distribution
        """
        
        # Calculate market shares (proportion of samples in each band)
        shares = band_stats['sample_pct'].values
        
        # HHI calculation
        hhi = np.sum(shares ** 2)
        
        # Normalized HHI (0 to 1)
        n_bands = len(band_stats)
        min_hhi = 1 / n_bands  # Perfect distribution
        max_hhi = 1  # All in one band
        
        normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi) if max_hhi > min_hhi else 0
        
        # Interpretation
        if normalized_hhi < 0.15:
            concentration = 'low'
        elif normalized_hhi < 0.25:
            concentration = 'moderate'
        else:
            concentration = 'high'
        
        return {
            'hhi': hhi,
            'normalized_hhi': normalized_hhi,
            'concentration': concentration,
            'well_distributed': normalized_hhi < 0.25
        }
    
    def _check_monotonicity(self, band_stats: pd.DataFrame) -> Dict[str, bool]:
        """Check if event rates are monotonic across bands"""
        
        event_rates = band_stats['event_rate'].values
        
        # Check for monotonic increase
        is_increasing = all(event_rates[i] <= event_rates[i + 1] 
                          for i in range(len(event_rates) - 1))
        
        # Check for monotonic decrease
        is_decreasing = all(event_rates[i] >= event_rates[i + 1] 
                          for i in range(len(event_rates) - 1))
        
        # Calculate Spearman correlation with band number
        band_numbers = band_stats['band'].values
        correlation, p_value = stats.spearmanr(band_numbers, event_rates)
        
        return {
            'is_monotonic': is_increasing or is_decreasing,
            'is_increasing': is_increasing,
            'is_decreasing': is_decreasing,
            'spearman_correlation': correlation,
            'correlation_p_value': p_value,
            'strong_monotonicity': abs(correlation) > 0.9
        }
    
    def _calculate_band_psi(self, band_stats: pd.DataFrame) -> float:
        """
        Calculate Population Stability Index for bands
        Compares actual vs expected distribution
        """
        
        # Use uniform distribution as reference
        expected_pct = 1 / len(band_stats)
        
        psi = 0
        for _, band in band_stats.iterrows():
            actual_pct = band['sample_pct']
            
            if actual_pct > 0 and expected_pct > 0:
                psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        
        return psi
    
    def _assign_risk_ratings(self, band_stats: pd.DataFrame) -> pd.DataFrame:
        """Assign business risk ratings to bands"""
        
        ratings = self.config.business_risk_ratings
        n_bands = len(band_stats)
        
        if len(ratings) >= n_bands:
            # Use first n_bands ratings
            band_stats['risk_rating'] = ratings[:n_bands]
        else:
            # Distribute ratings across bands
            rating_idx = np.linspace(0, len(ratings) - 1, n_bands).astype(int)
            band_stats['risk_rating'] = [ratings[i] for i in rating_idx]
        
        return band_stats
    
    def create_risk_score_mapping(
        self,
        bands: np.ndarray,
        band_stats: pd.DataFrame,
        scale_min: int = 300,
        scale_max: int = 850
    ) -> pd.DataFrame:
        """
        Create mapping from probability scores to scaled risk scores
        
        Parameters:
        -----------
        bands : np.ndarray
            Band boundaries
        band_stats : pd.DataFrame
            Band statistics
        scale_min : int
            Minimum scaled score
        scale_max : int
            Maximum scaled score
            
        Returns:
        --------
        pd.DataFrame
            Mapping table
        """
        
        # Create scaled scores for each band
        n_bands = len(band_stats)
        scaled_scores = np.linspace(scale_max, scale_min, n_bands).astype(int)
        
        mapping = band_stats.copy()
        mapping['scaled_score'] = scaled_scores
        mapping['score_range'] = [f"[{row['min_score']:.4f}, {row['max_score']:.4f})" 
                                  for _, row in mapping.iterrows()]
        
        # Add interpretation
        def get_risk_level(score):
            if score >= 750:
                return 'Very Low Risk'
            elif score >= 650:
                return 'Low Risk'
            elif score >= 550:
                return 'Medium Risk'
            elif score >= 450:
                return 'High Risk'
            else:
                return 'Very High Risk'
        
        mapping['risk_level'] = mapping['scaled_score'].apply(get_risk_level)
        
        return mapping[['band', 'score_range', 'scaled_score', 'risk_level', 
                       'event_rate', 'n_samples', 'sample_pct']]
    
    def apply_bands(self, scores: np.ndarray, bands: np.ndarray) -> np.ndarray:
        """
        Apply band assignment to scores
        
        Parameters:
        -----------
        scores : np.ndarray
            Scores to assign to bands
        bands : np.ndarray
            Band boundaries
            
        Returns:
        --------
        np.ndarray
            Band assignments (1-indexed)
        """
        
        band_assignments = np.zeros(len(scores), dtype=int)
        
        for i in range(len(bands) - 1):
            if i == len(bands) - 2:
                # Last band includes upper bound
                mask = (scores >= bands[i]) & (scores <= bands[i + 1])
            else:
                mask = (scores >= bands[i]) & (scores < bands[i + 1])
            
            band_assignments[mask] = i + 1
        
        return band_assignments