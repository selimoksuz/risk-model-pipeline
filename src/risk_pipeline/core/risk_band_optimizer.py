"""Risk band optimization module for creating data-driven risk segments"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


class RiskBandOptimizer:
    """Creates optimal risk bands based on data characteristics"""
    
    def create_risk_bands(self, scores: np.ndarray, y_true: np.ndarray, 
                         method: str = 'business', n_bands: int = 8) -> List[Dict]:
        """
        Create risk bands using different methods
        
        Parameters:
        -----------
        scores : array-like
            Model scores (probabilities 0-1)
        y_true : array-like
            True labels (0/1)
        method : str
            'business': Business-defined bands
            'quantile': Equal population bands
            'tree': Decision tree based splits
            'clustering': K-means clustering
            'monotonic': Monotonic risk progression
            'custom': User-defined thresholds
        n_bands : int
            Number of risk bands to create
        """
        
        if method == 'business':
            return self._create_business_bands()
        elif method == 'quantile':
            return self._create_quantile_bands(scores, y_true, n_bands)
        elif method == 'tree':
            return self._create_tree_bands(scores, y_true, n_bands)
        elif method == 'clustering':
            return self._create_cluster_bands(scores, y_true, n_bands)
        elif method == 'monotonic':
            return self._create_monotonic_bands(scores, y_true, n_bands)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_business_bands(self) -> List[Dict]:
        """Standard business risk bands based on industry standards"""
        
        # Based on credit rating agencies (S&P, Moody's, Fitch)
        return [
            {'name': 'AAA', 'min_score': 0.0000, 'max_score': 0.0010, 'target_pd': 0.0005, 'description': 'Prime'},
            {'name': 'AA+', 'min_score': 0.0010, 'max_score': 0.0025, 'target_pd': 0.00175, 'description': 'High grade'},
            {'name': 'AA', 'min_score': 0.0025, 'max_score': 0.0050, 'target_pd': 0.00375, 'description': 'High grade'},
            {'name': 'AA-', 'min_score': 0.0050, 'max_score': 0.0100, 'target_pd': 0.0075, 'description': 'High grade'},
            {'name': 'A+', 'min_score': 0.0100, 'max_score': 0.0175, 'target_pd': 0.01375, 'description': 'Upper medium'},
            {'name': 'A', 'min_score': 0.0175, 'max_score': 0.0300, 'target_pd': 0.02375, 'description': 'Upper medium'},
            {'name': 'A-', 'min_score': 0.0300, 'max_score': 0.0500, 'target_pd': 0.0400, 'description': 'Upper medium'},
            {'name': 'BBB+', 'min_score': 0.0500, 'max_score': 0.0750, 'target_pd': 0.0625, 'description': 'Lower medium'},
            {'name': 'BBB', 'min_score': 0.0750, 'max_score': 0.1000, 'target_pd': 0.0875, 'description': 'Lower medium'},
            {'name': 'BBB-', 'min_score': 0.1000, 'max_score': 0.1500, 'target_pd': 0.1250, 'description': 'Lower medium'},
            {'name': 'BB+', 'min_score': 0.1500, 'max_score': 0.2000, 'target_pd': 0.1750, 'description': 'Speculative'},
            {'name': 'BB', 'min_score': 0.2000, 'max_score': 0.2500, 'target_pd': 0.2250, 'description': 'Speculative'},
            {'name': 'BB-', 'min_score': 0.2500, 'max_score': 0.3000, 'target_pd': 0.2750, 'description': 'Speculative'},
            {'name': 'B+', 'min_score': 0.3000, 'max_score': 0.4000, 'target_pd': 0.3500, 'description': 'Highly spec'},
            {'name': 'B', 'min_score': 0.4000, 'max_score': 0.5000, 'target_pd': 0.4500, 'description': 'Highly spec'},
            {'name': 'B-', 'min_score': 0.5000, 'max_score': 0.6000, 'target_pd': 0.5500, 'description': 'Highly spec'},
            {'name': 'CCC', 'min_score': 0.6000, 'max_score': 0.7500, 'target_pd': 0.6750, 'description': 'Substantial risk'},
            {'name': 'CC', 'min_score': 0.7500, 'max_score': 0.9000, 'target_pd': 0.8250, 'description': 'Very high risk'},
            {'name': 'C', 'min_score': 0.9000, 'max_score': 1.0000, 'target_pd': 0.9500, 'description': 'Near default'},
        ]
    
    def _create_quantile_bands(self, scores: np.ndarray, y_true: np.ndarray, 
                               n_bands: int) -> List[Dict]:
        """Create bands with equal population (quantile-based)"""
        
        # Calculate quantiles
        quantiles = np.linspace(0, 100, n_bands + 1)
        thresholds = np.percentile(scores, quantiles)
        
        # Ensure unique thresholds
        thresholds = np.unique(thresholds)
        if len(thresholds) < n_bands + 1:
            print(f"Warning: Only {len(thresholds)-1} unique bands could be created")
        
        bands = []
        for i in range(len(thresholds) - 1):
            mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            if i == len(thresholds) - 2:  # Last band
                mask = (scores >= thresholds[i]) & (scores <= thresholds[i + 1])
            
            actual_pd = y_true[mask].mean() if mask.any() else 0
            
            bands.append({
                'name': f'Band_{i+1}',
                'min_score': thresholds[i],
                'max_score': thresholds[i + 1],
                'target_pd': actual_pd,
                'population_pct': mask.sum() / len(scores) * 100
            })
        
        return bands
    
    def _create_tree_bands(self, scores: np.ndarray, y_true: np.ndarray, 
                          n_bands: int) -> List[Dict]:
        """Use decision tree to find optimal split points"""
        
        # Fit a decision tree to find natural splits
        X = scores.reshape(-1, 1)
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_bands,
            min_samples_leaf=max(30, len(scores) // (n_bands * 10)),
            random_state=42
        )
        tree.fit(X, y_true)
        
        # Extract thresholds from tree
        thresholds = []
        
        def get_thresholds(node_id=0):
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]
            
            if left_child != right_child:  # Not a leaf
                threshold = tree.tree_.threshold[node_id]
                thresholds.append(threshold)
                get_thresholds(left_child)
                get_thresholds(right_child)
        
        get_thresholds()
        thresholds = sorted(set(thresholds))
        thresholds = [0] + thresholds + [1]
        
        # Create bands from thresholds
        bands = []
        for i in range(len(thresholds) - 1):
            mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            if i == len(thresholds) - 2:
                mask = (scores >= thresholds[i]) & (scores <= thresholds[i + 1])
            
            actual_pd = y_true[mask].mean() if mask.any() else 0
            
            bands.append({
                'name': f'Risk_{i+1}',
                'min_score': thresholds[i],
                'max_score': thresholds[i + 1],
                'target_pd': actual_pd,
                'count': mask.sum()
            })
        
        return bands
    
    def _create_cluster_bands(self, scores: np.ndarray, y_true: np.ndarray, 
                             n_bands: int) -> List[Dict]:
        """Use K-means clustering to create bands"""
        
        # Prepare data for clustering
        X = np.column_stack([
            scores,
            scores ** 2,  # Non-linear transformation
            np.log(scores + 0.001)  # Log transformation
        ])
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_bands, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Create bands from clusters
        bands = []
        cluster_scores = []
        
        for cluster_id in range(n_bands):
            mask = clusters == cluster_id
            cluster_score_mean = scores[mask].mean()
            cluster_scores.append((cluster_id, cluster_score_mean))
        
        # Sort clusters by mean score
        cluster_scores.sort(key=lambda x: x[1])
        
        for rank, (cluster_id, _) in enumerate(cluster_scores):
            mask = clusters == cluster_id
            min_score = scores[mask].min()
            max_score = scores[mask].max()
            actual_pd = y_true[mask].mean()
            
            bands.append({
                'name': f'Cluster_{rank+1}',
                'min_score': min_score,
                'max_score': max_score,
                'target_pd': actual_pd,
                'count': mask.sum()
            })
        
        # Adjust boundaries to avoid gaps
        bands = self._adjust_boundaries(bands)
        
        return bands
    
    def create_optimized_bands(self, scores: np.ndarray, y_true: np.ndarray,
                              optimization_metric: str = 'iv', n_bands: int = 10) -> List[Dict]:
        """
        Create optimized risk bands based on IV or Gini
        
        Parameters:
        -----------
        optimization_metric : str
            'iv': Optimize for maximum Information Value
            'gini': Optimize for maximum Gini coefficient
            'ks': Optimize for maximum KS statistic
        """
        
        if optimization_metric == 'iv':
            return self._optimize_bands_iv(scores, y_true, n_bands)
        elif optimization_metric == 'gini':
            return self._optimize_bands_gini(scores, y_true, n_bands)
        elif optimization_metric == 'ks':
            return self._optimize_bands_ks(scores, y_true, n_bands)
        else:
            return self._create_monotonic_bands(scores, y_true, n_bands)
    
    def _optimize_bands_iv(self, scores: np.ndarray, y_true: np.ndarray, 
                           n_bands: int) -> List[Dict]:
        """Optimize bands for maximum Information Value"""
        
        best_iv = -np.inf
        best_bands = None
        
        # Try different quantile combinations
        for n_quantiles in range(max(5, n_bands-3), min(20, n_bands+5)):
            try:
                # Create initial quantile splits
                quantiles = np.linspace(0, 100, n_quantiles + 1)
                thresholds = np.percentile(scores, quantiles)
                thresholds = np.unique(thresholds)
                
                # Calculate IV for this split
                total_iv = 0
                bands = []
                
                for i in range(len(thresholds) - 1):
                    mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
                    if i == len(thresholds) - 2:
                        mask = (scores >= thresholds[i]) & (scores <= thresholds[i + 1])
                    
                    if mask.sum() < 30:  # Minimum observations per band
                        continue
                    
                    # Calculate WOE and IV for this band
                    event_rate = y_true[mask].mean()
                    non_event_rate = 1 - event_rate
                    
                    total_events = y_true.sum()
                    total_non_events = len(y_true) - total_events
                    
                    pct_events = y_true[mask].sum() / total_events
                    pct_non_events = (mask.sum() - y_true[mask].sum()) / total_non_events
                    
                    if pct_events > 0 and pct_non_events > 0:
                        woe = np.log(pct_events / pct_non_events)
                        iv_contrib = (pct_events - pct_non_events) * woe
                        total_iv += iv_contrib
                        
                        bands.append({
                            'min_score': thresholds[i],
                            'max_score': thresholds[i + 1],
                            'woe': woe,
                            'iv_contribution': iv_contrib,
                            'event_rate': event_rate
                        })
                
                # Merge to get desired number of bands
                if len(bands) > n_bands:
                    bands = self._merge_bands_optimally(bands, n_bands, 'iv')
                
                if total_iv > best_iv and len(bands) >= n_bands // 2:
                    best_iv = total_iv
                    best_bands = bands
                    
            except Exception:
                continue
        
        # Format final bands
        if best_bands:
            return self._format_bands(best_bands, scores, y_true)
        else:
            return self._create_monotonic_bands(scores, y_true, n_bands)
    
    def _optimize_bands_gini(self, scores: np.ndarray, y_true: np.ndarray,
                             n_bands: int) -> List[Dict]:
        """Optimize bands for maximum Gini coefficient"""
        
        from sklearn.metrics import roc_auc_score
        
        best_gini = -np.inf
        best_bands = None
        
        # Try different splits
        for split_method in ['tree', 'quantile', 'monotonic']:
            if split_method == 'tree':
                candidate_bands = self._create_tree_bands(scores, y_true, n_bands)
            elif split_method == 'quantile':
                candidate_bands = self._create_quantile_bands(scores, y_true, n_bands)
            else:
                candidate_bands = self._create_monotonic_bands(scores, y_true, n_bands)
            
            # Calculate Gini for this banding
            band_predictions = np.zeros_like(scores)
            for band in candidate_bands:
                mask = (scores >= band['min_score']) & (scores < band['max_score'])
                if band['max_score'] == 1.0:
                    mask = (scores >= band['min_score']) & (scores <= band['max_score'])
                band_predictions[mask] = band['target_pd']
            
            try:
                auc = roc_auc_score(y_true, band_predictions)
                gini = 2 * auc - 1
                
                if gini > best_gini:
                    best_gini = gini
                    best_bands = candidate_bands
            except Exception:
                continue
        
        return best_bands if best_bands else self._create_monotonic_bands(scores, y_true, n_bands)
    
    def _optimize_bands_ks(self, scores: np.ndarray, y_true: np.ndarray,
                           n_bands: int) -> List[Dict]:
        """Optimize bands for maximum KS statistic"""
        
        # Calculate cumulative distributions
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_y = y_true[sorted_idx]
        
        # Find KS point
        cumsum_events = np.cumsum(sorted_y)
        cumsum_non_events = np.cumsum(1 - sorted_y)
        
        total_events = cumsum_events[-1]
        total_non_events = cumsum_non_events[-1]
        
        # Calculate KS at each point
        ks_values = np.abs(cumsum_events / total_events - 
                          cumsum_non_events / total_non_events)
        
        # Find optimal split points based on KS
        ks_peaks = self._find_peaks(ks_values, n_bands)
        thresholds = [0] + [sorted_scores[i] for i in ks_peaks] + [1]
        
        # Create bands
        bands = []
        for i in range(len(thresholds) - 1):
            mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            if i == len(thresholds) - 2:
                mask = (scores >= thresholds[i]) & (scores <= thresholds[i + 1])
            
            if mask.any():
                bands.append({
                    'name': self._get_rating_name(i, len(thresholds) - 1),
                    'min_score': thresholds[i],
                    'max_score': thresholds[i + 1],
                    'target_pd': y_true[mask].mean(),
                    'count': mask.sum()
                })
        
        return bands
    
    def _find_peaks(self, values: np.ndarray, n_peaks: int) -> List[int]:
        """Find peak indices in array"""
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(values, distance=len(values) // (n_peaks * 2))
        
        if len(peaks) < n_peaks:
            # Add more points evenly
            step = len(values) // n_peaks
            peaks = list(range(step, len(values), step))[:n_peaks-1]
        elif len(peaks) > n_peaks:
            # Keep strongest peaks
            peak_values = values[peaks]
            top_indices = np.argsort(peak_values)[-n_peaks+1:]
            peaks = sorted(peaks[top_indices])
        
        return peaks
    
    def _merge_bands_optimally(self, bands: List[Dict], target_n: int, 
                               metric: str = 'iv') -> List[Dict]:
        """Merge bands optimally to reach target number"""
        
        while len(bands) > target_n:
            # Find pair with minimum information loss
            min_loss = np.inf
            merge_idx = 0
            
            for i in range(len(bands) - 1):
                if metric == 'iv':
                    # Loss is reduction in IV when merging
                    loss = abs(bands[i].get('iv_contribution', 0) + 
                             bands[i+1].get('iv_contribution', 0))
                else:
                    # Loss is difference in event rates
                    loss = abs(bands[i].get('event_rate', 0) - 
                             bands[i+1].get('event_rate', 0))
                
                if loss < min_loss:
                    min_loss = loss
                    merge_idx = i
            
            # Merge bands
            bands[merge_idx]['max_score'] = bands[merge_idx + 1]['max_score']
            bands.pop(merge_idx + 1)
        
        return bands
    
    def _format_bands(self, bands: List[Dict], scores: np.ndarray, 
                     y_true: np.ndarray) -> List[Dict]:
        """Format bands with proper names and statistics"""
        
        formatted = []
        for i, band in enumerate(bands):
            mask = (scores >= band['min_score']) & (scores < band['max_score'])
            if band['max_score'] >= 1.0:
                mask = (scores >= band['min_score']) & (scores <= band['max_score'])
            
            formatted.append({
                'name': self._get_rating_name(i, len(bands)),
                'min_score': band['min_score'],
                'max_score': band['max_score'],
                'target_pd': y_true[mask].mean() if mask.any() else 0,
                'count': mask.sum(),
                'woe': band.get('woe', 0),
                'iv_contribution': band.get('iv_contribution', 0)
            })
        
        return formatted
    
    def _create_monotonic_bands(self, scores: np.ndarray, y_true: np.ndarray, 
                                n_bands: int) -> List[Dict]:
        """Create bands ensuring monotonic default rates"""
        
        # Sort scores and calculate cumulative default rate
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_y = y_true[sorted_idx]
        
        # Find points where cumulative default rate changes significantly
        window_size = max(30, len(scores) // (n_bands * 2))
        thresholds = [0]
        
        for i in range(window_size, len(scores) - window_size, window_size):
            left_dr = sorted_y[:i].mean()
            right_dr = sorted_y[i:].mean()
            
            # Significant change in default rate
            if abs(right_dr - left_dr) > 0.01:  # Lower threshold for sensitivity
                thresholds.append(sorted_scores[i])
        
        thresholds.append(1)
        thresholds = sorted(set(thresholds))
        
        # Ensure we have the right number of bands
        if len(thresholds) > n_bands + 1:
            # Merge closest thresholds
            while len(thresholds) > n_bands + 1:
                min_diff = float('inf')
                merge_idx = 0
                for i in range(1, len(thresholds) - 1):
                    diff = thresholds[i] - thresholds[i-1]
                    if diff < min_diff:
                        min_diff = diff
                        merge_idx = i
                thresholds.pop(merge_idx)
        
        # Create bands
        bands = []
        for i in range(len(thresholds) - 1):
            mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            if i == len(thresholds) - 2:
                mask = (scores >= thresholds[i]) & (scores <= thresholds[i + 1])
            
            actual_pd = y_true[mask].mean() if mask.any() else 0
            
            bands.append({
                'name': self._get_rating_name(i, len(thresholds) - 1),
                'min_score': thresholds[i],
                'max_score': thresholds[i + 1],
                'target_pd': actual_pd,
                'count': mask.sum()
            })
        
        return bands
    
    def _adjust_boundaries(self, bands: List[Dict]) -> List[Dict]:
        """Adjust band boundaries to avoid gaps and overlaps"""
        
        # Sort by min_score
        bands.sort(key=lambda x: x['min_score'])
        
        # Adjust boundaries
        for i in range(len(bands)):
            if i == 0:
                bands[i]['min_score'] = 0
            else:
                # Set min to previous max
                bands[i]['min_score'] = bands[i-1]['max_score']
            
            if i == len(bands) - 1:
                bands[i]['max_score'] = 1.0
        
        return bands
    
    def _get_rating_name(self, index: int, total: int) -> str:
        """Generate rating name based on position"""
        
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        
        if total <= len(ratings):
            return ratings[index]
        else:
            # Use numbered bands
            return f'Band_{index + 1}'
    
    def validate_bands(self, bands: List[Dict], scores: np.ndarray, 
                       y_true: np.ndarray) -> pd.DataFrame:
        """Validate and report on the quality of risk bands"""
        
        validation_results = []
        
        for band in bands:
            mask = (scores >= band['min_score']) & (scores < band['max_score'])
            if band['max_score'] == 1.0:
                mask = (scores >= band['min_score']) & (scores <= band['max_score'])
            
            n_obs = mask.sum()
            if n_obs > 0:
                actual_pd = y_true[mask].mean()
                avg_score = scores[mask].mean()
                std_score = scores[mask].std()
                
                # Gini within band
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true[mask])) > 1:
                    band_gini = 2 * roc_auc_score(y_true[mask], scores[mask]) - 1
                else:
                    band_gini = 0
                
                validation_results.append({
                    'band': band['name'],
                    'n_obs': n_obs,
                    'population_%': n_obs / len(scores) * 100,
                    'target_pd_%': band['target_pd'] * 100,
                    'actual_pd_%': actual_pd * 100,
                    'avg_score': avg_score,
                    'std_score': std_score,
                    'band_gini': band_gini,
                    'score_range': f"[{band['min_score']:.4f}, {band['max_score']:.4f}]"
                })
        
        df = pd.DataFrame(validation_results)
        
        # Check monotonicity
        df['monotonic'] = df['actual_pd_%'].diff() >= -0.1  # Allow small violations
        
        return df