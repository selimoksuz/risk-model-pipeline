"""
WOE Optimization Module - IV/Gini maximization with monotonicity
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')


class WOEOptimizer:
    """
    Optimizes WOE binning to maximize IV or Gini while maintaining interpretability
    """
    
    def __init__(self,
                 config: Optional[Any] = None,
                 optimization_metric: str = 'iv',
                 max_bins: int = 10,
                 min_bins: int = 2,
                 min_bin_size: float = 0.05,
                 monotonic: bool = True,
                 merge_insignificant: bool = True):
        """
        Initialize WOE Optimizer

        Parameters:
        -----------
        config : Config, optional
            Pipeline configuration object
        optimization_metric : str
            Metric to optimize ('iv' or 'gini')
        max_bins : int
            Maximum number of bins
        min_bins : int
            Minimum number of bins
        min_bin_size : float
            Minimum size of each bin (as fraction)
        monotonic : bool
            Enforce monotonic WOE for numeric variables
        merge_insignificant : bool
            Merge statistically insignificant bins for categorical
        """
        if config is not None and not isinstance(config, str):
            self.optimization_metric = getattr(config, 'woe_optimization_metric', optimization_metric)
            self.max_bins = getattr(config, 'woe_max_bins', max_bins)
            self.min_bins = getattr(config, 'woe_min_bins', min_bins)
            self.min_bin_size = getattr(config, 'woe_min_bin_size', min_bin_size)
            self.monotonic = getattr(config, 'woe_monotonic_numeric', monotonic)
            self.merge_insignificant = getattr(config, 'woe_merge_insignificant', merge_insignificant)
        else:
            self.optimization_metric = config if isinstance(config, str) else optimization_metric
            self.max_bins = max_bins
            self.min_bins = min_bins
            self.min_bin_size = min_bin_size
            self.monotonic = monotonic
            self.merge_insignificant = merge_insignificant

        self.woe_mapping_ = {}
        self.iv_values_ = {}
        self.gini_values_ = {}
        
    def optimize_numeric(self, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """
        Optimize WOE binning for numeric variable
        
        Approach:
        1. Start with max_bins using decision tree
        2. Calculate WOE and IV/Gini
        3. If monotonic, merge bins to maintain monotonicity
        4. Continue merging while IV/Gini improves or stays stable
        """
        
        variable_name = X.name
        
        # Remove missing values for binning
        mask = X.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            return self._create_single_bin(variable_name)
        
        # Find optimal number of bins
        best_bins = None
        best_metric = -np.inf
        best_woe_df = None
        
        for n_bins in range(min(self.max_bins, len(X_clean.unique())), self.min_bins - 1, -1):
            # Create initial bins using decision tree
            bins = self._create_initial_bins_numeric(X_clean, y_clean, n_bins)
            
            if bins is None or len(bins) < 2:
                continue
            
            # Calculate WOE for these bins
            woe_df = self._calculate_woe_numeric(X_clean, y_clean, bins)
            
            if woe_df is None or len(woe_df) < self.min_bins:
                continue
            
            # Check monotonicity if required
            if self.monotonic:
                woe_df = self._enforce_monotonicity(woe_df, X_clean, y_clean)
            
            # Calculate metric
            if self.optimization_metric == 'iv':
                metric = woe_df['iv'].sum()
            else:  # gini
                X_woe = self._apply_woe(X_clean, woe_df)
                metric = self._calculate_gini(y_clean, X_woe)
            
            # Check if this is better
            if metric > best_metric:
                best_metric = metric
                best_bins = bins
                best_woe_df = woe_df.copy()
        
        if best_woe_df is None:
            return self._create_single_bin(variable_name)

        # Handle missing values
        if X.isna().any():
            missing_woe = self._calculate_missing_woe(X, y)
            best_woe_df = pd.concat([best_woe_df, missing_woe], ignore_index=True)

        return best_woe_df.reset_index(drop=True)
    
    def optimize_categorical(self, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """
        Optimize WOE binning for categorical variable
        
        Approach:
        1. Calculate WOE for each category
        2. Merge categories with similar WOE values
        3. Use chi-square test to check if merge is valid
        4. Continue merging while maintaining statistical significance
        """
        
        variable_name = X.name
        
        # Calculate initial WOE for each category
        woe_df = self._calculate_woe_categorical(X, y)
        
        if woe_df is None or len(woe_df) <= self.min_bins:
            return woe_df
        
        # Merge insignificant categories if requested
        if self.merge_insignificant:
            woe_df = self._merge_insignificant_categories(woe_df, X, y)
        
        # Ensure we don't exceed max_bins
        while len(woe_df) > self.max_bins:
            woe_df = self._merge_closest_woe(woe_df)
        
        return woe_df
    
    def _create_initial_bins_numeric(self, X: pd.Series, y: pd.Series, n_bins: int) -> Optional[np.ndarray]:
        """Create initial bins using decision tree"""
        try:
            # Use decision tree to find optimal splits
            dt = DecisionTreeClassifier(
                max_leaf_nodes=n_bins,
                min_samples_leaf=int(len(X) * self.min_bin_size)
            )
            dt.fit(X.values.reshape(-1, 1), y)
            
            # Get split points
            tree = dt.tree_
            splits = []
            
            def get_splits(node=0):
                if tree.feature[node] != -2:  # Not a leaf
                    splits.append(tree.threshold[node])
                    get_splits(tree.children_left[node])
                    get_splits(tree.children_right[node])
            
            get_splits()
            
            if not splits:
                # Fall back to quantile binning
                return self._create_quantile_bins(X, n_bins)
            
            splits = sorted(splits)
            bins = [-np.inf] + splits + [np.inf]
            
            return np.array(bins)
            
        except Exception:
            return self._create_quantile_bins(X, n_bins)
    
    def _create_quantile_bins(self, X: pd.Series, n_bins: int) -> np.ndarray:
        """Create bins using quantiles"""
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = X.quantile(quantiles).unique()
        
        if len(bins) < 2:
            return None
        
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        return bins
    
    def _calculate_woe_numeric(self, X: pd.Series, y: pd.Series, bins: np.ndarray) -> pd.DataFrame:
        """Calculate WOE for numeric bins"""
        
        # Create bins
        X_binned = pd.cut(X, bins, include_lowest=True)
        
        # Calculate WOE
        crosstab = pd.crosstab(X_binned, y)
        
        if crosstab.shape[1] < 2:
            return None
        
        woe_df = self._compute_woe_iv(crosstab, bins=bins)
        woe_df['variable'] = X.name
        
        return woe_df
    
    def _calculate_woe_categorical(self, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Calculate WOE for categorical variable"""
        
        # Create crosstab
        crosstab = pd.crosstab(X, y)
        
        if crosstab.shape[1] < 2:
            return None
        
        woe_df = self._compute_woe_iv(crosstab, categorical=True)
        woe_df['variable'] = X.name
        
        return woe_df
    
    def _compute_woe_iv(self, crosstab: pd.DataFrame, bins=None, categorical=False) -> pd.DataFrame:
        """Compute WOE and IV from crosstab"""
        
        # Calculate distributions
        n_events = crosstab[1].values
        n_non_events = crosstab[0].values
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        
        # Calculate event and non-event rates
        event_rate = (n_events + eps) / (n_events.sum() + eps)
        non_event_rate = (n_non_events + eps) / (n_non_events.sum() + eps)
        
        # Calculate WOE
        woe = np.log(event_rate / non_event_rate)
        
        # Calculate IV
        iv = (event_rate - non_event_rate) * woe
        
        # Create DataFrame
        if categorical:
            woe_df = pd.DataFrame({
                'category': crosstab.index.astype(str),
                'n_obs': n_events + n_non_events,
                'n_events': n_events,
                'n_non_events': n_non_events,
                'event_rate': n_events / (n_events + n_non_events + eps),
                'woe': woe,
                'iv': iv
            })
        else:
            woe_df = pd.DataFrame({
                'min_value': bins[:-1] if bins is not None else crosstab.index,
                'max_value': bins[1:] if bins is not None else crosstab.index,
                'n_obs': n_events + n_non_events,
                'n_events': n_events,
                'n_non_events': n_non_events,
                'event_rate': n_events / (n_events + n_non_events + eps),
                'woe': woe,
                'iv': iv
            })
        
        return woe_df
    
    def _enforce_monotonicity(self, woe_df: pd.DataFrame, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Enforce monotonic WOE by merging bins"""
        
        woe_values = woe_df['woe'].values
        
        # Check if already monotonic
        if self._is_monotonic(woe_values):
            return woe_df
        
        # Merge bins to achieve monotonicity
        while not self._is_monotonic(woe_values) and len(woe_df) > self.min_bins:
            # Find the pair of adjacent bins with smallest WOE difference
            woe_diffs = np.abs(np.diff(woe_values))
            min_diff_idx = np.argmin(woe_diffs)
            
            # Merge these bins
            woe_df = self._merge_bins(woe_df, min_diff_idx, min_diff_idx + 1)
            woe_values = woe_df['woe'].values
        
        return woe_df
    
    def _is_monotonic(self, values: np.ndarray) -> bool:
        """Check if values are monotonic"""
        return np.all(np.diff(values) >= 0) or np.all(np.diff(values) <= 0)
    
    def _merge_bins(self, woe_df: pd.DataFrame, idx1: int, idx2: int) -> pd.DataFrame:
        """Merge two adjacent bins"""
        
        new_df = woe_df.copy()
        
        # Combine the bins
        new_df.loc[idx1, 'max_value'] = new_df.loc[idx2, 'max_value']
        new_df.loc[idx1, 'n_obs'] += new_df.loc[idx2, 'n_obs']
        new_df.loc[idx1, 'n_events'] += new_df.loc[idx2, 'n_events']
        new_df.loc[idx1, 'n_non_events'] += new_df.loc[idx2, 'n_non_events']
        
        # Recalculate WOE and IV
        eps = 1e-10
        total_events = new_df['n_events'].sum()
        total_non_events = new_df['n_non_events'].sum()
        
        event_rate = (new_df.loc[idx1, 'n_events'] + eps) / (total_events + eps)
        non_event_rate = (new_df.loc[idx1, 'n_non_events'] + eps) / (total_non_events + eps)
        
        new_df.loc[idx1, 'woe'] = np.log(event_rate / non_event_rate)
        new_df.loc[idx1, 'iv'] = (event_rate - non_event_rate) * new_df.loc[idx1, 'woe']
        new_df.loc[idx1, 'event_rate'] = new_df.loc[idx1, 'n_events'] / (new_df.loc[idx1, 'n_obs'] + eps)
        
        # Drop the merged bin
        new_df = new_df.drop(idx2).reset_index(drop=True)
        
        return new_df
    
    def _merge_insignificant_categories(self, woe_df: pd.DataFrame, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Merge categories that are not statistically different"""
        
        # Sort by WOE
        woe_df = woe_df.sort_values('woe').reset_index(drop=True)
        
        # Continue merging while we have more than min_bins
        while len(woe_df) > self.min_bins:
            # Find the pair with highest p-value (most similar)
            best_pair = None
            best_p_value = 0
            
            for i in range(len(woe_df) - 1):
                # Chi-square test between adjacent categories
                cat1 = woe_df.loc[i, 'category']
                cat2 = woe_df.loc[i + 1, 'category']
                
                # Create contingency table
                mask1 = X == cat1
                mask2 = X == cat2
                
                contingency = pd.crosstab(
                    pd.Series(['cat1'] * mask1.sum() + ['cat2'] * mask2.sum()),
                    pd.concat([y[mask1], y[mask2]])
                )
                
                # Chi-square test
                chi2, p_value, _, _ = chi2_contingency(contingency)
                
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_pair = i
            
            # If best p-value is not significant, merge
            if best_p_value > 0.05:  # 5% significance level
                woe_df = self._merge_categories(woe_df, best_pair, best_pair + 1)
            else:
                break  # No more insignificant pairs
        
        return woe_df
    
    def _merge_categories(self, woe_df: pd.DataFrame, idx1: int, idx2: int) -> pd.DataFrame:
        """Merge two categorical bins"""
        
        new_df = woe_df.copy()
        
        # Combine categories
        new_df.loc[idx1, 'category'] = f"{new_df.loc[idx1, 'category']}_{new_df.loc[idx2, 'category']}"
        new_df.loc[idx1, 'n_obs'] += new_df.loc[idx2, 'n_obs']
        new_df.loc[idx1, 'n_events'] += new_df.loc[idx2, 'n_events']
        new_df.loc[idx1, 'n_non_events'] += new_df.loc[idx2, 'n_non_events']
        
        # Recalculate WOE and IV
        eps = 1e-10
        total_events = new_df['n_events'].sum()
        total_non_events = new_df['n_non_events'].sum()
        
        event_rate = (new_df.loc[idx1, 'n_events'] + eps) / (total_events + eps)
        non_event_rate = (new_df.loc[idx1, 'n_non_events'] + eps) / (total_non_events + eps)
        
        new_df.loc[idx1, 'woe'] = np.log(event_rate / non_event_rate)
        new_df.loc[idx1, 'iv'] = (event_rate - non_event_rate) * new_df.loc[idx1, 'woe']
        new_df.loc[idx1, 'event_rate'] = new_df.loc[idx1, 'n_events'] / (new_df.loc[idx1, 'n_obs'] + eps)
        
        # Drop merged category
        new_df = new_df.drop(idx2).reset_index(drop=True)
        
        return new_df
    
    def _merge_closest_woe(self, woe_df: pd.DataFrame) -> pd.DataFrame:
        """Merge categories with closest WOE values"""
        
        # Sort by WOE
        woe_df = woe_df.sort_values('woe').reset_index(drop=True)
        
        # Find closest pair
        woe_diffs = np.abs(np.diff(woe_df['woe'].values))
        min_diff_idx = np.argmin(woe_diffs)
        
        # Merge
        if 'category' in woe_df.columns:
            return self._merge_categories(woe_df, min_diff_idx, min_diff_idx + 1)
        else:
            return self._merge_bins(woe_df, min_diff_idx, min_diff_idx + 1)
    
    def _calculate_missing_woe(self, X: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Calculate WOE for missing values"""
        
        mask = X.isna()
        n_events = y[mask].sum()
        n_non_events = (~y[mask]).sum()
        
        eps = 1e-10
        total_events = y.sum()
        total_non_events = (~y).sum()
        
        event_rate = (n_events + eps) / (total_events + eps)
        non_event_rate = (n_non_events + eps) / (total_non_events + eps)
        
        woe = np.log(event_rate / non_event_rate)
        iv = (event_rate - non_event_rate) * woe
        
        return pd.DataFrame({
            'min_value': [np.nan],
            'max_value': [np.nan],
            'category': ['MISSING'],
            'n_obs': [mask.sum()],
            'n_events': [n_events],
            'n_non_events': [n_non_events],
            'event_rate': [n_events / (n_events + n_non_events + eps)],
            'woe': [woe],
            'iv': [iv],
            'variable': [X.name]
        })
    
    def _apply_woe(self, X: pd.Series, woe_df: pd.DataFrame) -> pd.Series:
        """Apply WOE transformation"""
        
        X_woe = pd.Series(index=X.index, dtype=float)
        
        if 'category' in woe_df.columns:
            # Categorical
            for _, row in woe_df.iterrows():
                mask = X == row['category']
                X_woe[mask] = row['woe']
        else:
            # Numeric
            for _, row in woe_df.iterrows():
                mask = (X >= row['min_value']) & (X < row['max_value'])
                X_woe[mask] = row['woe']
        
        return X_woe
    
    def _calculate_gini(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        
        # Remove NaN values
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]
        
        if len(y_true) == 0:
            return 0.0
        
        # Sort by score
        order = np.argsort(y_score)
        y_true = y_true[order]
        y_score = y_score[order]
        
        # Calculate Gini
        n = len(y_true)
        cum_true = np.cumsum(y_true)
        
        return (2 * np.sum((n - np.arange(n)) * y_true)) / (n * cum_true[-1]) - 1
    
    def _create_single_bin(self, variable_name: str) -> pd.DataFrame:
        """Create a single bin for edge cases"""
        
        return pd.DataFrame({
            'min_value': [-np.inf],
            'max_value': [np.inf],
            'n_obs': [0],
            'n_events': [0],
            'n_non_events': [0],
            'event_rate': [0],
            'woe': [0],
            'iv': [0],
            'variable': [variable_name]
        })
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit WOE optimization and transform data
        
        Returns:
        --------
        X_woe : pd.DataFrame
            WOE transformed data
        woe_mappings : dict
            WOE mappings for each variable
        """
        
        X_woe = pd.DataFrame(index=X.index)
        woe_mappings = {}
        
        for col in X.columns:
            # Determine variable type
            if pd.api.types.is_numeric_dtype(X[col]):
                woe_df = self.optimize_numeric(X[col], y)
            else:
                woe_df = self.optimize_categorical(X[col], y)
            
            # Store mapping
            woe_mappings[col] = woe_df
            self.woe_mapping_[col] = woe_df
            
            # Apply transformation
            X_woe[col] = self._apply_woe(X[col], woe_df)
            
            # Store IV
            self.iv_values_[col] = woe_df['iv'].sum()
            
            # Calculate and store Gini
            self.gini_values_[col] = self._calculate_gini(y, X_woe[col])
        
        return X_woe, woe_mappings