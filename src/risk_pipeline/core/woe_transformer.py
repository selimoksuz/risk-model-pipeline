"""
Enhanced WOE Transformer with IV/Gini optimized binning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings

warnings.filterwarnings('ignore')


class EnhancedWOETransformer:
    """
    WOE Transformer with IV/Gini optimized binning.
    Handles both numeric and categorical variables.
    """

    def __init__(self, config):
        self.config = config
        self.woe_maps_ = {}
        self.iv_values_ = {}
        self.bin_stats_ = {}

    def fit_transform_single(self, X: pd.Series, y: pd.Series) -> Dict:
        """
        Fit and transform a single variable with WOE.

        Returns dict with:
        - transformed: WOE transformed values
        - woe_map: Mapping of bins to WOE values
        - iv: Information Value
        - bins: Bin edges or categories
        - stats: Binning statistics
        """

        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(X):
            return self._fit_transform_numeric(X, y)
        else:
            return self._fit_transform_categorical(X, y)

    def _fit_transform_numeric(self, X: pd.Series, y: pd.Series) -> Dict:
        """Fit and transform numeric variable."""

        # Initial binning
        if self.config.binning_method == 'optimized':
            bins = self._optimize_bins_iv(X, y)
        elif self.config.binning_method == 'quantile':
            bins = self._create_quantile_bins(X, self.config.max_bins)
        else:
            bins = self._create_equal_width_bins(X, self.config.max_bins)

        # Enforce monotonicity if configured
        if self.config.monotonic_woe:
            bins = self._enforce_monotonicity(bins, X, y)

        # Calculate WOE for each bin
        woe_map, iv, stats = self._calculate_woe(X, y, bins)

        # Transform
        transformed = self._apply_woe(X, bins, woe_map)

        return {
            'transformed': transformed,
            'woe_map': woe_map,
            'iv': iv,
            'bins': bins,
            'stats': stats,
            'type': 'numeric'
        }

    def _fit_transform_categorical(self, X: pd.Series, y: pd.Series) -> Dict:
        """Fit and transform categorical variable."""

        # Group rare categories
        X_grouped = self._group_rare_categories(X, y)

        # Calculate WOE for each category
        categories = X_grouped.unique()
        woe_map = {}
        stats = []

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for cat in categories:
            mask = X_grouped == cat
            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()

            # Calculate WOE
            woe = self._calculate_woe_value(good, bad, total_good, total_bad)
            woe_map[cat] = woe

            stats.append({
                'category': cat,
                'count': mask.sum(),
                'bad_rate': bad / (good + bad) if (good + bad) > 0 else 0,
                'woe': woe
            })

        # Merge categories with similar WOE if needed
        if self.config.binning_method == 'optimized':
            woe_map, stats = self._merge_similar_woe_categories(woe_map, stats, X_grouped, y)

        # Calculate IV
        iv = self._calculate_iv_categorical(X_grouped, y, woe_map)

        # Transform
        transformed = X_grouped.map(woe_map).fillna(0)

        return {
            'transformed': transformed,
            'woe_map': woe_map,
            'iv': iv,
            'categories': list(woe_map.keys()),
            'stats': stats,
            'type': 'categorical'
        }

    def _optimize_bins_iv(self, X: pd.Series, y: pd.Series) -> np.ndarray:
        """
        Optimize bins to maximize IV while maintaining interpretability.
        """

        best_iv = -np.inf
        best_bins = None

        # Try different number of bins
        for n_bins in range(3, min(self.config.max_bins + 1, 21)):
            # Create initial bins
            try:
                bins = pd.qcut(X.fillna(X.median()), n_bins, duplicates='drop').cat.categories
                bins = np.array([b.left for b in bins] + [bins[-1].right])

                # Calculate IV for these bins
                woe_map, iv, _ = self._calculate_woe(X, y, bins)

                # Check if better
                if iv > best_iv and iv < 10:  # Cap IV at 10 to avoid overfitting
                    best_iv = iv
                    best_bins = bins
            except:
                continue

        # If no valid bins found, use simple quantiles
        if best_bins is None:
            best_bins = self._create_quantile_bins(X, 5)

        # Fine-tune bins
        best_bins = self._fine_tune_bins(best_bins, X, y)

        return best_bins

    def _fine_tune_bins(self, bins: np.ndarray, X: pd.Series, y: pd.Series) -> np.ndarray:
        """Fine-tune bins by merging adjacent bins with similar bad rates."""

        if len(bins) <= 3:
            return bins

        # Calculate bad rate for each bin
        bad_rates = []
        for i in range(len(bins) - 1):
            if i == 0:
                mask = X <= bins[i + 1]
            elif i == len(bins) - 2:
                mask = X > bins[i]
            else:
                mask = (X > bins[i]) & (X <= bins[i + 1])

            bad_rate = y[mask].mean() if mask.sum() > 0 else 0
            bad_rates.append(bad_rate)

        # Merge bins with similar bad rates
        merged_bins = [bins[0]]
        for i in range(1, len(bins) - 1):
            # Check if should merge with previous
            if i < len(bad_rates) and i > 0:
                if abs(bad_rates[i] - bad_rates[i-1]) < 0.05:  # 5% threshold
                    continue
            merged_bins.append(bins[i])
        merged_bins.append(bins[-1])

        return np.array(merged_bins)

    def _enforce_monotonicity(self, bins: np.ndarray, X: pd.Series, y: pd.Series) -> np.ndarray:
        """Enforce monotonic WOE across bins."""

        # Calculate WOE for each bin
        woe_values = []
        for i in range(len(bins) - 1):
            if i == 0:
                mask = X <= bins[i + 1]
            elif i == len(bins) - 2:
                mask = X > bins[i]
            else:
                mask = (X > bins[i]) & (X <= bins[i + 1])

            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()
            total_good = (y == 0).sum()
            total_bad = (y == 1).sum()

            woe = self._calculate_woe_value(good, bad, total_good, total_bad)
            woe_values.append(woe)

        # Check monotonicity
        is_increasing = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1))
        is_decreasing = all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))

        if is_increasing or is_decreasing:
            return bins

        # Merge bins to enforce monotonicity
        # This is a simplified approach - could be enhanced
        while len(bins) > 3 and not (is_increasing or is_decreasing):
            # Find pair with smallest WOE difference
            min_diff = np.inf
            merge_idx = 0

            for i in range(len(woe_values) - 1):
                diff = abs(woe_values[i] - woe_values[i+1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            # Merge bins
            bins = np.delete(bins, merge_idx + 1)

            # Recalculate WOE values
            woe_values = []
            for i in range(len(bins) - 1):
                if i == 0:
                    mask = X <= bins[i + 1]
                elif i == len(bins) - 2:
                    mask = X > bins[i]
                else:
                    mask = (X > bins[i]) & (X <= bins[i + 1])

                good = ((y == 0) & mask).sum()
                bad = ((y == 1) & mask).sum()
                total_good = (y == 0).sum()
                total_bad = (y == 1).sum()

                woe = self._calculate_woe_value(good, bad, total_good, total_bad)
                woe_values.append(woe)

            is_increasing = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1))
            is_decreasing = all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))

        return bins

    def _calculate_woe(self, X: pd.Series, y: pd.Series, bins: np.ndarray) -> Tuple[Dict, float, List]:
        """Calculate WOE for given bins."""

        woe_map = {}
        iv_components = []
        stats = []

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for i in range(len(bins) - 1):
            # Define bin
            if i == 0:
                mask = X <= bins[i + 1]
                bin_label = f"<={bins[i + 1]:.2f}"
            elif i == len(bins) - 2:
                mask = X > bins[i]
                bin_label = f">{bins[i]:.2f}"
            else:
                mask = (X > bins[i]) & (X <= bins[i + 1])
                bin_label = f"({bins[i]:.2f}, {bins[i + 1]:.2f}]"

            # Calculate statistics
            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()

            # Calculate WOE and IV
            woe = self._calculate_woe_value(good, bad, total_good, total_bad)
            iv_component = ((bad/total_bad - good/total_good) * woe) if total_bad > 0 and total_good > 0 else 0

            woe_map[i] = woe
            iv_components.append(iv_component)

            stats.append({
                'bin': bin_label,
                'count': mask.sum(),
                'bad_rate': bad / (good + bad) if (good + bad) > 0 else 0,
                'woe': woe,
                'iv': iv_component
            })

        total_iv = sum(iv_components)

        return woe_map, total_iv, stats

    def _calculate_woe_value(self, good: int, bad: int, total_good: int, total_bad: int) -> float:
        """Calculate WOE value with smoothing."""

        # Add smoothing to avoid division by zero
        smooth_factor = 0.5

        pct_good = (good + smooth_factor) / (total_good + smooth_factor)
        pct_bad = (bad + smooth_factor) / (total_bad + smooth_factor)

        # Calculate WOE
        if pct_bad > 0 and pct_good > 0:
            woe = np.log(pct_bad / pct_good)
        else:
            woe = 0

        # Cap WOE values to avoid extreme values
        woe = np.clip(woe, -5, 5)

        return woe

    def _apply_woe(self, X: pd.Series, bins: np.ndarray, woe_map: Dict) -> pd.Series:
        """Apply WOE transformation."""

        transformed = pd.Series(index=X.index, dtype=float)

        for i in range(len(bins) - 1):
            if i == 0:
                mask = X <= bins[i + 1]
            elif i == len(bins) - 2:
                mask = X > bins[i]
            else:
                mask = (X > bins[i]) & (X <= bins[i + 1])

            transformed[mask] = woe_map[i]

        # Handle missing values
        transformed.fillna(0, inplace=True)

        return transformed

    def _create_quantile_bins(self, X: pd.Series, n_bins: int) -> np.ndarray:
        """Create quantile-based bins."""
        try:
            _, bins = pd.qcut(X.fillna(X.median()), n_bins, retbins=True, duplicates='drop')
            return bins
        except:
            return self._create_equal_width_bins(X, n_bins)

    def _create_equal_width_bins(self, X: pd.Series, n_bins: int) -> np.ndarray:
        """Create equal-width bins."""
        min_val = X.min()
        max_val = X.max()
        return np.linspace(min_val, max_val, n_bins + 1)

    def _group_rare_categories(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Group rare categories."""

        min_pct = self.config.min_bin_size

        # Calculate frequency
        freq = X.value_counts(normalize=True)
        rare = freq[freq < min_pct].index

        # Group rare categories
        X_grouped = X.copy()
        if len(rare) > 0:
            X_grouped[X_grouped.isin(rare)] = 'RARE'

        return X_grouped

    def _merge_similar_woe_categories(self, woe_map: Dict, stats: List, X: pd.Series, y: pd.Series) -> Tuple[Dict, List]:
        """Merge categories with similar WOE values."""

        # Sort categories by WOE
        sorted_cats = sorted(woe_map.items(), key=lambda x: x[1])

        # Merge if WOE difference < threshold
        merged_map = {}
        merged_stats = []
        merge_threshold = 0.1

        i = 0
        while i < len(sorted_cats):
            current_cat = sorted_cats[i][0]
            current_woe = sorted_cats[i][1]
            merged_cats = [current_cat]

            # Look for similar WOE values
            j = i + 1
            while j < len(sorted_cats) and abs(sorted_cats[j][1] - current_woe) < merge_threshold:
                merged_cats.append(sorted_cats[j][0])
                j += 1

            # Create merged category
            if len(merged_cats) > 1:
                merged_name = f"MERGED_{i}"
                for cat in merged_cats:
                    merged_map[cat] = current_woe
            else:
                merged_map[current_cat] = current_woe

            i = j

        return merged_map, stats

    def _calculate_iv_categorical(self, X: pd.Series, y: pd.Series, woe_map: Dict) -> float:
        """Calculate IV for categorical variable."""

        iv = 0
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for cat, woe in woe_map.items():
            mask = X == cat
            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()

            if total_bad > 0 and total_good > 0:
                pct_good = good / total_good
                pct_bad = bad / total_bad
                iv += (pct_bad - pct_good) * woe

        return iv

    def transform(self, df: pd.DataFrame, woe_values: Optional[Dict] = None) -> pd.DataFrame:
        """Transform entire dataframe with WOE."""

        if woe_values is None:
            woe_values = self.woe_maps_

        # Start with a copy of the original dataframe to preserve all columns
        df_woe = df.copy()

        # Transform columns that have WOE mappings
        for col, woe_info in woe_values.items():
            if col in df.columns:
                if woe_info['type'] == 'numeric':
                    df_woe[col] = self._apply_woe(
                        df[col], woe_info['bins'], woe_info['woe_map']
                    )
                else:
                    # For categorical, replace with WOE values
                    df_woe[col] = df[col].map(woe_info['woe_map']).fillna(0)

        return df_woe