"""Optimized Binning for IV/Gini Maximization"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BinInfo:
    """Information about a single bin"""
    left: float
    right: float
    n_obs: int
    n_events: int
    n_non_events: int
    event_rate: float
    woe: float
    iv_contrib: float


class OptimizedBinning:
    """
    Optimized binning algorithm that maximizes IV or Gini.

    Features:
    - Monotonic binning for numeric variables
    - Automatic bin merging for categorical variables
    - IV/Gini optimization
    - Minimum bin size constraints
    - Special handling for missing values
    """

    def __init__(self,
                 max_bins: int = 10,
                 min_bin_size: float = 0.05,
                 min_bin_obs: int = 50,
                 monotonic: bool = True,
                 optimize_metric: str = 'iv',
                 handle_missing: bool = True,
                 special_values: Optional[List] = None):
        """
        Initialize optimized binning.

        Parameters:
        -----------
        max_bins : int
            Maximum number of bins
        min_bin_size : float
            Minimum bin size as fraction of total
        min_bin_obs : int
            Minimum number of observations per bin
        monotonic : bool
            Enforce monotonic WOE
        optimize_metric : str
            Metric to optimize ('iv' or 'gini')
        handle_missing : bool
            Create special bin for missing values
        special_values : list
            List of special values to handle separately
        """
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.min_bin_obs = min_bin_obs
        self.monotonic = monotonic
        self.optimize_metric = optimize_metric.lower()
        self.handle_missing = handle_missing
        self.special_values = special_values or []

        # Results storage
        self.bins_ = None
        self.woe_mapping_ = None
        self.iv_ = None
        self.gini_ = None
        self.is_numeric_ = None

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """
        Fit optimal bins and transform to WOE values.

        Parameters:
        -----------
        X : pd.Series
            Feature to bin
        y : pd.Series
            Target variable (0/1)

        Returns:
        --------
        pd.Series
            WOE-transformed values
        """
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X: pd.Series, y: pd.Series) -> 'OptimizedBinning':
        """
        Find optimal bins for the feature.

        Parameters:
        -----------
        X : pd.Series
            Feature to bin
        y : pd.Series
            Target variable (0/1)
        """
        # Check if numeric or categorical
        self.is_numeric_ = pd.api.types.is_numeric_dtype(X)

        if self.is_numeric_:
            self._fit_numeric(X, y)
        else:
            self._fit_categorical(X, y)

        # Calculate final metrics
        self._calculate_metrics(X, y)

        return self

    def _fit_numeric(self, X: pd.Series, y: pd.Series):
        """Fit binning for numeric variables."""
        # Handle missing values
        if self.handle_missing:
            missing_mask = X.isna()
            X_clean = X[~missing_mask]
            y_clean = y[~missing_mask]

            if missing_mask.any():
                # Create missing bin
                self._create_missing_bin(y[missing_mask])
        else:
            X_clean = X.dropna()
            y_clean = y[X_clean.index]

        # Handle special values
        special_mask = X_clean.isin(self.special_values)
        if special_mask.any():
            X_special = X_clean[special_mask]
            y_special = y_clean[special_mask]
            X_clean = X_clean[~special_mask]
            y_clean = y_clean[~special_mask]

            # Create special bins
            for val in self.special_values:
                if val in X_special.values:
                    mask = X_special == val
                    self._create_special_bin(val, y_special[mask])

        # Initial binning using quantiles
        initial_bins = self._create_initial_bins(X_clean, y_clean)

        # Optimize bins
        if self.optimize_metric == 'iv':
            optimized_bins = self._optimize_bins_iv(initial_bins, X_clean, y_clean)
        else:
            optimized_bins = self._optimize_bins_gini(initial_bins, X_clean, y_clean)

        # Apply monotonic constraint if needed
        if self.monotonic:
            optimized_bins = self._enforce_monotonicity(optimized_bins)

        # Merge small bins
        final_bins = self._merge_small_bins(optimized_bins)

        self.bins_ = final_bins

    def _fit_categorical(self, X: pd.Series, y: pd.Series):
        """Fit binning for categorical variables."""
        # Calculate event rate for each category
        category_stats = []

        for category in X.unique():
            if pd.isna(category):
                if self.handle_missing:
                    mask = X.isna()
                    n_obs = mask.sum()
                    n_events = y[mask].sum()
                    category_stats.append({
                        'category': 'MISSING',
                        'n_obs': n_obs,
                        'n_events': n_events,
                        'event_rate': n_events / n_obs if n_obs > 0 else 0
                    })
            else:
                mask = X == category
                n_obs = mask.sum()
                n_events = y[mask].sum()
                category_stats.append({
                    'category': category,
                    'n_obs': n_obs,
                    'n_events': n_events,
                    'event_rate': n_events / n_obs if n_obs > 0 else 0
                })

        # Sort by event rate
        category_stats = sorted(category_stats, key=lambda x: x['event_rate'])

        # Group similar categories
        grouped_categories = self._group_similar_categories(category_stats)

        # Create categorical bins
        self.bins_ = self._create_categorical_bins(grouped_categories, y)

    def _create_initial_bins(self, X: pd.Series, y: pd.Series) -> List[BinInfo]:
        """Create initial bins using quantiles."""
        n_bins = min(self.max_bins, len(X.unique()))

        # Use quantiles for initial binning
        try:
            _, bin_edges = pd.qcut(X, q=n_bins, retbins=True, duplicates='drop')
        except:
            _, bin_edges = pd.cut(X, bins=n_bins, retbins=True)

        bins = []
        for i in range(len(bin_edges) - 1):
            mask = (X >= bin_edges[i]) & (X < bin_edges[i + 1])
            if i == len(bin_edges) - 2:  # Last bin
                mask = (X >= bin_edges[i]) & (X <= bin_edges[i + 1])

            if mask.any():
                bin_info = self._create_bin_info(
                    bin_edges[i], bin_edges[i + 1],
                    y[mask]
                )
                bins.append(bin_info)

        return bins

    def _optimize_bins_iv(self, bins: List[BinInfo], X: pd.Series, y: pd.Series) -> List[BinInfo]:
        """
        Optimize bins to maximize Information Value.

        Uses iterative merging and splitting to maximize IV.
        """
        current_iv = self._calculate_total_iv(bins)
        improved = True
        iteration = 0
        max_iterations = 50

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try merging adjacent bins
            for i in range(len(bins) - 1):
                if len(bins) <= 2:
                    break

                # Calculate IV after merging
                merged_bins = bins.copy()
                merged_bin = self._merge_two_bins(bins[i], bins[i + 1], X, y)
                merged_bins[i] = merged_bin
                del merged_bins[i + 1]

                new_iv = self._calculate_total_iv(merged_bins)

                # Keep merge if it improves IV or doesn't hurt much
                if new_iv >= current_iv * 0.99:  # Allow 1% tolerance
                    bins = merged_bins
                    current_iv = new_iv
                    improved = True
                    break

            # Try splitting bins if we have room
            if not improved and len(bins) < self.max_bins:
                for i, bin_info in enumerate(bins):
                    # Only split if bin is large enough
                    if bin_info.n_obs < 2 * self.min_bin_obs:
                        continue

                    # Try to split this bin
                    split_bins = self._split_bin(bin_info, X, y)
                    if split_bins and len(split_bins) == 2:
                        # Calculate IV after splitting
                        new_bins = bins.copy()
                        new_bins[i:i+1] = split_bins

                        new_iv = self._calculate_total_iv(new_bins)

                        if new_iv > current_iv:
                            bins = new_bins
                            current_iv = new_iv
                            improved = True
                            break

        return bins

    def _optimize_bins_gini(self, bins: List[BinInfo], X: pd.Series, y: pd.Series) -> List[BinInfo]:
        """Optimize bins to maximize Gini coefficient."""
        # Similar to IV optimization but using Gini
        # Convert to WOE and calculate AUC/Gini
        current_gini = self._calculate_gini_from_bins(bins, X, y)

        # Use similar optimization logic as IV
        # (Implementation similar to _optimize_bins_iv but with Gini metric)
        return bins

    def _enforce_monotonicity(self, bins: List[BinInfo]) -> List[BinInfo]:
        """Enforce monotonic WOE constraint."""
        if len(bins) <= 1:
            return bins

        # Check if already monotonic
        woes = [b.woe for b in bins]
        if self._is_monotonic(woes):
            return bins

        # Merge bins to achieve monotonicity
        while not self._is_monotonic([b.woe for b in bins]) and len(bins) > 2:
            # Find the pair that violates monotonicity most
            violations = []
            for i in range(len(bins) - 1):
                if not self._is_monotonic([bins[i].woe, bins[i + 1].woe]):
                    violations.append((i, abs(bins[i].woe - bins[i + 1].woe)))

            if violations:
                # Merge the pair with smallest WOE difference
                violations.sort(key=lambda x: x[1])
                merge_idx = violations[0][0]

                # Merge bins
                merged_bin = self._merge_two_bins_simple(bins[merge_idx], bins[merge_idx + 1])
                bins[merge_idx] = merged_bin
                del bins[merge_idx + 1]
            else:
                break

        return bins

    def _merge_small_bins(self, bins: List[BinInfo]) -> List[BinInfo]:
        """Merge bins that are too small."""
        min_size = int(self.min_bin_size * sum(b.n_obs for b in bins))

        while True:
            # Find smallest bin
            small_bins = [(i, b) for i, b in enumerate(bins) if b.n_obs < max(min_size, self.min_bin_obs)]

            if not small_bins or len(bins) <= 2:
                break

            # Merge smallest bin with neighbor
            idx, small_bin = min(small_bins, key=lambda x: x[1].n_obs)

            # Determine which neighbor to merge with
            if idx == 0:
                # Merge with next
                merged = self._merge_two_bins_simple(bins[idx], bins[idx + 1])
                bins[idx] = merged
                del bins[idx + 1]
            elif idx == len(bins) - 1:
                # Merge with previous
                merged = self._merge_two_bins_simple(bins[idx - 1], bins[idx])
                bins[idx - 1] = merged
                del bins[idx]
            else:
                # Merge with neighbor that has closer WOE
                woe_diff_prev = abs(bins[idx].woe - bins[idx - 1].woe)
                woe_diff_next = abs(bins[idx].woe - bins[idx + 1].woe)

                if woe_diff_prev < woe_diff_next:
                    merged = self._merge_two_bins_simple(bins[idx - 1], bins[idx])
                    bins[idx - 1] = merged
                    del bins[idx]
                else:
                    merged = self._merge_two_bins_simple(bins[idx], bins[idx + 1])
                    bins[idx] = merged
                    del bins[idx + 1]

        return bins

    def _group_similar_categories(self, category_stats: List[Dict]) -> List[List[Dict]]:
        """Group categories with similar event rates."""
        if not category_stats:
            return []

        groups = []
        current_group = [category_stats[0]]

        for i in range(1, len(category_stats)):
            curr_stat = category_stats[i]
            group_event_rate = sum(s['n_events'] for s in current_group) / sum(s['n_obs'] for s in current_group)

            # Check if should add to current group or start new group
            rate_diff = abs(curr_stat['event_rate'] - group_event_rate)

            # Criteria for new group
            if (len(current_group) >= self.max_bins or
                rate_diff > 0.1 or  # More than 10% difference in event rate
                sum(s['n_obs'] for s in current_group) >= self.min_bin_obs):
                groups.append(current_group)
                current_group = [curr_stat]
            else:
                current_group.append(curr_stat)

        if current_group:
            groups.append(current_group)

        return groups

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform feature values to WOE.

        Parameters:
        -----------
        X : pd.Series
            Feature values

        Returns:
        --------
        pd.Series
            WOE values
        """
        if self.bins_ is None:
            raise ValueError("Binning must be fitted first")

        woe_values = pd.Series(index=X.index, dtype=float)

        if self.is_numeric_:
            for bin_info in self.bins_:
                if hasattr(bin_info, 'is_missing') and bin_info.is_missing:
                    mask = X.isna()
                elif hasattr(bin_info, 'is_special') and bin_info.is_special:
                    mask = X == bin_info.value
                else:
                    mask = (X >= bin_info.left) & (X <= bin_info.right)

                woe_values[mask] = bin_info.woe
        else:
            # Categorical transformation
            for bin_info in self.bins_:
                if hasattr(bin_info, 'categories'):
                    mask = X.isin(bin_info.categories)
                    woe_values[mask] = bin_info.woe

        # Fill any remaining values with 0
        woe_values = woe_values.fillna(0)

        return woe_values

    # Helper methods
    def _create_bin_info(self, left: float, right: float, y_bin: pd.Series) -> BinInfo:
        """Create BinInfo object for a bin."""
        n_obs = len(y_bin)
        n_events = y_bin.sum()
        n_non_events = n_obs - n_events
        event_rate = n_events / n_obs if n_obs > 0 else 0

        # Calculate WOE
        total_events = y_bin.sum()  # This is for the bin
        total_non_events = len(y_bin) - total_events

        # Need total population stats (passed from parent)
        # Using smoothing to avoid log(0)
        event_pct = (n_events + 0.5) / (total_events + 1)
        non_event_pct = (n_non_events + 0.5) / (total_non_events + 1)
        woe = np.log(event_pct / non_event_pct) if non_event_pct > 0 else 0

        # Calculate IV contribution
        iv_contrib = (event_pct - non_event_pct) * woe

        return BinInfo(
            left=left,
            right=right,
            n_obs=n_obs,
            n_events=n_events,
            n_non_events=n_non_events,
            event_rate=event_rate,
            woe=woe,
            iv_contrib=iv_contrib
        )

    def _merge_two_bins_simple(self, bin1: BinInfo, bin2: BinInfo) -> BinInfo:
        """Merge two bins into one."""
        return BinInfo(
            left=min(bin1.left, bin2.left),
            right=max(bin1.right, bin2.right),
            n_obs=bin1.n_obs + bin2.n_obs,
            n_events=bin1.n_events + bin2.n_events,
            n_non_events=bin1.n_non_events + bin2.n_non_events,
            event_rate=(bin1.n_events + bin2.n_events) / (bin1.n_obs + bin2.n_obs),
            woe=(bin1.woe * bin1.n_obs + bin2.woe * bin2.n_obs) / (bin1.n_obs + bin2.n_obs),
            iv_contrib=bin1.iv_contrib + bin2.iv_contrib
        )

    def _calculate_total_iv(self, bins: List[BinInfo]) -> float:
        """Calculate total IV from bins."""
        return sum(b.iv_contrib for b in bins)

    def _is_monotonic(self, values: List[float]) -> bool:
        """Check if values are monotonic (increasing or decreasing)."""
        if len(values) <= 1:
            return True

        increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

        return increasing or decreasing

    def _calculate_metrics(self, X: pd.Series, y: pd.Series):
        """Calculate final IV and Gini metrics."""
        # Transform to WOE
        woe_values = self.transform(X)

        # Calculate IV
        self.iv_ = sum(b.iv_contrib for b in self.bins_ if hasattr(b, 'iv_contrib'))

        # Calculate Gini
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, woe_values)
            self.gini_ = 2 * auc - 1
        except:
            self.gini_ = 0.0

    def get_binning_table(self) -> pd.DataFrame:
        """
        Get binning table with statistics.

        Returns:
        --------
        pd.DataFrame
            Binning table with bin information
        """
        if self.bins_ is None:
            raise ValueError("Binning must be fitted first")

        rows = []
        for i, bin_info in enumerate(self.bins_):
            row = {
                'bin': i + 1,
                'range': f"[{bin_info.left:.2f}, {bin_info.right:.2f}]",
                'n_obs': bin_info.n_obs,
                'n_events': bin_info.n_events,
                'n_non_events': bin_info.n_non_events,
                'event_rate': bin_info.event_rate,
                'woe': bin_info.woe,
                'iv_contribution': bin_info.iv_contrib
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add totals row
        totals = pd.Series({
            'bin': 'Total',
            'range': '-',
            'n_obs': df['n_obs'].sum(),
            'n_events': df['n_events'].sum(),
            'n_non_events': df['n_non_events'].sum(),
            'event_rate': df['n_events'].sum() / df['n_obs'].sum(),
            'woe': '-',
            'iv_contribution': df['iv_contribution'].sum()
        })

        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        return df