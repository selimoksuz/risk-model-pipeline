"""
Smart Data Splitter with equal default rate capability
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class SmartDataSplitter:
    """
    Advanced data splitting with equal default rate option.

    Features:
    - Time-based OOT splitting
    - Equal default rate across splits
    - Stratified sampling
    - Configurable test/OOT ratios
    """

    def __init__(self, config):
        self.config = config
        self.split_stats_ = {}

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Standard split without equal default rate constraint."""

        result = {}

        # Time-based split if time column exists
        if self.config.time_col and self.config.time_col in df.columns:
            result = self._time_based_split(df)
        else:
            result = self._random_split(df)

        self._calculate_statistics(result)
        return result

    def split_equal_default_rate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split with equal default rates across train/test/OOT.

        This ensures each split has the same target distribution.
        """

        print("    Performing equal default rate split...")

        result = {}

        # First handle OOT if time-based
        if self.config.time_col and self.config.time_col in df.columns:
            # Sort by time
            df_sorted = df.sort_values(self.config.time_col)

            # Determine OOT period
            if hasattr(self.config, 'oot_months') and self.config.oot_months:
                max_date = pd.to_datetime(df_sorted[self.config.time_col]).max()
                oot_cutoff = max_date - pd.DateOffset(months=self.config.oot_months)

                # Split into in-time and OOT
                in_time_mask = pd.to_datetime(df_sorted[self.config.time_col]) < oot_cutoff
                in_time_df = df_sorted[in_time_mask]
                oot_df = df_sorted[~in_time_mask]

                # Check if OOT has enough samples
                min_oot_size = getattr(self.config, 'min_oot_size', 100)
                if len(oot_df) >= min_oot_size:
                    # Adjust OOT to match in-time default rate
                    target_rate = in_time_df[self.config.target_col].mean()
                    oot_adjusted = self._adjust_to_target_rate(oot_df, target_rate)
                    result['oot'] = oot_adjusted
                else:
                    # OOT too small, merge back
                    in_time_df = df_sorted
                    print(f"      OOT too small ({len(oot_df)}), skipping OOT split")
            else:
                in_time_df = df_sorted
        else:
            in_time_df = df

        # Split in-time into train/test with equal default rates
        if hasattr(self.config, 'test_ratio') and self.config.test_ratio > 0:
            # Use stratified split to maintain target distribution
            stratified_split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )

            # Get indices
            train_idx, test_idx = next(stratified_split.split(
                in_time_df,
                in_time_df[self.config.target_col]
            ))

            result['train'] = in_time_df.iloc[train_idx].reset_index(drop=True)
            result['test'] = in_time_df.iloc[test_idx].reset_index(drop=True)
        else:
            result['train'] = in_time_df.reset_index(drop=True)

        # Verify equal default rates
        self._calculate_statistics(result)
        self._verify_equal_rates(result)

        return result

    def _time_based_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split based on time column."""

        df_sorted = df.sort_values(self.config.time_col)
        result = {}

        # Calculate OOT cutoff
        if hasattr(self.config, 'oot_months') and self.config.oot_months > 0:
            max_date = pd.to_datetime(df_sorted[self.config.time_col]).max()
            oot_cutoff = max_date - pd.DateOffset(months=self.config.oot_months)

            # Split into in-time and out-of-time
            in_time = df_sorted[pd.to_datetime(df_sorted[self.config.time_col]) < oot_cutoff]
            oot = df_sorted[pd.to_datetime(df_sorted[self.config.time_col]) >= oot_cutoff]

            if len(oot) > 0:
                result['oot'] = oot.reset_index(drop=True)
        else:
            in_time = df_sorted

        # Split in-time into train/test
        if hasattr(self.config, 'test_ratio') and self.config.test_ratio > 0:
            n_test = int(len(in_time) * self.config.test_ratio)

            # Time-based train/test split
            train = in_time.iloc[:-n_test] if n_test > 0 else in_time
            test = in_time.iloc[-n_test:] if n_test > 0 else pd.DataFrame()

            result['train'] = train.reset_index(drop=True)
            if len(test) > 0:
                result['test'] = test.reset_index(drop=True)
        else:
            result['train'] = in_time.reset_index(drop=True)

        return result

    def _random_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Random stratified split."""

        result = {}

        if hasattr(self.config, 'test_ratio') and self.config.test_ratio > 0:
            # Stratified split
            train, test = train_test_split(
                df,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state,
                stratify=df[self.config.target_col] if self.config.target_col in df.columns else None
            )
            result['train'] = train.reset_index(drop=True)
            result['test'] = test.reset_index(drop=True)
        else:
            result['train'] = df.reset_index(drop=True)

        return result

    def _adjust_to_target_rate(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        """
        Adjust dataset to match target default rate through sampling.
        """

        current_rate = df[self.config.target_col].mean()

        # If rates are close enough, return as is
        if abs(current_rate - target_rate) < 0.001:
            return df

        # Separate defaults and non-defaults
        defaults = df[df[self.config.target_col] == 1]
        non_defaults = df[df[self.config.target_col] == 0]

        n_total = len(df)

        # Calculate how many of each class we need
        n_defaults_needed = int(n_total * target_rate)
        n_non_defaults_needed = n_total - n_defaults_needed

        # Sample appropriately
        if n_defaults_needed <= len(defaults):
            sampled_defaults = defaults.sample(
                n=n_defaults_needed,
                random_state=self.config.random_state,
                replace=False
            )
        else:
            # Need to oversample defaults
            sampled_defaults = defaults.sample(
                n=n_defaults_needed,
                random_state=self.config.random_state,
                replace=True
            )

        if n_non_defaults_needed <= len(non_defaults):
            sampled_non_defaults = non_defaults.sample(
                n=n_non_defaults_needed,
                random_state=self.config.random_state,
                replace=False
            )
        else:
            # Need to oversample non-defaults
            sampled_non_defaults = non_defaults.sample(
                n=n_non_defaults_needed,
                random_state=self.config.random_state,
                replace=True
            )

        # Combine and shuffle
        adjusted_df = pd.concat([sampled_defaults, sampled_non_defaults])
        adjusted_df = adjusted_df.sample(
            frac=1,
            random_state=self.config.random_state
        ).reset_index(drop=True)

        return adjusted_df

    def _calculate_statistics(self, splits: Dict[str, pd.DataFrame]):
        """Calculate split statistics."""

        for split_name, split_df in splits.items():
            if split_df is not None and len(split_df) > 0:
                self.split_stats_[split_name] = {
                    'n_samples': len(split_df),
                    'n_defaults': split_df[self.config.target_col].sum(),
                    'n_non_defaults': len(split_df) - split_df[self.config.target_col].sum(),
                    'default_rate': split_df[self.config.target_col].mean(),
                    'n_features': len(split_df.columns) - 3  # Exclude target, id, time
                }

                # Add time range if applicable
                if self.config.time_col and self.config.time_col in split_df.columns:
                    self.split_stats_[split_name]['date_min'] = split_df[self.config.time_col].min()
                    self.split_stats_[split_name]['date_max'] = split_df[self.config.time_col].max()

    def _verify_equal_rates(self, splits: Dict[str, pd.DataFrame]):
        """Verify that default rates are equal across splits."""

        rates = []
        for split_name, split_df in splits.items():
            if split_df is not None and len(split_df) > 0:
                rate = split_df[self.config.target_col].mean()
                rates.append(rate)

        if len(rates) > 1:
            rate_std = np.std(rates)
            rate_mean = np.mean(rates)

            if rate_std < 0.001:  # Less than 0.1% standard deviation
                print(f"      [OK] Equal default rates achieved (mean: {rate_mean:.2%}, std: {rate_std:.4f})")
            else:
                print(f"      [WARNING] Default rates not perfectly equal (mean: {rate_mean:.2%}, std: {rate_std:.4f})")
                for split_name in self.split_stats_:
                    stats = self.split_stats_[split_name]
                    print(f"        {split_name}: {stats['default_rate']:.2%}")

    def get_split_summary(self) -> pd.DataFrame:
        """Get summary of split statistics."""

        summary_data = []
        for split_name, stats in self.split_stats_.items():
            summary_data.append({
                'Split': split_name,
                'Samples': stats['n_samples'],
                'Defaults': stats['n_defaults'],
                'Non-Defaults': stats['n_non_defaults'],
                'Default Rate': f"{stats['default_rate']:.2%}",
                'Date Range': f"{stats.get('date_min', 'N/A')} to {stats.get('date_max', 'N/A')}"
            })

        return pd.DataFrame(summary_data)