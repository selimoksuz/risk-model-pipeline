"""Data splitting module for train/test/OOT splits"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class DataSplitter:
    """Handles data splitting strategies including equal default rate splits"""

    def __init__(self, config):
        self.config = config
        self.split_stats_ = {}
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/test/OOT sets"""

        result = {}

        # Time-based split if time column exists
        if self.config.time_col and self.config.time_col in df.columns:
            result = self._time_based_split(df)
        else:
            # Random split
            result = self._random_split(df)

        print(f"Data split - Train: {len(result['train'])}, ", end="")
        if 'test' in result:
            print(f"Test: {len(result['test'])}, ", end="")
        if 'oot' in result:
            print(f"OOT: {len(result['oot'])}")
        else:
            print()

        return result

    def split_stratified(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data with equal default rates across train/test/oot.

        Ensures each split has the same target distribution.
        """
        print("  Performing stratified split for equal default rates...")

        result = {}

        # First handle OOT if time-based
        if self.config.time_col and self.config.time_col in df.columns and self.config.oot_months:
            df = df.sort_values(self.config.time_col)
            max_date = pd.to_datetime(df[self.config.time_col]).max()
            oot_cutoff = max_date - pd.DateOffset(months=self.config.oot_months)

            # Initial OOT split
            in_time_mask = pd.to_datetime(df[self.config.time_col]) < oot_cutoff
            in_time = df[in_time_mask]
            oot = df[~in_time_mask]

            # Check if OOT has enough samples
            if len(oot) >= self.config.min_oot_size:
                # Stratify OOT to match in-time default rate
                oot = self._stratify_to_target_rate(
                    oot,
                    target_rate=in_time[self.config.target_col].mean()
                )
                result['oot'] = oot
            else:
                # Add OOT back to in-time if too small
                in_time = df
                result['oot'] = None
        else:
            in_time = df
            result['oot'] = None

        # Now split in-time into train/test with stratification
        if self.config.test_ratio > 0:
            # Use StratifiedShuffleSplit for equal default rates
            stratified_split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )

            # Get indices for train and test
            train_idx, test_idx = next(stratified_split.split(
                in_time,
                in_time[self.config.target_col]
            ))

            result['train'] = in_time.iloc[train_idx]
            result['test'] = in_time.iloc[test_idx]
        else:
            result['train'] = in_time
            result['test'] = None

        # Calculate and store split statistics
        self._calculate_split_stats(result)

        # Print statistics
        self._print_split_stats()

        return result

    def _stratify_to_target_rate(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        """
        Adjust a dataset to match a target default rate.

        This is done by sampling from each class appropriately.
        """
        current_rate = df[self.config.target_col].mean()

        if abs(current_rate - target_rate) < 0.01:  # Within 1% is good enough
            return df

        # Separate defaults and non-defaults
        defaults = df[df[self.config.target_col] == 1]
        non_defaults = df[df[self.config.target_col] == 0]

        n_total = len(df)

        # Calculate how many of each we need
        n_defaults_needed = int(n_total * target_rate)
        n_non_defaults_needed = n_total - n_defaults_needed

        # Sample appropriately
        if n_defaults_needed <= len(defaults):
            sampled_defaults = defaults.sample(n=n_defaults_needed, random_state=self.config.random_state)
        else:
            # Need all defaults plus some oversampling
            sampled_defaults = defaults

        if n_non_defaults_needed <= len(non_defaults):
            sampled_non_defaults = non_defaults.sample(n=n_non_defaults_needed, random_state=self.config.random_state)
        else:
            sampled_non_defaults = non_defaults

        # Combine and shuffle
        stratified_df = pd.concat([sampled_defaults, sampled_non_defaults])
        stratified_df = stratified_df.sample(frac=1, random_state=self.config.random_state)

        return stratified_df

    def _calculate_split_stats(self, result: Dict[str, pd.DataFrame]):
        """Calculate statistics for each split."""
        for split_name, split_df in result.items():
            if split_df is not None:
                self.split_stats_[split_name] = {
                    'n_samples': len(split_df),
                    'n_defaults': split_df[self.config.target_col].sum(),
                    'default_rate': split_df[self.config.target_col].mean(),
                    'n_features': len(split_df.columns) - 1  # Exclude target
                }

                # Add time range if time column exists
                if self.config.time_col in split_df.columns:
                    self.split_stats_[split_name]['date_min'] = split_df[self.config.time_col].min()
                    self.split_stats_[split_name]['date_max'] = split_df[self.config.time_col].max()

    def _print_split_stats(self):
        """Print split statistics."""
        print("\n  Split Statistics:")
        print("  " + "-" * 60)

        # Header
        print(f"  {'Split':<10} {'Samples':<10} {'Defaults':<10} {'Rate':<10}")
        print("  " + "-" * 60)

        # Data
        for split_name, stats in self.split_stats_.items():
            print(f"  {split_name:<10} {stats['n_samples']:<10} "
                  f"{stats['n_defaults']:<10} {stats['default_rate']:.2%}")

        # Check if rates are equal
        rates = [stats['default_rate'] for stats in self.split_stats_.values()]
        if len(rates) > 1:
            rate_std = np.std(rates)
            if rate_std < 0.001:
                print("\n  ✓ Default rates are equal across splits")
            else:
                print(f"\n  ⚠ Default rate standard deviation: {rate_std:.4f}")
    
    def _time_based_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data based on time column"""
        
        df = df.sort_values(self.config.time_col)
        
        # Calculate OOT cutoff
        if self.config.oot_months and self.config.oot_months > 0:
            max_date = pd.to_datetime(df[self.config.time_col]).max()
            oot_cutoff = max_date - pd.DateOffset(months=self.config.oot_months)
            
            # Split into in-time and out-of-time
            in_time = df[pd.to_datetime(df[self.config.time_col]) < oot_cutoff]
            oot = df[pd.to_datetime(df[self.config.time_col]) >= oot_cutoff]
        else:
            in_time = df
            oot = None
        
        # Split in-time into train/test
        if self.config.test_ratio and self.config.test_ratio > 0:
            n_test = int(len(in_time) * self.config.test_ratio)
            train = in_time.iloc[:-n_test]
            test = in_time.iloc[-n_test:]
        else:
            train = in_time
            test = None
        
        result = {'train': train}
        if test is not None and len(test) > 0:
            result['test'] = test
        if oot is not None and len(oot) > 0:
            result['oot'] = oot
        
        return result
    
    def _random_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Random train/test split"""
        
        from sklearn.model_selection import train_test_split
        
        if self.config.test_ratio and self.config.test_ratio > 0:
            train, test = train_test_split(
                df, 
                test_size=self.config.test_ratio,
                random_state=self.config.random_state,
                stratify=df[self.config.target_col] if self.config.target_col in df.columns else None
            )
            return {'train': train, 'test': test}
        else:
            return {'train': df}