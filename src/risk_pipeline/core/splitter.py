"""
Smart Data Splitter with equal default rate capability
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SmartDataSplitter:
    """Advanced data splitting utilities with risk-model specific rules."""

    def __init__(self, config):
        self.config = config
        self.split_stats_ = {}

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Standard split without forcing equal default rates."""

        if self.config.time_col and self.config.time_col in df.columns:
            result = self._time_based_split(df, ensure_equal=False)
        else:
            result = self._random_split(df, stratify=False)

        self._calculate_statistics(result)
        return result

    def split_equal_default_rate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split while keeping default rates aligned across the generated samples."""

        print("    Performing equal default rate split...")

        if self.config.time_col and self.config.time_col in df.columns:
            result = self._time_based_split(df, ensure_equal=True)
        else:
            result = self._random_split(df, stratify=True)

        # Align OOT default rate with train if both are present
        if "oot" in result and not result["oot"].empty and "train" in result:
            target_rate = result["train"][self.config.target_col].mean()
            result["oot"] = self._adjust_to_target_rate(result["oot"], target_rate)

        self._calculate_statistics(result)
        self._verify_equal_rates(result)
        return result

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _time_based_split(self, df: pd.DataFrame, *, ensure_equal: bool) -> Dict[str, pd.DataFrame]:
        df_local = df.copy()
        time_col = self.config.time_col
        if time_col in df_local.columns:
            df_local[time_col] = pd.to_datetime(df_local[time_col], errors='coerce')
            df_local['__month'] = df_local[time_col].apply(month_floor)
        else:
            df_local['__month'] = pd.NaT

        train_ratio, test_ratio, oot_ratio = self._get_split_ratios()
        oot_months = getattr(self.config, 'oot_months', None) or 0

        months = df_local['__month'].dropna().sort_values().unique()
        oot_df = pd.DataFrame(columns=df.columns)

        if oot_months > 0 and len(months) == 0:
            n_rows = len(df_local)
            k = max(1, int(n_rows * min(0.2, oot_ratio if oot_ratio > 0 else 0.2)))
            oot_df = df_local.tail(k)
            in_time_df = df_local.head(n_rows - k)
        elif oot_months > 0:
            anchor = months[-1]
            earliest_period = pd.Period(anchor, freq='M') - (oot_months - 1)
            cutoff_ts = earliest_period.to_timestamp()
            oot_mask = df_local['__month'] >= cutoff_ts
            oot_df = df_local.loc[oot_mask]
            in_time_df = df_local.loc[~oot_mask]
        elif oot_ratio > 0 and len(df_local) > 1:
            if time_col in df_local.columns and not df_local[time_col].isna().all():
                sorted_df = df_local.sort_values(time_col)
            else:
                sorted_df = df_local.sort_index()
            k = max(1, int(len(sorted_df) * oot_ratio))
            oot_df = sorted_df.tail(k)
            in_time_df = sorted_df.iloc[:-k]
        else:
            in_time_df = df_local

        stratify_flag = ensure_equal or getattr(self.config, 'stratify_test_split', getattr(self.config, 'stratify_test', True))
        seed = getattr(self.config, 'random_state', None)
        total_available = train_ratio + test_ratio
        test_fraction = test_ratio / total_available if total_available > 0 else 0.0

        train_df, test_df = self._monthly_split(
            in_time_df,
            test_ratio=test_fraction,
            stratify_flag=stratify_flag,
            seed=seed,
        )

        result: Dict[str, pd.DataFrame] = {}
        result['train'] = train_df.reset_index(drop=True)
        if not test_df.empty:
            result['test'] = test_df.reset_index(drop=True)
        if not oot_df.empty:
            result['oot'] = oot_df.drop(columns=['__month'], errors='ignore').sort_values(time_col).reset_index(drop=True) if time_col in oot_df.columns else oot_df.drop(columns=['__month'], errors='ignore').reset_index(drop=True)

        return result

    def _monthly_split(
        self,
        df: pd.DataFrame,
        *,
        test_ratio: float,
        stratify_flag: bool,
        seed: Optional[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty or test_ratio <= 0:
            cleaned = df.drop(columns=["__month"], errors="ignore")
            return cleaned, cleaned.iloc[0:0]

        time_col = self.config.time_col
        target_col = self.config.target_col

        working = df.copy()
        if time_col in working.columns:
            working["__split_month"] = working[time_col].apply(month_floor)
            valid = working.dropna(subset=["__split_month"])
            groups: List = list(valid.groupby("__split_month", sort=True))
            missing = working[working["__split_month"].isna()]
            if not missing.empty:
                groups.append((None, missing))
        else:
            groups = [(None, working)]

        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []

        for offset, (_, month_df) in enumerate(groups):
            if time_col in month_df.columns:
                month_df = month_df.sort_values(time_col)

            if len(month_df) < 2 or target_col not in month_df.columns:
                train_parts.append(month_df)
                continue

            has_two_classes = month_df[target_col].nunique() > 1
            random_state = None if seed is None else seed + offset

            if stratify_flag and has_two_classes:
                try:
                    train_idx, test_idx = train_test_split(
                        month_df.index,
                        test_size=test_ratio,
                        stratify=month_df[target_col],
                        random_state=random_state,
                    )
                except ValueError:
                    train_idx, test_idx = self._chronological_split(month_df, test_ratio)
            else:
                train_idx, test_idx = self._chronological_split(month_df, test_ratio)

            train_parts.append(month_df.loc[train_idx])
            if len(test_idx) > 0:
                test_parts.append(month_df.loc[test_idx])

        train_df = pd.concat(train_parts, ignore_index=False).sort_values(time_col)
        test_df = (
            pd.concat(test_parts, ignore_index=False).sort_values(time_col)
            if test_parts
            else train_df.iloc[0:0]
        )

        train_df = train_df.drop(columns=["__split_month", "__month"], errors="ignore")
        test_df = test_df.drop(columns=["__split_month", "__month"], errors="ignore")

        return train_df, test_df

    def _chronological_split(self, month_df: pd.DataFrame, test_ratio: float):
        k = max(1, int(len(month_df) * test_ratio))
        k = min(k, len(month_df) - 1) if len(month_df) > 1 else 0
        if k == 0:
            return month_df.index, month_df.iloc[0:0].index
        test_idx = month_df.index[-k:]
        train_idx = month_df.index[:-k]
        return train_idx, test_idx

    def _random_split(self, df: pd.DataFrame, stratify: bool) -> Dict[str, pd.DataFrame]:
        train_ratio, test_ratio, oot_ratio = self._get_split_ratios()
        result: Dict[str, pd.DataFrame] = {}

        work_df = df
        random_state = getattr(self.config, 'random_state', None)
        target_col = getattr(self.config, 'target_col', None)
        stratify_labels = None

        if oot_ratio > 0 and len(work_df) > 1:
            if stratify and target_col in work_df.columns:
                stratify_labels = work_df[target_col]
            work_df, oot_df = train_test_split(
                work_df,
                test_size=oot_ratio,
                random_state=random_state,
                stratify=stratify_labels
            )
            result['oot'] = oot_df.reset_index(drop=True)
            stratify_labels = None
        
        available = train_ratio + test_ratio
        test_fraction = test_ratio / available if available > 0 else 0.0

        if test_fraction > 0 and len(work_df) > 1:
            if stratify and target_col in work_df.columns:
                stratify_labels = work_df[target_col]
            train_df, test_df = train_test_split(
                work_df,
                test_size=test_fraction,
                random_state=random_state,
                stratify=stratify_labels
            )
            result['train'] = train_df.reset_index(drop=True)
            result['test'] = test_df.reset_index(drop=True)
        else:
            result['train'] = work_df.reset_index(drop=True)
            if test_fraction > 0:
                result['test'] = work_df.iloc[0:0].reset_index(drop=True)

        if 'oot' in result and result['oot'].empty:
            result.pop('oot')

        return result

    def _get_split_ratios(self) -> Tuple[float, float, float]:
        train_ratio = float(getattr(self.config, 'train_ratio', 0.7))
        test_ratio = float(getattr(self.config, 'test_ratio', getattr(self.config, 'test_size', 0.3)))
        oot_ratio = float(getattr(self.config, 'oot_ratio', getattr(self.config, 'oot_size', 0.0)))
        return train_ratio, test_ratio, oot_ratio

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _time_based_split(self, df: pd.DataFrame, *, ensure_equal: bool) -> Dict[str, pd.DataFrame]:
        df_local = df.copy()
        time_col = self.config.time_col
        df_local[time_col] = pd.to_datetime(df_local[time_col], errors="coerce")
        df_local["__month"] = df_local[time_col].apply(month_floor)

        months = df_local["__month"].dropna().sort_values().unique()
        oot_months = getattr(self.config, "oot_months", 0) or 0
        oot_df = pd.DataFrame(columns=df.columns)

        if oot_months > 0 and len(months) == 0:
            # Fallback when month information is not available: use last 20% as OOT.
            n_rows = len(df_local)
            k = max(1, int(n_rows * 0.2))
            oot_df = df_local.tail(k)
            in_time_df = df_local.head(n_rows - k)
        elif oot_months > 0:
            anchor = months[-1]
            earliest_period = pd.Period(anchor, freq="M") - (oot_months - 1)
            cutoff_ts = earliest_period.to_timestamp()
            oot_mask = df_local["__month"] >= cutoff_ts
            oot_df = df_local.loc[oot_mask]
            in_time_df = df_local.loc[~oot_mask]
        else:
            in_time_df = df_local

        test_ratio = getattr(self.config, "test_ratio", getattr(self.config, "test_size", 0.2))
        stratify_flag = getattr(self.config, "stratify_test_split", getattr(self.config, "stratify_test", True))

        train_df, test_df = self._monthly_split(
            in_time_df,
            test_ratio=test_ratio,
            stratify_flag=stratify_flag,
            seed=getattr(self.config, "random_state", None),
        )

        result: Dict[str, pd.DataFrame] = {}
        result["train"] = train_df.reset_index(drop=True)
        if not test_df.empty:
            result["test"] = test_df.reset_index(drop=True)
        if not oot_df.empty:
            result["oot"] = oot_df.sort_values(time_col).reset_index(drop=True)

        return result

    def _monthly_split(
        self,
        df: pd.DataFrame,
        *,
        test_ratio: float,
        stratify_flag: bool,
        seed: Optional[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty or test_ratio <= 0:
            cleaned = df.drop(columns=["__month"], errors="ignore")
            return cleaned, cleaned.iloc[0:0]

        time_col = self.config.time_col
        target_col = self.config.target_col

        working = df.copy()
        if time_col in working.columns:
            working["__split_month"] = working[time_col].apply(month_floor)
            valid = working.dropna(subset=["__split_month"])
            groups: List = list(valid.groupby("__split_month", sort=True))
            missing = working[working["__split_month"].isna()]
            if not missing.empty:
                groups.append((None, missing))
        else:
            groups = [(None, working)]

        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []

        for offset, (_, month_df) in enumerate(groups):
            if time_col in month_df.columns:
                month_df = month_df.sort_values(time_col)

            if len(month_df) < 2 or target_col not in month_df.columns:
                train_parts.append(month_df)
                continue

            has_two_classes = month_df[target_col].nunique() > 1
            random_state = None if seed is None else seed + offset

            if stratify_flag and has_two_classes:
                try:
                    train_idx, test_idx = train_test_split(
                        month_df.index,
                        test_size=test_ratio,
                        stratify=month_df[target_col],
                        random_state=random_state,
                    )
                except ValueError:
                    train_idx, test_idx = self._chronological_split(month_df, test_ratio)
            else:
                train_idx, test_idx = self._chronological_split(month_df, test_ratio)

            train_parts.append(month_df.loc[train_idx])
            if len(test_idx) > 0:
                test_parts.append(month_df.loc[test_idx])

        train_df = pd.concat(train_parts, ignore_index=False).sort_values(time_col)
        test_df = (
            pd.concat(test_parts, ignore_index=False).sort_values(time_col)
            if test_parts
            else train_df.iloc[0:0]
        )

        train_df = train_df.drop(columns=["__split_month", "__month"], errors="ignore")
        test_df = test_df.drop(columns=["__split_month", "__month"], errors="ignore")

        return train_df, test_df

    def _chronological_split(self, month_df: pd.DataFrame, test_ratio: float):
        k = max(1, int(len(month_df) * test_ratio))
        k = min(k, len(month_df) - 1) if len(month_df) > 1 else 0
        if k == 0:
            return month_df.index, month_df.iloc[0:0].index
        test_idx = month_df.index[-k:]
        train_idx = month_df.index[:-k]
        return train_idx, test_idx

    # ------------------------------------------------------------------
    # Existing utilities
    # ------------------------------------------------------------------
    def _adjust_to_target_rate(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        current_rate = df[self.config.target_col].mean() if len(df) else 0.0
        if abs(current_rate - target_rate) < 0.001 or len(df) == 0:
            return df

        defaults = df[df[self.config.target_col] == 1]
        non_defaults = df[df[self.config.target_col] == 0]

        n_total = len(df)
        n_defaults_needed = int(round(n_total * target_rate))
        n_non_defaults_needed = n_total - n_defaults_needed

        sampled_defaults = self._sample_with_replacement(
            defaults,
            n_defaults_needed,
            label="defaults"
        )
        sampled_non_defaults = self._sample_with_replacement(
            non_defaults,
            n_non_defaults_needed,
            label="non-defaults"
        )

        adjusted_df = pd.concat([sampled_defaults, sampled_non_defaults])
        adjusted_df = adjusted_df.sample(
            frac=1,
            random_state=getattr(self.config, 'random_state', None)
        ).reset_index(drop=True)
        return adjusted_df

    def _sample_with_replacement(self, df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=df.columns)
        replace = len(df) < n
        return df.sample(
            n=n,
            replace=replace,
            random_state=getattr(self.config, 'random_state', None)
        )

    def _calculate_statistics(self, splits: Dict[str, pd.DataFrame]):
        for split_name, split_df in splits.items():
            if split_df is not None and len(split_df) > 0:
                self.split_stats_[split_name] = {
                    'n_samples': len(split_df),
                    'n_defaults': split_df[self.config.target_col].sum(),
                    'n_non_defaults': len(split_df) - split_df[self.config.target_col].sum(),
                    'default_rate': split_df[self.config.target_col].mean(),
                    'n_features': len(split_df.columns) - 3  # rough indicator excluding id/time/target
                }

                if self.config.time_col and self.config.time_col in split_df.columns:
                    self.split_stats_[split_name]['date_min'] = split_df[self.config.time_col].min()
                    self.split_stats_[split_name]['date_max'] = split_df[self.config.time_col].max()

    def _verify_equal_rates(self, splits: Dict[str, pd.DataFrame]):
        rates = [
            split_df[self.config.target_col].mean()
            for split_df in splits.values()
            if split_df is not None and len(split_df) > 0
        ]

        if len(rates) > 1:
            rate_std = float(np.std(rates))
            rate_mean = float(np.mean(rates))

            if rate_std < 0.001:
                print(f"      [OK] Equal default rates achieved (mean: {rate_mean:.2%}, std: {rate_std:.4f})")
            else:
                print(f"      [WARNING] Default rates not perfectly equal (mean: {rate_mean:.2%}, std: {rate_std:.4f})")
                for split_name, stats in self.split_stats_.items():
                    print(f"        {split_name}: {stats['default_rate']:.2%}")

    def get_split_summary(self) -> pd.DataFrame:
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


def month_floor(ts) -> pd.Timestamp:
    try:
        if ts is None or pd.isna(ts):
            return pd.NaT
        if getattr(ts, "tzinfo", None):
            ts = ts.tz_localize(None)
        return pd.Timestamp(ts).to_period("M").to_timestamp()
    except Exception:
        ts2 = pd.to_datetime(ts, errors="coerce")
        return ts2.to_period("M").to_timestamp()


