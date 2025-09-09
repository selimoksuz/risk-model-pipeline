"""Data processing module for the pipeline"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class DataProcessor:
    """Handles data validation, splitting, and preprocessing"""

    def __init__(self, config):
        self.cfg = config
        self.var_catalog_ = None
        self.imputation_stats_ = {}  # Store imputation statistics

    def validate_and_freeze(self, df: pd.DataFrame):
        """Validate input data and freeze time column"""
        # Check required columns
        for c in [self.cfg.id_col, self.cfg.time_col, self.cfg.target_col]:
            if c not in df.columns:
                raise ValueError(f"Zorunlu kolon eksik: {c}")

        # Validate target values
        target_values = set(pd.Series(df[self.cfg.target_col]).dropna().unique())
        if not target_values.issubset({0, 1}):
            raise ValueError("target_col yalniz {0, 1} olmali.")

        # Create snapshot_month
        try:
            df["snapshot_month"] = pd.to_datetime(
                df[self.cfg.time_col], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        except Exception:
            df["snapshot_month"] = df[self.cfg.time_col].apply(month_floor)

    def downcast_inplace(self, df: pd.DataFrame):
        """Downcast numeric types to save memory"""
        for c in df.columns:
            s = df[c]
            try:
                if pd.api.types.is_integer_dtype(s):
                    df[c] = pd.to_numeric(s, downcast="integer")
                elif pd.api.types.is_float_dtype(s):
                    df[c] = pd.to_numeric(s, downcast="float")
            except Exception:
                pass

    def classify_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify variables as numeric or categorical"""
        try:
            from ..stages import classify_variables
            return classify_variables(
                df,
                id_col=self.cfg.id_col,
                time_col=self.cfg.time_col,
                target_col=self.cfg.target_col
            )
        except ImportError:
            # Fallback implementation
            rows = []
            exclude = [self.cfg.id_col, self.cfg.time_col, self.cfg.target_col, "snapshot_month"]

            for col in df.columns:
                if col in exclude:
                    continue

                dtype_name = str(df[col].dtype)
                unique_ratio = df[col].nunique() / len(df[col])

                if "int" in dtype_name or "float" in dtype_name:
                    var_group = "numeric"
                elif unique_ratio < 0.05:  # Less than 5% unique values
                    var_group = "categorical"
                else:
                    var_group = "categorical" if df[col].nunique() < 20 else "numeric"

                rows.append({
                    "variable": col,
                    "dtype": dtype_name,
                    "variable_group": var_group,
                    "nunique": df[col].nunique(),
                    "missing_pct": df[col].isnull().mean()
                })

            self.var_catalog_ = pd.DataFrame(rows)
            return self.var_catalog_

    def impute_missing_values(self, X: pd.DataFrame, y: Optional[np.ndarray] = None,
                              strategy: str = 'multiple', fit: bool = True) -> pd.DataFrame:
        """Apply multiple imputation strategies for missing values

        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : np.ndarray, optional
            Target variable (for target-based imputation)
        strategy : str
            Imputation strategy: 'multiple', 'median', 'mean', 'mode', 'forward_fill',
            'interpolate', 'target_mean', 'knn'
        fit : bool
            Whether to fit imputation statistics (True for training, False for test/OOT)

        Returns:
        --------
        pd.DataFrame : Imputed dataframe
        """
        X_imputed = X.copy()

        if strategy == 'multiple':
            # Apply multiple strategies and create new features
            strategies = ['median', 'mean', 'forward_fill', 'target_mean']

            for col in X.columns:
                if X[col].isnull().any():
                    # Create multiple imputed versions
                    for strat in strategies:
                        if strat == 'target_mean' and y is None:
                            continue

                        imputed_col = self._impute_column(X[col], y, strat, col, fit)

                        # Add as new feature for ensemble
                        if strat != 'median':  # Use median as base
                            X_imputed[f"{col}_imp_{strat}"] = imputed_col
                        else:
                            X_imputed[col] = imputed_col

                    # Add missing indicator
                    X_imputed[f"{col}_was_missing"] = X[col].isnull().astype(int)
        else:
            # Single strategy imputation
            for col in X.columns:
                if X[col].isnull().any():
                    X_imputed[col] = self._impute_column(X[col], y, strategy, col, fit)

                    # Always add missing indicator for important tracking
                    if strategy != 'drop':
                        X_imputed[f"{col}_was_missing"] = X[col].isnull().astype(int)

        return X_imputed

    def _impute_column(self, series: pd.Series, y: Optional[np.ndarray],
                       strategy: str, col_name: str, fit: bool) -> pd.Series:
        """Impute a single column with specified strategy"""

        if strategy == 'drop':
            # Never drop rows - fill with median instead
            strategy = 'median'

        if fit:
            # Calculate and store imputation value
            if strategy == 'median':
                fill_value = series.median()
            elif strategy == 'mean':
                fill_value = series.mean()
            elif strategy == 'mode':
                mode_vals = series.mode()
                fill_value = mode_vals[0] if len(mode_vals) > 0 else series.median()
            elif strategy == 'forward_fill':
                # For time series data
                return series.fillna(method='ffill').fillna(series.median())
            elif strategy == 'interpolate':
                # Linear interpolation
                return series.interpolate(method='linear').fillna(series.median())
            elif strategy == 'target_mean' and y is not None:
                # Impute with mean value for target = 1 vs target = 0
                mask_1 = (y == 1) & series.notna()
                mask_0 = (y == 0) & series.notna()

                mean_1 = series[mask_1].mean() if mask_1.any() else series.mean()
                mean_0 = series[mask_0].mean() if mask_0.any() else series.mean()

                self.imputation_stats_[f"{col_name}_target"] = {'mean_1': mean_1, 'mean_0': mean_0}

                # Apply target-based imputation
                result = series.copy()
                missing_mask = series.isnull()
                if y is not None and missing_mask.any():
                    result[missing_mask & (y == 1)] = mean_1
                    result[missing_mask & (y == 0)] = mean_0
                    result[missing_mask & pd.isnull(y)] = series.mean()
                return result
            elif strategy == 'knn':
                # Simple distance-based imputation (would need full implementation)
                fill_value = series.median()  # Fallback to median
            else:
                fill_value = series.median()  # Default fallback

            # Store imputation value for later use
            if strategy not in ['forward_fill', 'interpolate', 'target_mean']:
                self.imputation_stats_[f"{col_name}_{strategy}"] = fill_value
        else:
            # Use stored imputation value
            if strategy == 'target_mean' and f"{col_name}_target" in self.imputation_stats_:
                stats = self.imputation_stats_[f"{col_name}_target"]
                result = series.copy()
                missing_mask = series.isnull()
                if y is not None and missing_mask.any():
                    result[missing_mask & (y == 1)] = stats['mean_1']
                    result[missing_mask & (y == 0)] = stats['mean_0']
                    result[missing_mask & pd.isnull(y)] = (stats['mean_1'] + stats['mean_0']) / 2
                return result
            elif f"{col_name}_{strategy}" in self.imputation_stats_:
                fill_value = self.imputation_stats_[f"{col_name}_{strategy}"]
            else:
                # Fallback if no stored value
                fill_value = series.median()

        return series.fillna(fill_value)

    def split_time(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Split data: OOT by time, Train/Test by stratified sampling"""
        df_sorted = df.sort_values(self.cfg.time_col)
        n = len(df_sorted)

        # Step 1: Split OOT data based on time
        oot_window_months = getattr(self.cfg, 'oot_months', None)
        if oot_window_months:
            # Use last N months as OOT
            latest = pd.to_datetime(df_sorted[self.cfg.time_col]).max()
            oot_start = latest - pd.DateOffset(months=oot_window_months)
            oot_mask = pd.to_datetime(df_sorted[self.cfg.time_col]) >= oot_start
            oot_idx = df_sorted[oot_mask].index.values
            pre_oot_df = df_sorted[~oot_mask]
        else:
            # Use ratio-based splitting for OOT
            oot_ratio = getattr(self.cfg, 'oot_ratio', 0.20)
            oot_size = int(n * oot_ratio)
            oot_size = max(oot_size, getattr(self.cfg, 'min_oot_size', 50))
            oot_size = min(oot_size, n // 3)  # Max 33% for OOT

            oot_idx = df_sorted.index[-oot_size:].values
            pre_oot_df = df_sorted.iloc[:-oot_size]

        # Step 2: Split Train/Test from pre-OOT data
        use_test_split = getattr(self.cfg, 'use_test_split', True)

        if not use_test_split:
            # No test split - all pre-OOT data goes to train
            train_idx = pre_oot_df.index.values
            test_idx = None
        else:
            # Always use stratified split for train/test
            test_ratio = getattr(self.cfg, 'test_ratio', 0.20)

            # Stratified split maintaining target ratio across all months
            from sklearn.model_selection import train_test_split

            # First, ensure each month has representation in both train and test
            pre_oot_df['_month'] = pd.to_datetime(pre_oot_df[self.cfg.time_col]).dt.to_period('M')

            train_idx_list = []
            test_idx_list = []

            for month, month_group in pre_oot_df.groupby('_month'):
                month_indices = month_group.index.values

                # Skip if month has too few samples
                if len(month_group) < 2:
                    train_idx_list.extend(month_indices)
                    continue

                # Calculate test size for this month
                month_test_size = max(1, int(len(month_group) * test_ratio))

                # Ensure we have enough samples for stratification
                target_values = month_group[self.cfg.target_col].values
                unique_targets = np.unique(target_values)

                if len(unique_targets) > 1 and min(np.bincount(target_values.astype(int))) >= 2:
                    # Can do stratified split
                    try:
                        month_train_idx, month_test_idx = train_test_split(
                            month_indices,
                            test_size=month_test_size,
                            stratify=target_values,
                            random_state=self.cfg.random_state
                        )
                    except ValueError:
                        # Fallback to random if stratification fails
                        np.random.seed(self.cfg.random_state)
                        np.random.shuffle(month_indices)
                        month_test_idx = month_indices[:month_test_size]
                        month_train_idx = month_indices[month_test_size:]
                else:
                    # Random split if can't stratify
                    np.random.seed(self.cfg.random_state)
                    np.random.shuffle(month_indices)
                    month_test_idx = month_indices[:month_test_size]
                    month_train_idx = month_indices[month_test_size:]

                train_idx_list.extend(month_train_idx)
                test_idx_list.extend(month_test_idx)

            train_idx = np.array(train_idx_list)
            test_idx = np.array(test_idx_list) if test_idx_list else None

            # Clean up temporary column
            if '_month' in pre_oot_df.columns:
                pre_oot_df.drop('_month', axis=1, inplace=True)

        # Log split statistics
        if hasattr(self, '_log'):
            total = len(df)
            n_train = len(train_idx) if train_idx is not None else 0
            n_test = len(test_idx) if test_idx is not None else 0
            n_oot = len(oot_idx) if oot_idx is not None else 0

            self._log(f"   - Data split: Train={n_train} ({n_train/total:.1%}), "
                      f"Test={n_test} ({n_test/total:.1%}), "
                      f"OOT={n_oot} ({n_oot/total:.1%})")

            # Check target distribution
            if self.cfg.target_col in df.columns:
                train_target_rate = df.iloc[train_idx][self.cfg.target_col].mean()
                oot_target_rate = df.iloc[oot_idx][self.cfg.target_col].mean()
                self._log(f"   - Target rates: Train={train_target_rate:.2%}, ", end="")

                if test_idx is not None:
                    test_target_rate = df.iloc[test_idx][self.cfg.target_col].mean()
                    self._log(f"Test={test_target_rate:.2%}, ", end="")

                self._log(f"OOT={oot_target_rate:.2%}")

        return train_idx, test_idx, oot_idx

    def get_train_test_oot_data(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: Optional[np.ndarray],
        oot_idx: np.ndarray
    ) -> Tuple:
        """Extract train/test/OOT data"""
        X_train = df.iloc[train_idx].drop(columns=[self.cfg.target_col])
        y_train = df.iloc[train_idx][self.cfg.target_col].values

        if test_idx is not None and len(test_idx) > 0:
            X_test = df.iloc[test_idx].drop(columns=[self.cfg.target_col])
            y_test = df.iloc[test_idx][self.cfg.target_col].values
        else:
            X_test = None
            y_test = None

        X_oot = df.iloc[oot_idx].drop(columns=[self.cfg.target_col])
        y_oot = df.iloc[oot_idx][self.cfg.target_col].values

        return X_train, y_train, X_test, y_test, X_oot, y_oot

    def apply_raw_transformations(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        fit: bool = False,
        imputer=None,
        scaler=None
    ):
        """Apply imputation and outlier handling for raw variables"""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler

        X_transformed = X.copy()

        # Get imputation strategy
        raw_imputation_strategy = getattr(self.cfg, 'raw_imputation_strategy', 'median')

        # Use multiple imputation if specified
        if raw_imputation_strategy in ['multiple', 'all']:
            # Use our custom multiple imputation
            X_transformed = self.impute_missing_values(
                X_transformed,
                y=y,
                strategy='multiple',
                fit=fit
            )
            imputer = None  # Don't use sklearn imputer
        elif raw_imputation_strategy in ['median', 'mean', 'mode', 'forward_fill',
                                         'interpolate', 'target_mean', 'knn']:
            # Use custom single strategy imputation
            X_transformed = self.impute_missing_values(
                X_transformed,
                y=y,
                strategy=raw_imputation_strategy,
                fit=fit
            )
            imputer = None
        else:
            # Fallback to sklearn SimpleImputer for backward compatibility
            if fit:
                imputer = SimpleImputer(strategy=raw_imputation_strategy)
                X_transformed = pd.DataFrame(
                    imputer.fit_transform(X_transformed),
                    columns=X_transformed.columns,
                    index=X_transformed.index
                )
            elif imputer is not None:
                X_transformed = pd.DataFrame(
                    imputer.transform(X_transformed),
                    columns=X_transformed.columns,
                    index=X_transformed.index
                )

        # Outlier handling
        if self.cfg.raw_outlier_method and self.cfg.raw_outlier_method != "none":
            if fit:
                scaler = RobustScaler()
                X_transformed = pd.DataFrame(
                    scaler.fit_transform(X_transformed),
                    columns=X_transformed.columns,
                    index=X_transformed.index
                )
            elif scaler is not None:
                X_transformed = pd.DataFrame(
                    scaler.transform(X_transformed),
                    columns=X_transformed.columns,
                    index=X_transformed.index
                )

            # Apply outlier clipping
            if self.cfg.raw_outlier_method == "iqr":
                threshold = self.cfg.raw_outlier_threshold
                X_transformed = X_transformed.clip(lower=-threshold, upper=threshold)
            elif self.cfg.raw_outlier_method == "zscore":
                threshold = self.cfg.raw_outlier_threshold
                X_transformed = X_transformed.clip(lower=-threshold, upper=threshold)

        return X_transformed, imputer, scaler
