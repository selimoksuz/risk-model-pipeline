"""Data processing module for the pipeline"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime
from .utils import Timer, month_floor, safe_print

class DataProcessor:
    """Handles data validation, splitting, and preprocessing"""
    
    def __init__(self, config):
        self.cfg = config
        self.var_catalog_ = None
        
    def validate_and_freeze(self, df: pd.DataFrame):
        """Validate input data and freeze time column"""
        # Check required columns
        for c in [self.cfg.id_col, self.cfg.time_col, self.cfg.target_col]:
            if c not in df.columns:
                raise ValueError(f"Zorunlu kolon eksik: {c}")
        
        # Validate target values
        target_values = set(pd.Series(df[self.cfg.target_col]).dropna().unique())
        if not target_values.issubset({0, 1}):
            raise ValueError("target_col yalniz {0,1} olmali.")
        
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
    
    def split_time(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Split data based on time for train/test/OOT"""
        # Use fallback implementation for compatibility
        # The stages module has different parameters
        df_sorted = df.sort_values(self.cfg.time_col)
        n = len(df_sorted)
        
        # Determine OOT size
        oot_window_months = getattr(self.cfg, 'oot_window_months', 2)
        if oot_window_months:
            latest = df_sorted[self.cfg.time_col].max()
            oot_start = latest - pd.DateOffset(months=oot_window_months)
            oot_mask = df_sorted[self.cfg.time_col] >= oot_start
            oot_size = oot_mask.sum()
        else:
            oot_size_percent = getattr(self.cfg, 'oot_size_percent', 10)
            oot_size = int(n * oot_size_percent / 100)
        
        min_oot_size = getattr(self.cfg, 'min_oot_size', 50)
        oot_size = max(oot_size, min_oot_size)
        oot_size = min(oot_size, n // 3)  # Max 33% for OOT
        
        # Split indices
        train_test_size = n - oot_size
        
        use_test_split = getattr(self.cfg, 'use_test_split', True)
        if use_test_split:
            test_size_row_frac = getattr(self.cfg, 'test_size_row_frac', 0.2)
            test_size = int(train_test_size * test_size_row_frac)
            train_size = train_test_size - test_size
            
            train_idx = df_sorted.index[:train_size].values
            test_idx = df_sorted.index[train_size:train_size + test_size].values
            oot_idx = df_sorted.index[train_size + test_size:].values
        else:
            train_idx = df_sorted.index[:train_test_size].values
            test_idx = None
            oot_idx = df_sorted.index[train_test_size:].values
        
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
        fit: bool = False,
        imputer=None,
        scaler=None
    ):
        """Apply imputation and outlier handling for raw variables"""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        
        X_transformed = X.copy()
        
        # Imputation
        if fit:
            imputer = SimpleImputer(strategy=self.cfg.raw_imputation_strategy)
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