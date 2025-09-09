"""WOE transformation module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .feature_engineer import FeatureEngineer


class WOETransformer:
    """Handles WOE binning and transformation"""
    
    def __init__(self, config):
        self.config = config
        self.engineer = FeatureEngineer(config)
        self.woe_mapping_ = {}
        
    def fit_transform(self, train: pd.DataFrame, test: Optional[pd.DataFrame] = None,
                     oot: Optional[pd.DataFrame] = None, features: List[str] = None) -> Dict:
        """Fit WOE transformation on train and apply to all datasets"""
        
        if features is None:
            features = [col for col in train.columns 
                       if col not in [self.config.target_col, self.config.id_col, self.config.time_col]]
        
        print(f"Fitting WOE transformation for {len(features)} features...")
        
        # Fit WOE on training data
        target = train[self.config.target_col]
        self.woe_mapping_ = {}
        
        for feature in features:
            self.woe_mapping_[feature] = self.engineer.fit_woe(
                train[feature], target, is_numeric=train[feature].dtype in ['int64', 'float64']
            )
        
        # Transform datasets
        result = {
            'train': self.transform(train, self.woe_mapping_),
            'mapping': self.woe_mapping_
        }
        
        if test is not None:
            result['test'] = self.transform(test, self.woe_mapping_)
        
        if oot is not None:
            result['oot'] = self.transform(oot, self.woe_mapping_)
        
        return result
    
    def transform(self, df: pd.DataFrame, woe_mapping: Dict) -> pd.DataFrame:
        """Apply WOE transformation to dataframe"""
        
        df_woe = df.copy()
        
        for feature, mapping in woe_mapping.items():
            if feature in df.columns:
                df_woe[feature] = self._apply_woe_single(df[feature], mapping)
        
        return df_woe
    
    def _apply_woe_single(self, series: pd.Series, mapping) -> pd.Series:
        """Apply WOE transformation to a single column"""
        
        woe_values = pd.Series(index=series.index, dtype='float64')
        
        if hasattr(mapping, 'numeric_bins'):
            # Numeric variable
            for bin_info in mapping.numeric_bins:
                mask = (series >= bin_info.left) & (series <= bin_info.right)
                woe_values.loc[mask] = bin_info.woe
            
            # Handle missing
            woe_values.loc[series.isna()] = mapping.missing_woe if hasattr(mapping, 'missing_woe') else 0
            
        elif hasattr(mapping, 'categorical_groups'):
            # Categorical variable
            for group in mapping.categorical_groups:
                if group.label == 'MISSING':
                    woe_values.loc[series.isna()] = group.woe
                elif group.label == 'OTHER':
                    # Will be applied to unmatched values later
                    other_woe = group.woe
                else:
                    mask = series.isin(group.members)
                    woe_values.loc[mask] = group.woe
            
            # Apply OTHER woe to unmatched values
            woe_values.loc[woe_values.isna() & ~series.isna()] = other_woe if 'other_woe' in locals() else 0
        
        return woe_values