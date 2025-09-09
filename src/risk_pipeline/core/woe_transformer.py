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
        
        # Fit WOE for each feature
        for feature in features:
            # Simple WOE fitting - create bins and calculate WOE
            self.woe_mapping_[feature] = self._fit_woe_simple(
                train[feature], target
            )
        
        # Transform datasets - only keep WOE features and target/id columns
        keep_cols = list(self.woe_mapping_.keys()) + [self.config.target_col, self.config.id_col]
        
        result = {
            'train': self.transform(train, self.woe_mapping_)[keep_cols],
            'mapping': self.woe_mapping_
        }
        
        if test is not None:
            result['test'] = self.transform(test, self.woe_mapping_)[keep_cols]
        
        if oot is not None:
            result['oot'] = self.transform(oot, self.woe_mapping_)[keep_cols]
        
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
            
            # Handle values outside all bins (set to nearest bin's WOE)
            unmatched = woe_values.isna() & ~series.isna()
            if unmatched.any():
                # Use the WOE of the nearest bin
                for idx in series[unmatched].index:
                    val = series.loc[idx]
                    # Find nearest bin
                    min_dist = float('inf')
                    nearest_woe = 0
                    for bin_info in mapping.numeric_bins:
                        dist = min(abs(val - bin_info.left), abs(val - bin_info.right))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_woe = bin_info.woe
                    woe_values.loc[idx] = nearest_woe
            
        elif hasattr(mapping, 'categorical_groups'):
            # Categorical variable
            other_woe = 0
            for group in mapping.categorical_groups:
                if group.label == 'MISSING':
                    woe_values.loc[series.isna()] = group.woe
                elif group.label == 'OTHER':
                    other_woe = group.woe
                else:
                    mask = series.isin(group.members)
                    woe_values.loc[mask] = group.woe
            
            # Apply OTHER woe to unmatched values
            woe_values.loc[woe_values.isna() & ~series.isna()] = other_woe
        
        # Final safety check - fill any remaining NaNs with 0
        woe_values = woe_values.fillna(0)
        
        return woe_values
    
    def _fit_woe_simple(self, series: pd.Series, target: pd.Series) -> Dict:
        """Simple WOE fitting for a single feature"""
        from dataclasses import dataclass
        
        @dataclass
        class WOEBin:
            left: float
            right: float
            woe: float
            iv_contrib: float
            event_count: int
            nonevent_count: int
            event_rate: float
        
        @dataclass
        class WOEGroup:
            label: str
            members: list
            woe: float
            iv_contrib: float
            event_count: int
            nonevent_count: int
            event_rate: float
        
        @dataclass
        class VariableWOE:
            numeric_bins: list = None
            categorical_groups: list = None
            missing_woe: float = 0.0
            iv: float = 0.0
        
        result = VariableWOE()
        
        # Handle numeric variables
        if pd.api.types.is_numeric_dtype(series):
            # Create bins
            try:
                bins = pd.qcut(series.dropna(), q=min(10, len(series.dropna().unique())), duplicates='drop')
            except:
                bins = pd.cut(series.dropna(), bins=5)
            
            result.numeric_bins = []
            total_events = target.sum()
            total_nonevents = len(target) - total_events
            
            for interval in bins.cat.categories:
                mask = (series >= interval.left) & (series <= interval.right)
                events = target[mask].sum()
                nonevents = mask.sum() - events
                
                # Calculate WOE
                event_rate = (events + 0.5) / (total_events + 1)
                nonevent_rate = (nonevents + 0.5) / (total_nonevents + 1)
                woe = np.log(event_rate / nonevent_rate) if nonevent_rate > 0 else 0
                iv_contrib = (event_rate - nonevent_rate) * woe
                
                result.numeric_bins.append(WOEBin(
                    left=interval.left,
                    right=interval.right,
                    woe=woe,
                    iv_contrib=iv_contrib,
                    event_count=events,
                    nonevent_count=nonevents,
                    event_rate=events / (events + nonevents) if (events + nonevents) > 0 else 0
                ))
            
            # Handle missing
            if series.isna().any():
                missing_mask = series.isna()
                events = target[missing_mask].sum()
                nonevents = missing_mask.sum() - events
                event_rate = (events + 0.5) / (total_events + 1)
                nonevent_rate = (nonevents + 0.5) / (total_nonevents + 1)
                result.missing_woe = np.log(event_rate / nonevent_rate) if nonevent_rate > 0 else 0
            
        else:
            # Categorical variable
            result.categorical_groups = []
            total_events = target.sum()
            total_nonevents = len(target) - total_events
            
            for value in series.dropna().unique():
                mask = series == value
                events = target[mask].sum()
                nonevents = mask.sum() - events
                
                # Calculate WOE
                event_rate = (events + 0.5) / (total_events + 1)
                nonevent_rate = (nonevents + 0.5) / (total_nonevents + 1)
                woe = np.log(event_rate / nonevent_rate) if nonevent_rate > 0 else 0
                iv_contrib = (event_rate - nonevent_rate) * woe
                
                result.categorical_groups.append(WOEGroup(
                    label=str(value),
                    members=[value],
                    woe=woe,
                    iv_contrib=iv_contrib,
                    event_count=events,
                    nonevent_count=nonevents,
                    event_rate=events / (events + nonevents) if (events + nonevents) > 0 else 0
                ))
            
            # Handle missing
            if series.isna().any():
                missing_mask = series.isna()
                events = target[missing_mask].sum()
                nonevents = missing_mask.sum() - events
                event_rate = (events + 0.5) / (total_events + 1)
                nonevent_rate = (nonevents + 0.5) / (total_nonevents + 1)
                woe = np.log(event_rate / nonevent_rate) if nonevent_rate > 0 else 0
                
                result.categorical_groups.append(WOEGroup(
                    label='MISSING',
                    members=[],
                    woe=woe,
                    iv_contrib=(event_rate - nonevent_rate) * woe,
                    event_count=events,
                    nonevent_count=nonevents,
                    event_rate=events / (events + nonevents) if (events + nonevents) > 0 else 0
                ))
        
        # Calculate total IV
        if result.numeric_bins:
            result.iv = sum(abs(b.iv_contrib) for b in result.numeric_bins)
        elif result.categorical_groups:
            result.iv = sum(abs(g.iv_contrib) for g in result.categorical_groups)
        
        return result