"""Data splitting module for train/test/OOT splits"""

import pandas as pd
from typing import Dict, Optional


class DataSplitter:
    """Handles data splitting strategies"""
    
    def __init__(self, config):
        self.config = config
        
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