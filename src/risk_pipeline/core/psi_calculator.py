"""PSI calculation module for variables and scores"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class PSICalculator:
    """Handles PSI calculations for WOE variables and model scores"""
    
    def __init__(self):
        self.psi_results = {}
        
    def calculate_woe_psi(self, train_woe: pd.DataFrame, test_woe: pd.DataFrame, 
                         woe_mapping: Dict) -> Dict:
        """Calculate PSI for WOE-transformed variables using WOE bins"""
        
        psi_results = {}
        
        for var_name, var_mapping in woe_mapping.items():
            if var_name not in train_woe.columns or var_name not in test_woe.columns:
                continue
                
            # Get unique WOE values and their bins
            if hasattr(var_mapping, 'numeric_bins'):
                # For numeric variables, use WOE bin ranges
                psi_value, bin_details = self._calculate_numeric_woe_psi(
                    train_woe[var_name], test_woe[var_name], var_mapping.numeric_bins
                )
            elif hasattr(var_mapping, 'categorical_groups'):
                # For categorical variables, use WOE groups
                psi_value, bin_details = self._calculate_categorical_woe_psi(
                    train_woe[var_name], test_woe[var_name], var_mapping.categorical_groups
                )
            else:
                continue
                
            psi_results[var_name] = {
                'psi_value': psi_value,
                'bin_details': bin_details,
                'interpretation': self._interpret_psi(psi_value)
            }
            
        return psi_results
    
    def _calculate_numeric_woe_psi(self, train_values: pd.Series, test_values: pd.Series, 
                                   numeric_bins: List) -> Tuple[float, pd.DataFrame]:
        """Calculate PSI for numeric WOE variable"""
        
        bin_details = []
        total_psi = 0
        
        # Get unique WOE values from bins
        woe_values = {}
        for bin_info in numeric_bins:
            woe_values[bin_info.woe] = {
                'range': f"[{bin_info.left:.2f}, {bin_info.right:.2f}]",
                'woe': bin_info.woe
            }
        
        # Calculate distribution for each WOE value
        train_dist = train_values.value_counts(normalize=True)
        test_dist = test_values.value_counts(normalize=True)
        
        for woe_val, info in woe_values.items():
            train_pct = train_dist.get(woe_val, 0.0001)
            test_pct = test_dist.get(woe_val, 0.0001)
            
            # PSI calculation
            psi_contrib = (test_pct - train_pct) * np.log(test_pct / train_pct)
            total_psi += psi_contrib
            
            bin_details.append({
                'bin_range': info['range'],
                'woe_value': woe_val,
                'train_pct': train_pct * 100,
                'test_pct': test_pct * 100,
                'difference': (test_pct - train_pct) * 100,
                'psi_contribution': psi_contrib
            })
        
        return total_psi, pd.DataFrame(bin_details)
    
    def _calculate_categorical_woe_psi(self, train_values: pd.Series, test_values: pd.Series,
                                       categorical_groups: List) -> Tuple[float, pd.DataFrame]:
        """Calculate PSI for categorical WOE variable"""
        
        bin_details = []
        total_psi = 0
        
        # Get unique WOE values from groups
        woe_values = {}
        for group in categorical_groups:
            woe_values[group.woe] = {
                'label': group.label,
                'woe': group.woe
            }
        
        # Calculate distribution
        train_dist = train_values.value_counts(normalize=True)
        test_dist = test_values.value_counts(normalize=True)
        
        for woe_val, info in woe_values.items():
            train_pct = train_dist.get(woe_val, 0.0001)
            test_pct = test_dist.get(woe_val, 0.0001)
            
            # PSI calculation
            psi_contrib = (test_pct - train_pct) * np.log(test_pct / train_pct)
            total_psi += psi_contrib
            
            bin_details.append({
                'group': info['label'],
                'woe_value': woe_val,
                'train_pct': train_pct * 100,
                'test_pct': test_pct * 100,
                'difference': (test_pct - train_pct) * 100,
                'psi_contribution': psi_contrib
            })
        
        return total_psi, pd.DataFrame(bin_details)
    
    def calculate_score_psi(self, train_scores: np.ndarray, test_scores: np.ndarray,
                           n_bins: int = 10, method: str = 'quantile') -> Tuple[float, pd.DataFrame]:
        """Calculate PSI for model scores using deciles or equal width bins"""
        
        # Create bins
        if method == 'quantile':
            # Decile-based (10 equal population bins)
            _, bin_edges = pd.qcut(train_scores, q=n_bins, retbins=True, duplicates='drop')
        else:
            # Equal width bins
            _, bin_edges = pd.cut(train_scores, bins=n_bins, retbins=True)
        
        # Ensure full coverage
        bin_edges[0] = 0
        bin_edges[-1] = 1
        
        # Bin both distributions
        train_binned = pd.cut(train_scores, bins=bin_edges, include_lowest=True)
        test_binned = pd.cut(test_scores, bins=bin_edges, include_lowest=True)
        
        # Calculate distributions
        train_dist = train_binned.value_counts(normalize=True).sort_index()
        test_dist = test_binned.value_counts(normalize=True).sort_index()
        
        # Calculate PSI
        bin_details = []
        total_psi = 0
        
        for i, bin_range in enumerate(train_dist.index):
            train_pct = train_dist.iloc[i] if i < len(train_dist) else 0.0001
            test_pct = test_dist.iloc[i] if i < len(test_dist) else 0.0001
            
            # Avoid log(0)
            train_pct = max(train_pct, 0.0001)
            test_pct = max(test_pct, 0.0001)
            
            psi_contrib = (test_pct - train_pct) * np.log(test_pct / train_pct)
            total_psi += psi_contrib
            
            # Get scores in this bin
            train_mask = train_binned == bin_range
            test_mask = test_binned == bin_range
            
            bin_details.append({
                'decile': i + 1,
                'score_range': str(bin_range),
                'train_pct': train_pct * 100,
                'test_pct': test_pct * 100,
                'train_count': train_mask.sum(),
                'test_count': test_mask.sum(),
                'difference': (test_pct - train_pct) * 100,
                'psi_contribution': psi_contrib,
                'train_avg_score': train_scores[train_mask].mean() if train_mask.any() else 0,
                'test_avg_score': test_scores[test_mask].mean() if test_mask.any() else 0
            })
        
        bin_df = pd.DataFrame(bin_details)
        bin_df['cumulative_psi'] = bin_df['psi_contribution'].cumsum()
        
        return total_psi, bin_df
    
    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret PSI value"""
        if psi_value < 0.1:
            return "Insignificant change (PSI < 0.1) - No action required"
        elif psi_value < 0.25:
            return "Small change (0.1 ≤ PSI < 0.25) - Minor shift, investigate"
        else:
            return "Significant change (PSI ≥ 0.25) - Major shift, model retraining recommended"