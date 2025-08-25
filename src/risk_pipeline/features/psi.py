import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Calculate PSI between two series. For WOE values, use unique values directly.
    For raw values, use quantile binning.
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Check if values look like WOE (limited unique values, often negative)
    exp_unique = expected.nunique()
    act_unique = actual.nunique()
    
    if exp_unique <= 20 and act_unique <= 20:
        # Likely WOE values - use unique values as bins
        all_values = sorted(set(expected.unique()) | set(actual.unique()))
        if len(all_values) <= 1:
            return 0.0
            
        # Create categorical bins based on unique WOE values
        expected_binned = pd.Categorical(expected, categories=all_values)
        actual_binned = pd.Categorical(actual, categories=all_values)
        
        e_counts = expected_binned.value_counts().sort_index()
        a_counts = actual_binned.value_counts().sort_index()
        
        # Manual normalization
        e_dist = e_counts / e_counts.sum()
        a_dist = a_counts / a_counts.sum()
    else:
        # Raw values - use quantile binning
        try:
            e = pd.qcut(expected.rank(method="first"), q=bins, duplicates="drop")
            a = pd.qcut(actual.rank(method="first"), q=bins, duplicates="drop")
            e_dist = e.value_counts(normalize=True).sort_index()
            a_dist = a.value_counts(normalize=True).sort_index()
        except ValueError:
            # Fallback if qcut fails
            return 0.0
    
    # Align indexes and calculate PSI
    idx = sorted(set(e_dist.index) | set(a_dist.index))
    e_dist = e_dist.reindex(idx, fill_value=1e-6)
    a_dist = a_dist.reindex(idx, fill_value=1e-6)
    
    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    diff = a_dist - e_dist
    ln = np.log(a_dist / e_dist)
    return float((diff * ln).sum())
