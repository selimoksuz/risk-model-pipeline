"""Utility classes and functions for the pipeline"""

import os
import sys
import time
import psutil
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from datetime import datetime
import json

# Encoding utilities
def safe_print(msg, file=None):
    try:
        if file:
            file.write(msg + "\n")
            file.flush()
        else:
            print(msg)
    except Exception:
        try:
            encoded_msg = msg.encode("ascii", errors="replace").decode("ascii")
            if file:
                file.write(encoded_msg + "\n")
                file.flush()
            else:
                print(encoded_msg)
        except Exception:
            pass

def now_str() -> str:
    """Return current time as string"""
    return datetime.now().strftime("%H:%M:%S")

def sys_metrics() -> str:
    """Return system metrics as string"""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        return f" | CPU={int(cpu)}% RAM={int(mem)}%"
    except Exception:
        return ""

def month_floor(date_value) -> datetime:
    """Floor date to month start"""
    try:
        if hasattr(date_value, "year") and hasattr(date_value, "month"):
            return datetime(date_value.year, date_value.month, 1)
        return datetime(2020, 1, 1)
    except Exception:
        return datetime(2020, 1, 1)

def gini_from_auc(auc: float) -> float:
    """Convert AUC to Gini coefficient"""
    return 2 * auc - 1

def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray):
    """Calculate KS statistic"""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    sorted_idx = np.argsort(y_proba)
    y_true_sorted = y_true[sorted_idx]
    
    n_total = len(y_true)
    n_positives = np.sum(y_true)
    n_negatives = n_total - n_positives
    
    if n_positives == 0 or n_negatives == 0:
        return 0.0, 0.5
    
    tpr = np.cumsum(y_true_sorted) / n_positives
    fpr = np.cumsum(1 - y_true_sorted) / n_negatives
    
    ks_values = np.abs(tpr - fpr)
    max_ks_idx = np.argmax(ks_values)
    max_ks = ks_values[max_ks_idx]
    threshold = y_proba[sorted_idx[max_ks_idx]]
    
    return float(max_ks), float(threshold)

def ks_table(y_true: np.ndarray, y_proba: np.ndarray, n_bands: int = 10) -> Dict:
    """Generate KS table"""
    # Implementation for KS table generation
    return {}

def jeffreys_counts(total_event: int, total_nonevent: int, alpha: float = 0.5):
    """Jeffreys smoothing for counts"""
    te_s = total_event + alpha
    tne_s = total_nonevent + alpha
    return te_s, tne_s

def compute_woe_iv(event: int, nonevent: int, total_event: int, total_nonevent: int, alpha: float = 0.5):
    """Compute WOE and IV for a bin/group"""
    # Calculate actual event rate (without smoothing)
    if (event + nonevent) > 0:
        actual_rate = event / (event + nonevent)
    else:
        actual_rate = 0.0
    
    # Apply smoothing for WOE calculation
    e_s = event + alpha
    ne_s = nonevent + alpha
    te_s, tne_s = jeffreys_counts(total_event, total_nonevent, alpha)
    
    # Calculate WOE using smoothed values
    de = e_s/te_s
    dne = ne_s/tne_s
    
    # Prevent division by zero and extreme WOE values
    if dne <= 0:
        woe = 5.0  # Cap at maximum WOE
    elif de <= 0:
        woe = -5.0  # Cap at minimum WOE
    else:
        woe = float(np.log(de/dne))
        # Cap WOE values to prevent extreme values
        woe = max(-5.0, min(5.0, woe))
    
    # IV contribution
    iv_contrib = float((de - dne) * woe)
    
    return woe, float(actual_rate), iv_contrib

# Timer classes
class Timer:
    """Context manager for timing operations"""
    def __init__(self, label: str, logger=print):
        self.label, self.logger, self.t0 = label, logger, None
        
    def __enter__(self):
        self.t0 = time.time()
        self.logger(f"[{now_str()}] >> {self.label} starting{sys_metrics()}")
        
    def __exit__(self, exc_type, exc, tb):
        ok_fail = " — FAIL" if exc_type else " — OK"
        if exc_type:
            self.logger(f"[{now_str()}] â--  {self.label} completed ({time.time()-self.t0:.2f}s){ok_fail}: {exc}{sys_metrics()}")
        else:
            self.logger(f"[{now_str()}] â--  {self.label} completed ({time.time()-self.t0:.2f}s){ok_fail}{sys_metrics()}")

class Timer2:
    """Alternative timer implementation"""
    def __init__(self, logger=print):
        self.logger = logger
        
    def __call__(self, label: str):
        return Timer(label, self.logger)

# Data structures for WOE
@dataclass
class NumericBin:
    """Numeric bin for WOE transformation"""
    left: float
    right: float
    woe: float
    event_count: int = 0
    nonevent_count: int = 0
    total_count: int = 0
    event_rate: float = 0.0
    iv_contrib: float = 0.0

@dataclass
class CategoricalGroup:
    """Categorical group for WOE transformation"""
    label: str
    members: List[Any]
    woe: float
    event_count: int = 0
    nonevent_count: int = 0
    total_count: int = 0
    event_rate: float = 0.0
    iv_contrib: float = 0.0

@dataclass
class VariableWOE:
    """WOE mapping for a variable"""
    variable: str
    var_type: str  # "numeric" or "categorical"
    iv: float = 0.0
    numeric_bins: Optional[List[NumericBin]] = None
    categorical_groups: Optional[List[CategoricalGroup]] = None
    missing_woe: Optional[float] = None
    missing_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = {
            "variable": self.variable,
            "var_type": self.var_type,
            "iv": self.iv,
            "missing_woe": self.missing_woe,
            "missing_rate": self.missing_rate
        }
        
        if self.numeric_bins:
            result["numeric_bins"] = [
                {
                    "left": b.left,
                    "right": b.right,
                    "woe": b.woe,
                    "event_count": b.event_count,
                    "nonevent_count": b.nonevent_count,
                    "event_rate": b.event_rate,
                    "iv_contrib": b.iv_contrib
                }
                for b in self.numeric_bins
            ]
        
        if self.categorical_groups:
            result["categorical_groups"] = [
                {
                    "label": g.label,
                    "members": g.members,
                    "woe": g.woe,
                    "event_count": g.event_count,
                    "nonevent_count": g.nonevent_count,
                    "event_rate": g.event_rate,
                    "iv_contrib": g.iv_contrib
                }
                for g in self.categorical_groups
            ]
        
        return result