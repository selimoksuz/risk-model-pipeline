"""Feature engineering module for WOE, PSI, and feature selection"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from .utils import (
    VariableWOE, NumericBin, CategoricalGroup,
    compute_woe_iv, jeffreys_counts, ks_statistic, safe_print
)

class FeatureEngineer:
    """Handles WOE transformation, PSI calculation, and feature selection"""
    
    def __init__(self, config):
        self.cfg = config
        self.woe_map = {}
        self.psi_summary_ = None
        self.psi_dropped_ = None
        self.high_iv_flags_ = []
        self.iv_filter_log_ = []
        self.corr_dropped_ = []
        
    def fit_woe_mapping(
        self, 
        train_df: pd.DataFrame, 
        var_catalog: pd.DataFrame,
        policy: Dict[str, Any]
    ) -> Dict[str, VariableWOE]:
        """Fit WOE mapping on training data"""
        mapping = {}
        target_col = self.cfg.target_col
        
        for _, row in var_catalog.iterrows():
            var = row["variable"]
            var_type = row["variable_group"]
            
            if var == target_col:
                continue
                
            x = train_df[var]
            y = train_df[target_col].values  # Convert to numpy array
            
            # Remove missing values for fitting
            mask = x.notna()
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) == 0:
                continue
            
            vw = VariableWOE(variable=var, var_type=var_type)
            
            if var_type == "numeric":
                # Fit numeric WOE bins
                vw.numeric_bins = self._bin_numeric_adaptive(
                    x_clean, y_clean,
                    min_bins=getattr(self.cfg, 'min_bins_numeric', 3),
                    min_count_auto=20,
                    min_share=0.02,
                    max_share=0.98,
                    alpha=getattr(self.cfg, 'jeffreys_alpha', 0.5),
                    max_abs_woe=getattr(self.cfg, 'max_abs_woe', None),
                    monotonic=getattr(self.cfg, 'woe_monotonic', False)
                )
            else:
                # Fit categorical WOE groups
                vw.categorical_groups = self._group_categorical_adaptive(
                    x_clean, y_clean,
                    rare_threshold=getattr(self.cfg, 'rare_threshold', 0.01),
                    missing_label="<MISSING>",
                    other_label="<OTHER>",
                    alpha=getattr(self.cfg, 'jeffreys_alpha', 0.5),
                    max_abs_woe=getattr(self.cfg, 'max_abs_woe', 4.0)
                )
            
            # Calculate IV
            vw.iv = self._calculate_iv(vw, x_clean, y_clean)
            
            # Handle missing values
            missing_mask = x.isna()
            if missing_mask.any():
                event_missing = y[missing_mask].sum()
                nonevent_missing = (~y[missing_mask]).sum()
                total_event = y.sum()
                total_nonevent = (~y).sum()
                
                woe_missing, rate_missing, _ = compute_woe_iv(
                    event_missing, nonevent_missing,
                    total_event, total_nonevent,
                    getattr(self.cfg, 'jeffreys_alpha', 0.5)
                )
                vw.missing_woe = woe_missing
                vw.missing_rate = missing_mask.mean()
            
            mapping[var] = vw
        
        self.woe_map = mapping
        return mapping
    
    def _bin_numeric_adaptive(
        self, x, y, min_bins, min_count_auto, 
        min_share, max_share, alpha, max_abs_woe, monotonic
    ) -> List[NumericBin]:
        """Create adaptive WOE bins for numeric variable"""
        # Simple quantile-based binning
        n_bins = max(min_bins, min(10, len(x) // 100))
        
        try:
            # Get quantiles
            quantiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(x, quantiles)
            edges = np.unique(edges)  # Remove duplicates
            
            if len(edges) < 2:
                # Single bin if no variation
                return [NumericBin(left=-np.inf, right=np.inf, woe=0.0)]
            
            # Create bins
            bins = []
            total_event = y.sum()
            total_nonevent = len(y) - total_event
            
            for i in range(len(edges) - 1):
                left = edges[i] if i > 0 else -np.inf
                right = edges[i + 1] if i < len(edges) - 2 else np.inf
                
                # Get data in bin
                if i == 0:
                    mask = x <= edges[i + 1]
                elif i == len(edges) - 2:
                    mask = x > edges[i]
                else:
                    mask = (x > edges[i]) & (x <= edges[i + 1])
                
                event = y[mask].sum()
                nonevent = (~y[mask]).sum()
                
                # Calculate WOE
                woe, rate, iv_contrib = compute_woe_iv(
                    event, nonevent, total_event, total_nonevent, alpha
                )
                
                # Clip WOE if needed
                if max_abs_woe and abs(woe) > max_abs_woe:
                    woe = np.sign(woe) * max_abs_woe
                
                bin_obj = NumericBin(
                    left=float(left),
                    right=float(right),
                    woe=float(woe),
                    event_count=int(event),
                    nonevent_count=int(nonevent),
                    total_count=int(event + nonevent),
                    event_rate=float(rate),
                    iv_contrib=float(iv_contrib)
                )
                bins.append(bin_obj)
            
            return bins
            
        except Exception:
            # Fallback to single bin
            return [NumericBin(left=-np.inf, right=np.inf, woe=0.0)]
    
    def _group_categorical_adaptive(
        self, x, y, rare_threshold, missing_label, 
        other_label, alpha, max_abs_woe
    ) -> List[CategoricalGroup]:
        """Create adaptive WOE groups for categorical variable"""
        groups = []
        total_event = y.sum()
        total_nonevent = len(y) - total_event
        
        # Get value counts
        value_counts = x.value_counts()
        total_count = len(x)
        
        # Group rare categories
        rare_cats = []
        for cat, count in value_counts.items():
            if count / total_count < rare_threshold:
                rare_cats.append(cat)
        
        # Process each category
        processed_cats = set()
        
        for cat in value_counts.index:
            if cat in rare_cats:
                continue
                
            mask = x == cat
            event = y[mask].sum()
            nonevent = (~y[mask]).sum()
            
            # Calculate WOE
            woe, rate, iv_contrib = compute_woe_iv(
                event, nonevent, total_event, total_nonevent, alpha
            )
            
            # Clip WOE if needed
            if max_abs_woe and abs(woe) > max_abs_woe:
                woe = np.sign(woe) * max_abs_woe
            
            group = CategoricalGroup(
                label=str(cat),
                members=[cat],
                woe=float(woe),
                event_count=int(event),
                nonevent_count=int(nonevent),
                total_count=int(event + nonevent),
                event_rate=float(rate),
                iv_contrib=float(iv_contrib)
            )
            groups.append(group)
            processed_cats.add(cat)
        
        # Handle rare categories as OTHER
        if rare_cats:
            mask = x.isin(rare_cats)
            event = y[mask].sum()
            nonevent = (~y[mask]).sum()
            
            woe, rate, iv_contrib = compute_woe_iv(
                event, nonevent, total_event, total_nonevent, alpha
            )
            
            if max_abs_woe and abs(woe) > max_abs_woe:
                woe = np.sign(woe) * max_abs_woe
            
            group = CategoricalGroup(
                label=other_label,
                members=rare_cats,
                woe=float(woe),
                event_count=int(event),
                nonevent_count=int(nonevent),
                total_count=int(event + nonevent),
                event_rate=float(rate),
                iv_contrib=float(iv_contrib)
            )
            groups.append(group)
        
        return groups
    
    def _calculate_iv(self, vw: VariableWOE, x, y) -> float:
        """Calculate Information Value for a variable"""
        iv = 0.0
        
        if vw.numeric_bins:
            for bin_obj in vw.numeric_bins:
                iv += abs(bin_obj.iv_contrib) if hasattr(bin_obj, 'iv_contrib') else 0
        elif vw.categorical_groups:
            for group in vw.categorical_groups:
                iv += abs(group.iv_contrib) if hasattr(group, 'iv_contrib') else 0
        
        return iv
    
    def apply_woe_transform(
        self, 
        df: pd.DataFrame, 
        variables: List[str]
    ) -> pd.DataFrame:
        """Apply WOE transformation to dataframe"""
        result = pd.DataFrame(index=df.index)
        
        for var in variables:
            if var not in self.woe_map:
                continue
                
            vw = self.woe_map[var]
            x = df[var] if var in df.columns else pd.Series(index=df.index)
            
            # Initialize with 0.0 (will be overwritten for valid values)
            woe_values = pd.Series(0.0, index=df.index)
            
            if vw.var_type == "numeric" and vw.numeric_bins:
                # Apply numeric bins
                x_numeric = pd.to_numeric(x, errors='coerce')
                
                # Find missing WOE if exists
                missing_woe = 0.0
                for bin_obj in vw.numeric_bins:
                    # Check if this is the missing bin (both left and right are NaN)
                    if hasattr(bin_obj, 'left') and hasattr(bin_obj, 'right'):
                        if pd.isna(bin_obj.left) and pd.isna(bin_obj.right):
                            missing_woe = bin_obj.woe
                            break
                
                # Apply missing WOE
                missing_mask = pd.isna(x_numeric)
                woe_values[missing_mask] = missing_woe
                
                # Apply WOE for non-missing values
                for bin_obj in vw.numeric_bins:
                    # Skip missing bin
                    if hasattr(bin_obj, 'left') and hasattr(bin_obj, 'right'):
                        if pd.isna(bin_obj.left) and pd.isna(bin_obj.right):
                            continue
                    
                    if bin_obj.left == -np.inf:
                        mask = (~missing_mask) & (x_numeric <= bin_obj.right)
                    elif bin_obj.right == np.inf:
                        mask = (~missing_mask) & (x_numeric > bin_obj.left)
                    else:
                        mask = (~missing_mask) & (x_numeric > bin_obj.left) & (x_numeric <= bin_obj.right)
                    
                    woe_values[mask] = bin_obj.woe
                    
            elif vw.var_type == "categorical" and vw.categorical_groups:
                # Find special groups
                other_woe = 0.0
                missing_woe = 0.0
                
                for group in vw.categorical_groups:
                    if hasattr(group, 'label'):
                        if group.label == "OTHER":
                            other_woe = group.woe
                        elif group.label == "MISSING":
                            missing_woe = group.woe
                
                # Apply missing WOE
                missing_mask = pd.isna(x)
                woe_values[missing_mask] = missing_woe
                
                # Set default for unseen categories (OTHER)
                woe_values[~missing_mask] = other_woe
                
                # Apply categorical groups for known values
                for group in vw.categorical_groups:
                    if hasattr(group, 'label') and group.label in ["OTHER", "MISSING"]:
                        continue
                    if hasattr(group, 'members') and group.members:
                        mask = x.isin(group.members)
                        woe_values[mask] = group.woe
            
            result[var] = woe_values
        
        return result
    
    def calculate_psi(
        self,
        base_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        variables: List[str]
    ) -> pd.DataFrame:
        """Calculate PSI between base and comparison datasets"""
        psi_results = []
        
        for var in variables:
            if var not in self.woe_map:
                continue
                
            vw = self.woe_map[var]
            
            # Get WOE values
            base_woe = self.apply_woe_transform(base_df, [var])[var]
            compare_woe = self.apply_woe_transform(compare_df, [var])[var]
            
            # Calculate PSI
            base_dist = base_woe.value_counts(normalize=True)
            compare_dist = compare_woe.value_counts(normalize=True)
            
            psi = 0.0
            for woe_val in set(base_dist.index) | set(compare_dist.index):
                base_prop = base_dist.get(woe_val, 1e-6)
                compare_prop = compare_dist.get(woe_val, 1e-6)
                
                psi += (compare_prop - base_prop) * np.log(compare_prop / base_prop)
            
            psi_threshold = getattr(self.cfg, 'psi_threshold', 0.25)
            psi_results.append({
                'variable': var,
                'PSI': psi,
                'status': 'DROP' if psi > psi_threshold else 'KEEP'
            })
        
        return pd.DataFrame(psi_results)
    
    def correlation_clustering(
        self, 
        X: pd.DataFrame, 
        threshold: float = 0.95
    ) -> List[str]:
        """Perform correlation clustering to remove redundant features"""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Drop features with correlation above threshold
        to_drop = []
        for column in upper.columns:
            if column in to_drop:
                continue
            correlated = list(upper.index[upper[column] > threshold])
            to_drop.extend(correlated)
        
        keep_vars = [c for c in X.columns if c not in to_drop]
        
        # Store dropped variables info
        self.corr_dropped_ = []
        for dropped in to_drop:
            if dropped in X.columns:
                # Find which variable it's correlated with
                for kept in keep_vars:
                    if kept in corr_matrix.columns and dropped in corr_matrix.index:
                        corr_val = corr_matrix.loc[dropped, kept]
                        if corr_val > threshold:
                            self.corr_dropped_.append({
                                "variable": dropped,
                                "kept_with": kept,
                                "rho": float(corr_val)
                            })
                            break
        
        return keep_vars
    
    def forward_selection(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        max_features: int = 20,
        cv_folds: int = 5
    ) -> List[str]:
        """Forward feature selection with cross-validation"""
        selected = []
        remaining = list(X.columns)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.cfg.random_state)
        
        def cv_score(cols):
            if not cols:
                return 0.0
            scores = []
            for train_idx, val_idx in skf.split(X[cols], y):
                model = LogisticRegression(
                    penalty="l2", 
                    solver="lbfgs",
                    max_iter=300,
                    class_weight="balanced",
                    random_state=self.cfg.random_state
                )
                model.fit(X.iloc[train_idx][cols], y[train_idx])
                pred = model.predict_proba(X.iloc[val_idx][cols])[:, 1]
                
                try:
                    auc = roc_auc_score(y[val_idx], pred)
                    scores.append(auc)
                except Exception:
                    scores.append(0.5)
            
            return np.mean(scores)
        
        # Forward selection
        while remaining and len(selected) < max_features:
            best_score = -np.inf
            best_var = None
            
            for var in remaining:
                score = cv_score(selected + [var])
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var is None:
                break
                
            selected.append(best_var)
            remaining.remove(best_var)
            
            # Early stopping if score doesn't improve much
            if len(selected) > 3:
                prev_score = cv_score(selected[:-1])
                if best_score - prev_score < 0.001:
                    break
        
        return selected
    
    def boruta_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_iter: int = 100,
        use_lightgbm: bool = True
    ) -> List[str]:
        """Boruta feature selection with LightGBM or RandomForest"""
        try:
            from boruta import BorutaPy
            from lightgbm import LGBMClassifier
            
            # Use LightGBM by default for better performance
            if use_lightgbm:
                estimator = LGBMClassifier(
                    n_estimators=100,
                    random_state=getattr(self.cfg, 'random_state', 42),
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    max_depth=5,
                    learning_rate=0.1,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbosity=-1  # Suppress LightGBM warnings
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    random_state=getattr(self.cfg, 'random_state', 42),
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    max_depth=5
                )
            
            boruta = BorutaPy(
                estimator,
                n_estimators='auto',
                max_iter=max_iter,
                random_state=getattr(self.cfg, 'random_state', 42),
                verbose=0  # Suppress Boruta output
            )
            
            # Ensure X is numeric
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] == 0:
                return list(X.columns)[:min(20, X.shape[1])]
            
            boruta.fit(X_numeric.values, y)
            
            selected = X_numeric.columns[boruta.support_].tolist()
            tentative = X_numeric.columns[boruta.support_weak_].tolist()
            
            # Return at least some features if Boruta selected too few
            result = selected + tentative
            if len(result) < 5 and X_numeric.shape[1] >= 5:
                # Add top features by univariate score
                from sklearn.feature_selection import f_classif
                scores, _ = f_classif(X_numeric, y)
                top_indices = np.argsort(scores)[-5:]
                result = list(set(result + X_numeric.columns[top_indices].tolist()))
            
            return result
            
        except ImportError:
            print("   - Boruta not installed, using univariate selection")
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
            selector.fit(X, y)
            return X.columns[selector.get_support()].tolist()
        except Exception as e:
            print(f"   - Boruta failed: {e}, using fallback")
            # Fallback to simple univariate selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
            selector.fit(X, y)
            return X.columns[selector.get_support()].tolist()