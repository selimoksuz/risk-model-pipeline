"""
Complete Feature Selection Module
All selection methods: univariate, PSI, VIF, correlation, IV, Boruta, stepwise
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Comprehensive feature selection with all methods
    """
    
    def __init__(self, config):
        """Initialize feature selector with config"""
        self.config = config
        self.selection_history = {}
        self.feature_scores = {}
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        univariate_stats: Optional[Dict] = None,
        selection_steps: Optional[List[str]] = None
    ) -> List[str]:
        """
        Run complete feature selection pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        univariate_stats : Dict, optional
            Pre-calculated univariate statistics
        selection_steps : List[str], optional
            Selection steps to run (default from config)
            
        Returns:
        --------
        List[str]
            Selected features
        """
        selection_steps = selection_steps or self.config.selection_steps
        features = list(X.columns)
        initial_count = len(features)

        def _tsfresh_count(items):
            return sum(1 for name in items if str(name).endswith('_tsfresh'))

        def _log_tsfresh(step_name, before, after):
            if before != after:
                print(f"    tsfresh after {step_name}: {before} -> {after} (removed {before - after})")

        tsfresh_initial = _tsfresh_count(features)
        print(f"Starting feature selection with {initial_count} features ({tsfresh_initial} tsfresh)")

        for step in selection_steps:
            tsfresh_before = _tsfresh_count(features)
            if step == 'univariate' and univariate_stats:
                features = self._filter_univariate(features, univariate_stats)
                print(f"  After univariate filter: {len(features)} features")
                _log_tsfresh('univariate', tsfresh_before, _tsfresh_count(features))

            elif step == 'psi':
                pass

            elif step == 'vif':
                features = self._filter_vif(X[features])
                print(f"  After VIF filter: {len(features)} features")
                _log_tsfresh('vif', tsfresh_before, _tsfresh_count(features))

            elif step == 'correlation':
                features = self._filter_correlation(X[features], univariate_stats)
                print(f"  After correlation filter: {len(features)} features")
                _log_tsfresh('correlation', tsfresh_before, _tsfresh_count(features))

            elif step == 'iv' and univariate_stats:
                features = self._filter_iv(features, univariate_stats)
                print(f"  After IV filter: {len(features)} features")
                _log_tsfresh('iv', tsfresh_before, _tsfresh_count(features))

            elif step == 'boruta':
                features = self._run_boruta(X[features], y)
                print(f"  After Boruta: {len(features)} features")
                _log_tsfresh('boruta', tsfresh_before, _tsfresh_count(features))

            elif step == 'stepwise':
                method = self.config.stepwise_method
                features = self._run_stepwise(X[features], y, method)
                print(f"  After {method} selection: {len(features)} features")
                _log_tsfresh(method, tsfresh_before, _tsfresh_count(features))

        if self.config.use_noise_sentinel:
            print('  Noise sentinel analysis deferred to final model evaluation stage.')

        tsfresh_final = _tsfresh_count(features)
        print(f"Final selected features: {len(features)} (reduced from {initial_count})")
        if tsfresh_initial:
            print(f"  tsfresh summary: started {tsfresh_initial}, removed {tsfresh_initial - tsfresh_final}, remaining {tsfresh_final}")
        else:
            print("  tsfresh summary: no tsfresh features in input")
        return features
    
    def _filter_univariate(self, features: List[str], univariate_stats: Dict) -> List[str]:
        """Filter by univariate Gini"""
        selected = []
        
        for feature in features:
            if feature in univariate_stats:
                stats = univariate_stats[feature]
                # Use WOE gini if available, otherwise raw gini
                gini = stats.get('woe_gini', stats.get('raw_gini', 0))
                
                if abs(gini) >= self.config.min_univariate_gini:
                    selected.append(feature)
        
        return selected
    
    def _filter_vif(self, X: pd.DataFrame) -> List[str]:
        """Filter features by Variance Inflation Factor"""
        
        if len(X.columns) <= 1:
            return list(X.columns)
        
        # Calculate VIF for each feature
        vif_data = []
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) <= 1:
            return list(X.columns)
        
        for i in range(len(X_numeric.columns)):
            try:
                vif = variance_inflation_factor(X_numeric.values, i)
                vif_data.append({
                    'feature': X_numeric.columns[i],
                    'vif': vif
                })
            except:
                vif_data.append({
                    'feature': X_numeric.columns[i],
                    'vif': 1.0
                })
        
        vif_df = pd.DataFrame(vif_data)
        
        # Iteratively remove highest VIF until all below threshold
        features_to_keep = list(X.columns)
        
        while True:
            high_vif = vif_df[vif_df['vif'] > self.config.max_vif]
            
            if len(high_vif) == 0:
                break
            
            # Remove feature with highest VIF
            worst_feature = high_vif.nlargest(1, 'vif')['feature'].values[0]
            features_to_keep.remove(worst_feature)
            
            if len(features_to_keep) <= 1:
                break
            
            # Recalculate VIF
            X_temp = X[features_to_keep].select_dtypes(include=[np.number])
            
            if len(X_temp.columns) <= 1:
                break
            
            vif_data = []
            for i in range(len(X_temp.columns)):
                try:
                    vif = variance_inflation_factor(X_temp.values, i)
                    vif_data.append({
                        'feature': X_temp.columns[i],
                        'vif': vif
                    })
                except:
                    vif_data.append({
                        'feature': X_temp.columns[i],
                        'vif': 1.0
                    })
            
            vif_df = pd.DataFrame(vif_data)
        
        return features_to_keep
    
    def _filter_correlation(self, X: pd.DataFrame, univariate_stats: Optional[Dict]) -> List[str]:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs with high correlation
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            
            # Find correlated features
            correlated = list(upper_tri.index[upper_tri[column] > self.config.max_correlation])
            
            if correlated:
                # Keep the one with highest univariate score
                all_features = [column] + correlated
                
                if univariate_stats:
                    # Use IV or Gini to decide which to keep
                    scores = {}
                    for f in all_features:
                        if f in univariate_stats:
                            scores[f] = univariate_stats[f].get('iv', 
                                      univariate_stats[f].get('woe_gini', 0))
                        else:
                            scores[f] = 0
                    
                    # Keep best feature
                    best_feature = max(scores.items(), key=lambda x: abs(x[1]))[0]
                    for f in all_features:
                        if f != best_feature:
                            to_drop.add(f)
                else:
                    # Drop all but first
                    for f in correlated:
                        to_drop.add(f)
        
        return [f for f in X.columns if f not in to_drop]
    
    def _filter_iv(self, features: List[str], univariate_stats: Dict) -> List[str]:
        """Filter by Information Value"""
        selected = []
        
        for feature in features:
            if feature in univariate_stats:
                iv = univariate_stats[feature].get('iv', 0)
                
                if iv >= self.config.min_iv:
                    selected.append(feature)
        
        return selected
    
    def _run_boruta(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Boruta feature selection using LightGBM or RandomForest
        """
        from sklearn.utils import check_random_state
        
        np.random.seed(self.config.random_state)
        rng = check_random_state(self.config.random_state)
        
        # Create shadow features
        X_shadow = X.apply(rng.permutation)
        X_shadow.columns = ['shadow_' + str(col) for col in X.columns]
        
        # Combine original and shadow features
        X_combined = pd.concat([X, X_shadow], axis=1)
        
        # Train model
        if self.config.boruta_estimator == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                verbosity=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        # Run iterations
        feature_importance_history = []
        
        for i in range(min(self.config.boruta_max_iter, 100)):
            # Shuffle shadow features
            X_shadow = X.apply(rng.permutation)
            X_shadow.columns = ['shadow_' + str(col) for col in X.columns]
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # Fit model
            model.fit(X_combined, y)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.abs(model.coef_[0])
            
            feature_importance_history.append(importance)
        
        # Calculate statistics
        importance_df = pd.DataFrame(feature_importance_history, columns=X_combined.columns)
        
        # Get max shadow importance
        shadow_cols = [col for col in importance_df.columns if col.startswith('shadow_')]
        shadow_max = importance_df[shadow_cols].max(axis=1).max()
        
        # Select features that are consistently better than best shadow
        selected_features = []
        
        for col in X.columns:
            if col in importance_df.columns:
                # Use median importance
                median_importance = importance_df[col].median()
                
                if median_importance > shadow_max:
                    selected_features.append(col)
        
        # If too few features selected, use top features
        if len(selected_features) < self.config.stepwise_min_features:
            importance_median = importance_df[X.columns].median().sort_values(ascending=False)
            selected_features = list(importance_median.head(
                max(self.config.stepwise_min_features, 5)
            ).index)
        
        return selected_features
    
    def _run_stepwise(self, X: pd.DataFrame, y: pd.Series, method: str) -> List[str]:
        """
        Run stepwise selection (forward, backward, stepwise, forward_1se)
        """
        if method == 'forward':
            return self._forward_selection(X, y)
        elif method == 'backward':
            return self._backward_selection(X, y)
        elif method == 'stepwise':
            return self._stepwise_selection(X, y)
        elif method == 'forward_1se':
            return self._forward_selection_1se(X, y)
        else:
            print(f"Unknown stepwise method: {method}, using forward")
            return self._forward_selection(X, y)
    
    def _forward_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Forward feature selection"""
        
        selected = []
        remaining = list(X.columns)
        
        cv = StratifiedKFold(n_splits=self.config.stepwise_cv_folds, 
                             shuffle=True, random_state=self.config.random_state)
        
        best_score = 0
        
        while len(selected) < min(self.config.stepwise_max_features, len(X.columns)):
            scores = {}
            
            for feature in remaining:
                features_to_test = selected + [feature]
                X_subset = X[features_to_test]
                
                # Train simple model
                model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_subset, y, cv=cv, 
                                           scoring='roc_auc', n_jobs=-1)
                scores[feature] = cv_scores.mean()
            
            if not scores:
                break
            
            # Select best feature
            best_feature = max(scores.items(), key=lambda x: x[1])
            
            # Check if improvement is significant
            if best_feature[1] > best_score + 0.001:  # Min improvement threshold
                selected.append(best_feature[0])
                remaining.remove(best_feature[0])
                best_score = best_feature[1]
            else:
                break
            
            # Check minimum features
            if len(selected) >= self.config.stepwise_max_features:
                break
        
        return selected
    
    def _backward_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Backward feature selection"""
        
        selected = list(X.columns)
        
        cv = StratifiedKFold(n_splits=self.config.stepwise_cv_folds,
                             shuffle=True, random_state=self.config.random_state)
        
        while len(selected) > max(self.config.stepwise_min_features, 1):
            scores = {}
            
            for feature in selected:
                features_to_test = [f for f in selected if f != feature]
                
                if not features_to_test:
                    continue
                
                X_subset = X[features_to_test]
                
                # Train model
                model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_subset, y, cv=cv,
                                           scoring='roc_auc', n_jobs=-1)
                scores[feature] = cv_scores.mean()
            
            if not scores:
                break
            
            # Remove feature with highest score (least impact when removed)
            worst_feature = max(scores.items(), key=lambda x: x[1])
            
            # Check if we should stop
            if len(selected) <= self.config.stepwise_min_features:
                break
            
            selected.remove(worst_feature[0])
        
        return selected
    
    def _stepwise_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combined forward and backward selection"""
        
        # Start with forward selection
        selected = self._forward_selection(X, y)
        
        if len(selected) <= 2:
            return selected
        
        # Then apply backward elimination
        cv = StratifiedKFold(n_splits=self.config.stepwise_cv_folds,
                             shuffle=True, random_state=self.config.random_state)
        
        # Get baseline score
        model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        baseline_score = cross_val_score(model, X[selected], y, cv=cv,
                                        scoring='roc_auc', n_jobs=-1).mean()
        
        improved = True
        while improved and len(selected) > self.config.stepwise_min_features:
            improved = False
            scores = {}
            
            for feature in selected:
                features_to_test = [f for f in selected if f != feature]
                
                if not features_to_test:
                    continue
                
                X_subset = X[features_to_test]
                cv_scores = cross_val_score(model, X_subset, y, cv=cv,
                                           scoring='roc_auc', n_jobs=-1)
                scores[feature] = cv_scores.mean()
            
            if scores:
                best_removal = max(scores.items(), key=lambda x: x[1])
                
                if best_removal[1] > baseline_score - 0.001:  # Allow small degradation
                    selected.remove(best_removal[0])
                    baseline_score = best_removal[1]
                    improved = True
        
        return selected
    
    def _forward_selection_1se(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Forward selection with 1 standard error rule"""
        
        selected = []
        remaining = list(X.columns)
        
        cv = StratifiedKFold(n_splits=self.config.stepwise_cv_folds,
                             shuffle=True, random_state=self.config.random_state)
        
        best_score = 0
        best_se = 0
        
        while len(selected) < min(self.config.stepwise_max_features, len(X.columns)):
            scores = {}
            std_errors = {}
            
            for feature in remaining:
                features_to_test = selected + [feature]
                X_subset = X[features_to_test]
                
                model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
                cv_scores = cross_val_score(model, X_subset, y, cv=cv,
                                           scoring='roc_auc', n_jobs=-1)
                
                scores[feature] = cv_scores.mean()
                std_errors[feature] = cv_scores.std() / np.sqrt(len(cv_scores))
            
            if not scores:
                break
            
            # Find best feature
            best_feature = max(scores.items(), key=lambda x: x[1])
            
            # Apply 1SE rule: accept if within 1 SE of best
            threshold = best_score - best_se
            
            if best_feature[1] > threshold:
                selected.append(best_feature[0])
                remaining.remove(best_feature[0])
                best_score = best_feature[1]
                best_se = std_errors[best_feature[0]]
            else:
                break
            
            if len(selected) >= self.config.stepwise_max_features:
                break
        
        return selected
    
    def _noise_sentinel_check(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Check features against noise sentinels"""
        
        # Add random noise features
        n_noise = min(5, max(1, len(X.columns) // 10))
        noise_features = []
        
        X_with_noise = X.copy()
        
        for i in range(n_noise):
            noise_name = f'noise_sentinel_{i}'
            X_with_noise[noise_name] = np.random.randn(len(X))
            noise_features.append(noise_name)
        
        # Train model with all features including noise
        if self.config.boruta_estimator == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                verbosity=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        model.fit(X_with_noise, y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, 
                                 index=X_with_noise.columns)
        else:
            importance = pd.Series(np.abs(model.coef_[0]), 
                                 index=X_with_noise.columns)
        
        # Get noise threshold (max importance of noise features)
        noise_importance = importance[noise_features]
        threshold = noise_importance.quantile(self.config.noise_threshold)
        
        # Select features above noise threshold
        real_features = [f for f in X.columns if f not in noise_features]
        selected = [f for f in real_features if importance[f] > threshold]
        
        # Ensure minimum features
        if len(selected) < self.config.stepwise_min_features:
            feature_importance = importance[real_features].sort_values(ascending=False)
            selected = list(feature_importance.head(
                self.config.stepwise_min_features
            ).index)
        
        return selected
