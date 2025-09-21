"""
Advanced Feature Selector with all selection methods
PSI -> VIF -> Correlation -> IV -> Boruta -> Forward/Backward/Stepwise
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_object_dtype
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureSelector:
    """
    Complete feature selection with all methods:
    - PSI (Population Stability Index)
    - VIF (Variance Inflation Factor)
    - Correlation clustering
    - IV (Information Value)
    - Boruta with LightGBM
    - Forward/Backward/Stepwise selection
    """

    def __init__(self, config):
        self.config = config
        self.selection_history_ = []

    def _prepare_features_for_boruta(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical/datetime columns into numeric form for Boruta."""

        X_prepared = pd.DataFrame(index=X.index)

        for col in X.columns:
            series = X[col]

            if is_datetime64_any_dtype(series):
                X_prepared[col] = pd.to_datetime(series).view("int64")
            elif is_object_dtype(series) or is_categorical_dtype(series):
                cat = pd.Categorical(series)
                codes = cat.codes.astype(float)
                codes[codes == -1] = float("nan")
                X_prepared[col] = codes
            else:
                if hasattr(series, "dtype") and series.dtype.kind in "biu":
                    X_prepared[col] = series.astype(float)
                else:
                    X_prepared[col] = series

        return X_prepared

    def select_by_psi(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      threshold: float = 0.25) -> List[str]:
        """Select features with stable PSI."""

        stable_features = []

        for col in X_train.columns:
            psi = self._calculate_psi(X_train[col], X_test[col])

            if psi < threshold:
                stable_features.append(col)
            else:
                print(f"    Removing {col}: PSI={psi:.3f}")

        return stable_features

    def select_by_vif(self, X: pd.DataFrame, threshold: float = 10) -> List[str]:
        """Select features with low multicollinearity."""

        features = list(X.columns)
        removed = set()

        while True:
            # Calculate VIF for remaining features
            vif_data = pd.DataFrame()
            vif_data["Feature"] = features
            vif_data["VIF"] = [self._calculate_vif(X[features], i)
                              for i in range(len(features))]

            # Find max VIF
            max_vif = vif_data["VIF"].max()

            if max_vif < threshold:
                break

            # Remove feature with highest VIF
            worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            features.remove(worst_feature)
            removed.add(worst_feature)
            print(f"    Removing {worst_feature}: VIF={max_vif:.2f}")

        return features

    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series,
                             threshold: float = 0.9) -> List[str]:
        """Select best features from correlated clusters."""

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to remove
        to_remove = set()

        for column in upper_tri.columns:
            # Find correlated features
            correlated = list(upper_tri.index[upper_tri[column] > threshold])

            if correlated and column not in to_remove:
                # Keep the one with highest correlation with target
                cluster = [column] + correlated
                target_corrs = {feat: abs(X[feat].corr(y)) for feat in cluster}
                best = max(target_corrs, key=target_corrs.get)

                # Remove others
                for feat in cluster:
                    if feat != best:
                        to_remove.add(feat)
                        print(f"    Removing {feat}: Correlated with {best}")

        return [col for col in X.columns if col not in to_remove]

    def select_by_iv(self, woe_values: Dict, features: List[str],
                     threshold: float = 0.02) -> List[str]:
        """Select features with high Information Value."""

        selected = []

        for feat in features:
            if feat in woe_values:
                if isinstance(woe_values[feat], dict) and 'iv' in woe_values[feat]:
                    iv = woe_values[feat]['iv']
                    if iv >= threshold:
                        selected.append(feat)
                    else:
                        print(f"    Removing {feat}: IV={iv:.4f}")
                else:
                    # Debug: Show what we have
                    if isinstance(woe_values[feat], dict):
                        print(f"    WARNING: {feat} has no 'iv' key. Keys: {list(woe_values[feat].keys())[:5]}")
                    selected.append(feat)
            else:
                # Keep features without WOE info
                selected.append(feat)

        return selected

    def select_by_boruta_lgbm(self, X: pd.DataFrame, y: pd.Series,
                             n_iterations: int = 100) -> List[str]:
        """
        Boruta feature selection using LightGBM.
        """
        # Check if we have any features
        if X.shape[1] == 0:
            print("    Boruta skipped: No features to select from")
            return []

        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            print("    LightGBM not available, using RandomForest for Boruta")
            return self.select_by_boruta_rf(X, y, n_iterations)

        # Ensure data types are LightGBM-friendly
        X_prepared = self._prepare_features_for_boruta(X)

        # Create shadow features
        X_shadow = X_prepared.apply(np.random.permutation)
        X_shadow.columns = ['shadow_' + col for col in X_prepared.columns]

        # Combine original and shadow
        X_combined = pd.concat([X_prepared, X_shadow], axis=1)

        # Track hits
        hits = {col: 0 for col in X.columns}
        shadow_max_importance = []

        for i in range(n_iterations):
            # Train LightGBM
            lgb = LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                verbosity=-1,
                random_state=i
            )
            lgb.fit(X_combined, y)

            # Get importances
            importances = lgb.feature_importances_
            original_length = len(X_prepared.columns)
            original_imp = importances[:original_length]
            shadow_imp = importances[original_length:]

            # Max shadow importance
            max_shadow = shadow_imp.max()
            shadow_max_importance.append(max_shadow)

            # Count hits (feature more important than max shadow)
            for j, col in enumerate(X.columns):
                if original_imp[j] > max_shadow:
                    hits[col] += 1

            # Reshuffle shadow features
            X_shadow = X_prepared.apply(np.random.permutation)
            X_combined.iloc[:, len(X_prepared.columns):] = X_shadow.values

        # Select features with significant hits
        # Using binomial test logic
        threshold = self._binomial_threshold(n_iterations, 0.05)
        selected = [col for col, hit_count in hits.items()
                   if hit_count >= threshold]

        print(f"    Boruta selected {len(selected)} features from {len(X_prepared.columns)}")

        return selected

    def select_by_boruta_rf(self, X: pd.DataFrame, y: pd.Series,
                            n_iterations: int = 100) -> List[str]:
        """Fallback Boruta with RandomForest."""

        # Check if we have any features
        if X.shape[1] == 0:
            print("    Boruta RF skipped: No features to select from")
            return []

        from sklearn.ensemble import RandomForestClassifier

        # Apply the same preparation for RandomForest fallback
        X_prepared = self._prepare_features_for_boruta(X)

        # Similar logic as LGBM but with RF
        X_shadow = X_prepared.apply(np.random.permutation)
        X_shadow.columns = ['shadow_' + col for col in X_prepared.columns]
        X_combined = pd.concat([X_prepared, X_shadow], axis=1)

        hits = {col: 0 for col in X.columns}

        for i in range(n_iterations):
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=i
            )
            rf.fit(X_combined, y)

            importances = rf.feature_importances_
            original_length = len(X_prepared.columns)
            original_imp = importances[:original_length]
            shadow_imp = importances[original_length:]
            max_shadow = shadow_imp.max()

            for j, col in enumerate(X.columns):
                if original_imp[j] > max_shadow:
                    hits[col] += 1

            X_shadow = X.apply(np.random.permutation)
            X_combined.iloc[:, len(X.columns):] = X_shadow.values

        threshold = self._binomial_threshold(n_iterations, 0.05)
        selected = [col for col, hit_count in hits.items()
                   if hit_count >= threshold]

        return selected

    def forward_selection(self, X: pd.DataFrame, y: pd.Series,
                         max_features: int = 15, X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None) -> List[str]:
        """
        Forward feature selection with validation set for scoring.
        """
        # Check if we have any features
        if X.shape[1] == 0:
            print("      Forward selection skipped: No features to select from")
            return []

        # Use validation set if provided, otherwise split training data
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y

        features = list(X.columns)
        selected = []
        remaining = features.copy()

        print("    Starting Forward Selection...")

        while len(selected) < max_features and remaining:
            best_score = -np.inf
            best_feature = None

            for feature in remaining:
                # Try adding this feature
                current_features = selected + [feature]

                # Train model on training set and evaluate on validation set
                try:
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X_train[current_features], y_train)
                    y_pred = model.predict_proba(X_val[current_features])[:, 1]
                    score = roc_auc_score(y_val, y_pred)

                    if score > best_score:
                        best_score = score
                        best_feature = feature
                except:
                    continue

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                print(f"      Added {best_feature}: AUC={best_score:.4f}")
            else:
                break

        return selected

    def backward_selection(self, X: pd.DataFrame, y: pd.Series,
                          min_features: int = 5) -> List[str]:
        """
        Backward feature elimination.
        """
        # Check if we have any features
        if X.shape[1] == 0:
            print("      Backward selection skipped: No features to select from")
            return []

        features = list(X.columns)
        selected = features.copy()

        print("    Starting Backward Selection...")

        while len(selected) > min_features:
            worst_score = np.inf
            worst_feature = None

            for feature in selected:
                # Try removing this feature
                current_features = [f for f in selected if f != feature]

                if not current_features:
                    continue

                # Train model
                try:
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X[current_features], y)
                    y_pred = model.predict_proba(X[current_features])[:, 1]
                    score = roc_auc_score(y, y_pred)

                    # We want to remove the feature whose removal gives best score
                    if score > worst_score or worst_score == np.inf:
                        worst_score = score
                        worst_feature = feature
                except:
                    continue

            if worst_feature:
                selected.remove(worst_feature)
                print(f"      Removed {worst_feature}: New AUC={worst_score:.4f}")
            else:
                break

        return selected

    def stepwise_selection(self, X: pd.DataFrame, y: pd.Series,
                          max_features: int = 15) -> List[str]:
        """
        Stepwise selection (combination of forward and backward).
        """
        # Check if we have any features
        if X.shape[1] == 0:
            print("      Stepwise selection skipped: No features to select from")
            return []

        features = list(X.columns)
        selected = []
        remaining = features.copy()

        print("    Starting Stepwise Selection...")

        while len(selected) < max_features:
            # Forward step
            best_score = -np.inf
            best_feature = None

            for feature in remaining:
                current_features = selected + [feature]

                try:
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X[current_features], y)
                    y_pred = model.predict_proba(X[current_features])[:, 1]
                    score = roc_auc_score(y, y_pred)

                    if score > best_score:
                        best_score = score
                        best_feature = feature
                except:
                    continue

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                print(f"      Added {best_feature}: AUC={best_score:.4f}")

                # Backward step (only if we have more than 2 features)
                if len(selected) > 2:
                    worst_score = -np.inf
                    worst_feature = None

                    for feature in selected[:-1]:  # Don't remove the just-added feature
                        current_features = [f for f in selected if f != feature]

                        try:
                            model = LogisticRegression(max_iter=1000, solver='liblinear')
                            model.fit(X[current_features], y)
                            y_pred = model.predict_proba(X[current_features])[:, 1]
                            score = roc_auc_score(y, y_pred)

                            if score > worst_score:
                                worst_score = score
                                worst_feature = feature
                        except:
                            continue

                    # Remove if improvement is significant
                    if worst_feature and worst_score > best_score * 0.995:  # 0.5% tolerance
                        selected.remove(worst_feature)
                        remaining.append(worst_feature)
                        print(f"      Removed {worst_feature}: New AUC={worst_score:.4f}")
            else:
                break

        return selected

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series,
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index."""

        # Handle missing values
        expected = expected.fillna(expected.median() if expected.dtype in ['float64', 'int64'] else 'MISSING')
        actual = actual.fillna(actual.median() if actual.dtype in ['float64', 'int64'] else 'MISSING')

        # Create bins
        if expected.dtype in ['float64', 'int64']:
            # Numeric variable
            try:
                _, bins = pd.qcut(expected, buckets, retbins=True, duplicates='drop')
                expected_groups = pd.cut(expected, bins, include_lowest=True)
                actual_groups = pd.cut(actual, bins, include_lowest=True)
            except:
                return 0
        else:
            # Categorical variable
            expected_groups = expected
            actual_groups = actual

        # Calculate PSI
        psi_values = []

        for group in expected_groups.unique():
            expected_pct = (expected_groups == group).mean()
            actual_pct = (actual_groups == group).mean()

            # Avoid log(0)
            expected_pct = max(expected_pct, 0.0001)
            actual_pct = max(actual_pct, 0.0001)

            psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi_values.append(psi)

        return sum(psi_values)

    def _calculate_vif(self, X: pd.DataFrame, feature_idx: int) -> float:
        """Calculate VIF for a feature."""
        try:
            return variance_inflation_factor(X.values, feature_idx)
        except Exception:
            return 1.0

    def _binomial_threshold(self, n_trials: int, alpha: float = 0.05) -> int:
        """Calculate threshold for binomial test."""
        from scipy import stats
        # Approximate using normal distribution
        p = 0.5  # Null hypothesis: feature is as good as random
        mean = n_trials * p
        std = np.sqrt(n_trials * p * (1 - p))
        # One-tailed test
        threshold = stats.norm.ppf(1 - alpha, mean, std)
        return int(threshold)
