"""
Advanced Feature Selector with all selection methods
PSI -> VIF -> Correlation -> IV -> Boruta -> Forward/Backward/Stepwise
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_object_dtype
from typing import List, Dict, Optional, Tuple, Set, Any
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
        self.last_vif_summary_ = None
        self.last_correlation_clusters_ = None

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

    def select_by_psi(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        threshold: float = 0.25,
        oot_df: Optional[pd.DataFrame] = None,
        monthly_frames: Optional[Dict[str, pd.DataFrame]] = None,
        monthly_threshold: Optional[float] = None,
        oot_threshold: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Select features with stable PSI across test, OOT, and monthly splits."""

        if monthly_frames is None:
            monthly_frames = {}

        base_threshold = threshold if threshold is not None else 0.25
        oot_threshold = oot_threshold if oot_threshold is not None else base_threshold
        monthly_threshold = (
            monthly_threshold if monthly_threshold is not None else max(0.05, base_threshold / 2)
        )

        selected = []
        psi_details: Dict[str, Dict[str, Any]] = {}

        comparators: Dict[str, Tuple[pd.DataFrame, float]] = {}
        if X_test is not None and not X_test.empty:
            comparators['test'] = (X_test, base_threshold)
        if oot_df is not None and not oot_df.empty:
            comparators['oot'] = (oot_df, oot_threshold)

        valid_monthly = {
            str(label): frame
            for label, frame in monthly_frames.items()
            if frame is not None and not frame.empty
        }

        for col in X_train.columns:
            base_series = X_train[col].dropna()
            col_detail: Dict[str, Any] = {
                'status': 'kept',
                'comparisons': {},
                'drop_reasons': []
            }

            # Compare against test and OOT frames
            for name, (frame, cmp_threshold) in comparators.items():
                if col not in frame.columns:
                    continue
                psi_value = self._calculate_series_psi(base_series, frame[col].dropna())
                col_detail['comparisons'][name] = {
                    'psi': float(psi_value),
                    'threshold': float(cmp_threshold),
                }
                if psi_value > cmp_threshold:
                    col_detail['status'] = 'dropped'
                    col_detail['drop_reasons'].append(
                        f"{name} psi {psi_value:.3f} > {cmp_threshold:.3f}"
                    )

            # Compare against monthly slices
            monthly_results: Dict[str, float] = {}
            for label, frame in valid_monthly.items():
                if col not in frame.columns:
                    continue
                month_series = frame[col].dropna()
                psi_value = self._calculate_series_psi(base_series, month_series)
                monthly_results[label] = float(psi_value)
                if psi_value > monthly_threshold:
                    col_detail['status'] = 'dropped'
                    col_detail['drop_reasons'].append(
                        f"monthly({label}) psi {psi_value:.3f} > {monthly_threshold:.3f}"
                    )

            if monthly_results:
                col_detail['comparisons']['monthly'] = {
                    'psi_by_month': monthly_results,
                    'threshold': float(monthly_threshold),
                }

            if col_detail['status'] == 'kept':
                selected.append(col)
            else:
                reason_text = '; '.join(col_detail['drop_reasons'])
                if reason_text:
                    print(f"    Removing {col}: {reason_text}")

            psi_details[col] = col_detail

        return selected, psi_details

    def _calculate_series_psi(self, base: pd.Series, compare: pd.Series) -> float:
        """Calculate PSI between two series treating values as discrete buckets."""

        if base.empty or compare.empty:
            return 0.0

        eps = 1e-4
        base_dist = base.value_counts(normalize=True)
        compare_dist = compare.value_counts(normalize=True)
        categories = set(base_dist.index).union(compare_dist.index)

        psi = 0.0
        for val in categories:
            base_pct = float(base_dist.get(val, eps))
            compare_pct = float(compare_dist.get(val, eps))
            base_pct = max(base_pct, eps)
            compare_pct = max(compare_pct, eps)
            psi += (compare_pct - base_pct) * np.log(compare_pct / base_pct)

        return float(psi)


    def select_by_vif(self, X: pd.DataFrame, threshold: float = 10) -> List[str]:
        """Select features with low multicollinearity using a fast correlation inversion."""

        if X is None or X.empty:
            self.last_vif_summary_ = pd.DataFrame()
            return []

        working = X.apply(pd.to_numeric, errors="coerce")
        sample_size = getattr(self.config, "vif_sample_size", None)
        if sample_size and len(working) > sample_size:
            rng = getattr(self.config, "random_state", None)
            working = working.sample(sample_size, random_state=rng)
            print(f"    VIF subsample: using {len(working)} of {len(X)} rows")
        working = working.reset_index(drop=True).fillna(0.0)

        features = list(working.columns)
        if len(features) <= 1:
            self.last_vif_summary_ = pd.DataFrame({
                "feature": features,
                "vif": [1.0 for _ in features],
                "status": ["kept" for _ in features],
                "iteration": [0 for _ in features]
            })
            return features

        variances = working.var(axis=0)
        near_constant = variances[variances <= 1e-9].index.tolist()
        if near_constant:
            print(f"    Dropping {len(near_constant)} near-constant features before VIF")
        current = [f for f in features if f not in near_constant]

        latest_vif: Dict[str, float] = {}
        status: Dict[str, str] = {f: "candidate" for f in current}
        iteration_map: Dict[str, int] = {f: 0 for f in current}
        summary_rows: List[Dict[str, Any]] = []
        iteration = 0

        def _compute_vif_matrix(frame: pd.DataFrame) -> pd.Series:
            if frame.shape[1] == 1:
                return pd.Series([1.0], index=frame.columns, dtype=float)
            values = frame.to_numpy(dtype=float, copy=False)
            corr = np.corrcoef(values, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr += np.eye(corr.shape[0]) * 1e-6
            try:
                inv = np.linalg.inv(corr)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(corr)
            vif_values = np.diag(inv)
            vif_values = np.clip(vif_values, 1.0, None)
            return pd.Series(vif_values, index=frame.columns, dtype=float)

        while len(current) > 1:
            subset = working[current]
            vif_scores = _compute_vif_matrix(subset)
            for feat, value in vif_scores.items():
                latest_vif[feat] = float(value)
                iteration_map[feat] = iteration
            max_feature = vif_scores.idxmax()
            max_vif = float(vif_scores.loc[max_feature])
            if not np.isfinite(max_vif) or max_vif < threshold:
                break
            print(f"    Removing {max_feature}: VIF={max_vif:.2f}")
            status[max_feature] = "dropped"
            summary_rows.append({
                "feature": max_feature,
                "vif": max_vif,
                "status": "dropped",
                "iteration": iteration,
                "reason": f"VIF>{threshold}"
            })
            current.remove(max_feature)
            iteration += 1
            if len(current) == 1:
                break

        if current:
            final_scores = _compute_vif_matrix(working[current])
            for feat, value in final_scores.items():
                latest_vif[feat] = float(value)
                status[feat] = "kept"
                iteration_map[feat] = iteration
                summary_rows.append({
                    "feature": feat,
                    "vif": float(value),
                    "status": "kept",
                    "iteration": iteration,
                    "reason": ""
                })

        for feat in near_constant:
            summary_rows.append({
                "feature": feat,
                "vif": np.nan,
                "status": "dropped_constant",
                "iteration": 0,
                "reason": "variance~0"
            })

        summary_df = pd.DataFrame(summary_rows, columns=["feature", "vif", "status", "iteration", "reason"])
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["status", "vif"], ascending=[True, False]).reset_index(drop=True)
        self.last_vif_summary_ = summary_df

        kept_features = [feat for feat, state in status.items() if state == "kept"]
        preserved = [f for f in features if f not in status]
        return kept_features + preserved if kept_features or preserved else current

    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series,
                             threshold: float = 0.9, max_per_cluster: int = 1) -> List[str]:
        """Select best features from correlated clusters and store diagnostics."""

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_remove: Set[str] = set()
        processed: Set[str] = set()
        keep_count = max(1, int(max_per_cluster))
        clusters: List[Dict[str, Any]] = []

        for column in upper_tri.columns:
            if column in processed:
                continue

            correlated = list(upper_tri.index[upper_tri[column] > threshold])
            if not correlated:
                continue

            cluster = {column, *correlated}
            processed.update(cluster)

            target_corrs = {
                feat: abs(X[feat].corr(y)) if feat in X.columns else 0.0
                for feat in cluster
            }
            ordered = sorted(target_corrs.items(), key=lambda item: item[1], reverse=True)
            keep_features = {feat for feat, _ in ordered[:keep_count]}
            dropped_features = []

            for feat in cluster:
                if feat not in keep_features:
                    to_remove.add(feat)
                    leader = next(iter(keep_features))
                    print(
                        f"    Removing {feat}: Correlated with {leader} (|corr|>{threshold})"
                    )
                    dropped_features.append(feat)

            if cluster:
                sub_corr = corr_matrix.loc[list(cluster), list(cluster)].replace(1.0, 0.0)
                max_corr_value = float(sub_corr.values.max()) if not sub_corr.empty else 0.0
            else:
                max_corr_value = 0.0

            clusters.append({
                "cluster_lead": ordered[0][0] if ordered else column,
                "members": ", ".join(sorted(cluster)),
                "kept": ", ".join(sorted(keep_features)) if keep_features else "",
                "dropped": ", ".join(sorted(dropped_features)) if dropped_features else "",
                "max_correlation": max_corr_value,
                "threshold": threshold,
            })

        self.last_correlation_clusters_ = pd.DataFrame(clusters)
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
