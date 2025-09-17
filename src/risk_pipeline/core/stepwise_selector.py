"""Stepwise Feature Selection Methods (Forward, Backward, Stepwise)"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class StepwiseSelector:
    """
    Implements forward, backward, and stepwise feature selection methods.

    Methods:
    - Forward: Start with no features, add best feature at each step
    - Backward: Start with all features, remove worst feature at each step
    - Stepwise: Combination - can add or remove features at each step
    """

    def __init__(self, estimator=None, scoring='roc_auc', cv=5,
                 p_value_threshold=0.05, max_features=None, min_features=1):
        """
        Initialize stepwise selector.

        Parameters:
        -----------
        estimator : sklearn estimator
            Model to use for selection (default: LogisticRegression)
        scoring : str
            Scoring metric ('roc_auc', 'gini', 'accuracy')
        cv : int
            Cross-validation folds
        p_value_threshold : float
            P-value threshold for statistical significance
        max_features : int
            Maximum number of features to select
        min_features : int
            Minimum number of features to keep
        """
        self.estimator = estimator or LogisticRegression(max_iter=1000, solver='liblinear')
        self.scoring = scoring
        self.cv = cv
        self.p_value_threshold = p_value_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.selected_features_ = []
        self.selection_history_ = []

    def forward_selection(self, X: pd.DataFrame, y: pd.Series,
                         feature_names: List[str] = None) -> List[str]:
        """
        Forward feature selection.

        Start with empty set, add best feature at each iteration.
        """
        print("    Running forward selection...")

        if feature_names is None:
            feature_names = X.columns.tolist()

        selected = []
        remaining = feature_names.copy()

        # Convert to numpy for faster computation
        X_array = X[feature_names].values
        y_array = y.values

        best_score = 0
        iteration = 0

        while remaining and (self.max_features is None or len(selected) < self.max_features):
            iteration += 1
            scores = []

            # Evaluate each remaining feature
            for feature in remaining:
                # Create feature set with new feature
                test_features = selected + [feature]
                X_subset = X[test_features].values

                # Calculate score
                try:
                    if len(test_features) == 1:
                        # For single feature, use univariate score
                        score = self._calculate_univariate_score(X_subset.ravel(), y_array)
                    else:
                        # For multiple features, use cross-validation
                        cv_scores = cross_val_score(
                            self.estimator, X_subset, y_array,
                            cv=self.cv, scoring=self.scoring
                        )
                        score = cv_scores.mean()
                    scores.append(score)
                except:
                    scores.append(-np.inf)

            # Find best feature
            if scores:
                best_idx = np.argmax(scores)
                best_feature_score = scores[best_idx]

                # Check if improvement is significant
                if best_feature_score > best_score:
                    # Check statistical significance if we have existing features
                    if selected and self._is_significant_improvement(
                        best_score, best_feature_score, len(X)
                    ):
                        best_feature = remaining[best_idx]
                        selected.append(best_feature)
                        remaining.remove(best_feature)
                        best_score = best_feature_score

                        print(f"      Iteration {iteration}: Added '{best_feature}' "
                              f"(score: {best_score:.4f})")

                        self.selection_history_.append({
                            'iteration': iteration,
                            'action': 'add',
                            'feature': best_feature,
                            'score': best_score,
                            'n_features': len(selected)
                        })
                    elif not selected:
                        # First feature, add anyway
                        best_feature = remaining[best_idx]
                        selected.append(best_feature)
                        remaining.remove(best_feature)
                        best_score = best_feature_score

                        print(f"      Iteration {iteration}: Added '{best_feature}' "
                              f"(score: {best_score:.4f})")
                    else:
                        # No significant improvement
                        print(f"      Stopping: No significant improvement")
                        break
                else:
                    print(f"      Stopping: Score not improving")
                    break

        self.selected_features_ = selected
        print(f"    Selected {len(selected)} features via forward selection")
        return selected

    def backward_selection(self, X: pd.DataFrame, y: pd.Series,
                          feature_names: List[str] = None) -> List[str]:
        """
        Backward feature selection.

        Start with all features, remove worst feature at each iteration.
        """
        print("    Running backward selection...")

        if feature_names is None:
            feature_names = X.columns.tolist()

        selected = feature_names.copy()

        # Initial score with all features
        X_array = X[selected].values
        y_array = y.values

        try:
            cv_scores = cross_val_score(
                self.estimator, X_array, y_array,
                cv=self.cv, scoring=self.scoring
            )
            best_score = cv_scores.mean()
        except:
            best_score = 0

        print(f"      Initial score with {len(selected)} features: {best_score:.4f}")

        iteration = 0

        while len(selected) > max(self.min_features, 1):
            iteration += 1
            scores = []

            # Evaluate removing each feature
            for feature in selected:
                # Create feature set without this feature
                test_features = [f for f in selected if f != feature]

                if not test_features:
                    scores.append(-np.inf)
                    continue

                X_subset = X[test_features].values

                # Calculate score
                try:
                    cv_scores = cross_val_score(
                        self.estimator, X_subset, y_array,
                        cv=self.cv, scoring=self.scoring
                    )
                    score = cv_scores.mean()
                    scores.append(score)
                except:
                    scores.append(-np.inf)

            # Find feature whose removal gives best score
            if scores:
                best_idx = np.argmax(scores)
                best_score_after_removal = scores[best_idx]

                # Check if removal improves or maintains performance
                if best_score_after_removal >= best_score - 0.001:  # Small tolerance
                    worst_feature = selected[best_idx]
                    selected.remove(worst_feature)
                    best_score = best_score_after_removal

                    print(f"      Iteration {iteration}: Removed '{worst_feature}' "
                          f"(score: {best_score:.4f})")

                    self.selection_history_.append({
                        'iteration': iteration,
                        'action': 'remove',
                        'feature': worst_feature,
                        'score': best_score,
                        'n_features': len(selected)
                    })
                else:
                    print(f"      Stopping: Removal would decrease performance")
                    break

        self.selected_features_ = selected
        print(f"    Selected {len(selected)} features via backward selection")
        return selected

    def stepwise_selection(self, X: pd.DataFrame, y: pd.Series,
                          feature_names: List[str] = None) -> List[str]:
        """
        Stepwise feature selection (bidirectional).

        At each step, can either add or remove a feature.
        """
        print("    Running stepwise selection...")

        if feature_names is None:
            feature_names = X.columns.tolist()

        # Start with forward selection for initial features
        selected = []
        remaining = feature_names.copy()

        y_array = y.values
        best_score = 0
        iteration = 0
        last_action = None

        while True:
            iteration += 1
            improved = False

            # Try adding a feature (if we have remaining features and haven't hit max)
            if remaining and (self.max_features is None or len(selected) < self.max_features):
                add_scores = []

                for feature in remaining:
                    test_features = selected + [feature]
                    X_subset = X[test_features].values

                    try:
                        if len(test_features) == 1:
                            score = self._calculate_univariate_score(X_subset.ravel(), y_array)
                        else:
                            cv_scores = cross_val_score(
                                self.estimator, X_subset, y_array,
                                cv=self.cv, scoring=self.scoring
                            )
                            score = cv_scores.mean()
                        add_scores.append(score)
                    except:
                        add_scores.append(-np.inf)

                if add_scores:
                    best_add_idx = np.argmax(add_scores)
                    best_add_score = add_scores[best_add_idx]
                else:
                    best_add_score = -np.inf
            else:
                best_add_score = -np.inf

            # Try removing a feature (if we have selected features above minimum)
            if selected and len(selected) > self.min_features:
                remove_scores = []

                for feature in selected:
                    test_features = [f for f in selected if f != feature]

                    if test_features:
                        X_subset = X[test_features].values

                        try:
                            cv_scores = cross_val_score(
                                self.estimator, X_subset, y_array,
                                cv=self.cv, scoring=self.scoring
                            )
                            score = cv_scores.mean()
                            remove_scores.append(score)
                        except:
                            remove_scores.append(-np.inf)
                    else:
                        remove_scores.append(-np.inf)

                if remove_scores:
                    best_remove_idx = np.argmax(remove_scores)
                    best_remove_score = remove_scores[best_remove_idx]
                else:
                    best_remove_score = -np.inf
            else:
                best_remove_score = -np.inf

            # Decide whether to add or remove
            if best_add_score > best_score and best_add_score > best_remove_score:
                # Add feature
                best_feature = remaining[best_add_idx]
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_add_score
                improved = True

                print(f"      Iteration {iteration}: Added '{best_feature}' "
                      f"(score: {best_score:.4f})")

                self.selection_history_.append({
                    'iteration': iteration,
                    'action': 'add',
                    'feature': best_feature,
                    'score': best_score,
                    'n_features': len(selected)
                })
                last_action = 'add'

            elif best_remove_score > best_score:
                # Remove feature
                worst_feature = selected[best_remove_idx]
                selected.remove(worst_feature)
                remaining.append(worst_feature)
                best_score = best_remove_score
                improved = True

                print(f"      Iteration {iteration}: Removed '{worst_feature}' "
                      f"(score: {best_score:.4f})")

                self.selection_history_.append({
                    'iteration': iteration,
                    'action': 'remove',
                    'feature': worst_feature,
                    'score': best_score,
                    'n_features': len(selected)
                })
                last_action = 'remove'

            # Check stopping criteria
            if not improved:
                print(f"      Stopping: No improvement possible")
                break

            # Prevent infinite loops (alternating add/remove same feature)
            if iteration > 2 * len(feature_names):
                print(f"      Stopping: Maximum iterations reached")
                break

        self.selected_features_ = selected
        print(f"    Selected {len(selected)} features via stepwise selection")
        return selected

    def _calculate_univariate_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate univariate score for a single feature."""
        try:
            if self.scoring == 'roc_auc':
                return roc_auc_score(y, X)
            elif self.scoring == 'gini':
                return 2 * roc_auc_score(y, X) - 1
            else:
                # Use correlation for other metrics
                return abs(np.corrcoef(X, y)[0, 1])
        except:
            return 0.0

    def _is_significant_improvement(self, old_score: float, new_score: float,
                                   n_samples: int) -> bool:
        """
        Check if score improvement is statistically significant.

        Uses DeLong test approximation for AUC comparison.
        """
        if new_score <= old_score:
            return False

        # Simple significance test based on sample size
        # More sophisticated tests could be implemented (DeLong, bootstrap)
        improvement = new_score - old_score
        threshold = 1.96 * np.sqrt(1.0 / n_samples)  # Approximate

        return improvement > threshold

    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of selection process."""
        if not self.selection_history_:
            return pd.DataFrame()

        return pd.DataFrame(self.selection_history_)