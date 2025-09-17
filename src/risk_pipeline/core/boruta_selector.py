"""Boruta Feature Selection with LightGBM"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')


class BorutaSelector:
    """
    Boruta feature selection algorithm optimized for LightGBM.

    Boruta is an all-relevant feature selection method that finds all features
    that are relevant for prediction by comparing real features with shadow features.
    """

    def __init__(self,
                 estimator: Optional[Union[str, object]] = 'lightgbm',
                 n_estimators: int = 100,
                 max_iter: int = 100,
                 perc: int = 100,
                 alpha: float = 0.05,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize Boruta selector.

        Parameters:
        -----------
        estimator : str or sklearn estimator
            'lightgbm' (default), 'randomforest', or custom estimator
        n_estimators : int
            Number of estimators for the base model
        max_iter : int
            Maximum number of iterations
        perc : int
            Percentile for shadow feature importance (default 100 = max)
        alpha : float
            Significance level for statistical test
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print progress information
        """
        self.estimator_type = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        # Initialize estimator
        if estimator == 'lightgbm':
            self.estimator = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
                importance_type='gain'
            )
        elif estimator == 'randomforest':
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.estimator = estimator

        # Results storage
        self.selected_features_ = []
        self.rejected_features_ = []
        self.tentative_features_ = []
        self.feature_importances_ = {}
        self.iteration_results_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSelector':
        """
        Fit Boruta selector to find all relevant features.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable

        Returns:
        --------
        self : BorutaSelector
            Fitted selector
        """
        if self.verbose:
            print("    Running Boruta feature selection with LightGBM...")

        # Convert to numpy for faster computation
        X_array = X.values.copy()
        y_array = y.values
        n_samples, n_features = X_array.shape
        feature_names = X.columns.tolist()

        # Initialize decision arrays
        # 0: Tentative, 1: Confirmed, -1: Rejected
        decisions = np.zeros(n_features)
        hits = np.zeros(n_features)

        # Store importance history
        importance_history = np.zeros((self.max_iter, n_features))

        for iteration in range(self.max_iter):
            # Create shadow features (shuffled copies)
            shadow_features = self._create_shadow_features(X_array)

            # Combine real and shadow features
            X_combined = np.hstack([X_array, shadow_features])

            # Train model
            self.estimator.fit(X_combined, y_array)

            # Get feature importances
            if hasattr(self.estimator, 'feature_importances_'):
                importances = self.estimator.feature_importances_
            else:
                # Calculate importances using permutation if not available
                importances = self._calculate_permutation_importance(
                    X_combined, y_array
                )

            # Split importances into real and shadow
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # Get threshold (max or percentile of shadow importances)
            if self.perc == 100:
                threshold = np.max(shadow_importances)
            else:
                threshold = np.percentile(shadow_importances, self.perc)

            # Update hits (real importance > threshold)
            hits += (real_importances > threshold)

            # Store importance for this iteration
            importance_history[iteration] = real_importances

            # Perform statistical test (binomial test)
            # Under null hypothesis, feature has 50% chance to beat threshold
            for i in range(n_features):
                if decisions[i] == 0:  # Still tentative
                    # Two-sided binomial test
                    p_value = self._binomial_test(
                        int(hits[i]),
                        iteration + 1,
                        0.5
                    )

                    # Make decision based on p-value
                    if p_value < self.alpha:
                        if hits[i] > (iteration + 1) / 2:
                            decisions[i] = 1  # Confirm feature
                            if self.verbose:
                                print(f"      Iteration {iteration + 1}: "
                                      f"Confirmed '{feature_names[i]}'")
                        else:
                            decisions[i] = -1  # Reject feature
                            if self.verbose:
                                print(f"      Iteration {iteration + 1}: "
                                      f"Rejected '{feature_names[i]}'")

            # Check stopping criteria
            if np.all(decisions != 0):
                if self.verbose:
                    print(f"      All features decided after {iteration + 1} iterations")
                importance_history = importance_history[:iteration + 1]
                break

            # Early stopping if no changes for many iterations
            if iteration > 20:
                recent_decisions = np.sum(np.abs(decisions))
                if iteration > 40 and recent_decisions == np.sum(np.abs(decisions)):
                    if self.verbose:
                        print(f"      Early stopping at iteration {iteration + 1}")
                    importance_history = importance_history[:iteration + 1]
                    break

        # Finalize results
        self.selected_features_ = [
            feature_names[i] for i in range(n_features) if decisions[i] == 1
        ]
        self.rejected_features_ = [
            feature_names[i] for i in range(n_features) if decisions[i] == -1
        ]
        self.tentative_features_ = [
            feature_names[i] for i in range(n_features) if decisions[i] == 0
        ]

        # Calculate average importance for each feature
        avg_importances = np.mean(importance_history, axis=0)
        self.feature_importances_ = dict(zip(feature_names, avg_importances))

        # Store iteration results
        self.iteration_results_ = {
            'n_iterations': len(importance_history),
            'decisions': decisions,
            'hits': hits,
            'importance_history': importance_history
        }

        if self.verbose:
            print(f"    Boruta selection completed:")
            print(f"      • Confirmed features: {len(self.selected_features_)}")
            print(f"      • Rejected features: {len(self.rejected_features_)}")
            print(f"      • Tentative features: {len(self.tentative_features_)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset to selected features.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features

        Returns:
        --------
        pd.DataFrame
            Transformed dataset with selected features
        """
        if not self.selected_features_:
            raise ValueError("Boruta selector must be fitted before transform")

        # Include tentative features if specified
        features_to_keep = self.selected_features_.copy()
        if hasattr(self, 'include_tentative') and self.include_tentative:
            features_to_keep.extend(self.tentative_features_)

        return X[features_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     include_tentative: bool = False) -> pd.DataFrame:
        """
        Fit selector and transform dataset.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        include_tentative : bool
            Whether to include tentative features

        Returns:
        --------
        pd.DataFrame
            Transformed dataset
        """
        self.include_tentative = include_tentative
        self.fit(X, y)
        return self.transform(X)

    def _create_shadow_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create shadow features by shuffling each column independently.

        Parameters:
        -----------
        X : np.ndarray
            Original feature matrix

        Returns:
        --------
        np.ndarray
            Shadow feature matrix
        """
        n_samples, n_features = X.shape
        shadow = np.zeros_like(X)

        for i in range(n_features):
            shadow[:, i] = np.random.permutation(X[:, i])

        return shadow

    def _binomial_test(self, successes: int, trials: int, p: float) -> float:
        """
        Perform two-sided binomial test.

        Parameters:
        -----------
        successes : int
            Number of successes
        trials : int
            Number of trials
        p : float
            Probability under null hypothesis

        Returns:
        --------
        float
            P-value
        """
        from scipy import stats

        # Two-sided test
        result = stats.binomtest(successes, trials, p, alternative='two-sided')
        return result.pvalue

    def _calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate permutation importance if feature_importances_ not available.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable

        Returns:
        --------
        np.ndarray
            Feature importances
        """
        from sklearn.metrics import roc_auc_score

        n_features = X.shape[1]
        importances = np.zeros(n_features)

        # Get baseline score
        y_pred = self.estimator.predict_proba(X)[:, 1]
        baseline_score = roc_auc_score(y, y_pred)

        # Calculate importance for each feature
        for i in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            y_pred_permuted = self.estimator.predict_proba(X_permuted)[:, 1]
            permuted_score = roc_auc_score(y, y_pred_permuted)

            importances[i] = baseline_score - permuted_score

        return importances

    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Get feature ranking based on importance.

        Returns:
        --------
        pd.DataFrame
            Feature ranking with importance scores and decisions
        """
        if not self.feature_importances_:
            raise ValueError("Boruta selector must be fitted first")

        ranking = []
        for feature, importance in self.feature_importances_.items():
            if feature in self.selected_features_:
                status = 'Confirmed'
            elif feature in self.rejected_features_:
                status = 'Rejected'
            else:
                status = 'Tentative'

            ranking.append({
                'feature': feature,
                'importance': importance,
                'status': status,
                'hits': self.iteration_results_['hits'][
                    list(self.feature_importances_.keys()).index(feature)
                ],
                'hit_ratio': self.iteration_results_['hits'][
                    list(self.feature_importances_.keys()).index(feature)
                ] / self.iteration_results_['n_iterations']
            })

        return pd.DataFrame(ranking).sort_values('importance', ascending=False)

    def plot_importance_history(self, top_n: int = 20):
        """
        Plot importance history for top features.

        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        """
        import matplotlib.pyplot as plt

        history = self.iteration_results_['importance_history']
        feature_names = list(self.feature_importances_.keys())

        # Get top features by average importance
        avg_importances = np.mean(history, axis=0)
        top_indices = np.argsort(avg_importances)[-top_n:]

        plt.figure(figsize=(12, 8))
        for idx in top_indices:
            feature_name = feature_names[idx]
            if feature_name in self.selected_features_:
                linestyle = '-'
                alpha = 1.0
            elif feature_name in self.tentative_features_:
                linestyle = '--'
                alpha = 0.7
            else:
                linestyle = ':'
                alpha = 0.5

            plt.plot(history[:, idx], label=feature_name,
                    linestyle=linestyle, alpha=alpha)

        plt.xlabel('Iteration')
        plt.ylabel('Feature Importance')
        plt.title('Boruta Feature Importance Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()