"""Enhanced Model Builder with GAM, CatBoost, ExtraTrees and more"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
import warnings

warnings.filterwarnings('ignore')

# Optional imports
try:
    from pygam import LogisticGAM, s, f
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("Warning: pygam not installed. GAM models will not be available.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed. CatBoost models will not be available.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. LightGBM models will not be available.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. XGBoost models will not be available.")


class EnhancedModelBuilder:
    """
    Enhanced model builder with support for multiple algorithms:
    - Logistic Regression
    - GAM (Generalized Additive Models)
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost
    """

    def __init__(self, config):
        self.config = config
        self.models_ = {}
        self.scores_ = {}
        self.model_params_ = {}
        self.feature_importance_ = {}

    def build_models(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: Optional[pd.DataFrame] = None,
                    y_test: Optional[pd.Series] = None,
                    X_oot: Optional[pd.DataFrame] = None,
                    y_oot: Optional[pd.Series] = None,
                    model_types: Optional[List[str]] = None) -> Dict:
        """
        Build multiple models and select the best one.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame, optional
            Test features
        y_test : pd.Series, optional
            Test target
        X_oot : pd.DataFrame, optional
            Out-of-time features
        y_oot : pd.Series, optional
            Out-of-time target
        model_types : List[str], optional
            List of model types to train

        Returns:
        --------
        dict
            Dictionary with best model and scores
        """
        print(f"Training models with {X_train.shape[1]} features...")

        # Get models to train
        if model_types is None:
            model_types = self.config.model_types or self._get_default_models()

        models_to_train = self._get_model_configs(model_types)

        # Train each model
        for model_name, model_config in models_to_train.items():
            print(f"  Training {model_name}...")

            try:
                # Train model
                if self.config.use_optuna and model_name != 'LogisticRegression':
                    model = self._train_with_optuna(
                        X_train, y_train, X_test, y_test,
                        model_config['estimator_class'],
                        model_config['param_space']
                    )
                else:
                    model = model_config['estimator']
                    # Special handling for different model types
                    if model_name == 'GAM':
                        model = self._fit_gam(X_train, y_train)
                    elif model_name == 'CatBoost':
                        model = self._fit_catboost(X_train, y_train, X_test, y_test)
                    else:
                        model.fit(X_train, y_train)

                self.models_[model_name] = model

                # Evaluate model
                scores = self._evaluate_model(
                    model, X_train, y_train, X_test, y_test, X_oot, y_oot
                )
                self.scores_[model_name] = scores

                # Extract feature importance
                self.feature_importance_[model_name] = self._get_feature_importance(
                    model, X_train.columns
                )

                print(f"    Train AUC: {scores['train_auc']:.4f}", end="")
                if 'test_auc' in scores:
                    print(f", Test AUC: {scores['test_auc']:.4f}", end="")
                if 'oot_auc' in scores:
                    print(f", OOT AUC: {scores['oot_auc']:.4f}")
                else:
                    print()

            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue

        # Select best model
        best_model_name = self.select_best_model()

        return {
            'best_model': self.models_[best_model_name],
            'best_model_name': best_model_name,
            'best_score': self._get_best_score(best_model_name),
            'best_auc': self._get_best_score(best_model_name),
            'all_scores': self.scores_,
            'all_models': self.models_,
            'feature_importance': self.feature_importance_
        }

    def _get_default_models(self) -> List[str]:
        """Get default list of models to train."""
        models = ['LogisticRegression', 'RandomForest', 'GradientBoosting']

        if HAS_LIGHTGBM:
            models.append('LightGBM')
        if HAS_XGBOOST:
            models.append('XGBoost')
        if HAS_CATBOOST:
            models.append('CatBoost')
        if HAS_GAM:
            models.append('GAM')

        models.append('ExtraTrees')

        return models

    def _get_model_configs(self, model_types: List[str]) -> Dict:
        """Get model configurations for specified model types."""
        all_configs = {
            'LogisticRegression': {
                'estimator': LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    max_iter=1000,
                    random_state=self.config.random_state,
                    solver='lbfgs'
                ),
                'estimator_class': LogisticRegression,
                'param_space': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },

            'RandomForest': {
                'estimator': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ),
                'estimator_class': RandomForestClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [10, 20, 50],
                    'min_samples_leaf': [5, 10, 20]
                }
            },

            'ExtraTrees': {
                'estimator': ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ),
                'estimator_class': ExtraTreesClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [10, 20, 50],
                    'min_samples_leaf': [5, 10, 20]
                }
            },

            'GradientBoosting': {
                'estimator': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.config.random_state
                ),
                'estimator_class': GradientBoostingClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9]
                }
            }
        }

        # Add LightGBM if available
        if HAS_LIGHTGBM:
            all_configs['LightGBM'] = {
                'estimator': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=-1
                ),
                'estimator_class': lgb.LGBMClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            }

        # Add XGBoost if available
        if HAS_XGBOOST:
            all_configs['XGBoost'] = {
                'estimator': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    verbosity=0,
                    use_label_encoder=False
                ),
                'estimator_class': xgb.XGBClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            }

        # Add CatBoost if available
        if HAS_CATBOOST:
            all_configs['CatBoost'] = {
                'estimator': CatBoostClassifier(
                    iterations=100,
                    depth=5,
                    learning_rate=0.05,
                    random_state=self.config.random_state,
                    verbose=False,
                    allow_writing_files=False
                ),
                'estimator_class': CatBoostClassifier,
                'param_space': {
                    'iterations': [50, 100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
            }

        # Add GAM if available
        if HAS_GAM:
            all_configs['GAM'] = {
                'estimator': None,  # GAM needs special initialization
                'estimator_class': LogisticGAM,
                'param_space': {
                    'n_splines': [10, 20, 30],
                    'lam': [0.1, 0.5, 1.0, 5.0, 10.0]
                }
            }

        # Filter to requested models
        configs = {}
        for model_type in model_types:
            if model_type in all_configs:
                configs[model_type] = all_configs[model_type]
            else:
                print(f"  Warning: {model_type} not available or not installed")

        return configs

    def _fit_gam(self, X: pd.DataFrame, y: pd.Series):
        """Fit a GAM model with automatic spline selection."""
        if not HAS_GAM:
            raise ImportError("pygam is not installed")

        # Create GAM with splines for each feature
        gam = LogisticGAM()

        # Automatically determine splines
        for i, col in enumerate(X.columns):
            if X[col].nunique() > 10:
                # Use spline for continuous variables
                gam = gam + s(i, n_splines=min(20, X[col].nunique()))
            else:
                # Use factor for categorical-like variables
                gam = gam + f(i)

        # Fit model with automatic lambda selection
        gam.gridsearch(X.values, y.values, lam=np.logspace(-3, 2, 20))

        return gam

    def _fit_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: Optional[pd.DataFrame] = None,
                     y_test: Optional[pd.Series] = None):
        """Fit CatBoost with proper categorical handling."""
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed")

        # Identify categorical columns
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_indices = [X_train.columns.get_loc(c) for c in cat_features]

        model = CatBoostClassifier(
            iterations=self.config.n_estimators if hasattr(self.config, 'n_estimators') else 100,
            depth=5,
            learning_rate=0.05,
            random_state=self.config.random_state,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_indices if cat_indices else None
        )

        # Use validation set if available
        if X_test is not None and y_test is not None:
            model.fit(X_train, y_train,
                     eval_set=(X_test, y_test),
                     early_stopping_rounds=50,
                     verbose=False)
        else:
            model.fit(X_train, y_train)

        return model

    def _train_with_optuna(self, X_train, y_train, X_val, y_val,
                          estimator_class, param_space):
        """Train model with Optuna hyperparameter optimization."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("    Optuna not installed, using default parameters")
            return estimator_class().fit(X_train, y_train)

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], (int, float)):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_values), max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values)
                        )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

            # Create and train model
            model = estimator_class(**params, random_state=self.config.random_state)
            model.fit(X_train, y_train)

            # Evaluate
            if X_val is not None and y_val is not None:
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                # Use cross-validation on training set
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                score = scores.mean()

            return score

        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials)

        # Train final model with best parameters
        best_params = study.best_params
        model = estimator_class(**best_params, random_state=self.config.random_state)
        model.fit(X_train, y_train)

        return model

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, X_oot, y_oot):
        """Evaluate model on all datasets."""
        scores = {}

        # Train score
        y_pred_train = self._predict_proba(model, X_train)
        scores['train_auc'] = roc_auc_score(y_train, y_pred_train)
        scores['train_gini'] = 2 * scores['train_auc'] - 1

        # Test score
        if X_test is not None and y_test is not None:
            y_pred_test = self._predict_proba(model, X_test)
            scores['test_auc'] = roc_auc_score(y_test, y_pred_test)
            scores['test_gini'] = 2 * scores['test_auc'] - 1

        # OOT score
        if X_oot is not None and y_oot is not None:
            y_pred_oot = self._predict_proba(model, X_oot)
            scores['oot_auc'] = roc_auc_score(y_oot, y_pred_oot)
            scores['oot_gini'] = 2 * scores['oot_auc'] - 1

        return scores

    def _predict_proba(self, model, X):
        """Get probability predictions handling different model types."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif HAS_GAM and isinstance(model, LogisticGAM):
            # GAM returns probabilities directly
            return model.predict_mu(X.values)
        else:
            # Fallback to predict
            return model.predict(X)

    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from different model types."""
        importance = {}

        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            importance = dict(zip(feature_names, np.abs(coef)))
        elif HAS_GAM and isinstance(model, LogisticGAM):
            # GAM - use partial dependence as importance
            importance = {feat: 1.0 for feat in feature_names}  # Placeholder

        return importance

    def select_best_model(self) -> str:
        """Select best model based on configured strategy."""
        if not self.scores_:
            raise ValueError("No models have been trained yet")

        # Selection strategy
        if self.config.model_selection_method == 'oot_first':
            # Prefer OOT score if available
            score_key = 'oot_auc'
        elif self.config.model_selection_method == 'test_first':
            # Prefer test score if available
            score_key = 'test_auc'
        else:
            # Default: use best available score
            score_key = None

        best_model_name = None
        best_score = -np.inf

        for model_name, scores in self.scores_.items():
            if score_key and score_key in scores:
                score = scores[score_key]
            elif 'oot_auc' in scores:
                score = scores['oot_auc']
            elif 'test_auc' in scores:
                score = scores['test_auc']
            else:
                score = scores['train_auc']

            if score > best_score:
                best_score = score
                best_model_name = model_name

        print(f"\n  Best model: {best_model_name} (Score: {best_score:.4f})")
        return best_model_name

    def _get_best_score(self, model_name: str) -> float:
        """Get the best available score for a model."""
        scores = self.scores_[model_name]
        if 'oot_auc' in scores:
            return scores['oot_auc']
        elif 'test_auc' in scores:
            return scores['test_auc']
        else:
            return scores['train_auc']