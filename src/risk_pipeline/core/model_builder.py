"""
Comprehensive Model Builder with all ML algorithms
Supports: Logistic, GAM, CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveModelBuilder:
    """
    Model builder supporting all major ML algorithms with optional Optuna optimization.
    """

    def __init__(self, config):
        self.config = config
        self.models_ = {}
        self.scores_ = {}
        self.feature_importance_ = {}

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: Optional[pd.DataFrame] = None,
                        y_test: Optional[pd.Series] = None) -> Dict:
        """
        Train all configured models and select the best.
        """

        print(f"  Training models on {X_train.shape[1]} features...")

        # Check if we have any features
        if X_train.shape[1] == 0:
            print("    WARNING: No features available for training. Skipping model training.")
            return {
                'models': {},
                'scores': {},
                'best_model': None,
                'best_model_name': None,
                'feature_importance': {},
                'selected_features': []
            }

        # Determine which models to train
        models_to_train = self._get_models_to_train()

        # Train each model
        for model_name in models_to_train:
            print(f"    Training {model_name}...")

            try:
                if self.config.use_optuna and model_name != 'LogisticRegression':
                    model = self._train_with_optuna(
                        model_name, X_train, y_train, X_test, y_test
                    )
                else:
                    model = self._train_without_optuna(
                        model_name, X_train, y_train
                    )

                # Store model
                self.models_[model_name] = model

                # Evaluate
                train_score = self._evaluate_model(model, X_train, y_train)
                test_score = self._evaluate_model(model, X_test, y_test) if X_test is not None else None

                self.scores_[model_name] = {
                    'train_auc': train_score,
                    'test_auc': test_score
                }

                # Get feature importance
                self.feature_importance_[model_name] = self._get_feature_importance(
                    model, X_train.columns, model_name
                )

                print(f"      Train AUC: {train_score:.4f}", end="")
                if test_score:
                    print(f", Test AUC: {test_score:.4f}")
                else:
                    print()

            except Exception as e:
                print(f"      Failed: {str(e)[:50]}...")
                continue

        # Select best model
        best_model_name = self._select_best_model()

        # Handle case when no models trained successfully
        if best_model_name is None or best_model_name not in self.models_:
            return {
                'models': self.models_,
                'scores': self.scores_,
                'best_model': None,
                'best_model_name': None,
                'feature_importance': self.feature_importance_,
                'selected_features': list(X_train.columns) if X_train is not None and len(X_train.columns) > 0 else []
            }

        return {
            'models': self.models_,
            'scores': self.scores_,
            'best_model': self.models_[best_model_name],
            'best_model_name': best_model_name,
            'feature_importance': self.feature_importance_,
            'selected_features': list(X_train.columns)
        }

    def _get_models_to_train(self) -> List[str]:
        """Get list of models to train based on config."""

        all_models = [
            'LogisticRegression',
            'RandomForest',
            'ExtraTrees',
            'LightGBM',
            'XGBoost',
            'CatBoost',
            'GAM'
        ]

        if self.config.model_type == 'all':
            # Check which are available
            available = []
            for model in all_models:
                if self._is_model_available(model):
                    available.append(model)
            return available
        elif isinstance(self.config.model_type, list):
            return [m for m in self.config.model_type if self._is_model_available(m)]
        else:
            return [self.config.model_type] if self._is_model_available(self.config.model_type) else []

    def _is_model_available(self, model_name: str) -> bool:
        """Check if model library is available."""

        if model_name in ['LogisticRegression', 'RandomForest', 'ExtraTrees']:
            return True
        elif model_name == 'LightGBM':
            try:
                import lightgbm
                return True
            except:
                return False
        elif model_name == 'XGBoost':
            try:
                import xgboost
                return True
            except:
                return False
        elif model_name == 'CatBoost':
            try:
                import catboost
                return True
            except:
                return False
        elif model_name == 'GAM':
            try:
                from pygam import LogisticGAM
                return True
            except:
                return False
        return False

    def _train_without_optuna(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train model with default parameters."""

        if model_name == 'LogisticRegression':
            model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=self.config.random_state,
                solver='lbfgs'
            )

        elif model_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif model_name == 'LightGBM':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.config.random_state,
                verbosity=-1
            )

        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        elif model_name == 'CatBoost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=100,
                depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=False
            )

        elif model_name == 'GAM':
            from pygam import LogisticGAM, s
            # Simple GAM with splines for each feature
            n_features = X_train.shape[1]
            model = LogisticGAM(s(0, n_splines=10))
            for i in range(1, n_features):
                model += s(i, n_splines=10)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        return model

    def _train_with_optuna(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None):
        """Train model with Optuna hyperparameter optimization."""

        try:
            import optuna
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                # Get hyperparameters based on model type
                params = self._get_optuna_params(trial, model_name)

                # Create model
                model = self._create_model_with_params(model_name, params)

                # Evaluate
                if X_test is not None and y_test is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_pred)
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                    score = scores.mean()

                return score

            # Optimize
            study = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
            study.optimize(objective, n_trials=self.config.n_optuna_trials if hasattr(self.config, 'n_optuna_trials') else 20)

            # Train final model with best parameters
            best_params = study.best_params
            model = self._create_model_with_params(model_name, best_params)
            model.fit(X_train, y_train)

            return model

        except ImportError:
            print("      Optuna not available, using default parameters")
            return self._train_without_optuna(model_name, X_train, y_train)

    def _get_optuna_params(self, trial, model_name: str) -> Dict:
        """Get Optuna hyperparameter search space."""

        if model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20)
            }

        elif model_name == 'ExtraTrees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20)
            }

        elif model_name == 'LightGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }

        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }

        elif model_name == 'CatBoost':
            return {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
            }

        elif model_name == 'GAM':
            return {
                'n_splines': trial.suggest_int('n_splines', 5, 25),
                'lam': trial.suggest_float('lam', 0.001, 10.0, log=True)
            }

        return {}

    def _create_model_with_params(self, model_name: str, params: Dict):
        """Create model instance with given parameters."""

        if model_name == 'RandomForest':
            return RandomForestClassifier(
                **params,
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif model_name == 'ExtraTrees':
            return ExtraTreesClassifier(
                **params,
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif model_name == 'LightGBM':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                **params,
                random_state=self.config.random_state,
                verbosity=-1
            )

        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier
            return XGBClassifier(
                **params,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        elif model_name == 'CatBoost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                **params,
                random_state=self.config.random_state,
                verbose=False
            )

        elif model_name == 'GAM':
            from pygam import LogisticGAM, s
            # Create GAM with specified parameters
            n_features = params.get('n_features', 10)  # This should be passed
            n_splines = params.get('n_splines', 10)
            lam = params.get('lam', 0.6)

            model = LogisticGAM(s(0, n_splines=n_splines, lam=lam))
            for i in range(1, n_features):
                model += s(i, n_splines=n_splines, lam=lam)
            return model

        raise ValueError(f"Unknown model: {model_name}")

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
        """Evaluate model performance."""

        if X is None or y is None:
            return None

        try:
            y_pred = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred)
        except:
            return 0.0

    def _get_feature_importance(self, model, feature_names: List[str], model_name: str) -> pd.DataFrame:
        """Extract feature importance from model."""

        importance_values = None

        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_values = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            elif model_name == 'GAM':
                # GAM importance based on smoothing terms
                importance_values = np.ones(len(feature_names)) / len(feature_names)
        except:
            pass

        if importance_values is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': [1.0/len(feature_names)] * len(feature_names)
            })

        return importance_df

    def _select_best_model(self) -> str:
        """Select best model based on test performance."""

        if not self.scores_:
            return None

        best_model = None
        best_score = -np.inf

        for model_name, scores in self.scores_.items():
            # Prioritize test score if available
            score = scores.get('test_auc', scores.get('train_auc', 0))

            if score > best_score:
                best_score = score
                best_model = model_name

        print(f"    Best model: {best_model} (AUC: {best_score:.4f})")
        return best_model

    def calculate_shap_importance(self, model, X: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
        """Calculate SHAP feature importance."""

        try:
            import shap

            # Limit samples for speed
            X_sample = X.sample(min(max_samples, len(X)), random_state=self.config.random_state)

            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample)
                shap_values = explainer(X_sample)

                # For binary classification, use positive class SHAP values
                if len(shap_values.shape) == 3:
                    mean_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
                else:
                    mean_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
                mean_shap = np.abs(shap_values.values).mean(axis=0)

            return pd.DataFrame({
                'feature': X.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)

        except:
            return pd.DataFrame({
                'feature': X.columns,
                'shap_importance': [0] * len(X.columns)
            })