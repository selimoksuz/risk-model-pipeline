"""Model building and training module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')


class ModelBuilder:
    """Handles model training, evaluation and selection"""
    
    def __init__(self, config):
        self.config = config
        self.models_ = {}
        self.scores_ = {}
        
    def build_models(self, train: pd.DataFrame, test: Optional[pd.DataFrame] = None,
                    oot: Optional[pd.DataFrame] = None) -> Dict:
        """Train multiple models and select the best one"""
        
        # Prepare data
        feature_cols = [col for col in train.columns 
                       if col not in [self.config.target_col, self.config.id_col, self.config.time_col]]
        
        X_train = train[feature_cols]
        y_train = train[self.config.target_col]
        
        X_test = test[feature_cols] if test is not None else None
        y_test = test[self.config.target_col] if test is not None else None
        
        X_oot = oot[feature_cols] if oot is not None else None
        y_oot = oot[self.config.target_col] if oot is not None else None
        
        print(f"Training models with {len(feature_cols)} features...")
        
        # Define models to train
        models_to_train = self._get_model_configs()
        
        # Train each model
        for model_name, model_config in models_to_train.items():
            print(f"  Training {model_name}...")
            
            # Train model
            if self.config.use_optuna and model_name != 'LogisticRegression':
                model = self._train_with_optuna(
                    X_train, y_train, X_test, y_test, 
                    model_config['estimator'], model_config['param_space']
                )
            else:
                model = model_config['estimator']
                model.fit(X_train, y_train)
            
            self.models_[model_name] = model
            
            # Evaluate model
            scores = self._evaluate_model(
                model, X_train, y_train, X_test, y_test, X_oot, y_oot
            )
            self.scores_[model_name] = scores
            
            print(f"    Train AUC: {scores['train_auc']:.4f}, ", end="")
            if 'test_auc' in scores:
                print(f"Test AUC: {scores['test_auc']:.4f}, ", end="")
            if 'oot_auc' in scores:
                print(f"OOT AUC: {scores['oot_auc']:.4f}")
            else:
                print()
        
        # Select best model
        best_model_name = self.select_best_model()
        
        return {
            'best_model': self.models_[best_model_name],
            'best_model_name': best_model_name,
            'best_score': self.scores_[best_model_name].get('oot_auc', 
                         self.scores_[best_model_name].get('test_auc',
                         self.scores_[best_model_name]['train_auc'])),
            'best_auc': self.scores_[best_model_name].get('oot_auc',
                       self.scores_[best_model_name].get('test_auc',
                       self.scores_[best_model_name]['train_auc'])),
            'all_scores': self.scores_
        }
    
    def _get_model_configs(self) -> Dict:
        """Get model configurations"""
        
        models = {
            'LogisticRegression': {
                'estimator': LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    max_iter=1000,
                    random_state=self.config.random_state,
                    solver='lbfgs'
                ),
                'param_space': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
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
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [10, 20, 50],
                    'min_samples_leaf': [5, 10, 20]
                }
            }
        }
        
        # Add XGBoost if available
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = {
                'estimator': XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.8, 1.0]
                }
            }
        except ImportError:
            pass
        
        # Add LightGBM if available
        try:
            from lightgbm import LGBMClassifier
            models['LightGBM'] = {
                'estimator': LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbosity=-1
                ),
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [15, 31, 63]
                }
            }
        except ImportError:
            pass
        
        # Add CatBoost if available
        try:
            from catboost import CatBoostClassifier
            models['CatBoost'] = {
                'estimator': CatBoostClassifier(
                    iterations=100,
                    depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbose=False
                ),
                'param_space': {
                    'iterations': [50, 100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            }
        except ImportError:
            pass
        
        return models
    
    def _train_with_optuna(self, X_train, y_train, X_val, y_val, 
                          base_estimator, param_space) -> Any:
        """Train model with Optuna hyperparameter optimization"""
        
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            def objective(trial):
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
                
                model = base_estimator.__class__(**{**base_estimator.get_params(), **params})
                
                if X_val is not None:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, pred)
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                    score = scores.mean()
                
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_trials if hasattr(self.config, 'n_trials') else 20)
            
            # Train final model with best params
            best_params = study.best_params
            model = base_estimator.__class__(**{**base_estimator.get_params(), **best_params})
            model.fit(X_train, y_train)
            
            return model
            
        except ImportError:
            # Fallback to default parameters
            base_estimator.fit(X_train, y_train)
            return base_estimator
    
    def _evaluate_model(self, model, X_train, y_train, X_test=None, y_test=None,
                       X_oot=None, y_oot=None) -> Dict:
        """Evaluate model performance"""
        
        scores = {}
        
        # Train scores
        pred_train = model.predict_proba(X_train)[:, 1]
        scores['train_auc'] = roc_auc_score(y_train, pred_train)
        scores['train_gini'] = 2 * scores['train_auc'] - 1
        
        # Test scores
        if X_test is not None and y_test is not None:
            pred_test = model.predict_proba(X_test)[:, 1]
            scores['test_auc'] = roc_auc_score(y_test, pred_test)
            scores['test_gini'] = 2 * scores['test_auc'] - 1
        
        # OOT scores
        if X_oot is not None and y_oot is not None:
            pred_oot = model.predict_proba(X_oot)[:, 1]
            scores['oot_auc'] = roc_auc_score(y_oot, pred_oot)
            scores['oot_gini'] = 2 * scores['oot_auc'] - 1
        
        return scores
    
    def select_best_model(self) -> str:
        """Select best model based on OOT/Test performance"""
        
        if not self.scores_:
            return None
        
        # Prioritize OOT > Test > Train
        best_model = None
        best_score = -np.inf
        
        for model_name, scores in self.scores_.items():
            # Use OOT if available, else test, else train
            if 'oot_auc' in scores:
                score = scores['oot_auc']
            elif 'test_auc' in scores:
                score = scores['test_auc']
            else:
                score = scores['train_auc']
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        print(f"\nBest model: {best_model} (AUC: {best_score:.4f})")
        return best_model