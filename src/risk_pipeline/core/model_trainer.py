"""Model training and evaluation module"""

import numpy as np
import pandas as pd
import time
import optuna
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    
try:
    from pygam import LogisticGAM
except ImportError:
    LogisticGAM = None

from .utils import ks_statistic, ks_table, gini_from_auc, now_str, sys_metrics

class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation"""
    
    def __init__(self, config):
        self.cfg = config
        self.models_ = {}
        self.models_summary_ = None
        self.best_model_name_ = None
        self.calibrator_ = None
        
    def get_model_definitions(self) -> Dict:
        """Get model definitions and hyperparameter spaces"""
        models = {
            "Logit_L2": (
                LogisticRegression(
                    solver="lbfgs", 
                    max_iter=1000, 
                    class_weight="balanced"
                ),
                {"C": np.logspace(-3, 3, 7)},
            ),
            "RandomForest": (
                RandomForestClassifier(
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    random_state=getattr(self.cfg, 'random_state', 42),
                    class_weight="balanced_subsample",
                ),
                {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 5, 20],
                },
            ),
            "ExtraTrees": (
                ExtraTreesClassifier(
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    random_state=getattr(self.cfg, 'random_state', 42),
                    class_weight="balanced",
                ),
                {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 5, 20],
                },
            ),
        }
        
        if XGBClassifier:
            models["XGBoost"] = (
                XGBClassifier(
                    eval_metric="logloss",
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    random_state=getattr(self.cfg, 'random_state', 42),
                    tree_method="hist",
                    verbosity=0,
                ),
                {
                    "n_estimators": [200, 500],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.7, 1.0],
                },
            )
        
        if LGBMClassifier:
            models["LightGBM"] = (
                LGBMClassifier(
                    class_weight="balanced",
                    n_jobs=getattr(self.cfg, 'n_jobs', -1),
                    random_state=getattr(self.cfg, 'random_state', 42),
                    verbosity=-1,
                    min_child_samples=10,
                ),
                {
                    "n_estimators": [300, 500],
                    "num_leaves": [31, 63],
                    "max_depth": [-1, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.7, 1.0],
                },
            )
        
        if LogisticGAM:
            models["GAM"] = (
                LogisticGAM(max_iter=200),
                {"lam": np.logspace(-3, 3, 7)},
            )
        
        if getattr(self.cfg, 'try_mlp', False):
            models["MLP"] = (
                MLPClassifier(
                    random_state=self.cfg.random_state, 
                    max_iter=200
                ),
                {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]},
            )
        
        return models
    
    def hyperparameter_tune(
        self, 
        base_estimator, 
        param_dist, 
        X, 
        y
    ) -> Any:
        """Hyperparameter tuning with Optuna or random search"""
        if not param_dist:
            return base_estimator
        
        # Special handling for GAM with low feature count
        if LogisticGAM and isinstance(base_estimator, LogisticGAM) and X.shape[1] < 3:
            print(f"   - GAM: Skipping HPO (only {X.shape[1]} features)")
            return base_estimator
        
        # Use hpo_method from Config or default to optuna
        method = getattr(self.cfg, 'hpo_method', 'optuna')
        timeout = getattr(self.cfg, 'hpo_timeout_sec', 60)
        n_trials = getattr(self.cfg, 'hpo_trials', 10)
        
        if method == "optuna" and optuna:
            try:
                return self._tune_with_optuna(
                    base_estimator, param_dist, X, y, 
                    n_trials, timeout
                )
            except Exception:
                method = "random"  # Fallback to random search
        
        # Random search
        return self._tune_with_random_search(
            base_estimator, param_dist, X, y, 
            n_trials, timeout
        )
    
    def _tune_with_optuna(
        self, 
        base_estimator, 
        param_dist, 
        X, 
        y, 
        n_trials, 
        timeout
    ):
        """Hyperparameter tuning using Optuna"""
        skf = StratifiedKFold(
            n_splits=getattr(self.cfg, 'cv_folds', 5), 
            shuffle=True, 
            random_state=self.cfg.random_state
        )
        
        def objective(trial):
            params = {}
            for param, values in param_dist.items():
                if isinstance(values, list):
                    params[param] = trial.suggest_categorical(param, values)
                elif len(values) == 2:
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, values[0], values[1])
                    else:
                        params[param] = trial.suggest_float(param, values[0], values[1])
            
            mdl = clone(base_estimator).set_params(**params)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                mdl.fit(X.iloc[train_idx], y[train_idx])
                pred = self._predict_proba(mdl, X.iloc[val_idx])
                ks, _ = ks_statistic(y[val_idx], pred)
                scores.append(ks)
            
            return float(np.mean(scores)) if scores else -np.inf
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=getattr(self.cfg, 'random_state', 42)))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_trial.params if study.best_trial else {}
        return clone(base_estimator).set_params(**best_params)
    
    def _tune_with_random_search(
        self, 
        base_estimator, 
        param_dist, 
        X, 
        y, 
        n_trials, 
        timeout
    ):
        """Hyperparameter tuning using random search"""
        skf = StratifiedKFold(
            n_splits=getattr(self.cfg, 'cv_folds', 5), 
            shuffle=True, 
            random_state=self.cfg.random_state
        )
        
        # Generate random parameter combinations
        np.random.seed(getattr(self.cfg, 'random_state', 42))
        param_combinations = []
        
        for _ in range(n_trials):
            params = {}
            for param, values in param_dist.items():
                if isinstance(values, list):
                    params[param] = np.random.choice(values)
                else:
                    params[param] = values[np.random.randint(len(values))]
            param_combinations.append(params)
        
        best_score = -np.inf
        best_params = {}
        start = time.time()
        
        for params in param_combinations:
            mdl = clone(base_estimator).set_params(**params)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                mdl.fit(X.iloc[train_idx], y[train_idx])
                pred = self._predict_proba(mdl, X.iloc[val_idx])
                ks, _ = ks_statistic(y[val_idx], pred)
                scores.append(ks)
                
                if time.time() - start > timeout:
                    break
            
            if scores:
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_params = params
            
            if time.time() - start > timeout:
                break
        
        return clone(base_estimator).set_params(**best_params)
    
    def _predict_proba(self, model, X) -> np.ndarray:
        """Get probability predictions handling different model types"""
        try:
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            
            if proba.ndim == 1:
                return proba.ravel()
            if proba.shape[1] == 2:
                return proba[:, 1]
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba.max(axis=1)
            
        except Exception:
            # Handle GAM or models without predict_proba
            if hasattr(model, "predict_mu"):
                return np.asarray(model.predict_mu(X)).ravel()
            
            try:
                scores = np.asarray(model.decision_function(X)).ravel()
                return 1.0 / (1.0 + np.exp(-scores))
            except Exception:
                return np.asarray(model.predict(X)).ravel()
    
    def train_and_evaluate_models(
        self, 
        X_train, 
        y_train,
        X_test, 
        y_test,
        X_oot, 
        y_oot,
        feature_names: List[str],
        prefix: str = ""
    ) -> pd.DataFrame:
        """Train and evaluate all models"""
        models = self.get_model_definitions()
        rows = []
        
        skf = StratifiedKFold(
            n_splits=getattr(self.cfg, 'cv_folds', 5), 
            shuffle=True, 
            random_state=self.cfg.random_state
        )
        
        for name, (base_mdl, params) in models.items():
            # Skip GAM if we have less than 3 features
            if name == "GAM" and len(feature_names) < 3:
                print(f"   - Skipping GAM (requires >=3 features, have {len(feature_names)})")
                continue
            
            print(f"[{now_str()}]   - {prefix}{name} tuning{sys_metrics()}")
            
            # Hyperparameter tuning
            mdl = self.hyperparameter_tune(base_mdl, params, X_train[feature_names], y_train)
            
            print(f"[{now_str()}]   - {prefix}{name} CV basliyor{sys_metrics()}")
            
            # Cross-validation evaluation
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train[feature_names], y_train):
                m = clone(mdl)
                m.fit(X_train.iloc[train_idx][feature_names], y_train[train_idx])
                p = self._predict_proba(m, X_train.iloc[val_idx][feature_names])
                
                ks, _ = ks_statistic(y_train[val_idx], p)
                try:
                    auc = roc_auc_score(y_train[val_idx], p)
                except Exception:
                    auc = 0.5
                
                cv_scores.append((ks, auc))
            
            ks_cv = float(np.mean([s[0] for s in cv_scores])) if cv_scores else 0.0
            auc_cv = float(np.mean([s[1] for s in cv_scores])) if cv_scores else 0.5
            gini_cv = gini_from_auc(auc_cv) if auc_cv != 0.5 else 0.0
            
            # Train final model
            mdl.fit(X_train[feature_names], y_train)
            
            # Training metrics
            p_train = self._predict_proba(mdl, X_train[feature_names])
            ks_train, _ = ks_statistic(y_train, p_train)
            
            try:
                auc_train = roc_auc_score(y_train, p_train)
                gini_train = gini_from_auc(auc_train)
            except Exception:
                auc_train = 0.5
                gini_train = 0.0
            
            # Test metrics (if test set exists)
            ks_test = auc_test = gini_test = None
            if X_test is not None and y_test is not None and X_test.shape[0] > 0:
                p_test = self._predict_proba(mdl, X_test[feature_names])
                ks_test, _ = ks_statistic(y_test, p_test)
                
                try:
                    auc_test = roc_auc_score(y_test, p_test)
                    gini_test = gini_from_auc(auc_test)
                except Exception:
                    auc_test = 0.5
                    gini_test = 0.0
            
            # OOT metrics
            p_oot = self._predict_proba(mdl, X_oot[feature_names])
            ks_oot, thr_oot = ks_statistic(y_oot, p_oot)
            
            try:
                auc_oot = roc_auc_score(y_oot, p_oot)
                gini_oot = gini_from_auc(auc_oot)
            except Exception:
                auc_oot = 0.5
                gini_oot = 0.0
            
            # Store results
            rows.append({
                "model_name": f"{prefix}{name}",
                "KS_Train": ks_train,
                "AUC_Train": auc_train,
                "Gini_Train": gini_train,
                "KS_TrainCV": ks_cv,
                "AUC_TrainCV": auc_cv,
                "Gini_TrainCV": gini_cv,
                "KS_Test": ks_test,
                "AUC_Test": auc_test,
                "Gini_Test": gini_test,
                "KS_OOT": ks_oot,
                "AUC_OOT": auc_oot,
                "Gini_OOT": gini_oot,
                "KS_OOT_threshold": thr_oot,
            })
            
            # Store model
            self.models_[f"{prefix}{name}"] = mdl
        
        return pd.DataFrame(rows)
    
    def select_best_model(self, models_summary: pd.DataFrame) -> str:
        """Select best model based on Gini_OOT"""
        if models_summary is None or models_summary.empty:
            return None
        
        # Select model with highest Gini_OOT
        best_row = models_summary.sort_values(
            ["Gini_OOT", "KS_OOT", "AUC_OOT"], 
            ascending=[False, False, False]
        ).iloc[0]
        
        return str(best_row["model_name"])
    
    def calibrate_model(self, model, X_cal, y_cal):
        """Calibrate model probabilities"""
        try:
            from ..stages import fit_calibrator, apply_calibrator
            
            raw_proba = self._predict_proba(model, X_cal)
            
            self.calibrator_ = fit_calibrator(
                raw_proba, 
                y_cal, 
                method=getattr(self.cfg, 'calibration_method', 'isotonic')
            )
            
            calibrated_proba = apply_calibrator(self.calibrator_, raw_proba)
            
            from sklearn.calibration import brier_score_loss
            brier = brier_score_loss(y_cal, calibrated_proba)
            
            return self.calibrator_, {"brier": float(brier)}
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None, None