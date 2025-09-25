"""
Comprehensive Model Builder with all ML algorithms
Supports: Logistic, GAM, CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import mutual_info_classif
import warnings

from .utils import gini_from_auc

warnings.filterwarnings('ignore')



try:
    import xgboost as _xgb
    _XGBClassifier = getattr(_xgb, 'XGBClassifier', None)
    _XGB_VERSION = getattr(_xgb, '__version__', '0.0.0')
except ImportError:
    _xgb = None
    _XGBClassifier = None
    _XGB_VERSION = '0.0.0'


_USE_LABEL_ENCODER_PARAM = False


def _xgb_supports_label_encoder(version_str: str) -> bool:
    try:
        parts = [int(part) for part in version_str.split('.')[:2]]
        if len(parts) == 1:
            parts.append(0)
        major, minor = parts[0], parts[1]
    except Exception:
        return True
    return (major, minor) < (1, 6)


if _XGBClassifier is not None:
    _USE_LABEL_ENCODER_PARAM = _xgb_supports_label_encoder(_XGB_VERSION)


def _make_xgb_classifier(**kwargs):
    if _XGBClassifier is None:
        raise ImportError('xgboost is required for XGBoost-based models')
    params = dict(kwargs)
    if _USE_LABEL_ENCODER_PARAM:
        params.setdefault('use_label_encoder', False)
    else:
        params.pop('use_label_encoder', None)
    params.setdefault('eval_metric', params.get('eval_metric', 'logloss'))
    return _XGBClassifier(**params)


class WoeBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosted model configured for WOE-transformed features."""

    def __init__(
        self,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 1,
        max_depth: int = -1,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        num_boost_round: int = 120,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.max_depth = max_depth
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.booster_ = None
        self.feature_importances_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError('LightGBM is required for WoeBoost') from exc

        X_values = X.values if hasattr(X, 'values') else np.asarray(X)
        y_values = np.asarray(y)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': float(self.learning_rate),
            'num_leaves': int(self.num_leaves),
            'min_data_in_leaf': int(self.min_data_in_leaf),
            'feature_fraction': max(0.1, min(1.0, float(self.feature_fraction))),
            'bagging_fraction': max(0.1, min(1.0, float(self.bagging_fraction))),
            'bagging_freq': max(0, int(self.bagging_freq)),
            'max_depth': -1 if self.max_depth is None or int(self.max_depth) < 0 else int(self.max_depth),
            'lambda_l1': float(self.lambda_l1),
            'lambda_l2': float(self.lambda_l2),
            'feature_pre_filter': False,
            'verbosity': -1,
            'seed': int(self.random_state),
            'deterministic': True,
            'force_row_wise': True,
        }

        dataset = lgb.Dataset(X_values, label=y_values, free_raw_data=True)
        num_boost_round = max(10, int(self.num_boost_round))
        self.booster_ = lgb.train(params, dataset, num_boost_round=num_boost_round)

        self.classes_ = np.array(sorted(np.unique(y_values)))
        self.n_features_in_ = X_values.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f'f_{idx}' for idx in range(self.n_features_in_)]
        self.feature_importances_ = self.booster_.feature_importance(importance_type='gain')
        return self

    def predict_proba(self, X):
        if self.booster_ is None:
            raise ValueError('Model is not fitted.')

        X_values = X.values if hasattr(X, 'values') else np.asarray(X)
        preds = self.booster_.predict(X_values)
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class WoeLogisticInteractionClassifier(BaseEstimator, ClassifierMixin):
    """Logistic model on WOE features enriched with pairwise interactions."""

    def __init__(
        self,
        top_k: int = 5,
        include_original: bool = True,
        penalty: str = 'l2',
        C: float = 1.0,
        max_iter: int = 2000,
        solver: Optional[str] = None,
        random_state: int = 42,
    ):
        self.top_k = max(1, int(top_k))
        self.include_original = include_original
        self.penalty = penalty
        self.C = C
        self.max_iter = max(100, int(max_iter))
        self.solver = solver
        self.random_state = random_state
        self.model_ = None
        self.base_columns_: List[str] = []
        self.interaction_pairs_: List[Tuple[str, str]] = []
        self.feature_names_in_: List[str] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(np.asarray(X).shape[1])])
        self.base_columns_ = list(X_df.columns)

        data = X_df.fillna(0.0).astype(float)
        if data.shape[1] == 0:
            raise ValueError('WoeLogisticInteractionClassifier requires at least one feature.')

        try:
            scores = mutual_info_classif(
                data.values,
                np.asarray(y),
                discrete_features=False,
                random_state=self.random_state,
            )
        except Exception:
            scores = np.zeros(data.shape[1], dtype=float)

        order = np.argsort(scores)[::-1]
        top_indices = order[: min(self.top_k, len(order))]
        top_features = [self.base_columns_[i] for i in top_indices]

        design = data.copy() if self.include_original else pd.DataFrame(index=data.index)
        interaction_pairs: List[Tuple[str, str]] = []
        for f1, f2 in combinations(top_features, 2):
            col_name = f"{f1}__x__{f2}"
            design[col_name] = data[f1].values * data[f2].values
            interaction_pairs.append((f1, f2))
        self.interaction_pairs_ = interaction_pairs

        self.feature_names_in_ = list(design.columns)
        design_matrix = design.values
        y_array = np.asarray(y)

        solver = self.solver
        if solver is None:
            solver = 'liblinear' if self.penalty == 'l1' else 'lbfgs'

        model = LogisticRegression(
            penalty=self.penalty,
            C=float(self.C),
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(design_matrix, y_array)

        self.model_ = model
        self.classes_ = model.classes_
        return self

    def _prepare_features(self, X) -> pd.DataFrame:
        if hasattr(X, 'columns'):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.base_columns_)
        X_df = X_df.reindex(columns=self.base_columns_, fill_value=0.0).fillna(0.0).astype(float)

        if self.include_original:
            design = X_df.copy()
        else:
            design = pd.DataFrame(index=X_df.index)
        for f1, f2 in self.interaction_pairs_:
            col_name = f"{f1}__x__{f2}"
            design[col_name] = X_df[f1].values * X_df[f2].values

        if not self.feature_names_in_:
            self.feature_names_in_ = list(design.columns)
        design = design.reindex(columns=self.feature_names_in_, fill_value=0.0)
        return design

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError('Model is not fitted.')
        design = self._prepare_features(X)
        return self.model_.predict_proba(design.values)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ShaoLogitClassifier(BaseEstimator, ClassifierMixin):
    """Penalised logistic regression with CV-based regularisation (Shao et al.)."""

    def __init__(
        self,
        Cs: Optional[List[float]] = None,
        cv: int = 5,
        penalty: str = 'l1',
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.Cs = Cs or [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.cv = max(2, int(cv))
        self.penalty = penalty
        self.max_iter = max(100, int(max_iter))
        self.random_state = random_state
        self.model_: Optional[LogisticRegressionCV] = None
        self.feature_names_in_: List[str] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(np.asarray(X).shape[1])])
        data = X_df.fillna(0.0).astype(float)
        solver = 'liblinear' if self.penalty == 'l1' else 'lbfgs'

        model = LogisticRegressionCV(
            Cs=self.Cs,
            cv=self.cv,
            penalty=self.penalty,
            solver=solver,
            scoring='roc_auc',
            max_iter=self.max_iter,
            random_state=self.random_state,
            refit=True,
        )
        model.fit(data.values, np.asarray(y))

        self.model_ = model
        self.feature_names_in_ = list(data.columns)
        self.classes_ = model.classes_
        return self

    def _prepare_array(self, X) -> np.ndarray:
        if hasattr(X, 'columns'):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        X_df = X_df.reindex(columns=self.feature_names_in_, fill_value=0.0).fillna(0.0).astype(float)
        return X_df.values

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError('Model is not fitted.')
        return self.model_.predict_proba(self._prepare_array(X))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)



class XBoosterClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper around XGBoost with scorecard generation via xbooster."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        generate_scorecard: bool = True,
        scorecard_kwargs: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.generate_scorecard = generate_scorecard
        self.scorecard_kwargs = scorecard_kwargs
        self.random_state = random_state
        self.model_ = None
        self.scorecard_constructor_ = None
        self.scorecard_frame_ = None
        self.scorecard_points_ = None
        self.scorecard_error_: Optional[str] = None

    def fit(self, X, y):  # noqa: N803 - sklearn signature
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_array = np.asarray(X)
            X_df = pd.DataFrame(X_array, columns=[f'feature_{i}' for i in range(X_array.shape[1])])

        y_series = y if isinstance(y, pd.Series) else pd.Series(y)

        self.model_ = _make_xgb_classifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_df, y_series)
        self.feature_importances_ = getattr(self.model_, 'feature_importances_', None)
        self.classes_ = np.array(sorted(np.unique(y_series)))
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = list(X_df.columns)

        self.scorecard_constructor_ = None
        self.scorecard_frame_ = None
        self.scorecard_points_ = None
        self.scorecard_error_ = None

        if self.generate_scorecard:
            try:
                from xbooster.constructor import XGBScorecardConstructor  # type: ignore

                constructor = XGBScorecardConstructor(self.model_, X_df, y_series)
                constructor.construct_scorecard()
                kwargs = dict(self.scorecard_kwargs or {})
                scorecard_points = constructor.create_points(**kwargs)
                scorecard_points = constructor.add_detailed_split(scorecard_points.copy())
                self.scorecard_constructor_ = constructor
                self.scorecard_frame_ = constructor.xgb_scorecard.copy() if constructor.xgb_scorecard is not None else None
                self.scorecard_points_ = scorecard_points.copy() if hasattr(scorecard_points, 'copy') else scorecard_points
                self.scorecard_error_ = None
            except ImportError:
                self.scorecard_constructor_ = None
                self.scorecard_frame_ = None
                self.scorecard_points_ = None
                self.scorecard_error_ = 'xbooster is not installed'
            except Exception as exc:  # pragma: no cover - defensive
                self.scorecard_constructor_ = None
                self.scorecard_frame_ = None
                self.scorecard_points_ = None
                self.scorecard_error_ = str(exc)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError('Model is not fitted.')
        return self.model_.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_scorecard(self, with_points: bool = True):
        if with_points:
            return self.scorecard_points_
        return self.scorecard_frame_



class ComprehensiveModelBuilder:
    """
    Model builder supporting all major ML algorithms with optional Optuna optimization.
    """

    def __init__(self, config):
        self.config = config
        self.models_ = {}
        self.scores_ = {}
        self.feature_importance_ = {}
        self.interpretability_ = {}
        self.n_features_in_ = None
        self.feature_names_in_: List[str] = []

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: Optional[pd.DataFrame] = None,
                         y_test: Optional[pd.Series] = None,
                         X_oot: Optional[pd.DataFrame] = None,
                         y_oot: Optional[pd.Series] = None) -> Dict:
        """Train all configured models and select the best."""

        print(f"  Training models on {X_train.shape[1]} features...")

        self.interpretability_ = {}
        self.n_features_in_ = X_train.shape[1]
        if hasattr(X_train, 'columns'):
            self.feature_names_in_ = list(X_train.columns)
        else:
            self.feature_names_in_ = [f'feature_{idx}' for idx in range(self.n_features_in_)]

        if self.n_features_in_ == 0:
            print("    WARNING: No features available for training. Skipping model training.")
            return {
                'models': {},
                'scores': {},
                'best_model': None,
                'best_model_name': None,
                'feature_importance': {},
                'interpretability': {},
                'selected_features': []
            }

        models_to_train = self._get_models_to_train()

        for model_name in models_to_train:
            print(f"    Training {model_name}...")

            try:
                if self.config.use_optuna and self._supports_optuna(model_name):
                    model = self._train_with_optuna(
                        model_name, X_train, y_train, X_test, y_test
                    )
                else:
                    model = self._train_without_optuna(
                        model_name, X_train, y_train
                    )

                self.models_[model_name] = model

                train_score = self._evaluate_model(model, X_train, y_train)
                test_score = None
                if X_test is not None and y_test is not None and len(X_test) > 0:
                    test_score = self._evaluate_model(model, X_test, y_test)

                oot_score = None
                if X_oot is not None and y_oot is not None and len(X_oot) > 0:
                    oot_score = self._evaluate_model(model, X_oot, y_oot)

                def _safe_gini(value: Optional[float]) -> Optional[float]:
                    if value is None:
                        return None
                    try:
                        return gini_from_auc(value)
                    except Exception:
                        return None

                train_gini = _safe_gini(train_score)
                test_gini = _safe_gini(test_score)
                oot_gini = _safe_gini(oot_score)
                gap_reference = oot_gini if oot_gini is not None else test_gini
                gap_value = abs(train_gini - gap_reference) if train_gini is not None and gap_reference is not None else None

                self.scores_[model_name] = {
                    'train_auc': train_score,
                    'test_auc': test_score,
                    'oot_auc': oot_score,
                    'train_gini': train_gini,
                    'test_gini': test_gini,
                    'oot_gini': oot_gini,
                    'train_oot_gap': gap_value,
                }

                self.feature_importance_[model_name] = self._get_feature_importance(
                    model, X_train.columns, model_name
                )
                self.interpretability_[model_name] = self._collect_interpretability(
                    model, model_name
                )

                metrics_to_show = [f"Train AUC: {train_score:.4f}"]
                if oot_score is not None:
                    metrics_to_show.append(f"OOT AUC: {oot_score:.4f}")
                if test_score is not None:
                    metrics_to_show.append(f"Test AUC: {test_score:.4f}")
                if gap_value is not None and not np.isinf(gap_value):
                    metrics_to_show.append(f"|Train-OOT Gini gap|: {gap_value:.4f}")
                print("      " + ", ".join(metrics_to_show))

            except Exception as e:
                print(f"      Failed: {str(e)[:50]}...")
                continue

        best_model_name = self._select_best_model()

        if best_model_name is None or best_model_name not in self.models_:
            return {
                'models': self.models_,
                'scores': self.scores_,
                'best_model': None,
                'best_model_name': None,
                'feature_importance': self.feature_importance_,
                'interpretability': self.interpretability_,
                'selected_features': list(X_train.columns) if X_train is not None and len(X_train.columns) > 0 else []
            }

        return {
            'models': self.models_,
            'scores': self.scores_,
            'best_model': self.models_[best_model_name],
            'best_model_name': best_model_name,
            'feature_importance': self.feature_importance_,
            'interpretability': self.interpretability_,
            'selected_features': list(X_train.columns)
        }

    def _supports_optuna(self, model_name: str) -> bool:
        """Return True if the model has an Optuna search space."""

        return model_name in {
            'RandomForest',
            'ExtraTrees',
            'LightGBM',
            'XGBoost',
            'CatBoost',
            'GAM',
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
            'XBooster',
            'GAM',
            'WoeBoost',
            'WoeLogisticInteraction',
            'ShaoLogit'
        ]

        mapping = {
            'logistic': 'LogisticRegression',
            'logistic_regression': 'LogisticRegression',
            'logreg': 'LogisticRegression',
            'randomforest': 'RandomForest',
            'random_forest': 'RandomForest',
            'extratrees': 'ExtraTrees',
            'extra_trees': 'ExtraTrees',
            'lightgbm': 'LightGBM',
            'xgboost': 'XGBoost',
            'catboost': 'CatBoost',
            'xbooster': 'XBooster',
            'x_booster': 'XBooster',
            'gam': 'GAM',
            'woe_boost': 'WoeBoost',
            'woeboost': 'WoeBoost',
            'woe_li': 'WoeLogisticInteraction',
            'woeli': 'WoeLogisticInteraction',
            'woe_logistic_interaction': 'WoeLogisticInteraction',
            'shao': 'ShaoLogit',
            'shao_logit': 'ShaoLogit',
        }

        def _canonical(name: str) -> str:
            key = str(name).lower().replace('-', '_')
            return mapping.get(key, name)

        def _expand(selection) -> List[str]:
            if selection is None:
                return []
            if isinstance(selection, str):
                items = [selection]
            else:
                items = list(selection)
            expanded: List[str] = []
            for item in items:
                canonical = _canonical(item)
                if canonical.lower() == 'all':
                    expanded.extend(all_models)
                else:
                    expanded.append(canonical)
            return expanded

        default_algorithms = ['logistic', 'gam', 'catboost', 'lightgbm', 'xgboost', 'randomforest', 'extratrees', 'woe_boost', 'woe_li', 'shao', 'xbooster']
        default_selection = _expand(default_algorithms)

        model_type = getattr(self.config, 'model_type', 'all')
        algorithms_config = getattr(self.config, 'algorithms', default_algorithms)

        model_type_selection = _expand(model_type)
        algorithms_selection = _expand(algorithms_config)
        algorithms_customized = bool(algorithms_selection) and set(algorithms_selection) != set(default_selection)

        if isinstance(model_type, str) and model_type.lower() == 'all':
            requested = algorithms_selection if algorithms_customized else [m for m in all_models]
        elif isinstance(model_type, (list, tuple, set)):
            requested = model_type_selection
        else:
            requested = model_type_selection or (algorithms_selection if algorithms_customized else [m for m in all_models])

        if not requested:
            requested = algorithms_selection or [m for m in all_models]

        seen = set()
        result: List[str] = []
        for name in requested:
            canonical = _canonical(name)
            if canonical not in seen and self._is_model_available(canonical):
                seen.add(canonical)
                result.append(canonical)

        if not result:
            for name in all_models:
                if self._is_model_available(name):
                    result.append(name)
        return result

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
        elif model_name == 'XBooster':
            try:
                import xgboost
                import xbooster  # noqa: F401
                return True
            except Exception:
                return False
        elif model_name == 'WoeBoost':
            try:
                import lightgbm
                return True
            except:
                return False
        elif model_name == 'WoeLogisticInteraction':
            return True
        elif model_name == 'ShaoLogit':
            return True
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
            model = _make_xgb_classifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state
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

        elif model_name == 'XBooster':
            return self._train_xbooster(X_train, y_train)

        elif model_name == 'WoeBoost':
            return self._train_woe_boost(X_train, y_train)

        elif model_name == 'WoeLogisticInteraction':
            return self._train_woe_li(X_train, y_train)

        elif model_name == 'ShaoLogit':
            return self._train_shao_logit(X_train, y_train)

        elif model_name == 'GAM':
            model = self._build_gam_model(
                X_train.shape[1],
                n_splines=10,
                lam=0.6,
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        return model

    def _train_woe_boost(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = WoeBoostClassifier(random_state=self.config.random_state)
        if params:
            try:
                model = model.set_params(**params)
            except ValueError as exc:
                raise RuntimeError(f'Invalid WoeBoost parameters: {exc}') from exc
        return model.fit(X_train, y_train)

    def _train_woe_li(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = WoeLogisticInteractionClassifier(random_state=self.config.random_state)
        if params:
            try:
                model = model.set_params(**params)
            except ValueError as exc:
                raise RuntimeError(f'Invalid WoeLogisticInteraction parameters: {exc}') from exc
        return model.fit(X_train, y_train)

    def _train_shao_logit(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = ShaoLogitClassifier(random_state=self.config.random_state)
        if params:
            try:
                model = model.set_params(**params)
            except ValueError as exc:
                raise RuntimeError(f'Invalid ShaoLogit parameters: {exc}') from exc
        return model.fit(X_train, y_train)

    def _train_xbooster(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = XBoosterClassifier(random_state=self.config.random_state)
        if params:
            try:
                model = model.set_params(**params)
            except ValueError as exc:
                raise RuntimeError(f'Invalid XBooster parameters: {exc}') from exc
        return model.fit(X_train, y_train)

    def _train_with_optuna(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None):
        """Train model with Optuna hyperparameter optimization."""

        try:
            import optuna
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                params = self._get_optuna_params(trial, model_name)
                if model_name == 'GAM':
                    params = dict(params)
                    params['n_features'] = X_train.shape[1]

                model = self._create_model_with_params(model_name, params)

                if X_test is not None and y_test is not None and len(X_test) > 0:
                    model.fit(X_train, y_train)
                    y_pred = self._predict_positive_proba(model, X_test)
                    score = roc_auc_score(y_test, y_pred)
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                    score = scores.mean()

                return score

            study = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
            n_trials = getattr(self.config, 'n_optuna_trials', getattr(self.config, 'n_trials', 20))
            timeout = getattr(self.config, 'optuna_timeout', None)
            if timeout is not None and timeout <= 0:
                timeout = None
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            best_params = dict(study.best_params)
            if model_name == 'GAM':
                best_params['n_features'] = X_train.shape[1]
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

        elif model_name == 'XBooster':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 2, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
            }

        elif model_name == 'GAM':
            return {
                'n_splines': trial.suggest_int('n_splines', 5, 25),
                'lam': trial.suggest_float('lam', 0.001, 10.0, log=True)
            }

        return {}


    def _build_gam_model(self, n_features: int, *, n_splines: int, lam: float):
        from pygam import LogisticGAM, s

        n_features = int(n_features)
        if n_features <= 0:
            raise ValueError('GAM requires at least one feature for training.')

        n_splines = int(n_splines)
        lam = float(lam)
        terms = s(0, n_splines=n_splines, lam=lam)
        for idx in range(1, n_features):
            terms = terms + s(idx, n_splines=n_splines, lam=lam)
        return LogisticGAM(terms=terms, verbose=False)

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
            params = dict(params)
            params.setdefault('random_state', self.config.random_state)
            return _make_xgb_classifier(**params)

        elif model_name == 'CatBoost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                **params,
                random_state=self.config.random_state,
                verbose=False
            )

        elif model_name == 'XBooster':
            model = XBoosterClassifier(random_state=self.config.random_state)
            if params:
                model = model.set_params(**params)
            return model

        elif model_name == 'WoeBoost':
            model = WoeBoostClassifier(random_state=self.config.random_state)
            if params:
                model = model.set_params(**params)
            return model

        elif model_name == 'WoeLogisticInteraction':
            model = WoeLogisticInteractionClassifier(random_state=self.config.random_state)
            if params:
                model = model.set_params(**params)
            return model

        elif model_name == 'ShaoLogit':
            model = ShaoLogitClassifier(random_state=self.config.random_state)
            if params:
                model = model.set_params(**params)
            return model

        elif model_name == 'GAM':
            params = dict(params or {})
            n_features = params.get('n_features', self.n_features_in_)
            if n_features is None:
                raise ValueError("GAM requires the number of features to be provided via 'n_features'.")
            n_splines = params.get('n_splines', 10)
            lam = params.get('lam', 0.6)
            return self._build_gam_model(int(n_features), n_splines=int(n_splines), lam=float(lam))

        raise ValueError(f"Unknown model: {model_name}")


    def _predict_positive_proba(self, model, X):
        """Return probability estimates for the positive class as a 1D array."""

        if X is None:
            return None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                return proba
            if proba.ndim == 2:
                if proba.shape[1] == 1:
                    return proba[:, 0]
                return proba[:, -1]

        if hasattr(model, 'decision_function'):
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                from scipy.special import expit
                return expit(scores)
            if scores.ndim == 2:
                try:
                    from scipy.special import softmax
                except ImportError:
                    return np.asarray(scores)[:, -1]
                return softmax(scores, axis=1)[:, -1]

        preds = model.predict(X)
        return np.asarray(preds).ravel()


    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
        """Evaluate model performance."""

        if X is None or y is None:
            return None

        try:
            y_pred = self._predict_positive_proba(model, X)
            return roc_auc_score(y, y_pred)
        except Exception:
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

    def _collect_interpretability(self, model, model_name: str) -> Dict[str, Any]:
        """Collect supplementary interpretability artifacts for a model."""

        artifacts: Dict[str, Any] = {}

        if model_name == 'XBooster':
            if hasattr(model, 'scorecard_points_') and getattr(model, 'scorecard_points_', None) is not None:
                artifacts['scorecard_points'] = model.scorecard_points_
            if hasattr(model, 'scorecard_frame_') and getattr(model, 'scorecard_frame_', None) is not None:
                artifacts['scorecard'] = model.scorecard_frame_
            error = getattr(model, 'scorecard_error_', None)
            if error:
                artifacts['warnings'] = [error]

        return artifacts

    def _select_best_model(self) -> Optional[str]:
        """Select best model using configurable stability-aware criteria."""

        if not self.scores_:
            return None

        selection_method = str(getattr(self.config, 'model_selection_method', 'gini_oot') or 'gini_oot').lower()
        max_gap = getattr(self.config, 'max_train_oot_gap', None)
        stability_weight = float(getattr(self.config, 'model_stability_weight', 0.0) or 0.0)
        min_gini = float(getattr(self.config, 'min_gini_threshold', 0.5) or 0.0)

        def _effective_gini(entry):
            for key in ('oot_gini', 'test_gini', 'train_gini'):
                value = entry.get(key)
                if value is not None:
                    return value
            return None

        def _gap_value(entry):
            gap = entry.get('train_oot_gap')
            if gap is None:
                return np.inf
            return gap

        def _gini_key(entry):
            value = _effective_gini(entry)
            return value if value is not None else -np.inf

        stats = []
        for name, metrics in self.scores_.items():
            record = dict(metrics)
            if record.get('train_gini') is None and record.get('train_auc') is not None:
                record['train_gini'] = gini_from_auc(record['train_auc'])
            if record.get('test_gini') is None and record.get('test_auc') is not None:
                record['test_gini'] = gini_from_auc(record['test_auc'])
            if record.get('oot_gini') is None and record.get('oot_auc') is not None:
                record['oot_gini'] = gini_from_auc(record['oot_auc'])
            if record.get('train_oot_gap') is None:
                ref = record.get('oot_gini')
                if ref is None:
                    ref = record.get('test_gini')
                train_gini = record.get('train_gini')
                if train_gini is not None and ref is not None:
                    record['train_oot_gap'] = abs(train_gini - ref)
            stats.append({'model_name': name, **record})

        if not stats:
            return None

        candidates = stats
        if max_gap is not None:
            filtered = [rec for rec in stats if _gap_value(rec) <= max_gap]
            if filtered:
                candidates = filtered
            else:
                print(f"      Warning: No models satisfy stability gap <= {max_gap}. Using full set.")

        if not candidates:
            return None

        selection = selection_method
        if selection == 'balanced':
            for rec in candidates:
                gini_val = _effective_gini(rec)
                gap_val = _gap_value(rec)
                if gini_val is None:
                    rec['balanced_score'] = -np.inf
                else:
                    penalty = gap_val
                    if np.isinf(penalty):
                        penalty = 0.0 if stability_weight == 0 else np.inf
                    rec['balanced_score'] = (1 - stability_weight) * gini_val - stability_weight * penalty
            best = max(candidates, key=lambda rec: rec.get('balanced_score', -np.inf))
        elif selection == 'stable':
            eligible = [rec for rec in candidates if (_effective_gini(rec) is not None and _effective_gini(rec) >= min_gini)]
            if eligible:
                best = min(eligible, key=lambda rec: (_gap_value(rec), -_gini_key(rec)))
            else:
                best = max(candidates, key=lambda rec: (_gini_key(rec), -_gap_value(rec)))
        elif selection == 'conservative':
            best = min(candidates, key=lambda rec: (_gap_value(rec), -_gini_key(rec)))
        else:
            best = max(candidates, key=lambda rec: (_gini_key(rec), -_gap_value(rec)))

        metric_label = 'Train'
        best_auc = best.get('train_auc')
        if best.get('oot_auc') is not None:
            metric_label = 'OOT'
            best_auc = best['oot_auc']
        elif best.get('test_auc') is not None:
            metric_label = 'Test'
            best_auc = best['test_auc']

        if best_auc is not None:
            auc_display = f"{best_auc:.4f}"
        else:
            auc_display = 'n/a'
        gap_val = best.get('train_oot_gap')
        gap_str = ''
        if gap_val is not None and not np.isinf(gap_val):
            gap_str = f", |train-{metric_label.lower()} gini gap|: {gap_val:.4f}"

        print(f"    Best model: {best['model_name']} ({metric_label} AUC: {auc_display}, method: {selection_method}){gap_str}")
        return best['model_name']

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
