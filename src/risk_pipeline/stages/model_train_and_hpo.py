from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_logreg(X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X, y)
    return clf


def hpo_logreg(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 30, timeout: int | None = None, random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """Simple Optuna-based HPO for LogisticRegression(C, penalty='l2'). Falls back to fixed model if  # noqa: E501
    try:
        import optuna
    except Exception:
        clf = train_logreg(X, y)
        proba = clf.predict_proba(X)[:, 1]
        return clf, {"auc_train": float(roc_auc_score(y, proba)), "method": "fixed"}

    def objective(trial: optuna.Trial):
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 1000)
        clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=C)
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        return auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params
    best = LogisticRegression(max_iter=best_params.get("max_iter", 500), solver="lbfgs", C=best_params.get("C", 1.0))
    best.fit(X, y)
    proba = best.predict_proba(X)[:, 1]
    return best, {"auc_train": float(roc_auc_score(y, proba)), "best_params": best_params, "method": "optuna"}
