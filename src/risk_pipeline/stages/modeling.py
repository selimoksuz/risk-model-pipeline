from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_baseline_logreg(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42) -> Tuple[Dict[str, Any], str]:
    """Train a simple logistic regression as a baseline model.
    Returns (models_dict, best_name).
    """
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))
    name = "logreg"
    return ({name: {"estimator": clf, "auc_train": auc}}, name)
