from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_logreg(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    pred = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, pred)
    return {"model": clf, "auc": float(auc)}
