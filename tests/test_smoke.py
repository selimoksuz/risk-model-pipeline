import pandas as pd

from risk_pipeline.model.train import train_logreg


def test_smoke():
    X = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7], 'b': [1, 1, 0, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    res = train_logreg(X, y, test_size=0.25, seed=123)
    assert 'auc' in res and 0.0 <= res['auc'] <= 1.0
