import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_calibrator(scores, y, method="isotonic"):
    if method == "isotonic":
        calib = IsotonicRegression(out_of_bounds="clip")
        calib.fit(scores, y)
        return calib
    elif method == "sigmoid":
        lr = LogisticRegression(max_iter=200)
        lr.fit(scores.reshape(-1, 1), y)
        return lr
    else:
        raise ValueError("Unsupported calibration method")


def apply_calibrator(calib, scores):
    scores = np.asarray(scores)
    if hasattr(calib, "predict_proba"):
        return calib.predict_proba(scores.reshape(-1, 1))[:, 1]
    else:
        return calib.transform(scores)
