import numpy as np
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(scores, y):
    ir = IsotonicRegression(out_of_bounds="clip")
    return ir.fit(scores, y)
