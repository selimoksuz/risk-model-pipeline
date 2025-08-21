import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X, shap_sample=25000, random_state=42):
    if shap_sample and len(X) > shap_sample:
        Xs = X.sample(shap_sample, random_state=random_state)
    else:
        Xs = X
    explainer = shap.Explainer(model, Xs)
    return explainer(Xs)


def summarize_shap(shap_values, feature_names):
    vals = np.abs(shap_values.values).mean(axis=0)
    return {name: float(val) for name, val in zip(feature_names, vals)}
