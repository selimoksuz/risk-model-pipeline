import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X, shap_sample=25000, random_state=42):
    """Return SHAP explanation for a (possibly sampled) subset of X.

    The returned object is the standard :class:`shap.Explanation` instance, but
    the sampled feature matrix is also returned so callers can align SHAP values
    with the original rows for further diagnostics.
    """

    if shap_sample and len(X) > shap_sample:
        Xs = X.sample(shap_sample, random_state=random_state)
    else:
        Xs = X
    explainer = shap.Explainer(model, Xs)
    return explainer(Xs), Xs


def summarize_shap(shap_values, feature_names):
    """Average absolute SHAP value per feature."""
    vals = np.abs(shap_values.values).mean(axis=0)
    return {name: float(val) for name, val in zip(feature_names, vals)}


def analyze_shap(shap_values, Xs, split_series=None, psi_summary=None):
    """Generate advanced diagnostics for SHAP explanations.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation as returned by :func:`compute_shap_values`.
    Xs : pandas.DataFrame
        The feature matrix used for computing ``shap_values``.
    split_series : pandas.Series, optional
        Optional series indicating temporal or cohort splits corresponding to
        the rows of ``Xs``.  When provided, SHAP importances are computed per
        split and their rank correlation is reported to assess stability.
    psi_summary : pandas.DataFrame, optional
        PSI summary table produced during model development.  When supplied, the
        maximum PSI for each variable is included for cross-checking with SHAP
        importance.

    Returns
    -------
    tuple(pandas.DataFrame, dict)
        A pair of ``(details_df, stability)`` where ``details_df`` contains per
        feature diagnostics (mean |SHAP|, feature/SHAP Spearman correlation and
        PSI) and ``stability`` holds overall split stability indicators.
    """

    df_shap = pd.DataFrame(shap_values.values, columns=Xs.columns, index=Xs.index)
    mean_abs = df_shap.abs().mean()

    # Monotonicity / direction via Spearman correlation between feature and SHAP
    spearman = {}
    for c in Xs.columns:
        try:
            spearman[c] = float(pd.Series(Xs[c]).corr(df_shap[c], method="spearman"))
        except Exception:
            spearman[c] = np.nan

    # Cross-check with PSI results if available
    psi_map = {}
    if psi_summary is not None and not psi_summary.empty:
        psi_map = psi_summary.groupby("variable")["psi_value"].max().to_dict()

    # Split-based stability: rank correlation of SHAP importances across splits
    stability = None
    if split_series is not None:
        tmp = df_shap.abs().join(split_series.rename("__split"))
        grp = {}
        for g, d in tmp.groupby("__split"):
            grp[g] = d.drop(columns="__split").mean().rank(ascending=False)
        if len(grp) > 1:
            rank_df = pd.DataFrame(grp)
            corr = rank_df.corr(method="spearman")
            stability = {
                "mean_rank_corr": float(corr.where(~np.eye(len(corr), dtype=bool)).mean().mean()),
                "min_rank_corr": float(corr.where(~np.eye(len(corr), dtype=bool)).min().min()),
            }

    rows = []
    for c in Xs.columns:
        rows.append(
            {
                "variable": c,
                "mean_abs_shap": float(mean_abs.get(c, 0.0)),
                "spearman_corr": spearman.get(c),
                "psi_value": psi_map.get(c),
            }
        )

    return pd.DataFrame(rows), stability

