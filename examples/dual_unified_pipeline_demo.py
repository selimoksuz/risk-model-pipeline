"""
Dual Unified Pipeline Demo (synthetic data)

Generates a synthetic dataset, configures the unified pipeline to run in
dual mode (WOE + RAW), fits the pipeline, and prints a concise summary.

This script is self-contained for local runs by adding the repository's
src/ to sys.path when the package is not installed.
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def _ensure_local_import():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def generate_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt0 = datetime(2023, 1, 1)

    app_dt = [dt0 + timedelta(days=int(d)) for d in rng.integers(0, 365, size=n)]
    app_id = np.arange(1, n + 1)
    x1 = rng.normal(0, 1, n)
    x2 = rng.exponential(1.0, n)
    x3 = rng.binomial(1, 0.3, n)
    cat_choices = np.array(["A", "B", "C", "D", None], dtype=object)
    cat = rng.choice(cat_choices, size=n, p=[0.3, 0.3, 0.2, 0.15, 0.05])

    cat_map = {"A": 0.2, "B": 0.0, "C": -0.1, "D": 0.3}
    cat_term = pd.Series(cat).map(cat_map).fillna(0).values
    logit = -1.0 + 0.8 * x1 + 0.5 * (x2 > 1.0).astype(int) + 0.6 * x3 + cat_term
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    return pd.DataFrame(
        {
            "app_id": app_id,
            "app_dt": app_dt,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "cat": cat,
            "target": y,
        }
    )


def main():
    _ensure_local_import()

    from risk_pipeline.core.config import Config
    from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

    df = generate_synthetic(n=4000)

    cfg = Config(
        target_col="target",
        id_col="app_id",
        time_col="app_dt",
        enable_scoring=False,
        enable_calibration=True,
        stage2_method="lower_mean",
        enable_woe=True,
        selection_order=["psi", "vif", "correlation", "iv", "boruta", "stepwise"],
        use_optuna=False,
        model_type="LogisticRegression",  # keep deps minimal for demo
        use_test_split=True,
        oot_months=3,
        equal_default_splits=True,
        n_risk_bands=10,
        band_method="quantile",
        enable_dual_pipeline=True,
    )

    pipe = UnifiedRiskPipeline(cfg)
    results = pipe.fit(df)

    selected = results.get("selected_features", [])
    best_name = results.get("best_model_name")
    print("Selected features (n=", len(selected), "):", selected)
    print("Best model:", best_name)

    # If scoring is needed, one can enable config.enable_scoring and provide score_df
    # Here we just show that pipeline ran end-to-end.


if __name__ == "__main__":
    main()

