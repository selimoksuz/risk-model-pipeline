"""
Dual Unified Pipeline Demo (synthetic data)

Generates a richer synthetic dataset with:
- Strong and weak numeric predictors
- Correlated feature pairs (for VIF/correlation checks)
- Categorical predictors with rare levels
- A feature with controlled distribution shift (for PSI)
- Noise variables

Then configures the unified pipeline to run with full selection sequence
(PSI, VIF, Correlation, Boruta, Stepwise), fits the pipeline, and prints a
concise summary. Also prepares recent period data for Stage-2 calibration
and a separate scoring sample to exercise the scoring path.

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


def generate_synthetic(n: int = 12000, seed: int = 42, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt0 = datetime(2023, 1, 1)

    # Month index for drift control
    month_idx = rng.integers(0, months, size=n)
    app_dt = [dt0 + timedelta(days=int(m * 30 + rng.integers(0, 30))) for m in month_idx]

    # Base features
    x_num_strong = rng.normal(0, 1, n)
    x_num_corr = x_num_strong * 0.9 + rng.normal(0, 0.2, n)  # correlated with strong
    x_num_thresh = rng.exponential(1.0, n)
    x_num_weak = rng.normal(0, 1, n)
    x_num_noise1 = rng.normal(0, 1, n)
    x_num_noise2 = rng.normal(0, 1, n)

    # Drifted feature for PSI (drifts in last 3 months)
    x_num_psi = rng.normal(0, 1, n)
    drift_months = set(range(months - 3, months))
    drift_mask = np.array([m in drift_months for m in month_idx])
    x_num_psi[drift_mask] = rng.normal(1.5, 1.0, drift_mask.sum())  # shift mean up

    # Categoricals (with rare levels)
    cat1 = rng.choice(["A", "B", "C", "D", None], size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    cat2_levels = [f"K{i}" for i in range(10)] + [None]
    cat2_probs = np.array([0.10] * 5 + [0.04] * 5 + [0.06])  # create some rare levels
    cat2_probs = cat2_probs / cat2_probs.sum()
    cat2 = rng.choice(cat2_levels, size=n, p=cat2_probs)

    # Target generation with meaningful signal
    # Map categoricals to numeric effects
    cat1_map = {"A": 0.15, "B": 0.0, "C": -0.1, "D": 0.25}
    cat2_map = {lvl: (0.2 if lvl in ["K0", "K3"] else (0.05 if lvl in ["K1", "K7"] else 0.0)) for lvl in cat2_levels}
    cat1_term = pd.Series(cat1, dtype="object").map(cat1_map).fillna(0.0).values
    cat2_term = pd.Series(cat2, dtype="object").map(cat2_map).fillna(0.0).values

    # Base logit and month effect (seasonality)
    season = 0.1 * np.sin(2 * np.pi * (np.array(month_idx) % 12) / 12.0)
    logit = (
        -1.2
        + 1.2 * x_num_strong
        + 0.9 * (x_num_thresh > 1.0).astype(int)
        + 0.3 * x_num_weak
        + 0.25 * (x_num_psi > 0.5).astype(int)
        + cat1_term
        + cat2_term
        + season
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "app_id": np.arange(1, n + 1),
            "app_dt": app_dt,
            "x_num_strong": x_num_strong,
            "x_num_corr": x_num_corr,
            "x_num_thresh": x_num_thresh,
            "x_num_weak": x_num_weak,
            "x_num_psi": x_num_psi,
            "x_num_noise1": x_num_noise1,
            "x_num_noise2": x_num_noise2,
            "cat1": cat1,
            "cat2": cat2,
            "target": y,
        }
    )

    return df


def main():
    _ensure_local_import()

    from risk_pipeline.core.config import Config
    from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

    df = generate_synthetic(n=12000)

    cfg = Config(
        target_col="target",
        id_col="app_id",
        time_col="app_dt",
        enable_scoring=True,
        enable_calibration=True,
        stage2_method="lower_mean",
        enable_woe=True,
        # Full selection sequence incl. Boruta (LightGBM-based)
        selection_order=["psi", "vif", "correlation", "iv", "boruta", "stepwise"],
        iv_threshold=0.0,
        psi_threshold=10.0,
        use_boruta=True,
        forward_selection=True,
        max_features=12,
        use_optuna=False,
        model_type="all",
        use_test_split=True,
        oot_months=3,
        equal_default_splits=False,
        n_risk_bands=10,
        band_method="quantile",
        enable_dual_pipeline=True,
    )

    # Build auxiliary data for calibration and scoring
    # Stage-2 calibration on the most recent month window
    recent_cut = pd.Timestamp(max(df["app_dt"])) - pd.Timedelta(days=60)
    stage2_df = df[pd.to_datetime(df["app_dt"]) >= recent_cut].copy()

    # Separate scoring sample with similar distribution
    score_df = generate_synthetic(n=3000, seed=123)

    pipe = UnifiedRiskPipeline(cfg)
    results = pipe.fit(df=df, calibration_df=None, stage2_df=stage2_df, score_df=score_df)

    selected = results.get("selected_features", [])
    best_name = results.get("best_model_name")
    print("Selected features (n=", len(selected), "):", selected)
    print("Best model:", best_name)

    scored_df = results.get("scoring_results")
    if scored_df is not None and not scored_df.empty:
        preview_cols = [cfg.id_col, "risk_score", cfg.target_col] if cfg.target_col in scored_df.columns else [cfg.id_col, "risk_score"]
        print("\nScoring sample preview:")
        print(scored_df[preview_cols].head())
        band_counts = scored_df["risk_band"].value_counts().sort_index()
        print("\nRisk band distribution:")
        print(band_counts)
    else:
        print("No scoring results were generated.")


if __name__ == "__main__":
    main()
