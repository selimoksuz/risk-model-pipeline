"""Quickstart script for running the unified risk pipeline on a rich synthetic dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.data.sample import load_credit_risk_sample


def build_quickstart_config(output_dir: Path) -> Config:
    """Return a configuration that exercises all major pipeline stages."""
    algorithms = [
        "logistic",
        "gam",
        "catboost",
        "lightgbm",
        "xgboost",
        "randomforest",
        "extratrees",
        "woe_boost",
        "woe_li",
        "shao",
        "xbooster",
    ]
    config = Config(
        target_column="target",
        id_column="customer_id",
        time_column="app_dt",
        create_test_split=True,
        stratify_test=True,
        oot_months=2,
        enable_dual=True,
        enable_tsfresh_features=True,
        enable_scoring=True,
        enable_stage2_calibration=True,
        output_folder=str(output_dir),
        selection_steps=["psi", "univariate", "iv", "correlation", "boruta", "stepwise"],
        algorithms=algorithms,
        psi_threshold=0.25,
        iv_threshold=0.02,
        univariate_gini_threshold=0.05,
        correlation_threshold=0.95,
        vif_threshold=5.0,
        woe_binning_strategy="iv_optimal",
        use_optuna=True,
        n_trials=1,
        optuna_timeout=120,
        use_noise_sentinel=True,
        calculate_shap=True,
        shap_sample_size=500,
        risk_band_method="pd_constraints",
        n_risk_bands=8,
        risk_band_min_bins=7,
        risk_band_max_bins=10,
        risk_band_micro_bins=1000,
        risk_band_min_weight=0.05,
        risk_band_max_weight=0.30,
        risk_band_hhi_threshold=0.15,
        risk_band_binomial_pass_weight=0.85,
        risk_band_alpha=0.05,
        risk_band_pd_dr_tolerance=1e-4,
        risk_band_max_iterations=100,
        risk_band_max_phase_iterations=50,
        risk_band_early_stop_rounds=10,
        calibration_stage1_method="isotonic",
        calibration_stage2_method="lower_mean",
        random_state=42,
    )
    config.model_type = "all"
    return config


def run_quickstart(output_dir: Path | str) -> Dict[str, Any]:
    """Execute the pipeline end-to-end and return the results dictionary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_credit_risk_sample()

    cfg = build_quickstart_config(output_dir)
    pipe = UnifiedRiskPipeline(cfg)

    results = pipe.fit(
        sample.development,
        data_dictionary=sample.data_dictionary,
        calibration_df=sample.calibration_longrun,
        stage2_df=sample.calibration_recent,
        risk_band_df=sample.development,
        score_df=sample.scoring_future,
    )
    return results


def main() -> None:
    out_dir = Path("output/credit_risk_sample")
    results = run_quickstart(out_dir)

    print("Quickstart run completed.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    best_model = results.get("best_model_name")
    if best_model:
        print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
