"""Quickstart script for running the unified risk pipeline on a rich synthetic dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "credit_risk_sample"
DEVELOPMENT_CSV = DATA_DIR / "development.csv"
CALIBRATION_LONGRUN_CSV = DATA_DIR / "calibration_longrun.csv"
CALIBRATION_RECENT_CSV = DATA_DIR / "calibration_recent.csv"
SCORING_CSV = DATA_DIR / "scoring_future.csv"
DICTIONARY_CSV = DATA_DIR / "data_dictionary.csv"


def build_quickstart_config(output_dir: Path) -> Config:
    """Return a configuration that exercises all major pipeline stages."""
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
        n_risk_bands=6,
        risk_band_method="quantile",
        max_psi=0.6,
        selection_steps=["psi", "univariate", "iv", "correlation", "stepwise"],
        algorithms=["logistic", "lightgbm"],
        use_optuna=False,
        calculate_shap=False,
        use_noise_sentinel=False,
        random_state=42,
    )
    config.model_type = ['LogisticRegression', 'LightGBM']
    return config


def run_quickstart(output_dir: Path | str) -> Dict[str, Any]:
    """Execute the pipeline end-to-end and return the results dictionary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        DEVELOPMENT_CSV,
        CALIBRATION_LONGRUN_CSV,
        CALIBRATION_RECENT_CSV,
        SCORING_CSV,
        DICTIONARY_CSV,
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthetic dataset files: {missing}")

    development_df = pd.read_csv(DEVELOPMENT_CSV)
    calibration_longrun_df = pd.read_csv(CALIBRATION_LONGRUN_CSV)
    calibration_recent_df = pd.read_csv(CALIBRATION_RECENT_CSV)
    scoring_df = pd.read_csv(SCORING_CSV)
    data_dictionary = pd.read_csv(DICTIONARY_CSV)

    cfg = build_quickstart_config(output_dir)
    pipe = UnifiedRiskPipeline(cfg)

    results = pipe.fit(
        development_df,
        data_dictionary=data_dictionary,
        calibration_df=calibration_longrun_df,
        stage2_df=calibration_recent_df,
        score_df=scoring_df,
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
