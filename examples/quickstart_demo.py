"""Quickstart script for running the unified risk pipeline on a tiny synthetic dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "quickstart"
INPUT_CSV = DATA_DIR / "loan_applications.csv"
DICTIONARY_CSV = DATA_DIR / "data_dictionary.csv"


def build_quickstart_config(output_dir: Path) -> Config:
    """Return a lightweight configuration that keeps the full pipeline logic but runs fast."""
    config = Config(
        target_column="target",
        id_column="app_id",
        time_column="app_dt",
        create_test_split=True,
        test_size=0.25,
        stratify_test=True,
        oot_months=1,
        enable_dual=False,
        enable_tsfresh_features=False,
        enable_scoring=True,
        output_folder=str(output_dir),
        enable_stage2_calibration=False,
        n_risk_bands=5,
        risk_band_method="quantile",
        max_psi=0.6,
        selection_steps=["psi", "univariate", "iv", "correlation", "stepwise"],
        algorithms=["logistic"],
        use_optuna=False,
        calculate_shap=False,
        use_noise_sentinel=False,
        random_state=42,
    )
    config.model_type = 'LogisticRegression'
    return config


def run_quickstart(output_dir: Path | str) -> Dict[str, Any]:
    """Execute the pipeline end-to-end and return the results dictionary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists() or not DICTIONARY_CSV.exists():
        raise FileNotFoundError(
            "Quickstart input files are missing. Regenerate data via generate_quickstart_data helper."
        )

    df = pd.read_csv(INPUT_CSV)
    data_dictionary = pd.read_csv(DICTIONARY_CSV)

    cfg = build_quickstart_config(output_dir)
    pipe = UnifiedRiskPipeline(cfg)

    results = pipe.fit(df, data_dictionary=data_dictionary, score_df=df)
    return results


def main() -> None:
    out_dir = Path("output/quickstart")
    results = run_quickstart(out_dir)

    print("Quickstart run completed.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    best_model = results.get("best_model_name")
    if best_model:
        print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
