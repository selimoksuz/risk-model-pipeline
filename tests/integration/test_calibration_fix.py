#!/usr/bin/env python3
"""
Quick test to verify calibration fix
"""

import os
import sys

import pandas as pd

from risk_pipeline.utils.pipeline_runner import get_full_config, run_pipeline_from_dataframe

sys.path.append('src')


def main():
    print("🚀 === CALIBRATION FIX TEST ===")

    # Load training data
    df = pd.read_csv('data/input.csv')
    print(f"Training data: {df.shape[0]:, } rows x {df.shape[1]} columns")

    # Generate calibration data
    print("Generating calibration data...")
    # calibration_df = generate_calibration_data(n_samples=200, output_path="data/calibration_test.csv")
    print("✅ Calibration data generated")

    # Run pipeline with minimal settings but calibration enabled
    config = get_full_config(
        try_mlp=False,        # Skip MLP for speed
        ensemble=False,       # Skip ensemble for speed
        calibration_data_path="data/calibration_test.csv",
        hpo_trials=2,         # Very low for speed
        hpo_timeout_sec=10,   # Very low for speed
        output_folder="outputs_cal_test",
        output_excel_path="cal_test_report.xlsx",
    )

    print("Running pipeline with calibration...")
    try:
    # pipeline_results = run_pipeline_from_dataframe(
            df = df,
            id_col = "app_id",
            time_col = "app_dt",
            target_col = "target",
            output_folder = "outputs_cal_test",
            output_excel = "cal_test_report.xlsx",
            use_test_split = True,
            oot_months = 3,
            **{
                'try_mlp': config.try_mlp,
                'ensemble': config.ensemble,
                'calibration_data_path': config.calibration_data_path,
                'hpo_trials': config.hpo_trials,
                'hpo_timeout_sec': config.hpo_timeout_sec,
                'orchestrator': config.orchestrator
            }
        )

        print("✅ Pipeline completed successfully!")
        print(f"   Best Model: {pipeline_results['best_model']}")
        print(f"   Run ID: {pipeline_results['run_id']}")

        # Check if calibrator was created
        run_id = pipeline_results['run_id']
        calibrator_path = f"outputs_cal_test/calibrator_{run_id}.joblib"

        if os.path.exists(calibrator_path):
            print("✅ Calibrator file exists!")

            # Try to load it
            import joblib
            calibrator = joblib.load(calibrator_path)
            print(f"✅ Calibrator loaded: {type(calibrator)}")

        else:
            print(f"❌ Calibrator file not found: {calibrator_path}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
