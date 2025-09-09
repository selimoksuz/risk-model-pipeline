#!/usr/bin/env python3
"""
Test full pipeline with all features enabled
"""

from risk_pipeline.utils.pipeline_runner import run_pipeline_from_dataframe, get_full_config
import pandas as pd
import sys
import os
sys.path.append('src')


def main():
    print("=== FULL PIPELINE TEST ===")

    # Load data
    print("Loading data...")
    df = pd.read_csv('data/input.csv')
    print(f"Dataset: {df.shape[0]:, } rows x {df.shape[1]} columns")

    # Get full configuration with all stages enabled (disable calibration for now)
    config = get_full_config(
        # Enable advanced features
        try_mlp=True,
        ensemble=True,

        # Disable calibration temporarily
        # calibration_data_path="data/calibration.csv",

        # Moderate HPO for faster testing
        hpo_trials=20,
        hpo_timeout_sec=120,

        # Output settings
        output_folder="outputs_full",
        output_excel_path="full_pipeline_report.xlsx",
    )

    print("\nRunning FULL pipeline with ALL stages enabled...")
    print(f"- MLP: {config.try_mlp}")
    print(f"- Ensemble: {config.ensemble}")
    print(f"- Calibration: Disabled (data compatibility issue)")
    print(f"- Dictionary: {config.orchestrator.enable_dictionary}")
    print()

    # Run pipeline
    results = run_pipeline_from_dataframe(
        df=df,
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        output_folder="outputs_full",
        output_excel="full_pipeline_report.xlsx",
        use_test_split=True,
        oot_months=3,
        # Pass config overrides
        **{
            'try_mlp': config.try_mlp,
            'ensemble': config.ensemble,
            # 'calibration_data_path': config.calibration_data_path,
            'hpo_trials': config.hpo_trials,
            'hpo_timeout_sec': config.hpo_timeout_sec,
            'orchestrator': config.orchestrator
        }
    )

    print("\n=== PIPELINE COMPLETED ===")
    print(f"✅ Best Model: {results['best_model']}")
    print(f"✅ Final Features: {len(results['final_features'])} features")
    print(f"✅ Run ID: {results['run_id']}")
    print(f"✅ Output Folder: {results['output_folder']}")

    if results['final_features']:
        print("\n📊 Final Features:")
        for i, feature in enumerate(results['final_features'], 1):
            print(f"  {i}. {feature}")

    # Load and display results
    try:
        models_df = pd.read_excel(f"{results['output_folder']}/full_pipeline_report.xlsx", sheet_name='models_summary')
        print(f"\n🏆 Model Performance (AUC OOT):")
        for _, row in models_df.iterrows():
            print(f"  {row['model_name']}: {row['AUC_OOT']:.4f}")
    except Exception as e:
        print(f"Could not load results: {e}")

    print(f"\n📁 Check detailed results in: {results['output_folder']}/")


if __name__ == "__main__":
    main()
