#!/usr/bin/env python3
"""
Run minimal pipeline to generate model quickly
"""

from risk_pipeline.pipeline import Config, RiskModelPipeline
import pandas as pd
import sys
import os
import time
sys.path.append('src')


def main():
    print("=== RUNNING MINIMAL PIPELINE ===")

    # Load data
    df = pd.read_csv('data/input.csv')
    print(f"Data loaded: {df.shape}")

    # Minimal config for speed
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        use_test_split=True,
        oot_window_months=3,
        hpo_trials=3,  # Very low for speed
        hpo_timeout_sec=15,  # Very low for speed
        calibration_data_path=None,
        random_state=42
    )

    # Run pipeline
    print("Running pipeline...")
    start = time.time()

    pipeline = RiskModelPipeline(config)
    # results = pipeline.run(df)

    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.1f}s")

    # Save artifacts to outputs folder
    import joblib
    import json
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Save model
    if pipeline.best_model_name_ and pipeline.models_:
        model = pipeline.models_[pipeline.best_model_name_]
        model_path = f"{output_folder}/best_model_{run_id}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

    # Save final features
    if hasattr(pipeline, 'final_vars_'):
        features_path = f"{output_folder}/final_vars_{run_id}.json"
        with open(features_path, 'w') as f:
            json.dump(pipeline.final_vars_, f)
        print(f"Features saved: {features_path}")

    # Save WOE mapping
    if hasattr(pipeline, 'woe_map'):
        woe_path = f"{output_folder}/woe_mapping_{run_id}.json"

        # Convert WOE mapping to serializable format
        woe_dict = {}
        for var_name, var_info in pipeline.woe_map.items():
            woe_dict[var_name] = {
                'var': var_info.var if hasattr(var_info, 'var') else var_name,
                'bins': []
            }
            if hasattr(var_info, 'bins'):
                for bin_info in var_info.bins:
                    woe_dict[var_name]['bins'].append(
                        {
                            'range': list(
                                bin_info['range']) if isinstance(
                                bin_info['range'], (list, tuple)) else bin_info['range'], 'woe': float(
                                bin_info.get(
                                    'woe', 0)), 'count': int(
                                bin_info.get(
                                    'count', 0)) if 'count' in bin_info else 0, 'event_rate': float(
                                    bin_info.get(
                                        'event_rate', 0)) if 'event_rate' in bin_info else 0})

        with open(woe_path, 'w') as f:
            json.dump(woe_dict, f)
        print(f"WOE mapping saved: {woe_path}")

    print(f"\n✅ Pipeline artifacts saved to: {output_folder}/")
    print(f"   Run ID: {run_id}")
    print(f"   Best Model: {pipeline.best_model_name_}")
    print(f"   Final Features: {pipeline.final_vars_}")


if __name__ == "__main__":
    main()
