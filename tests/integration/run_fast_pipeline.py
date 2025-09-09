#!/usr/bin/env python3
"""
Fast pipeline run with minimal configuration for quick testing
"""

import pandas as pd
import sys
import os
import time
import joblib
import json
from datetime import datetime
sys.path.append('src')

from risk_pipeline.pipeline import Config, RiskModelPipeline

def main():
    print("="*60)
    print("FAST PIPELINE RUN")
    print("="*60)

    OUTPUT_FOLDER = "outputs"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('data/input.csv')
    print(f"  Data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Create MINIMAL config for maximum speed
    print("\n[2] Configuration (optimized for speed):")
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        use_test_split=True,
        oot_window_months=3,

        # SPEED OPTIMIZATIONS
        hpo_trials=1,           # Only 1 trial (no optimization)
        hpo_timeout_sec=5,      # 5 second timeout
        n_jobs=1,               # Single thread (sometimes faster for small data)
        cv_folds=2,             # Only 2 folds instead of 5

        # Skip some expensive operations
        calibration_data_path=None,  # Skip calibration for speed

        # Reduce feature engineering complexity
        min_bins_numeric=3,     # Fewer bins for WOE
        rare_threshold=0.05,    # Higher threshold = fewer categories

        random_state=42
    )

    print("  HPO trials: 1 (no optimization)")
    print("  HPO timeout: 5 seconds")
    print("  CV folds: 2")
    print("  Calibration: Disabled")
    print("  WOE bins: 3 (minimal)")

    # Run pipeline
    print("\n[3] Running pipeline...")
    start_time = time.time()

    # Limit model types in config
    config.orchestrator.enable_xgb = False
    config.orchestrator.enable_lgb = False
    config.orchestrator.enable_rf = False
    config.orchestrator.enable_et = False
    config.orchestrator.enable_mlp = False
    config.orchestrator.enable_gam = False
    config.orchestrator.enable_ensemble = False
    # Only keep Logistic Regression (fastest)
    config.orchestrator.enable_logreg = True

    pipeline = RiskModelPipeline(config)
    results = pipeline.run(df)

    pipeline_time = time.time() - start_time
    print(f"\n  Pipeline completed in {pipeline_time:.1f}s")

    # Save artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_fast")

    if pipeline.best_model_name_ and pipeline.models_:
        model = pipeline.models_[pipeline.best_model_name_]
        joblib.dump(model, f"{OUTPUT_FOLDER}/best_model_{run_id}.joblib")
        print(f"  Model saved: best_model_{run_id}.joblib")

    if hasattr(pipeline, 'final_vars_'):
        with open(f"{OUTPUT_FOLDER}/final_vars_{run_id}.json", 'w') as f:
            json.dump(pipeline.final_vars_, f)

    if hasattr(pipeline, 'woe_map'):
        woe_dict = {}
        for var_name, var_info in pipeline.woe_map.items():
            woe_dict[var_name] = {
                'var': var_info.var if hasattr(var_info, 'var') else var_name,
                'bins': []
            }
            if hasattr(var_info, 'bins'):
                for bin_info in var_info.bins:
                    woe_dict[var_name]['bins'].append({
                        'range': list(bin_info['range']) if isinstance(bin_info['range'], (list, tuple)) else bin_info['range'],
                        'woe': float(bin_info.get('woe', 0))
                    })
        with open(f"{OUTPUT_FOLDER}/woe_mapping_{run_id}.json", 'w') as f:
            json.dump(woe_dict, f)

    print(f"\n[4] Results:")
    print(f"  Best Model: {pipeline.best_model_name_}")
    print(f"  Final Features: {pipeline.final_vars_}")
    print(f"  Runtime: {pipeline_time:.1f} seconds")

    print(f"\n[SUCCESS] Fast pipeline completed!")

if __name__ == "__main__":
    main()
