#!/usr/bin/env python3
"""
Direct test of calibration functionality
"""

import pandas as pd
import sys
import os
sys.path.append('src')


def test_calibration_method():
    print("=== DIRECT CALIBRATION TEST ===")

    # Create a mock pipeline instance
    from risk_pipeline.pipeline import RiskPipeline16Config, RiskPipeline16

    # Load data
    df = pd.read_csv('data/input.csv')
    cal_df = pd.read_csv('data/calibration_test.csv')

    print(f"Training data: {df.shape}")
    print(f"Calibration data: {cal_df.shape}")

    # Create a minimal config
    config = RiskPipeline16Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        calibration_data_path="data/calibration_test.csv",
        calibration_method="isotonic"
    )

    # Initialize pipeline
    pipeline = RiskPipeline16(config)
    pipeline.df_ = df

    # Run minimal setup to get WOE mapping and final vars
    print("\nRunning minimal pipeline setup...")

    # Set up indices
    import numpy as np
    n = len(df)
    train_size = int(0.7 * n)
    pipeline.train_idx_ = np.arange(train_size)
    pipeline.test_idx_ = np.arange(train_size, n)
    pipeline.oot_idx_ = np.array([])

    # Create minimal variable catalog
    from risk_pipeline.stages import create_var_catalog
    pipeline.var_catalog_ = create_var_catalog(df, config.id_col, config.time_col, config.target_col)

    # Create minimal WOE mapping for the calibration test
    woe_map = {}
    for col in ['num1', 'num2', 'num3', 'num4']:
        if col in df.columns:
            # Create a simple WOE mapping - just for testing
            woe_map[f"{col}_corr95"] = type('WOEBin', (), {
                'var': col,
                'bins': [{'range': [-float('inf'), float('inf')], 'woe': 0.1}],
                'special': {},
                'iv': 0.1
            })()

    pipeline.woe_map = woe_map
    pipeline.final_vars_ = list(woe_map.keys())[:2]  # Use first 2 features

    print(f"Final vars: {pipeline.final_vars_}")
    print(f"WOE map keys: {list(pipeline.woe_map.keys())}")

    # Create a simple mock model
    class MockModel:
        def predict_proba(self, X):
            import numpy as np
            # Return random probabilities for testing
            return np.column_stack([
                1 - np.random.random(len(X)) * 0.5,  # Class 0 probs
                np.random.random(len(X)) * 0.5       # Class 1 probs
            ])

    pipeline.models_ = {"MockModel": MockModel()}
    pipeline.best_model_name_ = "MockModel"

    # Now test calibration method
    print("\nTesting calibration method...")
    try:
        pipeline._calibrate_model()

        if pipeline.calibrator_ is not None:
            print("SUCCESS: Calibration successful!")
            print(f"   Calibrator type: {type(pipeline.calibrator_)}")
            if hasattr(pipeline, 'calibration_report_'):
                print(f"   Brier score: {pipeline.calibration_report_.get('brier', 'N/A')}")
        else:
            print("ERROR: Calibration failed - no calibrator created")

    except Exception as e:
        print(f"ERROR: Calibration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_calibration_method()
