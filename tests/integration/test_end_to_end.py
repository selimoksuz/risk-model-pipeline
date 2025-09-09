#!/usr/bin/env python3
"""
Complete End-to-End Test: Pipeline + Scoring + Reporting
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from risk_pipeline.pipeline import Config, RiskModelPipeline
from risk_pipeline.utils.scoring import load_model_artifacts, score_data

sys.path.append('src')


def clean_outputs():
    """Clean all output folders"""
    folders = ['outputs', 'outputs_full', 'outputs_workflow', 'outputs_clean', 'outputs_cal_test']
    for folder in folders:
        if os.path.exists(folder) and folder != 'outputs':
            try:
                shutil.rmtree(folder)
                print(f"  Cleaned: {folder}")
            except Exception:
                pass


def main():
    print("=" * 60)
    print("END-TO-END TEST: Pipeline + Scoring + Excel Reporting")
    print("=" * 60)

    # Step 0: Clean old outputs
    print("\n[STEP 0] Cleaning old outputs...")
    clean_outputs()

    # Single output folder for everything
    OUTPUT_FOLDER = "outputs"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 1: Generate calibration data
    print("\n[STEP 1] Generating calibration data...")
    from scripts.make_calibration_data import generate_calibration_data
    cal_df = generate_calibration_data(n_samples=300, output_path="data/calibration_e2e.csv")
    print(f"  Generated: {len(cal_df)} samples")
    print(f"  Default rate: {cal_df['target'].mean():.3f}")

    # Step 2: Run pipeline
    print("\n[STEP 2] Running risk model pipeline...")

    # Load training data
    train_df = pd.read_csv('data/input.csv')
    print(f"  Training data: {train_df.shape[0]:, } rows x {train_df.shape[1]} columns")

    # Configure pipeline (minimal for speed)
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        use_test_split=True,
        oot_window_months=3,
        calibration_data_path="data/calibration_e2e.csv",
        calibration_method="isotonic",
        hpo_trials=2,  # Very low for speed
        hpo_timeout_sec=10,  # Very low for speed
        random_state=42
    )

    print("  Running pipeline...")
    start_time = time.time()

    pipeline = RiskModelPipeline(config)
    # results = pipeline.run(train_df)

    pipeline_time = time.time() - start_time
    print(f"  Pipeline completed in {pipeline_time:.1f}s")
    print(f"  Best model: {pipeline.best_model_name_}")
    print(f"  Final features: {pipeline.final_vars_}")
    print(f"  Run ID: {results['run_id']}")

    # Save additional artifacts for scoring
    import json
    from datetime import datetime

    import joblib

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_e2e")

    # Save model
    if pipeline.best_model_name_ and pipeline.models_:
        model = pipeline.models_[pipeline.best_model_name_]
        model_path = f"{OUTPUT_FOLDER}/best_model_{run_id}.joblib"
        joblib.dump(model, model_path)
        print(f"  Model saved: best_model_{run_id}.joblib")

    # Save final features
    if hasattr(pipeline, 'final_vars_'):
        features_path = f"{OUTPUT_FOLDER}/final_vars_{run_id}.json"
        with open(features_path, 'w') as f:
            json.dump(pipeline.final_vars_, f)
        print(f"  Features saved: final_vars_{run_id}.json")

    # Save WOE mapping
    if hasattr(pipeline, 'woe_map'):
        woe_path = f"{OUTPUT_FOLDER}/woe_mapping_{run_id}.json"
        woe_dict = {}
        for var_name, var_info in pipeline.woe_map.items():
            woe_dict[var_name] = {
                'var': var_info.var if hasattr(var_info, 'var') else var_name,
                'bins': []
            }
            if hasattr(var_info, 'bins'):
                for bin_info in var_info.bins:
                    woe_dict[var_name]['bins'].append({'range': list(bin_info['range']) if isinstance(
                        bin_info['range'], (list, tuple)) else bin_info['range'], 'woe': float(bin_info.get('woe', 0))})
        with open(woe_path, 'w') as f:
            json.dump(woe_dict, f)
        print(f"  WOE mapping saved: woe_mapping_{run_id}.json")

    # Save calibrator if exists
    if hasattr(pipeline, 'calibrator_') and pipeline.calibrator_ is not None:
        cal_path = f"{OUTPUT_FOLDER}/calibrator_{run_id}.joblib"
        joblib.dump(pipeline.calibrator_, cal_path)
        print(f"  Calibrator saved: calibrator_{run_id}.joblib")

    # Step 3: Score new data
    print("\n[STEP 3] Scoring new data...")

    # Load scoring data
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"  Scoring data: {scoring_df.shape[0]:, } rows")
    print(f"    With target: {(~scoring_df['target'].isna()).sum():, }")
    print(f"    Without target: {scoring_df['target'].isna().sum():, }")

    # Load model artifacts
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)

    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
    # final_features = list(final_features.values())[0]

print(f"  Model loaded: {type(model).__name__}")
    print(f"  Calibrator: {'Available' if calibrator else 'Not available'}")

    # Calculate training scores for PSI
    from risk_pipeline.utils.scoring import apply_woe_transform
    train_woe = apply_woe_transform(train_df, woe_mapping)

    # Map features
    feature_mapping = {}
    for feat in final_features:
        if feat not in train_woe.columns:
            base_name = feat.split('_')[0] if '_' in feat else feat
            if base_name in train_woe.columns:
                feature_mapping[feat] = base_name

    if feature_mapping:
        X_train = train_woe[[feature_mapping.get(f, f) for f in final_features]]
        X_train.columns = final_features
    else:
        X_train = train_woe[final_features] if all(f in train_woe.columns for f in final_features) else train_woe[[
            f for f in final_features if f in train_woe.columns]]

    # Clean NaN/Inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get training scores
    try:
        training_scores = model.predict_proba(X_train)
        if training_scores.ndim == 2:
            training_scores = training_scores[:, 1]
    except Exception:
        training_scores = model.predict(X_train)

    print(f"  Training scores calculated: {len(training_scores):, }")

    # Score new data
    start_time = time.time()
    # scoring_results = score_data(
        scoring_df = scoring_df,
        model = model,
    # final_features=final_features,
        woe_mapping = woe_mapping,
        calibrator = calibrator,
    # training_scores=training_scores,
        feature_mapping = feature_mapping if feature_mapping else None
    )
    scoring_time=time.time() - start_time
    print(f"  Scoring completed in {scoring_time:.1f}s")

    # Step 4: Display results
    print("\n[STEP 4] Results Summary:")
    print("-" * 40)
    print(f"Total scored: {scoring_results['n_total']:, }")
    print(f"Calibration applied: {'Yes' if scoring_results.get('calibration_applied') else 'No'}")

    if scoring_results.get('psi_score') is not None:
        psi=scoring_results['psi_score']
        print(f"PSI: {psi:.4f} ({'Stable' if psi < 0.1 else 'Some drift' if psi < 0.25 else 'Significant drift'})")

    if 'with_target' in scoring_results:
        wt=scoring_results['with_target']
        print(f"\nWith Target ({wt['n_records']:, } records):")
        print(f"  AUC: {wt['auc']:.4f}")
        print(f"  Gini: {wt['gini']:.4f}")
        print(f"  KS: {wt['ks']:.4f}")
        print(f"  Default Rate: {wt['default_rate']:.3f}")

    if 'without_target' in scoring_results:
        wot=scoring_results['without_target']
        print(f"\nWithout Target ({wot['n_records']:, } records):")
        print(f"  Score Mean: {wot['score_stats']['mean']:.4f}")
        print(f"  Score Std: {wot['score_stats']['std']:.4f}")

    # Save scored data
    scored_df=scoring_df.copy()
    scored_df['predicted_score']=scoring_results['scores']
    scored_df['raw_score']=scoring_results['raw_scores']
    scored_df.to_csv(f"{OUTPUT_FOLDER}/scored_data.csv", index = False)
    print(f"\nScored data saved: {OUTPUT_FOLDER}/scored_data.csv")

    # Step 5: Verify Excel report
    print("\n[STEP 5] Verifying Excel report...")
    excel_path=f"{OUTPUT_FOLDER}/model_report.xlsx"

    if os.path.exists(excel_path):
        xl_file=pd.ExcelFile(excel_path)
        print(f"  Excel report: {excel_path}")
        print(f"  Total sheets: {len(xl_file.sheet_names)}")
        print(f"  Size: {os.path.getsize(excel_path)/1024:.1f} KB")

        # List some key sheets
        key_sheets=['final_vars', 'models_summary', 'woe_mapping', 'psi_summary', 'oot_scores']
        available=[s for s in key_sheets if s in xl_file.sheet_names]
        print(f"  Key sheets: {', '.join(available)}")
    else:
        print(f"  [WARNING] Excel report not found: {excel_path}")

    # Step 6: Final summary
    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Pipeline time: {pipeline_time:.1f}s")
    print(f"Scoring time: {scoring_time:.1f}s")
    print(f"Total time: {pipeline_time + scoring_time:.1f}s")
    print(f"\nAll outputs in single folder: {OUTPUT_FOLDER}/")
    print(f"All reports in single Excel: {excel_path}")

    # List final outputs
    print("\nFinal outputs:")
    outputs=os.listdir(OUTPUT_FOLDER)
    for output in sorted(outputs)[:10]:  # Show first 10
        size=os.path.getsize(f"{OUTPUT_FOLDER}/{output}") / 1024
        print(f"  - {output} ({size:.1f} KB)")

    if len(outputs) > 10:
        print(f"  ... and {len(outputs)-10} more files")

    print("\n[SUCCESS] All systems working correctly!")


if __name__ == "__main__":
    main()
