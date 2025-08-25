#!/usr/bin/env python3
"""
Fast End-to-End Test: Pipeline + Scoring + Excel
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import joblib
import json
from datetime import datetime
sys.path.append('src')

from risk_pipeline.pipeline16 import Config, RiskModelPipeline
from risk_pipeline.utils.scoring import load_model_artifacts, score_data, create_scoring_report

def main():
    print("="*70)
    print("FAST END-TO-END TEST")
    print("="*70)
    
    OUTPUT_FOLDER = "outputs"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # PART 1: FAST PIPELINE
    print("\n[PART 1] RUNNING FAST PIPELINE")
    print("-" * 40)
    
    # Load data
    train_df = pd.read_csv('data/input.csv')
    print(f"Training data: {train_df.shape[0]:,} rows")
    
    # Fast config
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        use_test_split=True,
        oot_window_months=3,
        
        # SPEED SETTINGS
        hpo_trials=1,        # No optimization
        hpo_timeout_sec=5,   # 5 second timeout
        cv_folds=2,          # Only 2 folds
        n_jobs=1,            # Single thread
        
        # Simplifications
        min_bins_numeric=3,
        rare_threshold=0.05,
        calibration_data_path=None,  # No calibration
        
        random_state=42
    )
    
    print("Config: HPO=1 trial, CV=2 folds, No calibration")
    
    # Run pipeline
    print("Running pipeline...")
    start_time = time.time()
    
    pipeline = RiskModelPipeline(config)
    results = pipeline.run(train_df)
    
    pipeline_time = time.time() - start_time
    print(f"Pipeline completed in {pipeline_time:.1f}s")
    print(f"  Model: {pipeline.best_model_name_}")
    print(f"  Features: {pipeline.final_vars_}")
    
    # Save artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_e2e")
    
    if pipeline.best_model_name_ and pipeline.models_:
        model = pipeline.models_[pipeline.best_model_name_]
        joblib.dump(model, f"{OUTPUT_FOLDER}/best_model_{run_id}.joblib")
    
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
    
    # PART 2: SCORING
    print("\n[PART 2] SCORING")
    print("-" * 40)
    
    # Load scoring data
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"Scoring data: {scoring_df.shape[0]:,} rows")
    print(f"  With target: {(~scoring_df['target'].isna()).sum():,}")
    print(f"  Without target: {scoring_df['target'].isna().sum():,}")
    
    # Load model artifacts
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)
    
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
        final_features = list(final_features.values())[0]
    
    # Calculate training scores for PSI
    from risk_pipeline.utils.scoring import apply_woe_transform
    train_woe = apply_woe_transform(train_df, woe_mapping)
    
    # Feature mapping
    feature_mapping = {}
    for feat in final_features:
        if feat not in train_woe.columns:
            base_name = feat.split('_')[0]
            if base_name in train_woe.columns:
                feature_mapping[feat] = base_name
    
    if feature_mapping:
        X_train = train_woe[[feature_mapping.get(f, f) for f in final_features]]
        X_train.columns = final_features
    else:
        X_train = train_woe[final_features] if all(f in train_woe.columns for f in final_features) else pd.DataFrame()
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    try:
        training_scores = model.predict_proba(X_train)
        if training_scores.ndim == 2:
            training_scores = training_scores[:, 1]
    except:
        training_scores = model.predict(X_train)
    
    # Score new data
    print("Scoring...")
    start_time = time.time()
    
    scoring_results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=final_features,
        woe_mapping=woe_mapping,
        calibrator=calibrator,
        training_scores=training_scores,
        feature_mapping=feature_mapping if feature_mapping else None
    )
    
    scoring_time = time.time() - start_time
    print(f"Scoring completed in {scoring_time:.1f}s")
    
    # PART 3: RESULTS
    print("\n[PART 3] RESULTS")
    print("-" * 40)
    
    print(f"Total scored: {scoring_results['n_total']:,}")
    
    if scoring_results.get('psi_score') is not None:
        psi = scoring_results['psi_score']
        print(f"PSI: {psi:.4f} ({'Stable' if psi < 0.1 else 'Drift detected'})")
    
    if 'with_target' in scoring_results:
        wt = scoring_results['with_target']
        print(f"\nWith Target ({wt['n_records']:,} records):")
        print(f"  AUC={wt['auc']:.3f} | Gini={wt['gini']:.3f} | KS={wt['ks']:.3f}")
    
    if 'without_target' in scoring_results:
        wot = scoring_results['without_target']
        print(f"\nWithout Target ({wot['n_records']:,} records):")
        print(f"  Score Mean={wot['score_stats']['mean']:.3f}")
    
    # PART 4: EXCEL CHECK
    print("\n[PART 4] EXCEL REPORT")
    print("-" * 40)
    
    excel_path = f"{OUTPUT_FOLDER}/model_report.xlsx"
    if os.path.exists(excel_path):
        xl = pd.ExcelFile(excel_path)
        print(f"Excel: {excel_path}")
        print(f"  Sheets: {len(xl.sheet_names)}")
        print(f"  Size: {os.path.getsize(excel_path)/1024:.1f} KB")
        
        # Check for PSI sheets
        psi_sheets = [s for s in xl.sheet_names if 'psi' in s.lower()]
        scoring_sheets = [s for s in xl.sheet_names if 'scoring' in s.lower() or 'score' in s.lower()]
        
        print(f"  PSI sheets: {len(psi_sheets)}")
        print(f"  Scoring sheets: {len(scoring_sheets)}")
    
    # SUMMARY
    print("\n" + "="*70)
    print("FAST E2E TEST COMPLETED")
    print("="*70)
    print(f"Pipeline: {pipeline_time:.1f}s")
    print(f"Scoring: {scoring_time:.1f}s")
    print(f"Total: {pipeline_time + scoring_time:.1f}s")
    print(f"\nOutput folder: {OUTPUT_FOLDER}/")
    print(f"Excel report: model_report.xlsx")
    print("\n[SUCCESS] All systems working!")

if __name__ == "__main__":
    main()