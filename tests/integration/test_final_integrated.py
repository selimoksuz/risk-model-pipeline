#!/usr/bin/env python3
"""
Final integrated test - Complete workflow with single folder, enhanced Excel report
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import json
import joblib
from datetime import datetime
sys.path.append('src')

from risk_pipeline.utils.scoring import load_model_artifacts, score_data, create_scoring_report

def enhance_excel_report(excel_path, scoring_results, scoring_df):
    """Add scoring results to existing pipeline Excel report"""
    
    print(f"\nEnhancing Excel report with scoring results...")
    
    # Read existing Excel
    existing_sheets = {}
    with pd.ExcelFile(excel_path) as xl:
        for sheet_name in xl.sheet_names:
            existing_sheets[sheet_name] = pd.read_excel(xl, sheet_name)
    
    # Create scoring reports
    reports = create_scoring_report(scoring_results)
    
    # Write everything back with new sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Write existing sheets first
        for sheet_name, df in existing_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add new scoring sheets
        
        # 1. Scoring Summary
        reports['summary'].to_excel(writer, sheet_name='scoring_summary', index=False)
        print("  Added: scoring_summary")
        
        # 2. With Target Analysis
        if 'with_target' in reports:
            reports['with_target'].to_excel(writer, sheet_name='scoring_with_target', index=False)
            print("  Added: scoring_with_target")
        
        # 3. Without Target Analysis
        if 'without_target' in reports:
            reports['without_target'].to_excel(writer, sheet_name='scoring_without_target', index=False)
            print("  Added: scoring_without_target")
        
        # 4. Score Distribution Analysis
        scores_df = pd.DataFrame({
            'app_id': scoring_df['app_id'],
            'has_target': ~scoring_df['target'].isna(),
            'target': scoring_df['target'],
            'predicted_score': scoring_results['scores'],
            'raw_score': scoring_results['raw_scores']
        })
        
        # Create score bins
        try:
            scores_df['score_decile'] = pd.qcut(scores_df['predicted_score'], 
                                                q=10, 
                                                labels=False,
                                                duplicates='drop')
            scores_df['score_decile'] = scores_df['score_decile'].apply(lambda x: f'D{x+1}' if pd.notna(x) else 'D1')
        except:
            # If not enough unique values, use simple cut
            scores_df['score_decile'] = pd.cut(scores_df['predicted_score'], 
                                               bins=5, 
                                               labels=[f'D{i+1}' for i in range(5)])
        
        # Calculate statistics by decile
        decile_stats = scores_df.groupby('score_decile').agg({
            'app_id': 'count',
            'target': ['mean', 'sum', lambda x: x.isna().sum()],
            'predicted_score': ['mean', 'min', 'max'],
            'has_target': 'sum'
        }).round(4)
        
        decile_stats.columns = ['Count', 'Default_Rate', 'Defaults', 'Missing_Target', 
                                'Avg_Score', 'Min_Score', 'Max_Score', 'Has_Target_Count']
        decile_stats.to_excel(writer, sheet_name='score_distribution')
        print("  Added: score_distribution")
        
        # 5. PSI Analysis (if available)
        if scoring_results.get('psi_score') is not None:
            psi_df = pd.DataFrame({
                'Metric': ['PSI Score', 'PSI Status', 'Interpretation'],
                'Value': [
                    f"{scoring_results['psi_score']:.4f}" if scoring_results['psi_score'] is not None else 'N/A',
                    'Stable' if scoring_results.get('psi_score', 1) < 0.1 else 
                    'Some Drift' if scoring_results.get('psi_score', 1) < 0.25 else 'Significant Drift',
                    'PSI < 0.1: Stable | 0.1 <= PSI < 0.25: Some drift | PSI >= 0.25: Significant drift'
                ]
            })
            psi_df.to_excel(writer, sheet_name='psi_analysis', index=False)
            print("  Added: psi_analysis")
    
    print(f"Excel report enhanced: {excel_path}")

def main():
    print("=== FINAL INTEGRATED TEST ===")
    print("Complete workflow with single folder and enhanced Excel\n")
    
    # Single output folder
    OUTPUT_FOLDER = "outputs"
    
    # Find latest model
    model_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('best_model_') and f.endswith('.joblib')]
    if not model_files:
        print("No model found. Please run pipeline first.")
        return
    
    latest_model = sorted(model_files)[-1]
    run_id = latest_model.replace('best_model_', '').replace('.joblib', '')
    
    print(f"Using model from run: {run_id}")
    
    # Load artifacts
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)
    
    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
        final_features = list(final_features.values())[0]
    
    print(f"Model: {type(model).__name__}")
    print(f"Final features: {final_features}")
    print(f"Calibrator: {'Available' if calibrator else 'Not available'}")
    
    # Load data
    train_df = pd.read_csv('data/input.csv')
    scoring_df = pd.read_csv('data/scoring.csv')
    
    print(f"\nData:")
    print(f"  Training: {train_df.shape[0]:,} rows")
    print(f"  Scoring: {scoring_df.shape[0]:,} rows")
    print(f"    - With target: {(~scoring_df['target'].isna()).sum():,}")
    print(f"    - Without target: {scoring_df['target'].isna().sum():,}")
    
    # Calculate training scores for PSI
    print("\nCalculating training scores for PSI...")
    
    # Apply WOE to training data
    from risk_pipeline.utils.scoring import apply_woe_transform
    train_woe = apply_woe_transform(train_df, woe_mapping)
    
    # Select features
    available_features = []
    feature_mapping = {}
    
    for feat in final_features:
        if feat in train_woe.columns:
            available_features.append(feat)
        else:
            # Try to find base name
            base_name = feat.split('_')[0] if '_' in feat else feat
            if base_name in train_woe.columns:
                feature_mapping[feat] = base_name
                available_features.append(base_name)
    
    if feature_mapping:
        X_train = train_woe[[feature_mapping.get(f, f) for f in final_features]]
        X_train.columns = final_features
    else:
        X_train = train_woe[final_features] if all(f in train_woe.columns for f in final_features) else train_woe[available_features]
    
    # Clean any NaN or Inf values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(0)
    
    # Get training scores
    try:
        training_scores = model.predict_proba(X_train)
        if training_scores.ndim == 2:
            training_scores = training_scores[:, 1]
    except:
        training_scores = model.predict(X_train)
    
    print(f"Training scores calculated: {len(training_scores):,}")
    
    # Score new data
    print("\nScoring new data...")
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
    
    # Display results
    print("\n=== SCORING RESULTS ===")
    print(f"Total scored: {scoring_results['n_total']:,}")
    print(f"Calibration applied: {'Yes' if scoring_results.get('calibration_applied') else 'No'}")
    
    if scoring_results.get('psi_score') is not None:
        print(f"PSI: {scoring_results['psi_score']:.4f}")
    
    if 'with_target' in scoring_results:
        wt = scoring_results['with_target']
        print(f"\nWith Target ({wt['n_records']:,} records):")
        print(f"  AUC: {wt['auc']:.4f}")
        print(f"  Gini: {wt['gini']:.4f}")
        print(f"  KS: {wt['ks']:.4f}")
        print(f"  Default Rate: {wt['default_rate']:.3f}")
    
    if 'without_target' in scoring_results:
        wot = scoring_results['without_target']
        print(f"\nWithout Target ({wot['n_records']:,} records):")
        print(f"  Score Mean: {wot['score_stats']['mean']:.4f}")
        print(f"  Score Std: {wot['score_stats']['std']:.4f}")
    
    # Save scored data
    scored_df = scoring_df.copy()
    scored_df['predicted_score'] = scoring_results['scores']
    scored_df['raw_score'] = scoring_results['raw_scores']
    scored_df.to_csv(f"{OUTPUT_FOLDER}/scored_data.csv", index=False)
    print(f"\nScored data saved: {OUTPUT_FOLDER}/scored_data.csv")
    
    # Enhance the existing pipeline Excel report
    excel_path = f"{OUTPUT_FOLDER}/model_report.xlsx"
    if os.path.exists(excel_path):
        enhance_excel_report(excel_path, scoring_results, scoring_df)
        
        # Verify final Excel
        with pd.ExcelFile(excel_path) as xl:
            print(f"\nFinal Excel report: {excel_path}")
            print(f"Total sheets: {len(xl.sheet_names)}")
            print(f"Pipeline sheets: {', '.join([s for s in xl.sheet_names if not s.startswith('scoring')])}")
            print(f"Scoring sheets: {', '.join([s for s in xl.sheet_names if s.startswith('scoring') or s == 'score_distribution' or s == 'psi_analysis'])}")
    else:
        print(f"\nWarning: Pipeline Excel not found: {excel_path}")
    
    print("\n=== WORKFLOW COMPLETED ===")
    print(f"Single output folder: {OUTPUT_FOLDER}/")
    print(f"Single Excel report: {excel_path}")
    print("All pipeline and scoring results integrated in one place!")

if __name__ == "__main__":
    main()