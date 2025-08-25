#!/usr/bin/env python3
"""
Integrated workflow test: Single folder, single Excel with multiple sheets, proper WOE
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import shutil
from pathlib import Path
sys.path.append('src')

from risk_pipeline.utils.pipeline_runner import run_pipeline_from_dataframe, get_full_config
from risk_pipeline.utils.scoring import load_model_artifacts, score_data, create_scoring_report
from risk_pipeline.utils.report_updater import update_excel_with_scoring

def clean_previous_runs():
    """Clean previous output folders"""
    folders_to_clean = ['outputs', 'outputs_full', 'outputs_workflow', 'outputs_clean', 'outputs_cal_test']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Cleaned: {folder}")
            except Exception as e:
                print(f"Could not clean {folder}: {e}")

def main():
    print("=== INTEGRATED WORKFLOW TEST ===")
    print("Single folder, single Excel with multiple sheets\n")
    
    # Clean previous runs
    clean_previous_runs()
    
    # Single output folder for everything
    OUTPUT_FOLDER = "outputs"
    EXCEL_FILE = "risk_model_report.xlsx"
    
    # Step 1: Run pipeline
    print("Step 1: Running pipeline...")
    
    # Load training data
    df = pd.read_csv('data/input.csv')
    print(f"Training data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Generate calibration data
    print("Generating calibration data...")
    from scripts.make_calibration_data import generate_calibration_data
    calibration_df = generate_calibration_data(n_samples=500, output_path="data/calibration.csv")
    
    # Get configuration
    config = get_full_config(
        try_mlp=False,  # Skip for speed
        ensemble=False,  # Skip for speed
        calibration_data_path="data/calibration.csv",
        hpo_trials=5,  # Low for speed
        hpo_timeout_sec=20,
        output_folder=OUTPUT_FOLDER,
        output_excel_path=EXCEL_FILE,
    )
    
    print("Running pipeline...")
    start_time = time.time()
    
    # Run pipeline
    pipeline_results = run_pipeline_from_dataframe(
        df=df,
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        output_folder=OUTPUT_FOLDER,
        output_excel=EXCEL_FILE,
        use_test_split=True,
        oot_months=3,
        **{
            'try_mlp': config.try_mlp,
            'ensemble': config.ensemble,
            'calibration_data_path': config.calibration_data_path,
            'hpo_trials': config.hpo_trials,
            'hpo_timeout_sec': config.hpo_timeout_sec,
            'orchestrator': config.orchestrator
        }
    )
    
    pipeline_time = time.time() - start_time
    print(f"Pipeline completed in {pipeline_time:.1f}s")
    print(f"   Best Model: {pipeline_results['best_model']}")
    print(f"   Final Features: {pipeline_results['final_features']}")
    print(f"   Run ID: {pipeline_results['run_id']}")
    
    # Step 2: Load artifacts
    print(f"\nStep 2: Loading model artifacts...")
    
    run_id = pipeline_results['run_id']
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)
    
    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
        final_features = list(final_features.values())[0]
    
    print(f"Artifacts loaded:")
    print(f"   Model: {type(model).__name__}")
    print(f"   Features: {final_features}")
    print(f"   Calibrator: {'Available' if calibrator else 'Not found'}")
    
    # Step 3: Score new data
    print(f"\nStep 3: Scoring new data...")
    
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"Scoring data: {scoring_df.shape[0]:,} rows")
    print(f"   With target: {(~scoring_df['target'].isna()).sum():,}")
    print(f"   Without target: {scoring_df['target'].isna().sum():,}")
    
    # Get training scores for PSI
    print("Calculating training scores for PSI...")
    
    # Apply WOE to training data properly
    from risk_pipeline.stages.woe import apply_woe
    
    # Check WOE mapping format and fix if needed
    if "variables" not in woe_mapping:
        # Convert old format to new format
        print("Converting WOE mapping format...")
        new_woe_mapping = {"variables": {}}
        for var_name, var_info in woe_mapping.items():
            if hasattr(var_info, 'var'):
                original_var = var_info.var
                new_woe_mapping["variables"][original_var] = {
                    "type": "numeric" if hasattr(var_info, 'bins') else "categorical",
                    "bins": [] if hasattr(var_info, 'bins') else None,
                    "groups": [] if hasattr(var_info, 'special') else None
                }
                
                if hasattr(var_info, 'bins'):
                    for bin_info in var_info.bins:
                        new_woe_mapping["variables"][original_var]["bins"].append({
                            "left": bin_info['range'][0] if bin_info['range'][0] != -float('inf') else None,
                            "right": bin_info['range'][1] if bin_info['range'][1] != float('inf') else None,
                            "woe": bin_info.get('woe', 0)
                        })
        woe_mapping_for_transform = new_woe_mapping
    else:
        woe_mapping_for_transform = woe_mapping
    
    # Apply WOE to training data
    train_woe = apply_woe(df, woe_mapping_for_transform)
    
    # Get the original variable names that correspond to final features
    feature_mapping = {}
    for feat in final_features:
        # Extract base variable name (e.g., num1_corr95 -> num1)
        base_name = feat.split('_')[0] if '_' in feat else feat
        if base_name in train_woe.columns:
            feature_mapping[feat] = base_name
    
    print(f"Feature mapping: {feature_mapping}")
    
    # Select and rename columns for training scores
    if feature_mapping:
        X_train = train_woe[[feature_mapping[f] for f in final_features if f in feature_mapping]]
        X_train.columns = [f for f in final_features if f in feature_mapping]
    else:
        # Try direct selection
        available = [f for f in final_features if f in train_woe.columns]
        if available:
            X_train = train_woe[available]
        else:
            print("Warning: No features available for training scores")
            X_train = pd.DataFrame(np.random.random((len(train_woe), 2)), columns=['dummy1', 'dummy2'])
    
    # Get training scores
    try:
        training_scores = model.predict_proba(X_train)
        if training_scores.ndim == 2:
            training_scores = training_scores[:, 1]
    except:
        training_scores = model.predict(X_train)
    
    print(f"Training scores calculated: {len(training_scores):,}")
    
    # Score new data
    start_time = time.time()
    scoring_results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=final_features,
        woe_mapping=woe_mapping_for_transform,
        calibrator=calibrator,
        training_scores=training_scores,
        feature_mapping=feature_mapping
    )
    scoring_time = time.time() - start_time
    
    print(f"Scoring completed in {scoring_time:.1f}s")
    
    # Step 4: Create comprehensive Excel report with multiple sheets
    print(f"\nStep 4: Creating comprehensive Excel report...")
    
    excel_path = f"{OUTPUT_FOLDER}/{EXCEL_FILE}"
    
    # Load existing Excel or create new
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a' if os.path.exists(excel_path) else 'w') as writer:
        
        # Sheet 1: Model Summary (from pipeline)
        model_summary = pd.DataFrame({
            'Metric': ['Best Model', 'Run ID', 'Final Features', 'Training Rows', 'Test Rows', 'OOT Rows'],
            'Value': [
                pipeline_results['best_model'],
                pipeline_results['run_id'],
                ', '.join(final_features),
                pipeline_results.get('train_size', 'N/A'),
                pipeline_results.get('test_size', 'N/A'),
                pipeline_results.get('oot_size', 'N/A')
            ]
        })
        model_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
        
        # Sheet 2: Scoring Summary
        reports = create_scoring_report(scoring_results)
        reports['summary'].to_excel(writer, sheet_name='Scoring_Summary', index=False)
        
        # Sheet 3: With Target Analysis
        if 'with_target' in reports:
            reports['with_target'].to_excel(writer, sheet_name='With_Target', index=False)
        
        # Sheet 4: Without Target Analysis
        if 'without_target' in reports:
            reports['without_target'].to_excel(writer, sheet_name='Without_Target', index=False)
        
        # Sheet 5: Score Distribution
        scores_df = pd.DataFrame({
            'app_id': scoring_df['app_id'],
            'has_target': ~scoring_df['target'].isna(),
            'target': scoring_df['target'],
            'predicted_score': scoring_results['scores'],
            'raw_score': scoring_results['raw_scores']
        })
        
        # Add score bins
        scores_df['score_bin'] = pd.cut(scores_df['predicted_score'], 
                                        bins=10, 
                                        labels=[f'Bin_{i+1}' for i in range(10)])
        
        # Calculate statistics by bin
        bin_stats = scores_df.groupby('score_bin').agg({
            'app_id': 'count',
            'target': ['mean', 'sum'],
            'predicted_score': ['mean', 'min', 'max']
        }).round(4)
        bin_stats.columns = ['Count', 'Default_Rate', 'Defaults', 'Avg_Score', 'Min_Score', 'Max_Score']
        bin_stats.to_excel(writer, sheet_name='Score_Distribution')
        
        # Sheet 6: WOE Mapping Details
        woe_details = []
        for var_name, var_info in woe_mapping.items() if isinstance(woe_mapping, dict) and "variables" not in woe_mapping else woe_mapping.get("variables", {}).items():
            if isinstance(var_info, dict):
                var_type = var_info.get('type', 'unknown')
                if var_type == 'numeric' and 'bins' in var_info:
                    for bin_info in var_info['bins']:
                        woe_details.append({
                            'Variable': var_name,
                            'Type': 'Numeric',
                            'Range': f"[{bin_info.get('left', '-inf')}, {bin_info.get('right', 'inf')}]",
                            'WOE': bin_info.get('woe', 0)
                        })
            elif hasattr(var_info, 'bins'):
                for bin_info in var_info.bins:
                    woe_details.append({
                        'Variable': var_info.var if hasattr(var_info, 'var') else var_name,
                        'Type': 'Numeric',
                        'Range': str(bin_info.get('range', 'N/A')),
                        'WOE': bin_info.get('woe', 0)
                    })
        
        if woe_details:
            pd.DataFrame(woe_details).to_excel(writer, sheet_name='WOE_Mapping', index=False)
    
    print(f"Excel report saved: {excel_path}")
    
    # List all sheets in the Excel
    xl_file = pd.ExcelFile(excel_path)
    print(f"Excel sheets created: {', '.join(xl_file.sheet_names)}")
    
    # Step 5: Final summary
    total_time = pipeline_time + scoring_time
    print(f"\n=== WORKFLOW COMPLETED ===")
    print(f"Output folder: {OUTPUT_FOLDER}/")
    print(f"Excel report: {excel_path}")
    print(f"Pipeline time: {pipeline_time:.1f}s")
    print(f"Scoring time: {scoring_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    
    # Performance summary
    if 'with_target' in scoring_results:
        wt = scoring_results['with_target']
        print(f"\nModel Performance (with targets):")
        print(f"   AUC: {wt['auc']:.4f}")
        print(f"   Gini: {wt['gini']:.4f}")
        print(f"   KS: {wt['ks']:.4f}")
        print(f"   Default Rate: {wt['default_rate']:.3f}")
    
    print(f"\nAll outputs in single folder: {OUTPUT_FOLDER}/")
    print(f"All reports in single Excel: {excel_path}")

if __name__ == "__main__":
    main()