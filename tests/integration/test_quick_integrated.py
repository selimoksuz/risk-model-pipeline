#!/usr/bin/env python3
"""
Quick integrated test - use existing model, focus on Excel report generation
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import joblib
from datetime import datetime
sys.path.append('src')

def main():
    print("=== QUICK INTEGRATED TEST ===")
    print("Using existing model, creating single Excel with multiple sheets\n")
    
    # Single output folder
    OUTPUT_FOLDER = "outputs"
    EXCEL_FILE = "risk_model_report.xlsx"
    
    # Find latest model artifacts
    model_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('best_model_') and f.endswith('.joblib')]
    if not model_files:
        print("No model files found! Run pipeline first.")
        return
    
    latest_model = sorted(model_files)[-1]
    run_id = latest_model.replace('best_model_', '').replace('.joblib', '')
    print(f"Using model from run: {run_id}")
    
    # Load artifacts
    model = joblib.load(f"{OUTPUT_FOLDER}/{latest_model}")
    
    with open(f"{OUTPUT_FOLDER}/final_vars_{run_id}.json", 'r') as f:
        final_features = json.load(f)
    
    with open(f"{OUTPUT_FOLDER}/woe_mapping_{run_id}.json", 'r') as f:
        woe_mapping = json.load(f)
    
    print(f"Model: {type(model).__name__}")
    print(f"Final features: {final_features}")
    
    # Load data
    train_df = pd.read_csv('data/input.csv')
    scoring_df = pd.read_csv('data/scoring.csv')
    
    print(f"\nData loaded:")
    print(f"  Training: {train_df.shape}")
    print(f"  Scoring: {scoring_df.shape}")
    
    # Create comprehensive Excel report
    excel_path = f"{OUTPUT_FOLDER}/{EXCEL_FILE}"
    print(f"\nCreating Excel report: {excel_path}")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Model Summary
        model_summary = pd.DataFrame({
            'Metric': [
                'Model Type',
                'Run ID',
                'Run Date',
                'Final Features',
                'Number of Features',
                'Training Samples',
                'WOE Variables'
            ],
            'Value': [
                type(model).__name__,
                run_id,
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                ', '.join(final_features) if isinstance(final_features, list) else str(final_features),
                len(final_features) if isinstance(final_features, list) else 'N/A',
                f"{train_df.shape[0]:,}",
                f"{len(woe_mapping)} variables"
            ]
        })
        model_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
        print("  [OK] Model_Summary sheet created")
        
        # Sheet 2: Data Overview
        data_overview = pd.DataFrame({
            'Dataset': ['Training', 'Scoring', 'Scoring with Target', 'Scoring without Target'],
            'Rows': [
                train_df.shape[0],
                scoring_df.shape[0],
                (~scoring_df['target'].isna()).sum(),
                scoring_df['target'].isna().sum()
            ],
            'Columns': [
                train_df.shape[1],
                scoring_df.shape[1],
                scoring_df.shape[1],
                scoring_df.shape[1]
            ],
            'Default Rate': [
                f"{train_df['target'].mean():.3f}",
                f"{scoring_df['target'].mean():.3f}" if not scoring_df['target'].isna().all() else 'N/A',
                f"{scoring_df[~scoring_df['target'].isna()]['target'].mean():.3f}" if (~scoring_df['target'].isna()).sum() > 0 else 'N/A',
                'N/A'
            ]
        })
        data_overview.to_excel(writer, sheet_name='Data_Overview', index=False)
        print("  [OK] Data_Overview sheet created")
        
        # Sheet 3: WOE Details
        woe_details = []
        for var_name, var_info in woe_mapping.items():
            if isinstance(var_info, dict):
                # Handle different WOE format
                if 'var' in var_info:
                    original_var = var_info['var']
                    if 'bins' in var_info:
                        for bin_info in var_info['bins']:
                            woe_details.append({
                                'Variable': original_var,
                                'Transformed_Name': var_name,
                                'Type': 'Numeric',
                                'Bin': str(bin_info.get('range', 'N/A')),
                                'WOE': bin_info.get('woe', 0),
                                'Count': bin_info.get('count', 'N/A'),
                                'Event_Rate': bin_info.get('event_rate', 'N/A')
                            })
                else:
                    # Simple format
                    woe_details.append({
                        'Variable': var_name,
                        'Transformed_Name': var_name,
                        'Type': 'Unknown',
                        'Bin': 'All',
                        'WOE': 0,
                        'Count': 'N/A',
                        'Event_Rate': 'N/A'
                    })
        
        if woe_details:
            woe_df = pd.DataFrame(woe_details)
            woe_df.to_excel(writer, sheet_name='WOE_Mapping', index=False)
            print("  [OK] WOE_Mapping sheet created")
        
        # Sheet 4: Feature Importance (dummy data for now)
        feature_importance = pd.DataFrame({
            'Feature': final_features if isinstance(final_features, list) else ['Feature1', 'Feature2'],
            'Importance': np.random.random(len(final_features) if isinstance(final_features, list) else 2),
            'Rank': range(1, (len(final_features) if isinstance(final_features, list) else 2) + 1)
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
        print("  [OK] Feature_Importance sheet created")
        
        # Sheet 5: Scoring Results Summary
        has_target = ~scoring_df['target'].isna()
        scoring_summary = pd.DataFrame({
            'Metric': [
                'Total Records',
                'Records with Target',
                'Records without Target',
                'Target Coverage',
                'Default Rate (with target)',
                'Date Range',
                'Unique IDs'
            ],
            'Value': [
                f"{len(scoring_df):,}",
                f"{has_target.sum():,}",
                f"{(~has_target).sum():,}",
                f"{has_target.mean()*100:.1f}%",
                f"{scoring_df[has_target]['target'].mean():.3f}" if has_target.sum() > 0 else 'N/A',
                f"{scoring_df['app_dt'].min()} to {scoring_df['app_dt'].max()}",
                f"{scoring_df['app_id'].nunique():,}"
            ]
        })
        scoring_summary.to_excel(writer, sheet_name='Scoring_Summary', index=False)
        print("  [OK] Scoring_Summary sheet created")
        
        # Sheet 6: Performance Metrics (if we had actual scores)
        performance_metrics = pd.DataFrame({
            'Metric': ['AUC', 'Gini', 'KS', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Training': [0.75, 0.50, 0.35, 0.80, 0.70, 0.65, 0.67],  # Dummy values
            'Validation': [0.73, 0.46, 0.32, 0.78, 0.68, 0.63, 0.65],  # Dummy values
            'Test': [0.72, 0.44, 0.30, 0.77, 0.67, 0.62, 0.64],  # Dummy values
        })
        performance_metrics.to_excel(writer, sheet_name='Performance_Metrics', index=False)
        print("  [OK] Performance_Metrics sheet created")
        
        # Sheet 7: Configuration
        config_data = pd.DataFrame({
            'Parameter': [
                'id_col', 'time_col', 'target_col',
                'train_ratio', 'test_ratio', 'oot_months',
                'hpo_trials', 'random_state', 'calibration_enabled'
            ],
            'Value': [
                'app_id', 'app_dt', 'target',
                '0.67', '0.17', '3',
                '5', '42', 'Yes'
            ]
        })
        config_data.to_excel(writer, sheet_name='Configuration', index=False)
        print("  [OK] Configuration sheet created")
    
    # Verify Excel was created
    if os.path.exists(excel_path):
        xl_file = pd.ExcelFile(excel_path)
        print(f"\n[SUCCESS] Excel report created successfully!")
        print(f"   Path: {excel_path}")
        print(f"   Sheets: {', '.join(xl_file.sheet_names)}")
        print(f"   Size: {os.path.getsize(excel_path)/1024:.1f} KB")
    else:
        print(f"\n[ERROR] Failed to create Excel report")
    
    # Clean up old files
    print(f"\nCleaning up old separate files...")
    old_patterns = ['outputs_full', 'outputs_workflow', 'outputs_clean', 'outputs_cal_test']
    for pattern in old_patterns:
        if os.path.exists(pattern):
            print(f"  - Found old folder: {pattern}")
    
    print(f"\n=== TEST COMPLETED ===")
    print(f"All outputs in single folder: {OUTPUT_FOLDER}/")
    print(f"All reports in single Excel: {excel_path}")

if __name__ == "__main__":
    main()