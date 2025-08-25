#!/usr/bin/env python3
"""
Clean workflow test: Single output folder, proper calibration, target separation
"""

import pandas as pd
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
    folders_to_clean = ['outputs_full', 'outputs_workflow', 'outputs_clean']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"🧹 Cleaned: {folder}")
            except Exception as e:
                print(f"⚠️  Could not clean {folder}: {e}")

def main():
    print("🚀 === CLEAN WORKFLOW TEST ===")
    print("Single output folder, proper calibration, target separation\n")
    
    # Clean previous runs
    clean_previous_runs()
    
    # Single output folder
    OUTPUT_FOLDER = "outputs_clean"
    
    # Step 1: Run pipeline with calibration enabled
    print("📊 Step 1: Running pipeline with calibration...")
    
    # Load training data
    df = pd.read_csv('data/input.csv')
    print(f"Training data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Generate calibration data first
    print("Generating fresh calibration data...")
    from scripts.make_calibration_data import generate_calibration_data
    calibration_df = generate_calibration_data(n_samples=800, output_path="data/calibration_clean.csv")
    
    # Get configuration with calibration
    config = get_full_config(
        try_mlp=True,
        ensemble=True,
        calibration_data_path="data/calibration_clean.csv",  # Enable calibration!
        hpo_trials=10,  # Reduced for faster testing
        hpo_timeout_sec=30,
        output_folder=OUTPUT_FOLDER,
        output_excel_path="clean_report.xlsx",
    )
    
    print("Running pipeline with calibration enabled...")
    start_time = time.time()
    
    # Run pipeline
    pipeline_results = run_pipeline_from_dataframe(
        df=df,
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        output_folder=OUTPUT_FOLDER,
        output_excel="clean_report.xlsx",
        use_test_split=True,
        oot_months=3,
        **{
            'try_mlp': config.try_mlp,
            'ensemble': config.ensemble,
            'calibration_data_path': config.calibration_data_path,  # Important!
            'hpo_trials': config.hpo_trials,
            'hpo_timeout_sec': config.hpo_timeout_sec,
            'orchestrator': config.orchestrator
        }
    )
    
    pipeline_time = time.time() - start_time
    print(f"✅ Pipeline completed in {pipeline_time:.1f}s")
    print(f"   Best Model: {pipeline_results['best_model']}")
    print(f"   Final Features: {len(pipeline_results['final_features'])} features")
    print(f"   Run ID: {pipeline_results['run_id']}")
    
    # Step 2: Load artifacts with calibrator
    print(f"\n🔧 Step 2: Loading model artifacts...")
    
    run_id = pipeline_results['run_id']
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)
    
    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
        final_features = list(final_features.values())[0]
    
    print(f"✅ Artifacts loaded:")
    print(f"   Model: {type(model).__name__}")
    print(f"   Features: {final_features}")
    print(f"   Calibrator: {'Available' if calibrator else 'Not found'}")
    
    # Step 3: Load and score new data
    print(f"\n🎯 Step 3: Scoring new data...")
    
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"Scoring data: {scoring_df.shape[0]:,} rows x {scoring_df.shape[1]} columns")
    print(f"   With target: {(~scoring_df['target'].isna()).sum():,}")
    print(f"   Without target: {scoring_df['target'].isna().sum():,}")
    
    # Load training scores for PSI
    from risk_pipeline.stages import apply_woe
    train_woe = apply_woe(df, woe_mapping)
    
    # Map features for training data
    feature_mapping = {}
    for feat in final_features:
        if feat.startswith('num1'):
            feature_mapping[feat] = 'num1'
        elif feat.startswith('num2'):
            feature_mapping[feat] = 'num2'
        elif feat.startswith('num3'):
            feature_mapping[feat] = 'num3'
        elif feat.startswith('num4'):
            feature_mapping[feat] = 'num4'
    
    if feature_mapping:
        mapped_train_features = [feature_mapping.get(f, f) for f in final_features]
        X_train = train_woe[mapped_train_features]
        X_train.columns = final_features
    else:
        X_train = train_woe[final_features]
    
    try:
        training_scores = model.predict_proba(X_train)[:, 1]
    except:
        training_scores = model.predict(X_train)
    
    print(f"✅ Training scores calculated: {len(training_scores):,}")
    
    # Score new data with calibration
    start_time = time.time()
    scoring_results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=final_features,
        woe_mapping=woe_mapping,
        calibrator=calibrator,  # Pass calibrator
        training_scores=training_scores,
        feature_mapping=feature_mapping
    )
    scoring_time = time.time() - start_time
    
    print(f"✅ Scoring completed in {scoring_time:.1f}s")
    
    # Step 4: Display detailed results
    print(f"\n📈 Step 4: Results Analysis...")
    print(f"📊 Overall Statistics:")
    print(f"   Total scored: {scoring_results['n_total']:,}")
    print(f"   Calibration applied: {'✅ Yes' if scoring_results.get('calibration_applied', False) else '❌ No'}")
    
    psi = scoring_results.get('psi_score', float('inf'))
    if psi != float('inf'):
        print(f"   PSI: {psi:.4f}")
        if psi < 0.1:
            print("   📊 Population stability: GOOD")
        elif psi < 0.25:
            print("   📊 Population stability: MODERATE")
        else:
            print("   📊 Population stability: POOR (significant drift)")
    
    # Results for records WITH targets
    if 'with_target' in scoring_results:
        wt = scoring_results['with_target']
        print(f"\n🎯 Records WITH Targets ({wt['n_records']:,} records):")
        print(f"   Default Rate: {wt['default_rate']:.3f}")
        print(f"   AUC: {wt['auc']:.4f}")
        print(f"   Gini: {wt['gini']:.4f}")
        print(f"   KS: {wt['ks']:.4f}")
        print(f"   Score Range: {wt['score_stats']['min']:.4f} - {wt['score_stats']['max']:.4f}")
        print(f"   Score Mean: {wt['score_stats']['mean']:.4f}")
    
    # Results for records WITHOUT targets
    if 'without_target' in scoring_results:
        wot = scoring_results['without_target']
        print(f"\n📊 Records WITHOUT Targets ({wot['n_records']:,} records):")
        print(f"   Score Range: {wot['score_stats']['min']:.4f} - {wot['score_stats']['max']:.4f}")
        print(f"   Score Mean: {wot['score_stats']['mean']:.4f}")
    
    # Step 5: Save comprehensive reports
    print(f"\n💾 Step 5: Saving results...")
    
    results_folder = f"{OUTPUT_FOLDER}/scoring_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Save scored data
    scoring_with_scores = scoring_df.copy()
    scoring_with_scores['predicted_score'] = scoring_results['scores']
    scoring_with_scores['raw_score'] = scoring_results['raw_scores']
    scoring_with_scores.to_csv(f"{results_folder}/scored_data.csv", index=False)
    
    # Create and save comprehensive reports
    reports = create_scoring_report(scoring_results)
    
    for report_name, report_df in reports.items():
        report_df.to_csv(f"{results_folder}/report_{report_name}.csv", index=False)
        print(f"   📄 Saved: report_{report_name}.csv")
    
    # Update Excel report
    excel_path = f"{OUTPUT_FOLDER}/clean_report.xlsx"
    if os.path.exists(excel_path):
        # Use summary report for Excel (backward compatibility)
        success = update_excel_with_scoring(excel_path, scoring_results, reports['summary'])
        if success:
            print(f"✅ Excel report updated: {excel_path}")
        else:
            print(f"⚠️  Excel update failed")
    
    # Step 6: Final summary
    total_time = pipeline_time + scoring_time
    print(f"\n🎉 === WORKFLOW COMPLETED ===")
    print(f"📁 Output folder: {OUTPUT_FOLDER}/")
    print(f"📊 Pipeline: {pipeline_time:.1f}s")
    print(f"🎯 Scoring: {scoring_time:.1f}s")
    print(f"⏱️  Total: {total_time:.1f}s")
    print(f"📄 Main report: {excel_path}")
    print(f"📄 Detailed reports: {results_folder}/")
    print(f"📄 Scored data: {results_folder}/scored_data.csv")
    
    print(f"\n✅ All files in single output folder: {OUTPUT_FOLDER}")
    
    # List final contents
    print(f"\n📁 Final Output Structure:")
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        level = root.replace(OUTPUT_FOLDER, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    main()