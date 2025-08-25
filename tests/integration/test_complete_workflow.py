#!/usr/bin/env python3
"""
Test complete workflow: Pipeline training + Scoring + Reporting
"""

import pandas as pd
import sys
import os
import time
sys.path.append('src')

from risk_pipeline.utils.pipeline_runner import run_pipeline_from_dataframe, get_full_config
from risk_pipeline.utils.scoring import load_model_artifacts, score_data, create_scoring_report
from risk_pipeline.utils.report_updater import update_excel_with_scoring

def main():
    print("🚀 === COMPLETE WORKFLOW TEST ===")
    
    # Step 1: Run full pipeline
    print("\n📊 Step 1: Running full pipeline...")
    
    # Load training data
    df = pd.read_csv('data/input.csv')
    print(f"Training data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Get full configuration
    config = get_full_config(
        try_mlp=True,
        ensemble=True,
        hpo_trials=15,  # Reduced for faster testing
        hpo_timeout_sec=60,
        output_folder="outputs_workflow",
        output_excel_path="workflow_report.xlsx",
    )
    
    print("Running pipeline with all stages enabled...")
    start_time = time.time()
    
    # Run pipeline
    pipeline_results = run_pipeline_from_dataframe(
        df=df,
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        output_folder="outputs_workflow",
        output_excel="workflow_report.xlsx",
        use_test_split=True,
        oot_months=3,
        **{
            'try_mlp': config.try_mlp,
            'ensemble': config.ensemble,
            'hpo_trials': config.hpo_trials,
            'hpo_timeout_sec': config.hpo_timeout_sec,
            'orchestrator': config.orchestrator
        }
    )
    
    pipeline_time = time.time() - start_time
    print(f"✅ Pipeline completed in {pipeline_time:.1f}s")
    print(f"   Best Model: {pipeline_results['best_model']}")
    print(f"   Final Features: {len(pipeline_results['final_features'])} features")
    
    # Step 2: Load scoring data and score
    print(f"\n🎯 Step 2: Scoring new data...")
    
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"Scoring data: {scoring_df.shape[0]:,} rows x {scoring_df.shape[1]} columns")
    
    # Get artifacts from pipeline run
    output_folder = "outputs_workflow"
    run_id = pipeline_results['run_id']
    
    # Load model artifacts
    model, final_features, woe_mapping, calibrator = load_model_artifacts(output_folder, run_id)
    
    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get('final_vars', final_features)
    if isinstance(final_features, dict):
        final_features = list(final_features.values())[0]
    
    # Load training scores for PSI
    from risk_pipeline.stages import apply_woe
    train_woe = apply_woe(df, woe_mapping)
    
    # Map features for scoring
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
        X_train.columns = final_features  # Rename back
    else:
        X_train = train_woe[final_features]
    
    try:
        training_scores = model.predict_proba(X_train)[:, 1]
    except:
        training_scores = model.predict(X_train)
    
    # Score new data
    start_time = time.time()
    scoring_results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=final_features,
        woe_mapping=woe_mapping,
        calibrator=calibrator,
        training_scores=training_scores,
        feature_mapping=feature_mapping
    )
    scoring_time = time.time() - start_time
    
    print(f"✅ Scoring completed in {scoring_time:.1f}s")
    print(f"   Total scored: {scoring_results['n_total']:,}")
    print(f"   With target: {scoring_results['n_with_target']:,}")
    print(f"   AUC (with targets): {scoring_results.get('auc', 'N/A'):.4f}" if 'auc' in scoring_results else "   AUC: N/A")
    
    # Step 3: Update reports
    print(f"\n📈 Step 3: Updating reports...")
    
    # Create scoring summary
    scoring_summary = create_scoring_report(scoring_results)
    
    # Update Excel report
    excel_path = f"{output_folder}/workflow_report.xlsx"
    if os.path.exists(excel_path):
        success = update_excel_with_scoring(excel_path, scoring_results, scoring_summary)
        if success:
            print(f"✅ Excel report updated: {excel_path}")
        else:
            print(f"⚠️  Excel update failed")
    
    # Save additional outputs
    results_folder = f"{output_folder}/workflow_results"
    os.makedirs(results_folder, exist_ok=True)
    
    scoring_with_scores = scoring_df.copy()
    scoring_with_scores['predicted_score'] = scoring_results['scores']
    scoring_with_scores.to_csv(f"{results_folder}/scored_data.csv", index=False)
    scoring_summary.to_csv(f"{results_folder}/scoring_summary.csv", index=False)
    
    print(f"✅ Additional results saved to: {results_folder}/")
    
    # Step 4: Summary
    total_time = pipeline_time + scoring_time
    print(f"\n🎉 === WORKFLOW COMPLETED ===")
    print(f"📊 Pipeline: {pipeline_time:.1f}s")
    print(f"🎯 Scoring: {scoring_time:.1f}s") 
    print(f"⏱️  Total: {total_time:.1f}s")
    print(f"📁 Main report: {excel_path}")
    print(f"📁 Scored data: {results_folder}/scored_data.csv")
    
    # Performance summary
    if 'auc' in scoring_results:
        print(f"\n🏆 Model Performance:")
        print(f"   Training Model: {pipeline_results['best_model']}")
        print(f"   Features: {', '.join(final_features)}")
        print(f"   Scoring AUC: {scoring_results['auc']:.4f}")
        print(f"   Scoring Gini: {scoring_results['gini']:.4f}")
        print(f"   Default Rate: {scoring_results['default_rate']:.3f}")
        
        psi = scoring_results.get('psi_score', float('inf'))
        if psi != float('inf'):
            print(f"   PSI: {psi:.4f}")
            if psi < 0.1:
                print("   📊 Population stability: GOOD")
            elif psi < 0.25:
                print("   📊 Population stability: MODERATE")
            else:
                print("   📊 Population stability: POOR (significant drift)")
        else:
            print("   📊 PSI: Calculation error (likely data issue)")

if __name__ == "__main__":
    main()