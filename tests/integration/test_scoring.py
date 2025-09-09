#!/usr/bin/env python3
"""
Test scoring functionality with the trained model
"""

from risk_pipeline.utils.scoring import load_model_artifacts, score_data, create_scoring_report
import pandas as pd
import sys
import os
sys.path.append('src')


def main():
    print("=== SCORING TEST ===")

    # Load scoring data
    print("Loading scoring data...")
    scoring_df = pd.read_csv('data/scoring.csv')
    print(f"Scoring data: {scoring_df.shape[0]:, } rows x {scoring_df.shape[1]} columns")

    # Get the latest run artifacts
    output_folder = "outputs"
    run_id = "20250825_150724_3205a13a"  # From available run

    print(f"Loading model artifacts from {output_folder}...")
    try:
        model, final_features, woe_mapping, calibrator = load_model_artifacts(output_folder, run_id)
        print(f"✅ Model loaded: {type(model).__name__}")

        # Handle different final_features formats
        if isinstance(final_features, dict):
            final_features = final_features.get('final_vars', final_features)
        if isinstance(final_features, dict):
            final_features = list(final_features.values())[0]

        print(f"✅ Final features: {final_features}")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    # Load training scores for PSI calculation (from original dataset)
    print("Loading training data for PSI calculation...")
    train_df = pd.read_csv('data/input.csv')

    # Apply WOE and get training scores (simplified)
    from risk_pipeline.stages import apply_woe
    train_woe = apply_woe(train_df, woe_mapping)
    X_train = train_woe[final_features]

    try:
        training_scores = model.predict_proba(X_train)[:, 1]
    except Exception:
        training_scores = model.predict(X_train)

    print(f"✅ Training scores calculated: {len(training_scores):, } records")

    # Score the new data
    print("\nScoring new data...")
    print("Debug: Checking WOE transformation...")

    from risk_pipeline.stages import apply_woe
    df_woe = apply_woe(scoring_df, woe_mapping)
    print(f"WOE transformed columns: {list(df_woe.columns)}")
    print(f"Looking for features: {final_features}")

    # Check which features are available
    available_features = [f for f in final_features if f in df_woe.columns]
    missing_features = [f for f in final_features if f not in df_woe.columns]

    print(f"Available features: {available_features}")
    print(f"Missing features: {missing_features}")

    # Try to map final features to original feature names
    # num1_corr95 -> num1, num2_corr92 -> num2
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

    print(f"Feature mapping: {feature_mapping}")

    # Use mapped features if available
    if feature_mapping:
        mapped_features = [feature_mapping.get(f, f) for f in final_features]
        available_mapped = [f for f in mapped_features if f in df_woe.columns]
        print(f"Mapped features: {mapped_features}")
        print(f"Available mapped features: {available_mapped}")

        if available_mapped:
            # Update final_features to use mapped names
            final_features = available_mapped
            print(f"Using mapped features: {final_features}")
        else:
            print("❌ No mapped features available!")
            return
    elif not available_features:
        print("❌ No features available after WOE transformation!")
        print("This might be due to different data schema or WOE mapping issues.")
        return

    # Prepare original final features and mapping for model
    original_final_features = ['num1_corr95', 'num2_corr92']  # What model expects

    results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=original_final_features,
        woe_mapping=woe_mapping,
        calibrator=calibrator,  # Add calibrator
        training_scores=training_scores,
        feature_mapping=feature_mapping
    )

    print("=== SCORING RESULTS ===")
    print(f"📊 Total records scored: {results['n_total']:, }")
    print(f"📊 Records with target: {results['n_with_target']:, }")
    print(f"📊 Records without target: {results['n_without_target']:, }")

    if results.get('psi_score'):
        print(f"📊 PSI (Population Stability): {results['psi_score']:.4f}")
        if results['psi_score'] < 0.1:
            print("   ✅ PSI < 0.1: Population is stable")
        elif results['psi_score'] < 0.25:
            print("   ⚠️  0.1 ≤ PSI < 0.25: Some population shift")
        else:
            print("   🔴 PSI ≥ 0.25: Significant population shift")

    if 'auc' in results:
        print(f"📊 AUC (with targets): {results['auc']:.4f}")
        print(f"📊 Gini coefficient: {results['gini']:.4f}")
        print(f"📊 KS statistic: {results['ks']:.4f}")
        print(f"📊 Default rate: {results['default_rate']:.3f}")

    # Create detailed report
    print("\nCreating detailed report...")
    reports = create_scoring_report(results)

    # Save results
    results_folder = f"{output_folder}/scoring_results"
    os.makedirs(results_folder, exist_ok=True)

    # Save scores
    scoring_with_scores = scoring_df.copy()
    scoring_with_scores['predicted_score'] = results['scores']
    scoring_with_scores.to_csv(f"{results_folder}/scoring_results.csv", index=False)

    # Save all report types
    for report_name, report_df in reports.items():
        report_df.to_csv(f"{results_folder}/report_{report_name}.csv", index=False)
        print(f"   - report_{report_name}.csv")

    print(f"✅ Results saved to: {results_folder}/")
    print("   - scoring_results.csv (detailed scores)")
    print("   - scoring_summary.csv (summary metrics)")

    # Update Excel report with scoring metrics
    print("\nUpdating Excel report with scoring metrics...")
    excel_path = f"{output_folder}/risk_report_{run_id}.xlsx"

    if os.path.exists(excel_path):
        from risk_pipeline.utils.report_updater import update_excel_with_scoring
        # Use summary report for Excel update (backward compatibility)
        summary_report = reports.get('summary', list(reports.values())[0])
        success = update_excel_with_scoring(excel_path, results, summary_report)

        if success:
            print(f"✅ Excel report updated: {excel_path}")
        else:
            print(f"⚠️  Could not update Excel report")
    else:
        print(f"⚠️  Excel report not found: {excel_path}")

        # Create comprehensive report instead
        from risk_pipeline.utils.report_updater import create_comprehensive_report
        pipeline_results = {
            'best_model': 'MLP',
            'final_features': ['num1_corr95', 'num2_corr92'],
            'run_id': run_id
        }

        comprehensive_path = f"{results_folder}/comprehensive_report.xlsx"
        create_comprehensive_report(pipeline_results, results, comprehensive_path)

    # Display score distribution
    scores = results['scores']
    print(f"\n📈 Score Distribution:")
    print(f"   Min: {scores.min():.4f}")
    print(f"   25%: {np.percentile(scores, 25):.4f}")
    print(f"   50%: {np.percentile(scores, 50):.4f}")
    print(f"   75%: {np.percentile(scores, 75):.4f}")
    print(f"   Max: {scores.max():.4f}")
    print(f"   Mean: {scores.mean():.4f}")


if __name__ == "__main__":
    import numpy as np
    main()
