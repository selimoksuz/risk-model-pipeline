#!/usr/bin/env python3
"""
Add PSI results to Excel report
"""

import os
import sys

import numpy as np
import pandas as pd

from risk_pipeline.utils.scoring import load_model_artifacts, score_data

sys.path.append("src")


def add_psi_to_excel(excel_path, scoring_results):
    """Add PSI analysis to Excel report"""

    # Read existing sheets
    existing_sheets = {}
    with pd.ExcelFile(excel_path) as xl:
        for sheet_name in xl.sheet_names:
            existing_sheets[sheet_name] = pd.read_excel(xl, sheet_name)

    # Write back with PSI sheet
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Write existing sheets
        for sheet_name, df in existing_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Add PSI Analysis sheet
        psi_score = scoring_results.get("psi_score", None)
        psi_data = {
            "Metric": [
                "PSI Score",
                "PSI Status",
                "Population Stability",
                "Interpretation",
                "Action Required",
                "Total Scored",
                "With Target",
                "Without Target",
            ],
            "Value": [
                f"{psi_score:.4f}" if psi_score is not None else "N/A",
                "Stable"
                if psi_score and psi_score < 0.1
                else "Some Drift"
                if psi_score and psi_score < 0.25
                else "Significant Drift"
                if psi_score
                else "N/A",
                "Good - No shift detected"
                if psi_score and psi_score < 0.1
                else "Moderate - Some population shift"
                if psi_score and psi_score < 0.25
                else "Poor - Significant drift detected"
                if psi_score
                else "Cannot calculate",
                "PSI < 0.1: Population stable | 0.1 <= PSI < 0.25: Monitor closely | PSI >= 0.25: Model retraining recommended",
                "No action needed"
                if psi_score and psi_score < 0.1
                else "Monitor model performance"
                if psi_score and psi_score < 0.25
                else "Consider model retraining"
                if psi_score
                else "Check data quality",
                f"{scoring_results['n_total']:, }",
                f"{scoring_results['n_with_target']:, }",
                f"{scoring_results['n_without_target']:, }",
            ],
        }

        psi_df = pd.DataFrame(psi_data)
        psi_df.to_excel(writer, sheet_name="PSI_Analysis", index=False)

        # Add detailed PSI breakdown if available
        if "with_target" in scoring_results and "without_target" in scoring_results:
            wt = scoring_results["with_target"]
            wot = scoring_results["without_target"]

            comparison_data = {
                "Metric": [
                    "Records Count",
                    "Score Mean",
                    "Score Std",
                    "Score Min",
                    "Score Max",
                    "Default Rate",
                    "AUC",
                    "Gini",
                    "KS",
                ],
                "With_Target": [
                    wt["n_records"],
                    f"{wt['score_stats']['mean']:.4f}",
                    f"{wt['score_stats']['std']:.4f}",
                    f"{wt['score_stats']['min']:.4f}",
                    f"{wt['score_stats']['max']:.4f}",
                    f"{wt['default_rate']:.3f}",
                    f"{wt['auc']:.4f}",
                    f"{wt['gini']:.4f}",
                    f"{wt['ks']:.4f}",
                ],
                "Without_Target": [
                    wot["n_records"],
                    f"{wot['score_stats']['mean']:.4f}",
                    f"{wot['score_stats']['std']:.4f}",
                    f"{wot['score_stats']['min']:.4f}",
                    f"{wot['score_stats']['max']:.4f}",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                ],
            }

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name="Target_Comparison", index=False)
            print("  Added: Target_Comparison sheet")

        print("  Added: PSI_Analysis sheet")

    return True


def main():
    print("=== ADDING PSI TO EXCEL REPORT ===\n")

    OUTPUT_FOLDER = "outputs"

    # Find model
    model_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("best_model_") and f.endswith(".joblib")]
    if not model_files:
        print("No model found!")
        return

    latest_model = sorted(model_files)[-1]
    run_id = latest_model.replace("best_model_", "").replace(".joblib", "")

    print(f"Using model: {run_id}")

    # Load artifacts
    model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)

    # Fix feature format
    if isinstance(final_features, dict):
        final_features = final_features.get("final_vars", final_features)
    if isinstance(final_features, dict):
    # final_features = list(final_features.values())[0]

    # Load data
    train_df = pd.read_csv("data/input.csv")
    scoring_df = pd.read_csv("data/scoring.csv")

    print("Calculating PSI...")

    # Get training scores
    from risk_pipeline.utils.scoring import apply_woe_transform

    train_woe = apply_woe_transform(train_df, woe_mapping)

    # Map features
    feature_mapping = {}
    for feat in final_features:
        if feat not in train_woe.columns:
            base_name = feat.split("_")[0]
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
    except Exception:
    # training_scores = model.predict(X_train)

    # Score new data
    # scoring_results = score_data(
        scoring_df = scoring_df,
        model = model,
    # final_features=final_features,
        woe_mapping = woe_mapping,
        calibrator = calibrator,
    # training_scores=training_scores,
        feature_mapping = feature_mapping if feature_mapping else None,
    )

    print(f"\nPSI Score: {scoring_results.get('psi_score', 'N/A')}")

    # Add to Excel
    excel_path=f"{OUTPUT_FOLDER}/model_report.xlsx"
    if os.path.exists(excel_path):
        print(f"\nUpdating Excel: {excel_path}")
        add_psi_to_excel(excel_path, scoring_results)

        # Verify
        xl=pd.ExcelFile(excel_path)
        print(f"\nExcel updated successfully!")
        print(f"Total sheets: {len(xl.sheet_names)}")

        psi_sheets=[s for s in xl.sheet_names if "PSI" in s.upper() or "Target_Comparison" in s]
        print(f"PSI related sheets: {', '.join(psi_sheets)}")
    else:
        print(f"Excel not found: {excel_path}")

    print("\n[SUCCESS] PSI analysis added to Excel report!")


if __name__ == "__main__":
    main()
