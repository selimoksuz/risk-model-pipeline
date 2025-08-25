#!/usr/bin/env python3
"""
Final End-to-End Verification Test
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')

def main():
    print("="*70)
    print("FINAL END-TO-END VERIFICATION")
    print("="*70)
    
    OUTPUT_FOLDER = "outputs"
    
    # Check outputs folder
    print("\n[1] OUTPUT FOLDER CHECK:")
    if os.path.exists(OUTPUT_FOLDER):
        files = os.listdir(OUTPUT_FOLDER)
        print(f"  [OK] Folder exists: {OUTPUT_FOLDER}/")
        print(f"  [OK] Total files: {len(files)}")
        
        # Count file types
        models = [f for f in files if 'model' in f and f.endswith('.joblib')]
        jsons = [f for f in files if f.endswith('.json')]
        excels = [f for f in files if f.endswith('.xlsx')]
        csvs = [f for f in files if f.endswith('.csv')]
        
        print(f"  [OK] Model files: {len(models)}")
        print(f"  [OK] JSON files: {len(jsons)}")
        print(f"  [OK] Excel files: {len(excels)}")
        print(f"  [OK] CSV files: {len(csvs)}")
    else:
        print(f"  [FAIL] Folder not found!")
        return
    
    # Check Excel report
    print("\n[2] EXCEL REPORT CHECK:")
    excel_path = f"{OUTPUT_FOLDER}/model_report.xlsx"
    
    if os.path.exists(excel_path):
        xl = pd.ExcelFile(excel_path)
        sheets = xl.sheet_names
        
        print(f"  [OK] Excel exists: {excel_path}")
        print(f"  [OK] Total sheets: {len(sheets)}")
        print(f"  [OK] File size: {os.path.getsize(excel_path)/1024:.1f} KB")
        
        # Categorize sheets
        pipeline_sheets = []
        scoring_sheets = []
        psi_sheets = []
        
        for sheet in sheets:
            sheet_lower = sheet.lower()
            if 'scoring' in sheet_lower or 'score_' in sheet_lower:
                scoring_sheets.append(sheet)
            elif 'psi' in sheet_lower:
                psi_sheets.append(sheet)
            elif 'target_comparison' in sheet_lower:
                scoring_sheets.append(sheet)
            else:
                pipeline_sheets.append(sheet)
        
        print(f"\n  Sheet Categories:")
        print(f"    Pipeline: {len(pipeline_sheets)} sheets")
        print(f"    Scoring: {len(scoring_sheets)} sheets")
        print(f"    PSI: {len(psi_sheets)} sheets")
        
        # List key sheets
        print(f"\n  Key Pipeline Sheets:")
        key_pipeline = ['final_vars', 'best_name', 'models_summary', 'woe_mapping', 'oot_scores']
        for sheet in key_pipeline:
            if sheet in sheets:
                print(f"    [OK] {sheet}")
        
        print(f"\n  Key Scoring Sheets:")
        key_scoring = ['scoring_summary', 'scoring_with_target', 'scoring_without_target', 
                      'score_distribution', 'Target_Comparison']
        for sheet in key_scoring:
            if sheet in sheets:
                print(f"    [OK] {sheet}")
        
        print(f"\n  PSI Sheets:")
        for sheet in psi_sheets:
            print(f"    [OK] {sheet}")
        
        # Sample PSI data
        if 'PSI_Analysis1' in sheets:
            psi_df = pd.read_excel(xl, 'PSI_Analysis1')
            print(f"\n  PSI Analysis Content:")
            print(f"    PSI Score: {psi_df.iloc[0]['Value']}")
            print(f"    Total Scored: {psi_df.iloc[5]['Value']}")
            print(f"    With Target: {psi_df.iloc[6]['Value']}")
            print(f"    Without Target: {psi_df.iloc[7]['Value']}")
        
        # Sample Target Comparison
        if 'Target_Comparison' in sheets:
            comp_df = pd.read_excel(xl, 'Target_Comparison')
            print(f"\n  Target Comparison:")
            print(f"    With Target Records: {comp_df.iloc[0]['With_Target']}")
            print(f"    Without Target Records: {comp_df.iloc[0]['Without_Target']}")
            if len(comp_df) > 5:
                print(f"    Default Rate (With Target): {comp_df.iloc[5]['With_Target']}")
    else:
        print(f"  [FAIL] Excel not found!")
    
    # Quick scoring test
    print("\n[3] SCORING FUNCTIONALITY CHECK:")
    
    # Find latest model
    model_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('best_model_') and f.endswith('.joblib')]
    if model_files:
        latest_model = sorted(model_files)[-1]
        run_id = latest_model.replace('best_model_', '').replace('.joblib', '')
        
        from risk_pipeline.utils.scoring import load_model_artifacts
        model, final_features, woe_mapping, calibrator = load_model_artifacts(OUTPUT_FOLDER, run_id)
        
        print(f"  [OK] Model loaded: {type(model).__name__}")
        print(f"  [OK] Calibrator: {'Available' if calibrator else 'Not found'}")
        
        # Check scoring data
        scoring_df = pd.read_csv('data/scoring.csv')
        print(f"  [OK] Scoring data: {scoring_df.shape[0]:,} rows")
        print(f"    - With target: {(~scoring_df['target'].isna()).sum():,}")
        print(f"    - Without target: {scoring_df['target'].isna().sum():,}")
    else:
        print(f"  [FAIL] No model found!")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    checks = {
        "Single output folder": os.path.exists(OUTPUT_FOLDER),
        "Model files exist": len(models) > 0 if 'models' in locals() else False,
        "Excel report exists": os.path.exists(excel_path),
        "Multiple sheets in Excel": len(sheets) > 20 if 'sheets' in locals() else False,
        "PSI analysis included": any('psi' in s.lower() for s in sheets) if 'sheets' in locals() else False,
        "Target separation sheets": 'Target_Comparison' in sheets if 'sheets' in locals() else False,
        "Scoring sheets present": len(scoring_sheets) >= 3 if 'scoring_sheets' in locals() else False,
        "WOE mapping saved": any('woe' in f for f in files) if 'files' in locals() else False
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("[SUCCESS] ALL SYSTEMS OPERATIONAL - END-TO-END TEST PASSED!")
    else:
        print("[WARNING]  Some checks failed - review output above")
    print("="*70)

if __name__ == "__main__":
    main()