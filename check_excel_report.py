"""Check the contents of the latest Excel report"""

import pandas as pd
import os

# Find the latest report
report_path = "test_final/model_report_20250908_120955.xlsx"

if os.path.exists(report_path):
    print(f"Checking report: {report_path}\n")
    
    # Read all sheets
    xl = pd.ExcelFile(report_path)
    print(f"Available sheets: {xl.sheet_names}\n")
    
    # Check for best_model_details sheet
    if 'best_model_details' in xl.sheet_names:
        df = pd.read_excel(report_path, sheet_name='best_model_details')
        print("=" * 80)
        print("BEST MODEL DETAILS SHEET:")
        print("=" * 80)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}\n")
        
        # Show first few rows
        print("First 5 rows:")
        print(df.head())
        
        # Check for WOE-related columns
        woe_cols = ['bin_range', 'woe', 'event_rate', 'event_count', 'nonevent_count', 
                    'total_count', 'iv_contrib', 'total_iv']
        present = [c for c in woe_cols if c in df.columns]
        missing = [c for c in woe_cols if c not in df.columns]
        
        print(f"\nWOE columns present: {present}")
        if missing:
            print(f"WOE columns missing: {missing}")
            
        # Show a sample of WOE data
        if present:
            print("\nSample WOE data (first variable):")
            if not df.empty:
                first_var = df['variable'].iloc[0] if 'variable' in df.columns else None
                if first_var:
                    var_data = df[df['variable'] == first_var]
                    print(var_data[['variable'] + present[:5]])
    else:
        print("best_model_details sheet NOT FOUND!")
    
    # Check for SHAP sheet
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS:")
    print("=" * 80)
    if 'shap_importance' in xl.sheet_names:
        df = pd.read_excel(report_path, sheet_name='shap_importance')
        print(f"SHAP sheet found! Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nTop 5 important features by SHAP:")
        print(df.head())
    else:
        print("SHAP sheet NOT FOUND!")
    
    # Check WOE mapping sheet
    print("\n" + "=" * 80)
    print("WOE MAPPING:")
    print("=" * 80)
    if 'woe_mapping' in xl.sheet_names:
        df = pd.read_excel(report_path, sheet_name='woe_mapping')
        print(f"WOE mapping sheet found! Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 WOE mappings:")
        print(df.head())
    else:
        print("WOE mapping sheet NOT FOUND!")
        
else:
    print(f"Report not found: {report_path}")