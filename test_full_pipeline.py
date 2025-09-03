#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test full pipeline with realistic data
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from src.risk_pipeline.pipeline16 import RiskModelPipeline, Config

warnings.filterwarnings('ignore')

print("="*80)
print("FULL PIPELINE TEST")
print("="*80)

# Set seed
np.random.seed(42)

# Create realistic dataset
n_samples = 2000

def create_realistic_data(n_samples):
    """Create realistic credit risk data"""
    
    # Base features that affect default
    age = np.random.normal(40, 12, n_samples).clip(18, 70)
    income = np.random.lognormal(10, 0.5, n_samples)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    
    # Default probability based on features
    default_logit = (
        -3.0 +
        0.02 * (40 - age) +  # Younger = higher risk
        -0.0001 * income +   # Higher income = lower risk
        -0.005 * credit_score  # Higher score = lower risk
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    default_prob = default_prob.clip(0.05, 0.4)
    
    # Generate target
    target = np.random.binomial(1, default_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'app_id': [f'APP_{i:06d}' for i in range(n_samples)],
        'app_dt': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'target': target,
        
        # Core features
        'age': age.astype(int),
        'income': income,
        'credit_score': credit_score,
        'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
        'loan_amount': np.random.exponential(50000, n_samples),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'num_credit_lines': np.random.poisson(3, n_samples).clip(0, 10),
        
        # Categorical features  
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Retired'], 
                                           n_samples, p=[0.6, 0.3, 0.1]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                          n_samples, p=[0.3, 0.5, 0.2]),
        'home_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], 
                                         n_samples, p=[0.3, 0.4, 0.3]),
    })
    
    # Add some missing values
    missing_cols = ['employment_years', 'num_credit_lines']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

# Create data
print("\n1. Creating realistic data...")
df = create_realistic_data(n_samples)
print(f"   Shape: {df.shape}")
print(f"   Target rate: {df['target'].mean():.2%}")

# Create calibration data
print("\n2. Creating calibration data...")
cal_df = create_realistic_data(500)
print(f"   Shape: {cal_df.shape}")
print(f"   Target rate: {cal_df['target'].mean():.2%}")

# Create data dictionary
print("\n3. Creating data dictionary...")
data_dict = pd.DataFrame({
    'alan_adi': [
        'age', 'income', 'credit_score', 'debt_to_income', 'loan_amount',
        'employment_years', 'num_credit_lines', 'education', 'employment_type',
        'marital_status', 'home_ownership'
    ],
    'alan_aciklamasi': [
        'Customer age (years)',
        'Monthly income (USD)', 
        'Credit score (300-850)',
        'Debt to income ratio (%)',
        'Requested loan amount (USD)',
        'Years of employment',
        'Number of credit lines',
        'Education level',
        'Employment type',
        'Marital status',
        'Home ownership status'
    ]
})
print(f"   Variables described: {len(data_dict)}")

# Configure pipeline
print("\n4. Configuring pipeline...")
cfg = Config(
    # Core settings
    id_col='app_id',
    time_col='app_dt', 
    target_col='target',
    
    # Data split
    use_test_split=True,
    test_size_row_frac=0.2,
    oot_window_months=2,
    
    # Data dictionary and calibration
    data_dictionary_df=data_dict,
    calibration_df=cal_df,
    calibration_method='isotonic',
    
    # Model settings
    cv_folds=3,
    random_state=42,
    n_jobs=2,
    
    # HPO settings
    hpo_timeout_sec=30,
    hpo_trials=10,
    
    # Feature engineering
    rare_threshold=0.02,
    psi_threshold=0.20,
    iv_min=0.02,
    rho_threshold=0.95,
    
    # Output
    output_folder='outputs_full_test',
    output_excel_path='full_test_report.xlsx',
    log_file='outputs_full_test/pipeline.log',
    write_parquet=True,
    write_csv=True
)

print("   Config ready!")

# Run pipeline
print("\n5. Running pipeline...")
print("="*60)
pipeline = RiskModelPipeline(cfg)
pipeline.run(df)

# Check results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

if pipeline.best_model_name_:
    print(f"Best Model: {pipeline.best_model_name_}")
    print(f"Final Features: {len(pipeline.final_vars_)}")
    
    if pipeline.models_summary_ is not None and not pipeline.models_summary_.empty:
        best = pipeline.models_summary_[pipeline.models_summary_['model_name'] == pipeline.best_model_name_].iloc[0]
        print(f"\nPerformance:")
        print(f"  AUC (OOT): {best.get('AUC_OOT', 'N/A')}")
        print(f"  Gini (OOT): {best.get('Gini_OOT', 'N/A')}")
        print(f"  KS (OOT): {best.get('KS_OOT', 'N/A')}")
else:
    print("WARNING: No model was selected!")
    print(f"Final vars: {pipeline.final_vars_}")

# Check Excel output
excel_path = os.path.join(cfg.output_folder, cfg.output_excel_path)
if os.path.exists(excel_path):
    print(f"\nExcel report created: {excel_path}")
    excel_file = pd.ExcelFile(excel_path)
    print(f"Sheets: {len(excel_file.sheet_names)}")
else:
    print(f"\nExcel report not found at {excel_path}")

print("\n" + "="*80)
print("TEST COMPLETED!")
print("="*80)