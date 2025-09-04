"""Compare old pipeline16 with new modular pipeline to ensure identical functionality"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.insert(0, 'src')

def create_test_data(n_samples=3000, seed=42):
    """Create reproducible test data"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    data = {
        'app_id': range(1, n_samples + 1),
        'app_dt': pd.date_range(start='2022-01-01', periods=n_samples, freq='D')[:n_samples],
    }
    
    # Numeric features
    data['risk_score'] = np.random.beta(2, 5, n_samples)
    data['payment_score'] = np.random.beta(3, 2, n_samples)
    data['debt_ratio'] = np.random.beta(2, 3, n_samples)
    data['income_level'] = np.random.lognormal(10, 1.5, n_samples)
    data['credit_months'] = np.random.gamma(3, 10, n_samples)
    
    # Categorical features
    data['employment_type'] = np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    # Create target
    risk_factor = (
        2.0 * data['risk_score'] + 
        1.5 * data['payment_score'] + 
        1.0 * data['debt_ratio'] +
        -0.3 * np.log1p(data['income_level'] / 10000) +
        np.random.normal(0, 0.3, n_samples)
    )
    
    data['target'] = (risk_factor > np.percentile(risk_factor, 75)).astype(int)
    
    # Add some missing values
    missing_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data['income_level'][missing_idx] = np.nan
    
    return pd.DataFrame(data)

print("="*80)
print("PIPELINE COMPARISON TEST")
print("="*80)

# Create test data
df = create_test_data(n_samples=3000)
print(f"\nTest dataset created:")
print(f"  Shape: {df.shape}")
print(f"  Target rate: {df['target'].mean():.2%}")
print(f"  Missing values: {df.isnull().sum().sum()}")

# Import both pipelines
from risk_pipeline.pipeline16 import Config, RiskModelPipeline as OldPipeline
from risk_pipeline.pipeline import RiskModelPipeline as NewPipeline

# Create identical configs
config_old = Config(
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    output_folder='test_old_pipeline',
    random_state=42,
    cv_folds=3,
    hpo_timeout_sec=30,
    hpo_trials=5,
    enable_dual_pipeline=True
)

config_new = Config(
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    output_folder='test_new_pipeline',
    random_state=42,
    cv_folds=3,
    hpo_timeout_sec=30,
    hpo_trials=5,
    enable_dual_pipeline=True
)

# Run OLD pipeline
print("\n" + "="*80)
print("RUNNING OLD PIPELINE (pipeline16.py)")
print("="*80)

start_old = time.time()
try:
    old_pipeline = OldPipeline(config_old)
    old_pipeline.run(df)
    time_old = time.time() - start_old
    
    print(f"\n✓ Old pipeline completed in {time_old:.2f} seconds")
    
    old_results = {
        'time': time_old,
        'final_vars': old_pipeline.final_vars_ if hasattr(old_pipeline, 'final_vars_') else [],
        'models': list(old_pipeline.models_.keys()) if hasattr(old_pipeline, 'models_') else [],
        'best_model': old_pipeline.best_model_name_ if hasattr(old_pipeline, 'best_model_name_') else None,
        'models_summary': old_pipeline.models_summary_ if hasattr(old_pipeline, 'models_summary_') else None
    }
    
except Exception as e:
    print(f"✗ Old pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    old_results = None

# Run NEW pipeline
print("\n" + "="*80)
print("RUNNING NEW MODULAR PIPELINE")
print("="*80)

start_new = time.time()
try:
    new_pipeline = NewPipeline(config_new)
    new_pipeline.run(df)
    time_new = time.time() - start_new
    
    print(f"\n✓ New pipeline completed in {time_new:.2f} seconds")
    
    new_results = {
        'time': time_new,
        'final_vars': new_pipeline.final_vars_ if hasattr(new_pipeline, 'final_vars_') else [],
        'models': list(new_pipeline.models_.keys()) if hasattr(new_pipeline, 'models_') else [],
        'best_model': new_pipeline.best_model_name_ if hasattr(new_pipeline, 'best_model_name_') else None,
        'models_summary': new_pipeline.models_summary_ if hasattr(new_pipeline, 'models_summary_') else None
    }
    
except Exception as e:
    print(f"✗ New pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    new_results = None

# COMPARISON RESULTS
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

if old_results and new_results:
    print("\n1. EXECUTION TIME:")
    print(f"   Old pipeline: {old_results['time']:.2f} seconds")
    print(f"   New pipeline: {new_results['time']:.2f} seconds")
    print(f"   Difference: {abs(old_results['time'] - new_results['time']):.2f} seconds")
    
    print("\n2. FINAL VARIABLES:")
    print(f"   Old pipeline: {len(old_results['final_vars'])} variables")
    print(f"   New pipeline: {len(new_results['final_vars'])} variables")
    
    if set(old_results['final_vars']) == set(new_results['final_vars']):
        print("   ✓ Variables match exactly!")
    else:
        print("   ⚠ Variables differ:")
        print(f"     Only in old: {set(old_results['final_vars']) - set(new_results['final_vars'])}")
        print(f"     Only in new: {set(new_results['final_vars']) - set(old_results['final_vars'])}")
    
    print("\n3. MODELS TRAINED:")
    print(f"   Old pipeline: {len(old_results['models'])} models")
    print(f"   New pipeline: {len(new_results['models'])} models")
    
    if set(old_results['models']) == set(new_results['models']):
        print("   ✓ Model names match!")
    else:
        print("   ⚠ Model names differ:")
        print(f"     Only in old: {set(old_results['models']) - set(new_results['models'])}")
        print(f"     Only in new: {set(new_results['models']) - set(old_results['models'])}")
    
    print("\n4. BEST MODEL:")
    print(f"   Old pipeline: {old_results['best_model']}")
    print(f"   New pipeline: {new_results['best_model']}")
    
    if old_results['best_model'] == new_results['best_model']:
        print("   ✓ Best model selection matches!")
    else:
        print("   ⚠ Different best model selected")
    
    print("\n5. MODEL PERFORMANCE:")
    if old_results['models_summary'] is not None and new_results['models_summary'] is not None:
        # Compare Gini_OOT scores
        old_gini = old_results['models_summary'].set_index('model_name')['Gini_OOT']
        new_gini = new_results['models_summary'].set_index('model_name')['Gini_OOT']
        
        common_models = set(old_gini.index) & set(new_gini.index)
        
        if common_models:
            print("   Gini_OOT comparison for common models:")
            for model in sorted(common_models):
                old_val = old_gini.get(model, np.nan)
                new_val = new_gini.get(model, np.nan)
                diff = abs(old_val - new_val)
                
                status = "✓" if diff < 0.01 else "⚠"
                print(f"     {model:20s}: Old={old_val:.4f}, New={new_val:.4f}, Diff={diff:.4f} {status}")
    
    print("\n6. OUTPUT FILES:")
    # Check output files
    old_files = os.listdir(config_old.output_folder) if os.path.exists(config_old.output_folder) else []
    new_files = os.listdir(config_new.output_folder) if os.path.exists(config_new.output_folder) else []
    
    print(f"   Old pipeline: {len(old_files)} files")
    print(f"   New pipeline: {len(new_files)} files")
    
    # Check for Excel files
    old_excel = [f for f in old_files if f.endswith('.xlsx')]
    new_excel = [f for f in new_files if f.endswith('.xlsx')]
    
    if old_excel and new_excel:
        print(f"   ✓ Both created Excel reports")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Overall assessment
    issues = []
    
    if set(old_results['final_vars']) != set(new_results['final_vars']):
        issues.append("Different final variables")
    
    if old_results['best_model'] != new_results['best_model']:
        issues.append("Different best model selection")
    
    if len(old_results['models']) != len(new_results['models']):
        issues.append("Different number of models")
    
    if not issues:
        print("✓ NEW PIPELINE WORKS IDENTICALLY TO OLD PIPELINE!")
        print("✓ Safe to move pipeline16.py to old_process folder")
    else:
        print("⚠ Some differences detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nReview these differences before replacing pipeline16.py")

else:
    print("✗ Could not complete comparison due to pipeline failures")

print("\n" + "="*80)