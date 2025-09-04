"""Quick test to verify new pipeline works"""

import sys
import os
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

# Create simple test data
np.random.seed(42)
df = pd.DataFrame({
    'app_id': range(1, 1001),
    'app_dt': pd.date_range('2022-01-01', periods=1000),
    'risk_score': np.random.beta(2, 5, 1000),
    'payment_score': np.random.beta(3, 2, 1000),
    'debt_ratio': np.random.beta(2, 3, 1000),
    'target': np.random.binomial(1, 0.2, 1000)
})

print(f"Test data: {df.shape}, Target rate: {df['target'].mean():.2%}")

# Test old pipeline
print("\n--- Testing OLD pipeline ---")
try:
    from risk_pipeline.pipeline16 import Config, RiskModelPipeline as OldPipeline
    
    config = Config(
        id_col='app_id',
        time_col='app_dt',
        target_col='target',
        output_folder='test_old',
        random_state=42,
        cv_folds=2,
        hpo_timeout_sec=10,
        hpo_trials=2
    )
    
    old_pipeline = OldPipeline(config)
    old_pipeline.run(df)
    
    print(f"[OK] Old pipeline works")
    print(f"  Final vars: {len(old_pipeline.final_vars_)}")
    print(f"  Best model: {old_pipeline.best_model_name_}")
    
except Exception as e:
    print(f"[FAIL] Old pipeline failed: {e}")

# Test new pipeline
print("\n--- Testing NEW pipeline ---")
try:
    from risk_pipeline.pipeline import RiskModelPipeline as NewPipeline
    
    config = Config(
        id_col='app_id',
        time_col='app_dt',
        target_col='target',
        output_folder='test_new',
        random_state=42,
        cv_folds=2,
        hpo_timeout_sec=10,
        hpo_trials=2
    )
    
    new_pipeline = NewPipeline(config)
    new_pipeline.run(df)
    
    print(f"[OK] New pipeline works")
    print(f"  Final vars: {len(new_pipeline.final_vars_)}")
    print(f"  Best model: {new_pipeline.best_model_name_}")
    
except Exception as e:
    print(f"[FAIL] New pipeline failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")