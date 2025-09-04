"""
Advanced Usage Example - Dual Pipeline with Custom Settings
"""

import pandas as pd
import numpy as np
from risk_pipeline import Config, RiskModelPipeline

# Generate sample data with missing values
np.random.seed(42)
n_samples = 10000

df = pd.DataFrame({
    'app_id': range(1, n_samples + 1),
    'app_date': pd.date_range(start='2022-01-01', periods=n_samples, freq='H')[:n_samples],
    'income': np.random.lognormal(10, 1.5, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'payment_history': np.random.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.6, 0.3, 0.1]),
    'employment': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
    'target': np.random.binomial(1, 0.2, n_samples)
})

# Add missing values
missing_idx = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
df.loc[missing_idx, 'income'] = np.nan

# Advanced configuration
config = Config(
    # Data columns
    id_col='app_id',
    time_col='app_date',
    target_col='target',
    
    # Enable dual pipeline
    enable_dual_pipeline=True,
    
    # Multiple imputation for RAW pipeline
    raw_imputation_strategy='multiple',
    
    # Balanced model selection (70% performance + 30% stability)
    model_selection_method='balanced',
    model_stability_weight=0.3,
    max_train_oot_gap=0.15,  # Max 15% Train-OOT gap
    
    # Feature engineering
    rare_threshold=0.01,
    psi_threshold=0.25,
    iv_min=0.02,
    rho_threshold=0.90,
    
    # Model training
    cv_folds=5,
    hpo_method='optuna',
    hpo_timeout_sec=300,
    hpo_trials=50,
    
    # Output
    output_folder='outputs_advanced',
    output_excel_path='risk_model_advanced.xlsx',
    random_state=42
)

# Run pipeline
pipeline = RiskModelPipeline(config)
results = pipeline.run(df)

# Export comprehensive reports
pipeline.export_reports()

# Access results
print(f"Best model: {pipeline.best_model_name_}")
print(f"Selected features: {pipeline.final_vars_}")
print(f"Models summary shape: {pipeline.models_summary_.shape}")
print("\nPipeline completed! Check 'outputs_advanced' folder for detailed reports.")