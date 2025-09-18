"""
Configuration to fix the "no features selected" issue
"""

import sys
sys.path.insert(0, 'src')

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

def get_ultra_permissive_config(**overrides):
    """
    Ultra-permissive configuration that will keep features even with very low IV
    """
    config_dict = {
        # Basic settings
        'target_col': 'target',
        'id_col': 'app_id',
        'time_col': 'app_dt',
        'random_state': 42,

        # EXTREMELY PERMISSIVE THRESHOLDS
        'iv_min': 0.0,  # Accept ANY IV value, even 0
        'iv_threshold': 0.0,  # No IV filtering at all
        'iv_high_threshold': 100.0,  # No upper limit
        'psi_threshold': 100.0,  # Effectively disable PSI filtering
        'rho_threshold': 0.99,  # Only remove perfect correlations
        'correlation_threshold': 0.99,
        'vif_threshold': 100.0,  # Effectively disable VIF filtering
        'rare_threshold': 0.001,  # 0.1% threshold

        # WOE settings
        'enable_woe': True,
        'n_bins': 5,  # Fewer bins for stability
        'max_bins': 10,
        'min_bin_size': 0.05,
        'woe_monotonic': False,

        # Pipeline settings
        'enable_dual_pipeline': False,
        'enable_noise_sentinel': False,  # Don't add noise
        'enable_calibration': True,
        'enable_scoring': False,

        # SIMPLIFIED SELECTION - Remove problematic filters
        'selection_order': [],  # NO FILTERING AT ALL - keep all features
        'selection_method': 'forward',
        'max_features': 50,  # Allow many features
        'min_features': 1,

        # Model settings
        'model_type': 'logistic',  # Start with simple model
        'use_optuna': False,
        'n_trials': 10,
        'cv_folds': 3,

        # Split settings
        'test_ratio': 0.2,
        'oot_months': 0,  # No OOT
        'equal_default_splits': False,  # Don't force equal splits

        # Output settings
        'output_folder': 'output_fixed',
        'save_plots': False,
        'save_model': True
    }

    # Apply overrides
    config_dict.update(overrides)

    return Config(**config_dict)

def get_minimal_filtering_config(**overrides):
    """
    Minimal filtering - only remove truly problematic features
    """
    config_dict = {
        # Basic settings
        'target_col': 'target',
        'id_col': 'app_id',
        'time_col': 'app_dt',
        'random_state': 42,

        # Very low thresholds
        'iv_min': 0.00001,  # Extremely low IV threshold
        'iv_threshold': 0.00001,
        'iv_high_threshold': 100.0,
        'psi_threshold': 10.0,  # High PSI tolerance
        'rho_threshold': 0.95,  # High correlation tolerance
        'correlation_threshold': 0.95,
        'vif_threshold': 50.0,  # High VIF tolerance
        'rare_threshold': 0.001,

        # WOE settings
        'enable_woe': True,
        'n_bins': 5,
        'max_bins': 10,
        'min_bin_size': 0.05,
        'woe_monotonic': False,

        # Pipeline settings
        'enable_dual_pipeline': False,
        'enable_noise_sentinel': False,
        'enable_calibration': True,
        'enable_scoring': False,

        # Only basic selection
        'selection_order': ['iv'],  # Only IV filtering with very low threshold
        'selection_method': 'forward',
        'max_features': 30,
        'min_features': 3,

        # Model settings
        'model_type': 'all',  # Try all models
        'use_optuna': False,
        'n_trials': 10,
        'cv_folds': 3,

        # Split settings
        'test_ratio': 0.2,
        'oot_months': 0,
        'equal_default_splits': False,

        # Output settings
        'output_folder': 'output_minimal',
        'save_plots': False,
        'save_model': True
    }

    # Apply overrides
    config_dict.update(overrides)

    return Config(**config_dict)

def test_with_sample_data():
    """Test the configuration with sample data"""

    import pandas as pd
    import numpy as np

    print("Creating sample data...")
    np.random.seed(42)
    n = 5000

    df = pd.DataFrame({
        'app_id': range(n),
        'app_dt': pd.date_range(start='2022-01-01', periods=n, freq='H')[:n],
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n),
        'loan_amount': np.random.lognormal(9, 0.5, n),
        'employment_years': np.random.gamma(2, 2, n),
        'credit_score': np.random.normal(650, 100, n),
        'debt_to_income': np.random.beta(2, 5, n) * 100,
        'num_credit_lines': np.random.poisson(5, n),
        'months_since_delinquent': np.where(np.random.rand(n) < 0.7, np.nan, np.random.exponential(12, n)),
        'total_credit_limit': np.random.lognormal(10, 0.5, n) * 1000,
        'utilization_rate': np.random.beta(2, 3, n) * 100,
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n),
        'loan_purpose': np.random.choice(['debt', 'credit', 'home'], n),
        'employment_type': np.random.choice(['Full', 'Part', 'Self'], n),
        'education': np.random.choice(['HS', 'BS', 'MS'], n),
        'marital_status': np.random.choice(['S', 'M', 'D'], n),
        'state': np.random.choice(['CA', 'NY', 'TX'], n),
        'target': np.random.binomial(1, 0.13, n)
    })

    print(f"Created {n} samples with {df['target'].mean():.2%} default rate")

    # Test 1: Ultra permissive (no filtering)
    print("\n" + "="*60)
    print("TEST 1: ULTRA PERMISSIVE CONFIG (No filtering)")
    print("="*60)

    config1 = get_ultra_permissive_config()
    pipeline1 = UnifiedRiskPipeline(config1)

    try:
        results1 = pipeline1.fit(df)
        if 'selected_features' in results1:
            print(f"[OK] Selected {len(results1['selected_features'])} features")
            print(f"  First 5: {results1['selected_features'][:5]}")
        else:
            print("[FAIL] No features selected")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 2: Minimal filtering
    print("\n" + "="*60)
    print("TEST 2: MINIMAL FILTERING CONFIG")
    print("="*60)

    config2 = get_minimal_filtering_config()
    pipeline2 = UnifiedRiskPipeline(config2)

    try:
        results2 = pipeline2.fit(df)
        if 'selected_features' in results2:
            print(f"[OK] Selected {len(results2['selected_features'])} features")
            print(f"  Features: {results2['selected_features']}")
        else:
            print("[FAIL] No features selected")
    except Exception as e:
        print(f"[ERROR] {e}")

    return results1 if 'selected_features' in results1 else results2

if __name__ == "__main__":
    print("="*80)
    print("TESTING CONFIGURATIONS TO FIX NO FEATURES ISSUE")
    print("="*80)

    results = test_with_sample_data()

    print("\n" + "="*80)
    print("RECOMMENDED USAGE FOR YOUR DATA:")
    print("="*80)
    print("""
from fix_no_features_config import get_ultra_permissive_config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

# Use ultra-permissive config to keep all features
config = get_ultra_permissive_config(
    target_col='target',  # Your target column
    id_col='app_id',      # Your ID column
    time_col='app_dt',    # Your date column
)

pipeline = UnifiedRiskPipeline(config)
results = pipeline.fit(your_dataframe)
    """)