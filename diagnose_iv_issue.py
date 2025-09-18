"""
Diagnostic script to identify why IV values are 0
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from risk_pipeline.core.woe_transformer import EnhancedWOETransformer
from risk_pipeline.core.config import Config

def diagnose_woe_iv(df, feature_col, target_col='target'):
    """Diagnose WOE and IV calculation for a single feature"""

    print(f"\n{'='*60}")
    print(f"Diagnosing: {feature_col}")
    print(f"{'='*60}")

    X = df[feature_col]
    y = df[target_col]

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Type: {X.dtype}")
    print(f"  Unique values: {X.nunique()}")
    print(f"  Missing: {X.isna().sum()} ({X.isna().mean():.1%})")
    print(f"  Target rate: {y.mean():.2%}")

    if pd.api.types.is_numeric_dtype(X):
        print(f"  Min: {X.min():.2f}")
        print(f"  Max: {X.max():.2f}")
        print(f"  Mean: {X.mean():.2f}")
        print(f"  Std: {X.std():.2f}")

    # Check target distribution
    print(f"\nTarget Distribution:")
    print(f"  Class 0: {(y==0).sum()} samples")
    print(f"  Class 1: {(y==1).sum()} samples")

    # Create WOE transformer
    config = Config(
        n_bins=5,
        min_bin_size=0.05,
        binning_method='quantile'
    )

    transformer = EnhancedWOETransformer(config)

    try:
        # Calculate WOE
        result = transformer.fit_transform_single(X, y)

        print(f"\nWOE Calculation Result:")
        print(f"  IV: {result.get('iv', 'NOT FOUND')}")
        print(f"  Type: {result.get('type', 'UNKNOWN')}")

        if 'stats' in result and result['stats']:
            print(f"\nBin Statistics:")
            for stat in result['stats'][:5]:  # Show first 5 bins
                print(f"  {stat.get('bin', 'N/A')}: "
                      f"Count={stat.get('count', 0)}, "
                      f"BadRate={stat.get('bad_rate', 0):.2%}, "
                      f"WOE={stat.get('woe', 0):.3f}, "
                      f"IV={stat.get('iv', 0):.4f}")

        # Manual IV calculation for verification
        if pd.api.types.is_numeric_dtype(X):
            manual_iv = calculate_iv_manual(X, y)
            print(f"\nManual IV Calculation: {manual_iv:.4f}")

        return result.get('iv', 0)

    except Exception as e:
        print(f"\nERROR in WOE calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0

def calculate_iv_manual(X, y):
    """Manual IV calculation for verification"""

    # Create bins
    X_clean = X.fillna(X.median())
    bins = pd.qcut(X_clean, 5, duplicates='drop')

    iv = 0
    total_good = (y == 0).sum()
    total_bad = (y == 1).sum()

    if total_good == 0 or total_bad == 0:
        return 0

    for bin_label in bins.cat.categories:
        mask = bins == bin_label
        good = ((y == 0) & mask).sum()
        bad = ((y == 1) & mask).sum()

        # Add smoothing
        pct_good = (good + 0.5) / (total_good + 0.5)
        pct_bad = (bad + 0.5) / (total_bad + 0.5)

        if pct_good > 0 and pct_bad > 0:
            woe = np.log(pct_bad / pct_good)
            iv_component = (pct_bad - pct_good) * woe
            iv += iv_component

    return iv

def main():
    """Main diagnostic function"""

    print("="*80)
    print("WOE/IV DIAGNOSTIC TEST")
    print("="*80)

    # Create sample data similar to your case
    np.random.seed(42)
    n = 15000

    df = pd.DataFrame({
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

    print(f"\nDataset: {n} samples, {df['target'].mean():.2%} default rate")

    # Test numeric features
    numeric_features = ['age', 'income', 'credit_score', 'debt_to_income']
    categorical_features = ['home_ownership', 'loan_purpose']

    iv_results = {}

    print("\n" + "="*80)
    print("TESTING NUMERIC FEATURES")
    print("="*80)

    for feature in numeric_features:
        iv = diagnose_woe_iv(df, feature)
        iv_results[feature] = iv

    print("\n" + "="*80)
    print("TESTING CATEGORICAL FEATURES")
    print("="*80)

    for feature in categorical_features:
        iv = diagnose_woe_iv(df, feature)
        iv_results[feature] = iv

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nIV Values Summary:")
    for feature, iv in iv_results.items():
        status = "✓ OK" if iv > 0 else "✗ PROBLEM"
        print(f"  {feature:20s}: {iv:8.4f}  {status}")

    # Test with real relationship
    print("\n" + "="*80)
    print("TESTING WITH REAL RELATIONSHIP")
    print("="*80)

    # Create target with real relationship
    risk_score = (
        -0.005 * df['credit_score'] +
        0.02 * df['debt_to_income'] +
        np.where(df['home_ownership'] == 'RENT', 0.5, 0) +
        np.random.randn(n) * 0.5
    )
    df['target_real'] = (risk_score > np.percentile(risk_score, 87)).astype(int)

    print(f"\nReal target rate: {df['target_real'].mean():.2%}")

    for feature in ['credit_score', 'debt_to_income', 'home_ownership']:
        iv = diagnose_woe_iv(df, feature, 'target_real')
        print(f"  {feature} with real target: IV={iv:.4f}")

if __name__ == "__main__":
    main()