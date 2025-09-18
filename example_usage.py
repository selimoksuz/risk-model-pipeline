"""
Example usage of the risk model pipeline with recommended configurations
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from recommended_config import get_conservative_config, get_standard_config

def main():
    """Example pipeline usage"""

    print("="*80)
    print("RISK MODEL PIPELINE - EXAMPLE USAGE")
    print("="*80)

    # Load your data
    # df = pd.read_csv('your_data.csv')
    # For demo, create sample data
    df = create_sample_data()

    # Option 1: Use conservative config for initial exploration
    print("\n1. CONSERVATIVE APPROACH (keeps more features)")
    print("-"*40)

    config = get_conservative_config(
        target_col='target',
        id_col='app_id',
        time_col='app_dt',
        output_folder='output_conservative'
    )

    pipeline = UnifiedRiskPipeline(config)
    results = pipeline.fit(df)

    if 'selected_features' in results:
        print(f"\nSelected {len(results['selected_features'])} features:")
        for i, feat in enumerate(results['selected_features'][:10], 1):
            print(f"  {i}. {feat}")
        if len(results['selected_features']) > 10:
            print(f"  ... and {len(results['selected_features']) - 10} more")

    if 'best_model_name' in results:
        print(f"\nBest model: {results['best_model_name']}")
        if 'scores' in results and results['best_model_name'] in results['scores']:
            scores = results['scores'][results['best_model_name']]
            if 'test_auc' in scores:
                print(f"Test AUC: {scores['test_auc']:.4f}")

    # Option 2: If too few features selected, adjust thresholds
    if 'selected_features' not in results or len(results['selected_features']) < 5:
        print("\n2. ADJUSTED APPROACH (very permissive thresholds)")
        print("-"*40)

        config = get_conservative_config(
            target_col='target',
            id_col='app_id',
            time_col='app_dt',

            # Make thresholds even more permissive
            iv_threshold=0.00001,  # Almost no IV filtering
            psi_threshold=10.0,     # Very high PSI tolerance
            vif_threshold=50.0,     # Very high VIF tolerance

            # Simplify selection
            selection_order=['iv'],  # Only IV selection

            output_folder='output_adjusted'
        )

        pipeline = UnifiedRiskPipeline(config)
        results = pipeline.fit(df)

        if 'selected_features' in results:
            print(f"\nWith adjusted thresholds, selected {len(results['selected_features'])} features")

    return results


def create_sample_data(n_samples=5000):
    """Create sample credit risk data"""

    np.random.seed(42)

    df = pd.DataFrame({
        'app_id': range(n_samples),
        'app_dt': pd.date_range(start='2022-01-01', periods=n_samples, freq='H')[:n_samples]
    })

    # Numeric features
    df['age'] = np.random.randint(18, 70, n_samples)
    df['income'] = np.random.lognormal(10.5, 0.6, n_samples)
    df['loan_amount'] = np.random.lognormal(9, 0.7, n_samples)
    df['employment_years'] = np.random.gamma(2, 2, n_samples)
    df['credit_score'] = np.random.normal(650, 100, n_samples)
    df['debt_to_income'] = np.random.beta(2, 5, n_samples) * 100
    df['num_credit_lines'] = np.random.poisson(5, n_samples)
    df['months_since_delinquent'] = np.where(
        np.random.rand(n_samples) < 0.7,
        np.nan,
        np.random.exponential(12, n_samples)
    )
    df['total_credit_limit'] = df['income'] * np.random.uniform(0.3, 0.8, n_samples)
    df['utilization_rate'] = np.random.beta(2, 3, n_samples) * 100

    # Categorical features
    df['home_ownership'] = np.random.choice(
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        n_samples,
        p=[0.4, 0.2, 0.35, 0.05]
    )
    df['loan_purpose'] = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'other'],
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    df['employment_type'] = np.random.choice(
        ['Full-time', 'Part-time', 'Self-employed', 'Retired'],
        n_samples,
        p=[0.6, 0.1, 0.2, 0.1]
    )
    df['education'] = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'Other'],
        n_samples,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    df['marital_status'] = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'],
        n_samples,
        p=[0.35, 0.45, 0.15, 0.05]
    )
    df['state'] = np.random.choice(
        ['CA', 'NY', 'TX', 'FL', 'IL', 'Other'],
        n_samples,
        p=[0.15, 0.12, 0.11, 0.09, 0.08, 0.45]
    )

    # Create target with relationships to features
    risk_score = (
        -0.004 * df['credit_score'] +
        -0.00002 * df['income'] +
        0.01 * df['debt_to_income'] +
        0.01 * df['utilization_rate'] +
        -0.01 * df['employment_years'] +
        0.0001 * df['loan_amount'] +
        np.where(df['home_ownership'] == 'RENT', 0.3, 0) +
        np.where(df['employment_type'] == 'Part-time', 0.2, 0) +
        np.where(pd.notna(df['months_since_delinquent']), 0.5, 0) +
        np.random.randn(n_samples) * 0.5
    )

    # Create binary target
    threshold = np.percentile(risk_score, 87)  # ~13% default rate
    df['target'] = (risk_score > threshold).astype(int)

    print(f"\nCreated {n_samples} samples")
    print(f"Default rate: {df['target'].mean():.2%}")
    print(f"Features: {len(df.columns) - 3} ({len(df.select_dtypes(include=[np.number]).columns) - 3} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical)")

    return df


if __name__ == "__main__":
    results = main()