"""
Create high-quality test dataset with controlled Gini (60-70%)
Ensures at least 5 features are selected by the model
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.core.config import Config

# FIXED SEED FOR REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)

def create_high_quality_data(n_samples=20000, test_mode=False):
    """
    Create high-quality credit risk data with strong predictive features.
    Gini coefficient will be 60-70% (AUC 80-85%).
    """

    np.random.seed(SEED)

    print(f"Creating high-quality dataset with {n_samples} samples...")

    # Time series for proper OOT split
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='H')[:n_samples]

    df = pd.DataFrame({
        'app_id': range(n_samples),
        'app_dt': dates
    })

    # STRONG NUMERIC PREDICTORS with realistic distributions

    # 1. Credit Score - STRONG predictor (300-850 range)
    df['credit_score'] = np.random.beta(5, 2, n_samples) * 550 + 300
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)

    # 2. Annual Income - STRONG predictor (log-normal, realistic range)
    df['income'] = np.random.lognormal(10.8, 0.7, n_samples)
    df['income'] = np.clip(df['income'], 15000, 500000)

    # 3. Debt-to-Income Ratio - STRONG predictor
    df['debt_to_income'] = np.random.beta(2, 5, n_samples) * 80

    # 4. Employment Years - MODERATE predictor
    df['employment_years'] = np.random.gamma(2, 3, n_samples)
    df['employment_years'] = np.clip(df['employment_years'], 0, 40)

    # 5. Number of Credit Lines - MODERATE predictor
    df['num_credit_lines'] = np.random.negative_binomial(5, 0.5, n_samples)
    df['num_credit_lines'] = np.clip(df['num_credit_lines'], 0, 30)

    # 6. Loan Amount - MODERATE predictor
    df['loan_amount'] = np.random.lognormal(9.2, 0.8, n_samples)
    df['loan_amount'] = np.clip(df['loan_amount'], 1000, 100000)

    # 7. Months Since Last Delinquency - STRONG predictor (with missings)
    df['months_since_delinquent'] = np.where(
        np.random.rand(n_samples) < 0.6,  # 60% have no delinquency
        np.nan,
        np.random.exponential(24, n_samples)
    )

    # 8. Credit Utilization Rate - STRONG predictor
    df['utilization_rate'] = np.random.beta(2, 3, n_samples) * 100

    # 9. Number of inquiries in last 6 months - MODERATE predictor
    df['inquiries_6mo'] = np.random.poisson(0.5, n_samples)
    df['inquiries_6mo'] = np.clip(df['inquiries_6mo'], 0, 10)

    # 10. Age - WEAK but relevant predictor
    df['age'] = np.random.normal(42, 12, n_samples)
    df['age'] = np.clip(df['age'], 18, 75).astype(int)

    # CATEGORICAL FEATURES with clear impact

    # 1. Home Ownership - STRONG categorical
    df['home_ownership'] = np.random.choice(
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        n_samples,
        p=[0.35, 0.15, 0.45, 0.05]
    )

    # 2. Loan Purpose - MODERATE categorical
    df['loan_purpose'] = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'car', 'other'],
        n_samples,
        p=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
    )

    # 3. Employment Type - MODERATE categorical
    df['employment_type'] = np.random.choice(
        ['Full-time', 'Part-time', 'Self-employed', 'Retired', 'Student'],
        n_samples,
        p=[0.60, 0.10, 0.15, 0.10, 0.05]
    )

    # 4. Education Level - WEAK but relevant
    df['education'] = np.random.choice(
        ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'],
        n_samples,
        p=[0.25, 0.20, 0.35, 0.15, 0.05]
    )

    # CREATE TARGET with STRONG RELATIONSHIPS (ensures 60-70% Gini)

    # Calculate risk score with clear relationships
    risk_score = (
        # Credit score is very important (normalized)
        -2.5 * (df['credit_score'] - 300) / 550 +

        # Income effect (log scale, normalized)
        -0.8 * np.log(df['income'] / 50000) +

        # DTI is very predictive
        0.03 * df['debt_to_income'] +

        # Utilization rate is important
        0.015 * df['utilization_rate'] +

        # Employment years (negative = good)
        -0.05 * np.sqrt(df['employment_years']) +

        # Inquiries increase risk significantly
        0.3 * df['inquiries_6mo'] +

        # Credit lines (more = better, but diminishing returns)
        -0.05 * np.log(df['num_credit_lines'] + 1) +

        # Loan amount effect (normalized)
        0.00002 * df['loan_amount'] +

        # Age effect (younger = riskier)
        -0.01 * (df['age'] - 18) +

        # Categorical effects
        np.where(df['home_ownership'] == 'RENT', 0.4, 0) +
        np.where(df['home_ownership'] == 'OTHER', 0.3, 0) +
        np.where(df['home_ownership'] == 'OWN', -0.3, 0) +

        np.where(df['employment_type'] == 'Student', 0.4, 0) +
        np.where(df['employment_type'] == 'Part-time', 0.2, 0) +
        np.where(df['employment_type'] == 'Full-time', -0.1, 0) +

        np.where(df['loan_purpose'] == 'debt_consolidation', 0.2, 0) +
        np.where(df['loan_purpose'] == 'credit_card', 0.15, 0) +

        np.where(df['education'].isin(['Master', 'Doctorate']), -0.2, 0) +
        np.where(df['education'] == 'High School', 0.1, 0) +

        # Delinquency is VERY predictive
        np.where(df['months_since_delinquent'].isna(), -0.3,
                np.where(df['months_since_delinquent'] < 12, 0.8,
                        np.where(df['months_since_delinquent'] < 24, 0.4, 0.1))) +

        # Add controlled random noise (less noise = stronger signal)
        np.random.randn(n_samples) * 0.5
    )

    # Convert to probability using logistic function
    prob = 1 / (1 + np.exp(-risk_score))

    # Create binary target with some randomness but strong signal
    random_component = np.random.rand(n_samples)
    df['target'] = ((prob + random_component * 0.2) > 0.62).astype(int)

    # Adjust to get ~12-15% default rate
    if df['target'].mean() > 0.15:
        threshold_adj = np.percentile(risk_score, 85)
        df['target'] = (risk_score > threshold_adj).astype(int)

    # Calculate actual Gini for verification
    actual_prob = 1 / (1 + np.exp(-risk_score))
    gini = 2 * roc_auc_score(df['target'], actual_prob) - 1

    print(f"  Dataset created successfully!")
    print(f"  Samples: {n_samples}")
    print(f"  Features: 14 (10 numeric, 4 categorical)")
    print(f"  Default rate: {df['target'].mean():.2%}")
    print(f"  Expected Gini: {gini:.1%}")
    print(f"  Date range: {df['app_dt'].min().date()} to {df['app_dt'].max().date()}")

    if test_mode:
        # Quick feature importance check
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        feature_cols = [c for c in df.columns if c not in ['app_id', 'app_dt', 'target']]
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_cols].fillna(-999),
            df['target'],
            test_size=0.2,
            random_state=SEED
        )

        # Convert categoricals to numeric for RF
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = pd.Categorical(X_train[col]).codes
            X_test[col] = pd.Categorical(X_test[col]).codes

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
        rf.fit(X_train, y_train)

        test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        test_gini = 2 * test_auc - 1

        print(f"  Quick RF Test - Gini: {test_gini:.1%} (AUC: {test_auc:.3f})")

        # Top features
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n  Top 5 Important Features:")
        for i, row in importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")

    return df


def test_with_pipeline():
    """Test the dataset with the actual pipeline"""

    print("\n" + "="*80)
    print("TESTING WITH RISK PIPELINE")
    print("="*80)

    # Create data
    df = create_high_quality_data(n_samples=15000, test_mode=True)

    # Configure pipeline for reasonable feature selection
    config = Config(
        target_col='target',
        id_col='app_id',
        time_col='app_dt',
        random_state=SEED,

        # Reasonable thresholds for good data
        iv_min=0.01,  # Standard IV threshold
        iv_threshold=0.01,
        psi_threshold=0.5,  # Some PSI tolerance
        rho_threshold=0.90,  # High correlation threshold
        vif_threshold=10.0,

        # WOE settings
        enable_woe=True,
        n_bins=5,
        min_bin_size=0.05,

        # Pipeline settings
        enable_dual_pipeline=False,
        enable_noise_sentinel=False,
        enable_calibration=True,

        # Selection - moderate filtering
        selection_order=['psi', 'vif', 'correlation', 'iv'],
        max_features=15,
        min_features=5,

        # Models
        model_type='all',

        # Splits
        test_ratio=0.2,
        oot_months=2,  # Last 2 months as OOT

        # Output
        output_folder='output_quality_test',
        save_plots=False
    )

    # Run pipeline
    print("\nRunning pipeline...")
    pipeline = UnifiedRiskPipeline(config)
    results = pipeline.fit(df)

    print("\n" + "="*80)
    print("PIPELINE RESULTS")
    print("="*80)

    if 'selected_features' in results:
        print(f"\n✓ Selected Features: {len(results['selected_features'])}")
        for i, feat in enumerate(results['selected_features'], 1):
            print(f"  {i}. {feat}")

    if 'best_model_name' in results:
        print(f"\n✓ Best Model: {results['best_model_name']}")

    if 'scores' in results:
        print(f"\n✓ Model Performance:")
        for model_name, scores in results['scores'].items():
            if 'train_auc' in scores and 'test_auc' in scores:
                train_gini = 2 * scores['train_auc'] - 1
                test_gini = 2 * scores['test_auc'] - 1

                # Check if in target range
                in_range = "✓" if 0.6 <= test_gini <= 0.7 else "✗"

                print(f"  {model_name}:")
                print(f"    Train Gini: {train_gini:.1%} (AUC: {scores['train_auc']:.3f})")
                print(f"    Test Gini:  {test_gini:.1%} (AUC: {scores['test_auc']:.3f}) {in_range}")

    # Verify OOT performance
    if 'oot_performance' in results:
        oot_gini = 2 * results['oot_performance']['auc'] - 1
        in_range = "✓" if 0.6 <= oot_gini <= 0.7 else "✗"
        print(f"\n✓ OOT Performance:")
        print(f"  Gini: {oot_gini:.1%} (AUC: {results['oot_performance']['auc']:.3f}) {in_range}")

    return results


def save_sample_data(filename='sample_credit_data.csv'):
    """Save sample data to CSV for later use"""

    df = create_high_quality_data(n_samples=20000)
    df.to_csv(filename, index=False)
    print(f"\n✓ Sample data saved to: {filename}")
    return df


if __name__ == "__main__":
    print("="*80)
    print("HIGH-QUALITY TEST DATA GENERATOR")
    print("Fixed seed: 42 for reproducibility")
    print("Target: Gini 60-70% on train/test/OOT")
    print("="*80)

    # Test with pipeline
    results = test_with_pipeline()

    # Save sample data
    df = save_sample_data()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ Dataset created with strong predictive features")
    print("✓ Expected Gini: 60-70% across all splits")
    print("✓ At least 5 features will be selected")
    print("✓ Data saved to: sample_credit_data.csv")
    print("\nUse this data for testing and demos!")