"""Test stratified splitting functionality."""

import numpy as np
import pandas as pd

from risk_pipeline.stages.split import time_based_split


def test_stratified_split_preserves_target_distribution():
    """Test that stratified split preserves target distribution."""
    # Create test data with known target distribution
    np.random.seed(42)
    n_samples = 1000

    # Create imbalanced target (30% positive, 70% negative)
    target = np.concatenate([
        np.ones(300),  # 30% positive
        np.zeros(700)  # 70% negative
    ])
    np.random.shuffle(target)

    df = pd.DataFrame({
        'app_id': range(n_samples),
        'app_dt': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'target': target,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })

    # Split with test set enabled
    train_idx, test_idx, oot_idx = time_based_split(
        df,
        time_col='app_dt',
        target_col='target',
        use_test_split=True,
        oot_window_months=1,
        test_size_row_frac=0.2
    )

    # Check that we have both train and test sets
    assert test_idx is not None, "Test set should be created when use_test_split = True"
    assert len(train_idx) > 0, "Train set should not be empty"
    assert len(test_idx) > 0, "Test set should not be empty"

    # Get target distributions
    train_targets = df.loc[train_idx, 'target']
    test_targets = df.loc[test_idx, 'target']

    # Calculate positive rates
    train_pos_rate = train_targets.mean()
    test_pos_rate = test_targets.mean()

    # Assert that target distributions are similar (within 5% tolerance)
    diff = abs(train_pos_rate - test_pos_rate)
    tolerance = 0.05

    print(f"Train positive rate: {train_pos_rate:.3f}")
    print(f"Test positive rate: {test_pos_rate:.3f}")
    print(f"Difference: {diff:.3f}")

    assert diff < tolerance, f"Target distribution difference ({diff:.3f}) exceeds tolerance ({tolerance})"


def test_fallback_when_stratification_fails():
    """Test fallback to simple split when stratification is not possible."""
    # Create data with only one class (stratification will fail)
    df = pd.DataFrame({
        'app_id': range(100),
        'app_dt': pd.date_range('2024-01-01', periods=100, freq='D'),
        'target': [1] * 100,  # All positive - stratification impossible
        'feature1': np.random.randn(100)
    })

    # Should not crash and should still create train/test split
    train_idx, test_idx, oot_idx = time_based_split(
        df,
        time_col='app_dt',
        target_col='target',
        use_test_split=True,
        oot_window_months=1,
        test_size_row_frac=0.2
    )

    assert test_idx is not None, "Should create test set even when stratification fails"
    assert len(train_idx) > 0, "Train set should not be empty"
    assert len(test_idx) > 0, "Test set should not be empty"


def test_no_test_split_when_disabled():
    """Test that no test split is created when use_test_split = False."""
    df = pd.DataFrame({
        'app_id': range(100),
        'app_dt': pd.date_range('2024-01-01', periods=100, freq='D'),
        'target': [0, 1] * 50,
        'feature1': np.random.randn(100)
    })

    train_idx, test_idx, oot_idx = time_based_split(
        df,
        time_col='app_dt',
        target_col='target',
        use_test_split=False,  # Disable test split
        oot_window_months=1
    )

    assert test_idx is None, "No test set should be created when use_test_split = False"
    assert len(train_idx) > 0, "Train set should not be empty"
    assert len(oot_idx) > 0, "OOT set should not be empty"


def test_monthly_event_rate_stability():
    """Ensure monthly stratified split keeps per-month event rates aligned."""
    rng = np.random.default_rng(2024)
    months = pd.date_range('2024-01-01', periods=90, freq='D')
    df = pd.DataFrame({
        'app_id': range(len(months)),
        'app_dt': months,
        'target': rng.choice([0, 1], size=len(months), p=[0.7, 0.3]),
        'feature1': rng.normal(size=len(months))
    })

    train_idx, test_idx, _ = time_based_split(
        df,
        time_col='app_dt',
        target_col='target',
        use_test_split=True,
        oot_window_months=0,
        test_size_row_frac=0.3
    )

    assert test_idx is not None and len(test_idx) > 0

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    train_df['month'] = train_df['app_dt'].dt.to_period('M')
    test_df['month'] = test_df['app_dt'].dt.to_period('M')

    common_months = sorted(set(train_df['month']).intersection(test_df['month']))
    for month in common_months:
        train_rate = train_df.loc[train_df['month'] == month, 'target'].mean()
        test_rate = test_df.loc[test_df['month'] == month, 'target'].mean()
        assert abs(train_rate - test_rate) < 0.1, f"Month {month}: rates diverged ({train_rate:.2f} vs {test_rate:.2f})"
