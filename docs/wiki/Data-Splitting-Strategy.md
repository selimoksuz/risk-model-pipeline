# Data Splitting Strategy

This page explains the data splitting methodology used in the Risk Model Pipeline.

## Table of Contents
- [Overview](#overview)
- [Splitting Philosophy](#splitting-philosophy)
- [Configuration Parameters](#configuration-parameters)
- [Implementation Details](#implementation-details)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Overview

The Risk Model Pipeline uses a sophisticated data splitting strategy that combines:
- **Time-based splitting** for Out-of-Time (OOT) validation
- **Stratified sampling** for Train/Test split within each month
- **Optional test set** configuration

This ensures:
1. No data leakage (future information never used in training)
2. Consistent target rates across train/test splits
3. Representative sampling from each time period

## Splitting Philosophy

### Three-Way Split
```
Timeline: [=========== Historical Data ===========][=== OOT ===]
           ↓                                         ↓
          [== Train ==][= Test =]                  [=== OOT ===]
           ↓            ↓                            ↓
        Stratified   Stratified                Time-based
```

### Key Principles

1. **OOT is Always Time-Based**
   - Last N months or last X% of data
   - Simulates real production scenario
   - Never randomly sampled

2. **Train/Test is Always Stratified**
   - Maintains target ratio within each month
   - Ensures temporal representation
   - Random sampling within strata

3. **Test Set is Optional**
   - Can disable test split if not needed
   - All pre-OOT data goes to training
   - Useful for maximum training data

## Configuration Parameters

```python
from risk_pipeline import Config

config = Config(
    # Test split configuration
    use_test_split=True,      # Whether to create test set
    test_ratio=0.20,          # 20% of pre-OOT data for test
    
    # OOT configuration
    oot_months=3,             # Use last 3 months as OOT
    # OR
    oot_ratio=0.20,          # Use last 20% as OOT
    min_oot_size=50,         # Minimum OOT samples
    
    # General settings
    time_col='app_dt',       # Date column for sorting
    target_col='target',     # Target column for stratification
    random_state=42          # For reproducibility
)
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_test_split` | bool | True | Create test split from pre-OOT data |
| `test_ratio` | float | 0.20 | Test size as ratio of pre-OOT data |
| `oot_months` | int/None | None | Number of months for OOT (overrides ratio) |
| `oot_ratio` | float | 0.20 | OOT size as ratio of total data |
| `min_oot_size` | int | 50 | Minimum samples in OOT |

## Implementation Details

### Step 1: OOT Split (Time-Based)

```python
# If oot_months is specified
if oot_months:
    latest_date = df[time_col].max()
    oot_start = latest_date - pd.DateOffset(months=oot_months)
    oot_data = df[df[time_col] >= oot_start]
    pre_oot_data = df[df[time_col] < oot_start]

# Otherwise use ratio
else:
    n_oot = int(len(df) * oot_ratio)
    n_oot = max(n_oot, min_oot_size)  # Ensure minimum
    oot_data = df.iloc[-n_oot:]  # Last n_oot rows
    pre_oot_data = df.iloc[:-n_oot]
```

### Step 2: Train/Test Split (Stratified by Month)

```python
# For each month in pre-OOT data
for month in pre_oot_data.groupby(pd.Grouper(freq='M')):
    # Stratified split maintaining target ratio
    train_month, test_month = train_test_split(
        month_data,
        test_size=test_ratio,
        stratify=month_data[target_col],
        random_state=random_state
    )
```

### Stratification Logic

The stratified splitting ensures:
1. **Monthly representation**: Each month appears in both train and test
2. **Target balance**: Target ratio preserved across splits
3. **Temporal coverage**: No time periods missing from train/test

## Examples

### Example 1: Standard Configuration
```python
config = Config(
    use_test_split=True,
    test_ratio=0.20,
    oot_months=3
)

# Result with 12 months of data:
# Months 1-9: Split 80/20 into Train/Test (stratified)
# Months 10-12: OOT (time-based)
```

### Example 2: No Test Split
```python
config = Config(
    use_test_split=False,  # No test set
    oot_ratio=0.15
)

# Result:
# First 85%: All goes to Train
# Last 15%: OOT
```

### Example 3: Fixed OOT Window
```python
config = Config(
    use_test_split=True,
    test_ratio=0.25,
    oot_months=2  # Always use last 2 months
)

# Result:
# All data except last 2 months: Split 75/25 Train/Test
# Last 2 months: OOT
```

## Best Practices

### 1. Choosing OOT Strategy

**Use `oot_months` when:**
- You have seasonal patterns
- Business requires specific time window
- Regulatory requirements specify months

**Use `oot_ratio` when:**
- Data doesn't have strong seasonality  
- You want consistent split ratios
- Dataset size varies

### 2. Test Split Decisions

**Enable test split (`use_test_split=True`) when:**
- You need hyperparameter tuning
- Model selection between alternatives
- Early stopping for neural networks

**Disable test split (`use_test_split=False`) when:**
- Dataset is small
- Using cross-validation instead
- Production uses different validation

### 3. Recommended Ratios

#### Credit Risk Models
```python
config = Config(
    use_test_split=True,
    test_ratio=0.15,    # 15% test
    oot_months=3        # 3 months OOT
)
```

#### Fraud Detection
```python
config = Config(
    use_test_split=True,
    test_ratio=0.20,    # 20% test
    oot_months=1        # 1 month OOT (rapid change)
)
```

#### Marketing Models
```python
config = Config(
    use_test_split=False,  # No test needed
    oot_ratio=0.30        # 30% OOT for seasonality
)
```

### 4. Validation Checks

Always verify split quality:

```python
# Check target distribution
train_target_rate = y_train.mean()
test_target_rate = y_test.mean() if y_test else None
oot_target_rate = y_oot.mean()

print(f"Target rates - Train: {train_target_rate:.2%}")
if test_target_rate:
    print(f"Target rates - Test: {test_target_rate:.2%}")
print(f"Target rates - OOT: {oot_target_rate:.2%}")

# Check temporal coverage
train_months = pd.to_datetime(X_train.index).to_period('M').nunique()
oot_months = pd.to_datetime(X_oot.index).to_period('M').nunique()

print(f"Months in Train: {train_months}")
print(f"Months in OOT: {oot_months}")
```

## Common Issues and Solutions

### Issue 1: Unbalanced Target in Small Months
**Problem**: Some months have too few positive cases for stratification
**Solution**: Pipeline automatically falls back to random splitting for those months

### Issue 2: OOT Too Small
**Problem**: Last N months have very few samples
**Solution**: Use `min_oot_size` parameter to ensure minimum samples

### Issue 3: Target Drift in OOT
**Problem**: OOT target rate very different from train
**Solution**: This is actually valuable information about model stability!

## Code Example

Complete example of custom splitting:

```python
from risk_pipeline import Config, DualPipeline
import pandas as pd

# Load data
df = pd.read_csv('credit_data.csv')

# Configure splitting
config = Config(
    # Columns
    target_col='default',
    time_col='application_date',
    
    # Split configuration  
    use_test_split=True,     # Want test set
    test_ratio=0.20,         # 20% for test
    oot_months=3,            # Last 3 months for OOT
    min_oot_size=100,        # At least 100 OOT samples
    
    # Other settings
    random_state=42
)

# Run pipeline
pipeline = DualPipeline(config)
pipeline.run(df)

# Check splits
print(f"Train samples: {len(pipeline.train_idx)}")
print(f"Test samples: {len(pipeline.test_idx) if pipeline.test_idx else 0}")
print(f"OOT samples: {len(pipeline.oot_idx)}")
```

## Related Topics
- [Feature Selection Parameters](Feature-Selection-Parameters)
- [Model Selection Criteria](Model-Selection-Criteria)
- [PSI Monitoring](PSI-Monitoring)