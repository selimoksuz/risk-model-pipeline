# Weight of Evidence (WOE) Methodology

## Overview

Weight of Evidence (WOE) is a powerful transformation technique used in credit risk modeling to convert categorical and continuous variables into a standardized scale that represents the strength of relationship with the target variable.

## Formula

```
WOE = ln(% of Good / % of Bad)
```

Where:
- **Good**: Non-default cases (target = 0)
- **Bad**: Default cases (target = 1)

## Why Use WOE?

1. **Handles Non-linear Relationships**: Captures complex patterns
2. **Standardized Scale**: All features on same scale (-∞ to +∞)
3. **Interpretability**: Positive WOE = lower risk, Negative WOE = higher risk
4. **Missing Values**: Natural handling through separate bin
5. **Outliers**: Robust through binning

## WOE Calculation Process

### Step 1: Binning
```python
# Numeric features: Create bins
Bins: [0-20], [20-40], [40-60], [60+], [Missing]

# Categorical features: Group rare categories
Groups: ['A'], ['B'], ['C'], ['RARE'], ['Missing']
```

### Step 2: Calculate Distribution
```python
# For each bin/group:
Good_count = count where target = 0
Bad_count = count where target = 1
Good_pct = Good_count / Total_good
Bad_pct = Bad_count / Total_bad
```

### Step 3: Apply WOE Formula
```python
# With smoothing (Laplace/Jeffreys)
WOE = ln((Good_pct + 0.5) / (Bad_pct + 0.5))
```

## Example Calculation

### Income Feature
```
Bin         | Good | Bad | Good% | Bad% | WOE
------------|------|-----|-------|------|-------
[0-30k]     | 100  | 80  | 10%   | 40%  | -1.39
[30-50k]    | 300  | 60  | 30%   | 30%  | 0.00
[50-100k]   | 400  | 40  | 40%   | 20%  | 0.69
[100k+]     | 200  | 20  | 20%   | 10%  | 0.69
Missing     | 0    | 0   | 0%    | 0%   | 0.00
```

## WOE Interpretation

| WOE Range | Interpretation | Risk Level |
|-----------|---------------|------------|
| WOE < -0.5 | Strong bad indicator | High Risk |
| -0.5 ≤ WOE < -0.1 | Weak bad indicator | Medium-High Risk |
| -0.1 ≤ WOE ≤ 0.1 | Neutral | Medium Risk |
| 0.1 < WOE ≤ 0.5 | Weak good indicator | Medium-Low Risk |
| WOE > 0.5 | Strong good indicator | Low Risk |

## Binning Strategies

### 1. Equal Width Binning
```python
# Divide range into equal intervals
bins = np.linspace(min_value, max_value, n_bins)
```

### 2. Equal Frequency (Quantile) Binning
```python
# Equal number of observations per bin
bins = pd.qcut(data, q=n_bins)
```

### 3. Optimal Binning (Used in Pipeline)
```python
# Maximize IV while maintaining monotonicity
- Start with fine bins
- Merge adjacent bins to maximize IV
- Ensure monotonic WOE trend
```

## Monotonicity Constraint

The pipeline enforces monotonic WOE trends:

```
Age: [18-25]: -0.5 → [25-35]: -0.2 → [35-50]: 0.1 → [50+]: 0.4
```

This ensures logical business interpretation.

## Advantages and Limitations

### Advantages ✅
- Handles missing values naturally
- Robust to outliers
- Creates linear relationship with log-odds
- Interpretable transformation
- Works well with linear models

### Limitations ❌
- Information loss through binning
- Requires sufficient samples per bin
- May overfit with too many bins
- Less effective with tree-based models

## Configuration in Pipeline

```python
config = Config(
    # Binning parameters
    min_bin_size=0.05,      # Minimum 5% of samples per bin
    max_bins=10,            # Maximum number of bins
    
    # WOE parameters
    woe_monotonic=True,     # Enforce monotonicity
    woe_smoothing=0.5,      # Laplace smoothing parameter
)
```

## Best Practices

1. **Minimum 30 observations per bin** for statistical significance
2. **Maximum 10-20 bins** to prevent overfitting
3. **Check IV after WOE** to validate transformation
4. **Monitor PSI** for WOE stability over time
5. **Visualize WOE trends** for business validation