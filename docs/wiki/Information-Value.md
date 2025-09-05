# Information Value (IV)

Information Value is one of the most important metrics in credit risk modeling for measuring the predictive power of features.

## Table of Contents
- [Understanding Information Value](#understanding-information-value)
- [Mathematical Foundation](#mathematical-foundation)
- [Calculation Process](#calculation-process)
- [Interpretation Guidelines](#interpretation-guidelines)
- [IV in Feature Selection](#iv-in-feature-selection)
- [Implementation Examples](#implementation-examples)
- [Common Pitfalls](#common-pitfalls)

---

## Understanding Information Value

Information Value (IV) quantifies the relationship between a feature and the target variable (good/bad outcomes). It measures how well a feature can separate good and bad populations.

### Key Properties
- **Range**: 0 to infinity (practically 0 to 2)
- **Higher = Better**: Higher IV indicates stronger predictive power
- **Additive**: Total IV = Sum of bin IVs
- **Symmetric**: Works for both binary and binned features

## Mathematical Foundation

### Basic Formula
```
IV = Σ [(Good% - Bad%) × WOE]
```

Where:
- **Good%**: Percentage of good outcomes in the bin
- **Bad%**: Percentage of bad outcomes in the bin
- **WOE**: Weight of Evidence = ln(Good% / Bad%)

### Detailed Calculation
For each bin i:
```
IV_i = (Good_i/Total_Good - Bad_i/Total_Bad) × ln(Good_i/Total_Good ÷ Bad_i/Total_Bad)
```

Total IV:
```
IV = Σ IV_i for all bins
```

## Calculation Process

### Step 1: Binning
Features must be binned before IV calculation:
- **Numeric features**: Equal frequency, equal width, or optimal binning
- **Categorical features**: Each category is a bin

### Step 2: Calculate Distributions
For each bin, calculate:
```python
good_rate = count_good_in_bin / total_good
bad_rate = count_bad_in_bin / total_bad
```

### Step 3: Calculate WOE
```python
woe = np.log(good_rate / bad_rate)
```

### Step 4: Calculate IV Contribution
```python
iv_contribution = (good_rate - bad_rate) * woe
```

### Step 5: Sum IV Contributions
```python
total_iv = sum(iv_contributions)
```

## Interpretation Guidelines

### IV Ranges and Predictive Power

| IV Range | Predictive Power | Usage Recommendation | Typical Action |
|----------|------------------|---------------------|----------------|
| IV < 0.02 | Not Predictive | Not useful | Drop feature |
| 0.02 ≤ IV < 0.10 | Weak | Marginal predictor | Use with caution |
| 0.10 ≤ IV < 0.30 | Medium | Good predictor | Include in model |
| 0.30 ≤ IV < 0.50 | Strong | Excellent predictor | Definitely include |
| IV ≥ 0.50 | Too Good | Suspicious | Check for leakage |

### Industry Standards
- **Credit Risk**: Typically use IV > 0.02
- **Fraud Detection**: Often require IV > 0.05
- **Marketing**: May accept IV > 0.01

## IV in Feature Selection

### Pipeline Integration
```python
from risk_pipeline import Config

config = Config(
    iv_min=0.02,  # Minimum IV threshold
    iv_top_k=50   # Keep top 50 features by IV
)
```

### Selection Strategy
1. **Calculate IV** for all features
2. **Filter** features below iv_min
3. **Rank** remaining features by IV
4. **Select** top K features
5. **Check** for multicollinearity

### Multi-stage Selection
```python
# Stage 1: Loose filter
initial_filter = features[features.iv > 0.01]

# Stage 2: Correlation check
uncorrelated = remove_correlated(initial_filter)

# Stage 3: Strict filter
final_features = uncorrelated[uncorrelated.iv > 0.05]
```

## Implementation Examples

### Example 1: Simple Binary Feature
```python
# Feature: has_default (Yes/No)
# Good outcomes: 800 (No default: 700, Yes default: 100)
# Bad outcomes: 200 (No default: 50, Yes default: 150)

# No default bin:
good_rate = 700/800 = 0.875
bad_rate = 50/200 = 0.250
woe = ln(0.875/0.250) = 1.253
iv_contrib = (0.875 - 0.250) * 1.253 = 0.783

# Yes default bin:
good_rate = 100/800 = 0.125
bad_rate = 150/200 = 0.750
woe = ln(0.125/0.750) = -1.792
iv_contrib = (0.125 - 0.750) * (-1.792) = 1.120

# Total IV = 0.783 + 1.120 = 1.903 (Very strong predictor!)
```

### Example 2: Income Feature (Binned)
```python
# Income bins: Low, Medium, High
# Distribution:
#   Low:    Good=10%, Bad=40%
#   Medium: Good=50%, Bad=40%
#   High:   Good=40%, Bad=20%

iv_calculations = []

# Low income:
woe_low = np.log(0.10/0.40) = -1.386
iv_low = (0.10 - 0.40) * (-1.386) = 0.416

# Medium income:
woe_med = np.log(0.50/0.40) = 0.223
iv_med = (0.50 - 0.40) * 0.223 = 0.022

# High income:
woe_high = np.log(0.40/0.20) = 0.693
iv_high = (0.40 - 0.20) * 0.693 = 0.139

# Total IV = 0.416 + 0.022 + 0.139 = 0.577
# Interpretation: Strong predictor (0.30 < IV < 0.50)
```

### Example 3: Pipeline Usage
```python
from risk_pipeline.core.feature_engineer import FeatureEngineer

# Initialize with configuration
fe = FeatureEngineer(config)

# Calculate IV for all features
iv_scores = fe.calculate_information_values(X_train, y_train)

# Filter by IV
selected_features = iv_scores[iv_scores['IV'] > config.iv_min]['Feature'].tolist()

print(f"Selected {len(selected_features)} features with IV > {config.iv_min}")
```

## Common Pitfalls

### 1. Target Leakage
**Problem**: Extremely high IV (> 0.5)
```python
# Suspicious feature with IV = 1.2
# Often indicates the feature contains future information
```
**Solution**: Review feature definition and data collection timing

### 2. Overfitting on Small Samples
**Problem**: High IV on small bins
```python
# Bin with 5 goods, 1 bad → WOE = ln(5/1) = 1.609
# Small sample leads to unstable IV
```
**Solution**: Set minimum bin size (e.g., 5% of population)

### 3. Missing Value Treatment
**Problem**: Incorrect handling of missing values
```python
# Don't drop missing values - they can be informative!
# Treat missing as separate category
```
**Solution**: Create separate bin for missing values

### 4. Inconsistent Binning
**Problem**: Different binning between train and test
```python
# Train: [0-100], [100-200], [200+]
# Test:  [0-150], [150-300], [300+]  # Different bins!
```
**Solution**: Save binning rules from training and apply to test

### 5. Ignoring Business Logic
**Problem**: Purely statistical selection
```python
# Feature with IV = 0.015 (below threshold)
# But critical for regulatory compliance
```
**Solution**: Combine IV with business rules

## Advanced Topics

### IV Stability Over Time
Monitor IV changes across different time periods:
```python
# Calculate IV for different periods
iv_2023_q1 = calculate_iv(data_2023_q1)
iv_2023_q2 = calculate_iv(data_2023_q2)
iv_change = abs(iv_2023_q2 - iv_2023_q1)

# Flag unstable features
if iv_change > 0.1:
    print(f"Warning: IV changed by {iv_change:.2f}")
```

### Multivariate Information Value (MIV)
Consider interactions between features:
```python
# Joint IV for feature combination
joint_iv = calculate_iv(feature1 * feature2)
individual_sum = iv_feature1 + iv_feature2
synergy = joint_iv - individual_sum
```

### IV for Continuous Targets
Adapt IV for regression problems:
```python
# Bin the continuous target
y_binned = pd.qcut(y_continuous, q=2, labels=['Low', 'High'])
# Calculate IV as usual
iv = calculate_iv(X, y_binned)
```

## Best Practices

1. **Always validate IV on out-of-time data**
   ```python
   iv_train = calculate_iv(X_train, y_train)
   iv_oot = calculate_iv(X_oot, y_oot)
   stability = 1 - abs(iv_train - iv_oot) / iv_train
   ```

2. **Combine IV with other metrics**
   ```python
   feature_score = 0.6 * iv_score + 0.3 * gini_score + 0.1 * stability_score
   ```

3. **Document IV thresholds**
   ```python
   # config.yaml
   feature_selection:
     iv_min: 0.02  # Justified by historical model performance
     iv_max: 0.5   # Prevent target leakage
   ```

4. **Monitor IV in production**
   ```python
   # Monthly IV monitoring
   current_iv = calculate_iv(current_month_data)
   if abs(current_iv - baseline_iv) > 0.1:
       alert("Significant IV shift detected")
   ```

## Related Topics
- [WOE Methodology](WOE-Methodology) - Understanding Weight of Evidence
- [Feature Selection Parameters](Feature-Selection-Parameters) - IV thresholds
- [PSI Monitoring](PSI-Monitoring) - Feature stability over time