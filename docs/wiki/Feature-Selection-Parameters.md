# Feature Selection Parameters

This page explains all feature selection parameters used in the Risk Model Pipeline.

## Table of Contents
- [PSI Threshold](#psi-threshold)
- [IV Minimum](#iv-minimum)
- [Correlation Threshold](#correlation-threshold)
- [VIF Threshold](#vif-threshold)
- [Rare Threshold](#rare-threshold)
- [Cluster Top K](#cluster-top-k)

---

## PSI Threshold

**Parameter**: `psi_threshold`  
**Default**: `0.25`  
**Range**: `0.0 - 1.0`

### What is PSI?
Population Stability Index (PSI) measures the distribution shift of a feature between training and out-of-time (OOT) periods. It helps identify features that have become unstable over time.

### How it works
```python
config = Config(
    psi_threshold=0.25  # Features with PSI > 0.25 are dropped
)
```

### Interpretation
| PSI Range | Interpretation | Action |
|-----------|---------------|--------|
| PSI < 0.10 | No significant population change | Keep feature |
| 0.10 ≤ PSI < 0.25 | Small population change | Monitor feature |
| PSI ≥ 0.25 | Significant population change | Drop feature |

### Formula
```
PSI = Σ ((%OOT - %Train) × ln(%OOT / %Train))
```

### Best Practices
- **Credit Risk**: Use 0.25 (industry standard)
- **Fraud Detection**: Use 0.20 (more sensitive to shifts)
- **Marketing Models**: Use 0.30 (more tolerant)

### Example Scenarios

#### Scenario 1: Stable Feature
```
Training: [0-20]: 30%, [20-40]: 40%, [40+]: 30%
OOT:      [0-20]: 32%, [20-40]: 38%, [40+]: 30%
PSI = 0.05 → Keep feature ✓
```

#### Scenario 2: Unstable Feature
```
Training: [0-20]: 30%, [20-40]: 40%, [40+]: 30%
OOT:      [0-20]: 10%, [20-40]: 30%, [40+]: 60%
PSI = 0.35 → Drop feature ✗
```

---

## IV Minimum

**Parameter**: `iv_min`  
**Default**: `0.02`  
**Range**: `0.0 - 1.0`

### What is IV?
Information Value (IV) measures the predictive power of a feature. It quantifies the feature's ability to separate good and bad outcomes.

### How it works
```python
config = Config(
    iv_min=0.02  # Features with IV < 0.02 are dropped
)
```

### Interpretation
| IV Range | Predictive Power | Recommendation |
|----------|-----------------|----------------|
| IV < 0.02 | Not useful | Drop |
| 0.02 ≤ IV < 0.1 | Weak | Consider keeping |
| 0.1 ≤ IV < 0.3 | Medium | Keep |
| 0.3 ≤ IV < 0.5 | Strong | Keep |
| IV ≥ 0.5 | Suspicious | Check for overfitting |

### Formula
```
IV = Σ ((%Good - %Bad) × WOE)
WOE = ln(%Good / %Bad)
```

### Best Practices
- **High-stakes models**: Use `iv_min=0.05` (only meaningful features)
- **Exploratory models**: Use `iv_min=0.01` (keep more features)
- **Parsimonious models**: Use `iv_min=0.10` (only strong predictors)

### Example
```python
# Feature: Income
# Bins: Low, Medium, High
# IV calculation:
# Low:    Good=10%, Bad=40%, WOE=-1.39, IV_contrib=0.42
# Medium: Good=50%, Bad=40%, WOE=0.22,  IV_contrib=0.02  
# High:   Good=40%, Bad=20%, WOE=0.69,  IV_contrib=0.14
# Total IV = 0.58 → Strong predictor ✓
```

---

## Correlation Threshold

**Parameter**: `rho_threshold`  
**Default**: `0.90`  
**Range**: `0.0 - 1.0`

### What is it?
Maximum allowed Pearson correlation between features. Helps remove redundant features that provide similar information.

### How it works
```python
config = Config(
    rho_threshold=0.90  # Features with correlation > 0.90 are clustered
)
```

### Process
1. Calculate correlation matrix
2. Find feature pairs with |correlation| > threshold
3. Create correlation clusters
4. Keep only top features from each cluster (based on IV)

### Best Practices
- **Strict**: `0.80` - Remove most redundancy
- **Balanced**: `0.90` - Good trade-off
- **Relaxed**: `0.95` - Keep more features

### Example
```python
# Correlation matrix:
#                  debt_ratio  utilization_rate  credit_usage
# debt_ratio           1.00         0.92            0.88
# utilization_rate     0.92         1.00            0.91
# credit_usage         0.88         0.91            1.00

# With rho_threshold=0.90:
# - Cluster 1: {debt_ratio, utilization_rate, credit_usage}
# - Action: Keep only the feature with highest IV
```

---

## VIF Threshold

**Parameter**: `vif_threshold`  
**Default**: `5.0`  
**Range**: `1.0 - ∞`

### What is VIF?
Variance Inflation Factor measures multicollinearity. High VIF indicates a feature can be predicted from other features.

### How it works
```python
config = Config(
    vif_threshold=5.0  # Features with VIF > 5 may be dropped
)
```

### Interpretation
| VIF | Multicollinearity | Action |
|-----|------------------|--------|
| VIF < 5 | Low | Keep |
| 5 ≤ VIF < 10 | Moderate | Monitor |
| VIF ≥ 10 | High | Consider dropping |

### Formula
```
VIF_i = 1 / (1 - R²_i)
```
Where R²_i is the R-squared from regressing feature i on all other features.

### Example
```python
# Feature: debt_ratio
# R² when predicted by other features: 0.80
# VIF = 1/(1-0.80) = 5.0
# Action: At threshold, monitor carefully
```

---

## Rare Threshold

**Parameter**: `rare_threshold`  
**Default**: `0.01`  
**Range**: `0.0 - 0.1`

### What is it?
Minimum frequency for categorical values. Values below this threshold are grouped as "RARE".

### How it works
```python
config = Config(
    rare_threshold=0.01  # Categories with < 1% frequency become "RARE"
)
```

### Purpose
- Prevents overfitting on rare categories
- Ensures statistical significance
- Improves model stability

### Example
```python
# Feature: Region
# Distribution:
#   North: 40%
#   South: 35%
#   East: 20%
#   West: 4.5%
#   Central: 0.3%  → Grouped as "RARE"
#   Island: 0.2%   → Grouped as "RARE"

# After grouping:
#   North: 40%
#   South: 35%
#   East: 20%
#   West: 4.5%
#   RARE: 0.5%
```

---

## Cluster Top K

**Parameter**: `cluster_top_k`  
**Default**: `2`  
**Range**: `1 - n_features`

### What is it?
Number of features to keep from each correlation cluster.

### How it works
```python
config = Config(
    cluster_top_k=2  # Keep top 2 features from each correlation cluster
)
```

### Selection Criteria
Features are ranked within each cluster by:
1. Information Value (IV)
2. PSI stability
3. Missing rate

### Example
```python
# Cluster: Credit utilization metrics
# Members: [utilization_rate (IV=0.25), 
#           debt_ratio (IV=0.22), 
#           credit_usage (IV=0.20),
#           balance_ratio (IV=0.18)]
# 
# With cluster_top_k=2:
# Keep: utilization_rate, debt_ratio
# Drop: credit_usage, balance_ratio
```

## Feature Selection Pipeline Flow

```mermaid
graph LR
    A[All Features] --> B[PSI Filter]
    B --> C[IV Filter]
    C --> D[Correlation Clustering]
    D --> E[VIF Check]
    E --> F[Boruta Selection]
    F --> G[Forward Selection]
    G --> H[Noise Sentinel]
    H --> I[Final Features]
```

## Configuration Examples

### Conservative (Keep more features)
```python
config = Config(
    psi_threshold=0.30,      # More tolerant to shifts
    iv_min=0.01,            # Keep weak predictors
    rho_threshold=0.95,     # Allow high correlation
    vif_threshold=10.0,     # Tolerant to multicollinearity
    rare_threshold=0.005    # Keep rarer categories
)
```

### Aggressive (Keep fewer features)
```python
config = Config(
    psi_threshold=0.15,      # Strict stability requirement
    iv_min=0.05,            # Only meaningful predictors
    rho_threshold=0.80,     # Remove most redundancy
    vif_threshold=3.0,      # Strict multicollinearity check
    rare_threshold=0.02     # Group more as rare
)
```

### Balanced (Recommended)
```python
config = Config(
    psi_threshold=0.25,
    iv_min=0.02,
    rho_threshold=0.90,
    vif_threshold=5.0,
    rare_threshold=0.01
)
```