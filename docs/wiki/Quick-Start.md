# Quick Start Guide

Get started with the Risk Model Pipeline in under 5 minutes!

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Running Your First Model](#running-your-first-model)
- [Understanding Results](#understanding-results)
- [Next Steps](#next-steps)

---

## Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git
```

### For Development
```bash
git clone https://github.com/selimoksuz/risk-model-pipeline.git
cd risk-model-pipeline
pip install -e .
```

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn
- lightgbm, xgboost, catboost
- optuna (for hyperparameter tuning)

## Basic Usage

### 1. Prepare Your Data

Your data should have:
- **Features**: X variables (numeric and categorical)
- **Target**: Binary target (0/1) 
- **Time splits**: Train, validation, and out-of-time (OOT) sets

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv("credit_data.csv")

# Create time-based splits
train_data = data[data['date'] < '2023-01-01']
valid_data = data[(data['date'] >= '2023-01-01') & (data['date'] < '2023-07-01')]
oot_data = data[data['date'] >= '2023-07-01']

# Separate features and target
X_train = train_data.drop(['target', 'date'], axis=1)
y_train = train_data['target']

X_valid = valid_data.drop(['target', 'date'], axis=1)
y_valid = valid_data['target']

X_oot = oot_data.drop(['target', 'date'], axis=1)
y_oot = oot_data['target']
```

### 2. Quick Configuration

```python
from risk_pipeline import Config

# Create configuration with default settings
config = Config(
    # Model settings
    model_type="lightgbm",  # or "xgboost", "catboost"
    
    # Feature selection
    iv_min=0.02,           # Minimum Information Value
    psi_threshold=0.25,    # Maximum PSI for stability
    
    # Model selection
    model_selection_method="balanced"  # or "gini_oot", "stable", "conservative"
)
```

### 3. Run the Pipeline

```python
from risk_pipeline import DualPipeline

# Initialize pipeline
pipeline = DualPipeline(config)

# Fit on training data
pipeline.fit(X_train, y_train, X_valid, y_valid, X_oot, y_oot)

# Get predictions
predictions_woe = pipeline.predict_woe(X_oot)
predictions_raw = pipeline.predict_raw(X_oot)

# Get best predictions (automatically selected)
best_predictions = pipeline.predict(X_oot)
```

## Configuration

### Minimal Configuration
```python
# Bare minimum - uses all defaults
config = Config()
```

### Typical Configuration
```python
config = Config(
    # Feature engineering
    iv_min=0.02,              # Drop features with IV < 0.02
    psi_threshold=0.25,       # Drop features with PSI > 0.25
    
    # Model training
    model_type="lightgbm",
    use_optuna=True,          # Hyperparameter optimization
    n_trials=100,             # Optuna trials
    
    # Model selection
    model_selection_method="balanced",
    model_stability_weight=0.3,  # 30% weight on stability
)
```

### Advanced Configuration
```python
config = Config(
    # Feature selection
    iv_min=0.02,
    psi_threshold=0.25,
    rho_threshold=0.90,       # Correlation threshold
    vif_threshold=5.0,        # Multicollinearity check
    
    # WOE settings
    n_bins=10,                # Number of bins
    min_bin_size=0.05,        # Minimum 5% in each bin
    woe_monotonic=True,       # Enforce monotonic WOE
    
    # Model settings
    model_type="ensemble",    # Use all models
    use_optuna=True,
    n_trials=200,
    
    # Selection criteria
    model_selection_method="stable",
    min_gini_threshold=0.5,   # Minimum acceptable Gini
    max_train_oot_gap=0.1,    # Max 10% Gini drop
    
    # Imputation
    imputation_strategy="multiple",  # Ensemble imputation
)
```

## Running Your First Model

### Complete Example
```python
import pandas as pd
from risk_pipeline import Config, DualPipeline

# 1. Load data
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv") 
oot_df = pd.read_csv("oot.csv")

# 2. Prepare features and target
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_valid = valid_df.drop('target', axis=1)
y_valid = valid_df['target']

X_oot = oot_df.drop('target', axis=1)
y_oot = oot_df['target']

# 3. Configure pipeline
config = Config(
    model_type="lightgbm",
    iv_min=0.02,
    psi_threshold=0.25,
    model_selection_method="balanced"
)

# 4. Run pipeline
pipeline = DualPipeline(config)
results = pipeline.fit(X_train, y_train, X_valid, y_valid, X_oot, y_oot)

# 5. View results
print("\n=== Model Performance ===")
print(results['model_comparison'])

print("\n=== Feature Importance ===")
print(results['feature_importance'].head(10))

# 6. Make predictions
predictions = pipeline.predict(X_oot)
probabilities = pipeline.predict_proba(X_oot)
```

### Using Pre-saved Models
```python
# Save fitted pipeline
import joblib
joblib.dump(pipeline, 'risk_model.pkl')

# Load and use
loaded_pipeline = joblib.load('risk_model.pkl')
predictions = loaded_pipeline.predict(new_data)
```

## Understanding Results

### Model Comparison Output
```
                    gini_train  gini_valid  gini_oot  train_oot_gap  selected
model                                                                        
lightgbm_woe           0.752      0.748      0.741        0.011         ✓
xgboost_woe            0.745      0.740      0.735        0.010         
lightgbm_raw           0.798      0.765      0.710        0.088         
xgboost_raw            0.791      0.758      0.705        0.086         
```

**Interpreting the columns:**
- **gini_train**: Performance on training data (higher is better)
- **gini_valid**: Performance on validation data
- **gini_oot**: Performance on out-of-time data (most important)
- **train_oot_gap**: Difference between train and OOT (lower is better)
- **selected**: ✓ indicates the chosen model

### Feature Importance
```
         feature    importance    iv    psi    correlation_cluster
0    income_woe        0.152   0.45   0.08                     1
1      age_woe         0.098   0.32   0.12                     2
2     debt_woe         0.087   0.28   0.09                     3
```

**Understanding the metrics:**
- **importance**: Model feature importance (0-1)
- **iv**: Information Value (>0.3 is strong)
- **psi**: Population Stability Index (<0.25 is stable)
- **correlation_cluster**: Features in same cluster are correlated

### Pipeline Summary
```python
# Get comprehensive summary
summary = pipeline.get_summary()

print(f"Selected Model: {summary['best_model']}")
print(f"OOT Gini: {summary['best_gini']:.3f}")
print(f"Number of Features: {summary['n_features']}")
print(f"Pipeline Type: {summary['pipeline_type']}")
```

## Next Steps

### 1. Explore Advanced Features

#### Custom Binning
```python
config = Config(
    binning_strategy="quantile",  # or "uniform", "kmeans"
    n_bins=10,
    min_bin_size=0.05
)
```

#### Ensemble Models
```python
config = Config(
    model_type="ensemble",
    ensemble_method="voting"  # or "stacking", "blending"
)
```

#### Custom Thresholds
```python
# Adjust for your use case
config = Config(
    # Strict feature selection
    iv_min=0.05,          # Only strong predictors
    psi_threshold=0.15,   # Very stable features
    
    # Conservative model selection
    model_selection_method="conservative",
    min_gini_threshold=0.6,  # High performance requirement
)
```

### 2. Monitor in Production

```python
from risk_pipeline.monitoring import ModelMonitor

# Set up monitoring
monitor = ModelMonitor(pipeline, config)

# Check monthly performance
monthly_data = pd.read_csv("2024_01_data.csv")
monitoring_report = monitor.evaluate(monthly_data)

print(monitoring_report['psi_alerts'])
print(monitoring_report['performance_metrics'])
```

### 3. Optimize Performance

```python
# Hyperparameter tuning
config = Config(
    use_optuna=True,
    n_trials=500,           # More trials for better optimization
    optuna_timeout=3600,    # 1 hour timeout
)

# Feature engineering
config = Config(
    create_interactions=True,    # Create feature interactions
    polynomial_features=True,    # Add polynomial features
    feature_scaling="robust"     # Scale features
)
```

### 4. Custom Preprocessing

```python
from risk_pipeline.core import DataProcessor

class CustomProcessor(DataProcessor):
    def custom_transform(self, X):
        # Add your custom logic
        X['custom_feature'] = X['feature1'] / X['feature2']
        return X

# Use custom processor
pipeline = DualPipeline(config, processor_class=CustomProcessor)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Gini Scores
```python
# Check data quality
print(f"Missing values: {X_train.isnull().sum().sum()}")
print(f"Target distribution: {y_train.value_counts(normalize=True)}")

# Try different settings
config = Config(
    iv_min=0.01,           # Keep more features
    model_type="xgboost",  # Try different algorithm
    use_optuna=True        # Optimize hyperparameters
)
```

#### 2. High Train-OOT Gap
```python
# Improve stability
config = Config(
    model_selection_method="stable",
    regularization_alpha=1.0,  # Increase regularization
    max_depth=5,               # Limit model complexity
)
```

#### 3. PSI Warnings
```python
# Check feature stability
from risk_pipeline.utils import check_psi

psi_report = check_psi(X_train, X_oot)
unstable_features = psi_report[psi_report['psi'] > 0.25]['feature'].tolist()

# Remove unstable features
X_train_stable = X_train.drop(columns=unstable_features)
```

## Best Practices Checklist

- [ ] **Data Quality**
  - Remove outliers
  - Handle missing values
  - Check for data leakage

- [ ] **Feature Engineering**
  - Create domain-specific features
  - Test different binning strategies
  - Monitor feature stability

- [ ] **Model Training**
  - Use time-based validation
  - Optimize hyperparameters
  - Check for overfitting

- [ ] **Model Selection**
  - Prioritize OOT performance
  - Consider stability metrics
  - Validate business logic

- [ ] **Production Deployment**
  - Save preprocessing steps
  - Monitor PSI regularly
  - Track model performance

## Getting Help

### Documentation
- [Feature Selection Parameters](Feature-Selection-Parameters)
- [Model Selection Criteria](Model-Selection-Criteria)
- [Best Practices](Best-Practices)

### Support
- GitHub Issues: [Report bugs or request features](https://github.com/selimoksuz/risk-model-pipeline/issues)
- Documentation: [Full documentation](https://github.com/selimoksuz/risk-model-pipeline/wiki)

### Examples
Check the `notebooks/` folder for complete examples:
- `01_dual_pipeline_example.ipynb` - Complete pipeline walkthrough
- `02_advanced_features.ipynb` - Advanced techniques
- `03_monitoring_example.ipynb` - Production monitoring