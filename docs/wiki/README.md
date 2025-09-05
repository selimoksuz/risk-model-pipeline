# Risk Model Pipeline Wiki

## 📚 Documentation Structure

This wiki provides comprehensive documentation for the Risk Model Pipeline, explaining all parameters, methodologies, and best practices.

## 📖 Quick Links

### Configuration Parameters
- **[Feature Selection Parameters](Feature-Selection-Parameters.md)**
  - PSI Threshold - Population stability monitoring
  - IV Minimum - Feature importance filtering
  - Correlation Threshold - Redundancy removal
  - VIF Threshold - Multicollinearity detection
  - Rare Threshold - Category grouping
  - Cluster Top K - Correlation cluster selection

- **[Model Selection Criteria](Model-Selection-Criteria.md)**
  - Model Selection Method - Strategy choice
  - Model Stability Weight - Performance vs stability balance
  - Max Train-OOT Gap - Overfitting prevention
  - Min Gini Threshold - Performance floor

- **[Imputation Strategies](Imputation-Strategies.md)**
  - Single methods (median, mean, mode)
  - Multiple imputation ensemble
  - Target-based imputation
  - Time series methods

### Methodologies
- **[WOE Methodology](WOE-Methodology.md)**
  - Weight of Evidence transformation
  - Binning strategies
  - Monotonicity constraints
  - Best practices

## 🎯 Parameter Quick Reference

### Feature Selection
```python
config = Config(
    psi_threshold=0.25,      # Stability check
    iv_min=0.02,            # Importance filter
    rho_threshold=0.90,     # Correlation limit
    vif_threshold=5.0,      # Multicollinearity
    rare_threshold=0.01,    # Rare categories
    cluster_top_k=2         # Features per cluster
)
```

### Model Selection
```python
config = Config(
    model_selection_method='balanced',  # Strategy
    model_stability_weight=0.3,        # 30% stability
    max_train_oot_gap=0.15,            # Max 15% gap
    min_gini_threshold=0.5             # Min performance
)
```

### Imputation
```python
config = Config(
    raw_imputation_strategy='multiple'  # Ensemble approach
)
```

## 🔍 How to Use This Wiki

1. **New Users**: Start with the [Home](Home.md) page for overview
2. **Configuration**: Check specific parameter pages for detailed explanations
3. **Best Practices**: Each page includes recommended settings for different use cases
4. **Examples**: Real-world configuration examples throughout

## 📊 Common Configurations

### Conservative (Stable Models)
```python
Config(
    psi_threshold=0.20,
    model_selection_method='conservative',
    max_train_oot_gap=0.10
)
```

### Aggressive (High Performance)
```python
Config(
    iv_min=0.05,
    model_selection_method='gini_oot',
    rho_threshold=0.95
)
```

### Balanced (Recommended)
```python
Config(
    psi_threshold=0.25,
    model_selection_method='balanced',
    model_stability_weight=0.3
)
```

## 📝 Contributing

To improve documentation:
1. Fork the repository
2. Edit/add markdown files in `docs/wiki/`
3. Submit a pull request

## 🔗 Additional Resources

- [Main Repository](https://github.com/selimoksuz/risk-model-pipeline)
- [Example Notebooks](../../notebooks/)
- [Issue Tracker](https://github.com/selimoksuz/risk-model-pipeline/issues)