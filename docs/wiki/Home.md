# Risk Model Pipeline Wiki

Welcome to the Risk Model Pipeline documentation! This wiki provides comprehensive information about all configuration parameters, feature selection methods, and model selection strategies.

## 📚 Documentation Structure

### Configuration Parameters
- [Feature Selection Parameters](Feature-Selection-Parameters.md) - PSI, IV, correlation thresholds
- [Model Selection Criteria](Model-Selection-Criteria.md) - Performance vs stability trade-offs
- [Imputation Strategies](Imputation-Strategies.md) - Data preprocessing without loss
- [Pipeline Settings](Pipeline-Settings.md) - Dual pipeline and general settings

### Concepts & Methodology
- [Weight of Evidence (WOE)](WOE-Methodology.md) - Binning and transformation
- [Information Value (IV)](Information-Value.md) - Feature importance measurement
- [Population Stability Index (PSI)](PSI-Monitoring.md) - Distribution drift detection
- [Model Stability](Model-Stability.md) - Train-OOT gap analysis

### Tutorials
- [Quick Start Guide](Quick-Start.md)
- [Advanced Configuration](Advanced-Configuration.md)
- [Best Practices](Best-Practices.md)

## 🚀 Quick Navigation

| Topic | Description |
|-------|-------------|
| [PSI Threshold](#) | How to set population stability thresholds |
| [IV Minimum](#) | Understanding information value cutoffs |
| [Model Selection](#) | Choosing between performance and stability |
| [Dual Pipeline](#) | When to use WOE vs RAW pipelines |

## 📊 Key Concepts

### Feature Selection Pipeline
```
Data → PSI Filter → IV Filter → Correlation Filter → Boruta → Forward Selection → Final Features
```

### Model Selection Strategies
- **Traditional**: Highest OOT performance
- **Stable**: Minimum Train-OOT gap
- **Balanced**: Weighted combination
- **Conservative**: Stability with constraints

## 🔗 Links
- [GitHub Repository](https://github.com/selimoksuz/risk-model-pipeline)
- [Issue Tracker](https://github.com/selimoksuz/risk-model-pipeline/issues)
- [Examples](https://github.com/selimoksuz/risk-model-pipeline/tree/main/examples)