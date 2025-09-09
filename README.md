# Risk Model Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/selimoksuz/risk-model-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/selimoksuz/risk-model-pipeline/actions/workflows/ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20quality-99.5%25-brightgreen)](https://github.com/selimoksuz/risk-model-pipeline)

Production-ready risk modeling pipeline with WOE transformation and advanced ML features.

## Features

- **WOE Transformation**: Automatic Weight of Evidence binning and transformation
- **Advanced ML Models**: XGBoost, LightGBM, CatBoost, Random Forest, and more
- **Dual Pipeline**: Simultaneous raw and WOE-transformed feature processing
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Model Interpretability**: SHAP values and feature importance analysis
- **Comprehensive Reporting**: Excel reports with model metrics, WOE bins, and visualizations
- **Production Ready**: Modular design with proper error handling and logging

## Installation

### From PyPI (Coming Soon)
```bash
pip install risk-model-pipeline
```

### From GitHub (Latest Development Version)
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git@main
```

### With Optional Dependencies
```bash
# Full installation with all features
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@main#egg=risk-model-pipeline[all]"

# Only visualization tools
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@main#egg=risk-model-pipeline[viz]"

# Only advanced ML models
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@main#egg=risk-model-pipeline[ml]"
```

## Quick Start

```python
from risk_pipeline import Config, RiskModelPipeline
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Configure pipeline
config = Config(
    target_col='target',
    enable_dual_pipeline=True,
    use_optuna=True,
    n_trials=50
)

# Run pipeline
pipeline = RiskModelPipeline(config)
pipeline.run(df)

# Get predictions
predictions = pipeline.predict(df)

# Access results
print(f"Best model: {pipeline.best_model_name_}")
print(f"Best score: {pipeline.best_score_}")
```

## Advanced Configuration

```python
config = Config(
    # Data columns
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    
    # Feature Engineering
    rare_threshold=0.01,       # Rare category threshold
    psi_threshold=0.25,        # PSI stability threshold  
    iv_min=0.02,              # Minimum Information Value
    rho_threshold=0.90,       # Correlation threshold
    
    # Model Training
    cv_folds=5,
    use_optuna=True,
    n_trials=100,
    
    # Model Selection
    model_selection_method='balanced',
    model_stability_weight=0.3,
    
    # Output
    output_folder='outputs',
    random_state=42
)
```

## Model Selection Strategies

### Traditional (Highest Performance)
```python
config = Config(model_selection_method='gini_oot')
```

### Stability-Focused
```python
config = Config(
    model_selection_method='stable',
    min_gini_threshold=0.5
)
```

### Balanced (Performance + Stability)
```python
config = Config(
    model_selection_method='balanced',
    model_stability_weight=0.3  # 30% stability, 70% performance
)
```

## Documentation

- [Installation Guide](INSTALL_GUIDE.md)
- [Publishing to PyPI](PUBLISH_TO_PYPI.md)
- [Example Notebooks](notebooks/)

## Development

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/selimoksuz/risk-model-pipeline.git
cd risk-model-pipeline

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=risk_pipeline

# Run specific test file
pytest tests/test_pipeline.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Selim Öksüz**
- GitHub: [@selimoksuz](https://github.com/selimoksuz)

## Support

For bugs and feature requests, please use the [GitHub Issues](https://github.com/selimoksuz/risk-model-pipeline/issues).

## Acknowledgments

- Built with scikit-learn, XGBoost, LightGBM, and CatBoost
- SHAP for model interpretability
- Optuna for hyperparameter optimization