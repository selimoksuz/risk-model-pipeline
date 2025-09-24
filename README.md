# Risk Model Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/selimoksuz/risk-model-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/selimoksuz/risk-model-pipeline/actions/workflows/ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20quality-99.5%25-brightgreen)](https://github.com/selimoksuz/risk-model-pipeline)

Production-ready risk modeling pipeline with WOE transformation and advanced ML features.

## Features

- **WOE Transformation**: Automatic Weight of Evidence binning and transformation
- **Advanced ML Models**: XGBoost, LightGBM, CatBoost, Random Forest, WoE-LI interactions, Shao penalised logistic, and more
- **Dual Pipeline**: Simultaneous raw and WOE-transformed feature processing
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Model Interpretability**: SHAP values and feature importance analysis
- **Comprehensive Reporting**: Excel reports with model metrics, WOE bins, and visualizations
- **Bundled Sample Dataset**: Synthetic credit risk data shipped under `risk_pipeline.data.sample` for reproducible quickstarts
- **Production Ready**: Modular design with proper error handling and logging

## Installation

### From PyPI (Coming Soon)
```bash
pip install risk-model-pipeline
```

### From GitHub (Latest Development Version)
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git@development
```

### With Optional Dependencies

> Not: Kurulum, LightGBM/XGBoost/CatBoost, PyGAM, Optuna, SHAP, xbooster ve nbformat gibi istege bagli kutuphaneleri de otomatik olarak yukler; ayri paket kurulumuna gerek kalmaz.

```bash
# Full installation with all features
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@development#egg=risk-model-pipeline[all]"

# Only visualization tools
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@development#egg=risk-model-pipeline[viz]"

# Only advanced ML models
pip install "git+https://github.com/selimoksuz/risk-model-pipeline.git@development#egg=risk-model-pipeline[ml]"
```

## Quick Start

```bash
pip install --no-cache-dir --upgrade --force-reinstall git+https://github.com/selimoksuz/risk-model-pipeline.git@development#egg=risk-pipeline[ml,notebook]
```

```python
import pandas as pd
from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline

# Load your data
df = pd.read_csv("your_data.csv")

# Configure the unified pipeline
config = Config(
    target_column="target",
    id_column="app_id",
    time_column="app_dt",
    create_test_split=True,
    use_test_split=True,
    train_ratio=0.6,
    test_ratio=0.2,
    oot_ratio=0.2,
    enable_dual=True,
    selection_steps=["psi", "univariate", "iv", "correlation", "boruta", "stepwise"],
    algorithms=[
        "logistic", "gam", "catboost", "lightgbm", "xgboost",
        "randomforest", "extratrees", "woe_boost", "woe_li", "shao", "xbooster",
    ],
    use_optuna=True,
    n_trials=10,
    model_selection_method="balanced",
    model_stability_weight=0.25,
    risk_band_method="pd_constraints",
    n_risk_bands=8,
    random_state=42,
)
config.model_type = 'all'


# Train and optionally score
demo = UnifiedRiskPipeline(config)
results = demo.fit(df=df, score_df=df)

print("Best model:", results.get("best_model_name"))
print("Selected features:", results.get("selected_features"))
```

## Advanced Configuration

```python
config = Config(
    # Core columns
    target_column="target",
    id_column="app_id",
    time_column="app_dt",

    # Dataset splitting
    create_test_split=True,
    use_test_split=True,
    train_ratio=0.6,
    test_ratio=0.2,
    oot_ratio=0.2,
    stratify_test=True,

    # Feature engineering safeguards
    rare_category_threshold=0.01,
    psi_threshold=0.25,
    iv_threshold=0.02,
    correlation_threshold=0.95,
    vif_threshold=5.0,

    # Model training
    algorithms=["lightgbm", "xgboost", "catboost", "woe_boost"],
    model_type='all',
    cv_folds=5,
    use_optuna=True,
    n_trials=50,

    # Model selection
    model_selection_method="balanced",
    model_stability_weight=0.3,
    max_train_oot_gap=0.08,

    # Outputs
    output_folder="output_reports",
    random_state=42,
)


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
- [End-to-End Demo Notebook](notebooks/Risk_Model_Pipeline_End_to_End.ipynb)

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

**Selim Oksuz
- GitHub: [@selimoksuz](https://github.com/selimoksuz)

## Support

For bugs and feature requests, please use the [GitHub Issues](https://github.com/selimoksuz/risk-model-pipeline/issues).

## Acknowledgments

- Built with scikit-learn, XGBoost, LightGBM, and CatBoost
- SHAP for model interpretability
- Optuna for hyperparameter optimization

