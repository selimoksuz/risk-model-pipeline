# Risk Model Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready risk modeling pipeline with dual approach: WOE (Weight of Evidence) transformation and raw variables processing. Designed for credit risk, fraud detection, and other binary classification problems in financial services.

## 🚀 Key Features

### Dual Pipeline Architecture
- **WOE Pipeline**: Interpretable models with binning and Weight of Evidence transformation
- **Raw Pipeline**: High-performance models with automated imputation and outlier handling
- Run both pipelines simultaneously for comprehensive model comparison

### Advanced Risk Modeling
- **Adaptive WOE Binning**: Intelligent binning with monotonic constraints
- **PSI Monitoring**: Population Stability Index for drift detection  
- **Feature Engineering**: Boruta selection, forward selection with 1SE rule
- **Correlation Clustering**: Automatic handling of multicollinearity
- **Model Calibration**: Isotonic/Platt calibration for probability adjustment

### Production Ready
- **Multi-language Support**: Full UTF-8 support for Turkish and other languages
- **Comprehensive Reporting**: Excel reports with WOE tables, model metrics, SHAP values
- **Data Dictionary Integration**: Variable descriptions and business context
- **Flexible Configuration**: YAML/JSON configuration with dataclass validation

## 📦 Installation

### From Source
```bash
git clone https://github.com/selimoksuz/risk-model-pipeline.git
cd risk-model-pipeline
pip install -e .
```

### Using pip
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from risk_pipeline.pipeline16 import Config, RiskModelPipeline
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Configure pipeline
config = Config(
    id_col='app_id',
    time_col='app_dt', 
    target_col='target',
    
    # Enable dual pipeline
    enable_dual_pipeline=True,
    
    # Output settings
    output_folder='outputs',
    output_excel_path='risk_model_report.xlsx'
)

# Run pipeline
pipeline = RiskModelPipeline(config)
pipeline.run(df)

# Export comprehensive reports
pipeline.export_reports()
```

## 📊 Pipeline Workflow

```
Data Input → Validation → Feature Classification → Missing Value Policy
    ↓
Time-based Split (Train/Test/OOT)
    ↓
    ├── WOE Pipeline
    │   ├── Adaptive Binning
    │   ├── WOE Transformation
    │   ├── PSI Calculation
    │   └── Feature Selection
    │
    └── Raw Pipeline (if enabled)
        ├── Imputation (median/mean/mode)
        ├── Outlier Handling (IQR/Z-score)
        ├── Feature Scaling
        └── Feature Selection
    ↓
Model Training (Logistic, RF, XGBoost, LightGBM, GAM)
    ↓
Model Evaluation & Selection
    ↓
Calibration (Optional)
    ↓
Comprehensive Reporting (Excel + Parquet)
```

## 🛠️ Configuration Options

### Basic Configuration
```python
config = Config(
    # Data columns
    id_col='customer_id',
    time_col='application_date',
    target_col='default_flag',
    
    # Splitting strategy
    use_test_split=True,
    test_size_row_frac=0.2,
    oot_window_months=3,
    
    # Feature engineering thresholds
    rare_threshold=0.01,      # Minimum category frequency
    psi_threshold=0.25,        # PSI stability threshold
    iv_min=0.02,              # Minimum Information Value
    rho_threshold=0.95,       # Correlation threshold
    
    # Model settings
    cv_folds=5,
    hpo_timeout_sec=600,
    hpo_trials=50,
)
```

### Dual Pipeline Configuration
```python
config = Config(
    # Enable dual pipeline
    enable_dual_pipeline=True,
    
    # Raw pipeline settings
    raw_imputation_strategy='median',  # median, mean, zero
    raw_outlier_method='iqr',         # iqr, zscore, percentile
    raw_outlier_threshold=1.5,        # IQR multiplier
)
```

### Data Dictionary Support
```python
# Create data dictionary
data_dict = pd.DataFrame([
    {'alan_adi': 'income', 'alan_aciklamasi': 'Monthly income'},
    {'alan_adi': 'age', 'alan_aciklamasi': 'Customer age'},
])

config = Config(
    data_dictionary_df=data_dict,
    # or from file
    data_dictionary_path='data_dictionary.xlsx'
)
```

## 📈 Model Outputs

### Excel Report Sheets
- **models_summary**: All model performances with CV/Test/OOT metrics
- **woe_mapping**: Complete WOE transformation tables
- **psi_summary**: Feature-level PSI analysis
- **best_model_vars**: Selected features with importance scores
- **ks_info_[train/test/oot]**: KS tables and performance bands
- **shap_summary**: SHAP-based feature importance
- **run_meta**: Pipeline execution metadata

### Metrics Provided
- **Gini Coefficient**: Primary performance metric
- **KS Statistic**: Kolmogorov-Smirnov with optimal threshold
- **AUC-ROC**: Area under ROC curve
- **PSI**: Population Stability Index
- **IV**: Information Value per feature
- **WOE**: Weight of Evidence per bin/category

## 📂 Project Structure

```
risk-model-pipeline/
├── src/
│   └── risk_pipeline/
│       ├── pipeline16.py      # Main pipeline orchestrator
│       ├── stages.py          # Core pipeline stages
│       ├── model/             # Model implementations
│       ├── reporting/         # Report generation
│       └── cli.py            # Command-line interface
├── notebooks/
│   ├── 00_master_end_to_end.ipynb    # Complete example
│   ├── 01-05_simulation_*.ipynb      # Various scenarios
│   ├── 06_realistic_gini_test.ipynb  # Performance testing
│   └── 07_dual_pipeline_test.ipynb   # Dual pipeline demo
├── tests/                     # Unit and integration tests
├── examples/                  # Example configurations
└── data/                     # Sample datasets
```

## 🔧 Advanced Usage

### Custom Model Configuration
```python
# Modify model hyperparameter grids
pipeline.models_config = {
    "XGBoost": {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }
}
```

### Scoring on New Data
```python
# Score new applications
new_data = pd.read_csv("new_applications.csv")
scores = pipeline.score(new_data)

# With calibration
calibrated_scores = pipeline.score(new_data, apply_calibration=True)
```

### Custom Feature Engineering
```python
# Add custom features before pipeline
df['custom_feature'] = df['feature1'] / df['feature2']

# Configure to keep custom features
config.keep_custom_features = ['custom_feature']
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test scenarios:
```bash
python -m pytest tests/test_pipeline.py::test_dual_pipeline
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Selim Oksuz - [GitHub](https://github.com/selimoksuz)

Project Link: [https://github.com/selimoksuz/risk-model-pipeline](https://github.com/selimoksuz/risk-model-pipeline)

## 🙏 Acknowledgments

- Weight of Evidence methodology from credit risk literature
- Boruta feature selection algorithm
- SHAP for model interpretability
- scikit-learn, XGBoost, LightGBM communities