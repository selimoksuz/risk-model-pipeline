# Installation Guide 📦

## Quick Install from PyPI (Coming Soon)

Once published to PyPI, you'll be able to install directly:

```bash
# Basic installation
pip install risk-model-pipeline

# With visualization support
pip install risk-model-pipeline[viz]

# With advanced ML features
pip install risk-model-pipeline[ml]

# Full installation with all features
pip install risk-model-pipeline[all]
```

## Install from GitHub (Current Method)

### Latest Development Version
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git
```

### Specific Release
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git@v0.3.0
```

### Editable Installation (for development)
```bash
git clone https://github.com/selimoksuz/risk-model-pipeline.git
cd risk-model-pipeline
pip install -e .

# With all development tools
pip install -e .[dev]
```

## Installation Options

### 1. Minimal Installation (Core Only)
```bash
pip install risk-model-pipeline
```
✅ Features included:
- Model training pipeline
- WOE transformation
- Feature selection
- Excel reporting
- Basic models (Logistic, XGBoost, LightGBM)

❌ Not included:
- Visualization
- SHAP analysis
- Advanced HPO with Optuna

### 2. With Visualization Support
```bash
pip install risk-model-pipeline[viz]
```
Additionally includes:
- Matplotlib plotting
- Seaborn charts
- Interactive Plotly dashboards

### 3. With Advanced ML Features
```bash
pip install risk-model-pipeline[ml]
```
Additionally includes:
- Optuna hyperparameter optimization
- SHAP explainability
- Imbalanced learning techniques
- Extra scikit-learn models

### 4. For Notebook Users
```bash
pip install risk-model-pipeline[notebook]
```
Additionally includes:
- Jupyter notebook support
- IPython widgets
- Enhanced notebook displays

### 5. Full Installation
```bash
pip install risk-model-pipeline[all]
```
Includes everything!

## Environment Setup

### Using Conda
```bash
# Create new environment
conda create -n risk-pipeline python=3.9
conda activate risk-pipeline

# Install package
pip install risk-model-pipeline[all]
```

### Using venv
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install package
pip install risk-model-pipeline[all]
```

### Using Poetry
```bash
poetry add risk-model-pipeline
# With extras
poetry add risk-model-pipeline[viz,ml]
```

## Verify Installation

```python
# Check installation
import risk_pipeline
print(risk_pipeline.__version__)

# Test imports
from risk_pipeline import Config, DualPipeline
from risk_pipeline.core import DataProcessor, FeatureEngineer

# Check available features
from risk_pipeline.utils.safe_imports import check_dependencies
check_dependencies(verbose=True)
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 1GB free disk space

### Recommended for Production
- Python 3.9 or 3.10
- 8GB+ RAM
- 2GB+ free disk space
- 64-bit operating system

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Ubuntu 20.04+
- ✅ CentOS 7+
- ✅ Google Colab
- ✅ AWS SageMaker

## Troubleshooting

### ImportError: No module named 'risk_pipeline'
```bash
# Ensure package is installed
pip list | grep risk-model-pipeline

# If not found, reinstall
pip install --force-reinstall risk-model-pipeline
```

### Version Conflicts
```bash
# Create clean environment
python -m venv clean_env
clean_env\Scripts\activate  # Windows
pip install risk-model-pipeline
```

### Memory Issues
For large datasets (>1M rows):
```python
# Use chunking
config = Config(
    chunk_size=50000,
    use_memory_optimization=True
)
```

### Slow Installation
```bash
# Use wheels for faster installation
pip install --only-binary :all: risk-model-pipeline
```

## Usage After Installation

### Command Line Interface
```bash
# Check installation
risk-pipeline --version

# Run pipeline
risk-pipeline run --config config.yaml --data data.csv

# Check environment
risk-pipeline-check
```

### Python API
```python
from risk_pipeline import Config, DualPipeline
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Configure
config = Config(
    target_col="target",
    enable_dual_pipeline=True
)

# Run pipeline
pipeline = DualPipeline(config)
pipeline.run(df)

# Get results
results = pipeline.get_results()
```

## Upgrading

### Upgrade to Latest Version
```bash
pip install --upgrade risk-model-pipeline
```

### Check Current Version
```python
import risk_pipeline
print(risk_pipeline.__version__)
```

### Migration Guide
See [CHANGELOG.md](CHANGELOG.md) for breaking changes between versions.

## Uninstallation

```bash
pip uninstall risk-model-pipeline
```

## Support

- 📖 Documentation: [GitHub README](https://github.com/selimoksuz/risk-model-pipeline)
- 🐛 Issues: [GitHub Issues](https://github.com/selimoksuz/risk-model-pipeline/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/selimoksuz/risk-model-pipeline/discussions)

## License

MIT License - see [LICENSE](LICENSE) file for details.