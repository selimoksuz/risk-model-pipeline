# Notebooks

This directory contains Jupyter notebooks for interactive analysis and experimentation.

## Available Notebooks

### 1. `risk_model_demo.ipynb`
- Demonstrates the complete pipeline workflow
- Shows feature engineering and WOE transformation
- Visualizes model performance metrics

### 2. `data_exploration.ipynb`
- Exploratory data analysis
- Feature distribution analysis
- Target variable analysis

### 3. `model_comparison.ipynb`
- Compares different model types (LightGBM, XGBoost, CatBoost)
- Performance benchmarking
- Feature importance analysis

## Setup

To run these notebooks:

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook
```

## Requirements

All notebooks require the risk-model-pipeline package to be installed:

```bash
pip install -e ..
```