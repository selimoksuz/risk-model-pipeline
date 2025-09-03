#!/bin/bash

echo "======================================"
echo "Risk Model Pipeline Environment Setup"
echo "======================================"
echo ""

echo "Creating virtual environment..."
python3 -m venv risk_env

echo ""
echo "Activating environment..."
source risk_env/bin/activate

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing compatible packages..."
pip uninstall -y numpy pandas scikit-learn
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install scikit-learn==1.3.0

echo ""
echo "Installing other requirements..."
pip install xgboost>=2.0.0
pip install lightgbm>=4.3.0
pip install matplotlib>=3.8.0
pip install shap>=0.45.0
pip install openpyxl>=3.1.0
pip install xlsxwriter>=3.2.0
pip install boruta>=0.3
pip install pygam>=0.9.0
pip install optuna>=3.6.0
pip install scipy>=1.10.0
pip install statsmodels>=0.14.0
pip install typer>=0.12.0
pip install pydantic>=2.7.0
pip install pyyaml>=6.0
pip install joblib>=1.3.0
pip install psutil>=5.9.0

echo ""
echo "Installing Jupyter..."
pip install jupyter notebook

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To use this environment:"
echo "  1. Run: source risk_env/bin/activate"
echo "  2. Start Jupyter: jupyter notebook"
echo ""