# Risk Model Pipeline Notebooks

## Available Notebooks

### 1. working_demo.ipynb ✅
**Complete working demonstration** of the risk model pipeline with all features:
- Synthetic data generation
- Data processing and validation  
- Train/Test/OOT splitting
- Comprehensive feature selection (IV, PSI, Boruta, Forward Selection, VIF)
- WOE transformation with binning
- Model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- PSI calculation (WOE-based and Score-based)
- Feature importance analysis
- Model saving and export

## How to Run

```bash
# 1. Install the package (from project root)
pip install -e .

# 2. Start Jupyter
jupyter notebook

# 3. Open notebooks/working_demo.ipynb and run all cells
```

## Features Demonstrated

- **Data Validation**: Automatic validation of target, ID, and time columns
- **Feature Selection**: 7 different methods including:
  - Information Value (IV) filtering
  - PSI stability check
  - Correlation removal
  - Boruta selection
  - Forward selection with 1SE rule
  - Noise sentinel check
  - VIF multicollinearity check
- **WOE Transformation**: Automatic binning and WOE calculation
- **Model Training**: Multiple algorithms with optional hyperparameter optimization
- **PSI Analysis**: Population Stability Index for monitoring
- **Model Export**: Save trained models and WOE mappings

## Configuration Options

The notebook uses minimal settings for fast execution:
- `n_trials=1`: Single trial for Optuna (increase for better optimization)
- `cv_folds=3`: 3-fold cross-validation (increase for more robust selection)
- `use_boruta=True`: Enable/disable Boruta feature selection
- `forward_selection=True`: Enable/disable forward selection
- `use_noise_sentinel=True`: Enable/disable overfitting detection

## Output Files

The notebook creates:
- `output_notebook/best_model_*.pkl`: Trained model
- `output_notebook/woe_mapping.pkl`: WOE transformation mappings
- `output_notebook/results_summary.csv`: Performance summary

## Test Results

✅ **All features tested and working**
- Data processing: ✅
- Feature selection (7 methods): ✅
- WOE transformation: ✅
- Model training (4 algorithms): ✅
- PSI calculation: ✅
- Model export: ✅