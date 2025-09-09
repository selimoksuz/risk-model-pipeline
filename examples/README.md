# End-to-End Pipeline Example

## Available File

### end_to_end_pipeline.py
Complete working demonstration of the entire risk model pipeline with all features.

## How to Run

```bash
# From project root directory
python examples/end_to_end_pipeline.py
```

## What It Does

The script runs a complete risk modeling pipeline:

1. **Creates synthetic data** (2000 samples, 35 features)
2. **Processes and validates** the data
3. **Splits** into Train/Test sets
4. **Selects features** using 7 different methods
5. **Applies WOE transformation** with automatic binning
6. **Trains 4 models** (LR, RF, XGBoost, LightGBM)
7. **Calculates PSI** for stability monitoring
8. **Saves the best model** and results

## Output

Creates an `output` folder with:
- `model_*.pkl` - Best trained model
- `woe_mapping.pkl` - WOE transformation mappings
- `selected_features.txt` - List of selected features
- `model_summary.csv` - Performance metrics

## Expected Results

- Feature reduction: ~35 features → ~6 features
- Test AUC: ~0.74-0.75
- Score PSI: <0.1 (stable)
- Execution time: ~30 seconds
