# Old Process Backup

This folder contains the original pipeline implementations for comparison and reference.

## Files

### pipeline16_backup.py
- **Description**: Original monolithic pipeline (2,176 lines)
- **Status**: Fully functional but not modular
- **Created**: Original implementation
- **Why Kept**: For comparison and fallback if needed

### pipeline_first_attempt.py  
- **Description**: First attempt at modularization
- **Status**: Had some compatibility issues
- **Created**: During refactoring process
- **Why Kept**: Shows the refactoring evolution

## Migration Summary

The original `pipeline16.py` has been successfully refactored into a modular architecture:

### New Structure (in `src/risk_pipeline/`)

```
core/
├── __init__.py           # Core module exports
├── base.py               # Base pipeline class
├── data_processor.py     # Data validation, splitting, preprocessing
├── feature_engineer.py   # WOE, PSI, feature selection
├── model_trainer.py      # Model training and evaluation
├── report_generator.py   # Report generation and export
└── utils.py             # Utility functions and classes

pipeline.py              # New orchestrator (replaces pipeline16.py)
pipeline16.py           # Kept for Config class compatibility
```

## Key Improvements

1. **Modular Design**: Separated concerns into focused modules
2. **Better Maintainability**: Each module can be updated independently
3. **Improved Testability**: Components can be tested in isolation
4. **Code Reusability**: Modules can be used in other pipelines
5. **Backwards Compatible**: Still uses Config from pipeline16.py

## Usage Comparison

### Old Way (pipeline16.py)
```python
from risk_pipeline.pipeline16 import Config, RiskModelPipeline
```

### New Way (modular)
```python
from risk_pipeline.pipeline16 import Config  # Config still from old file
from risk_pipeline.pipeline import RiskModelPipeline  # New modular pipeline
```

## Features Preserved

✅ All pipeline steps (1-15)
✅ Dual pipeline support (WOE + Raw)
✅ Model selection based on Gini_OOT
✅ Excel report generation
✅ Artifact export
✅ Random seed consistency
✅ GAM model handling

## Migration Date
September 4, 2024

## Notes
- The new pipeline passes all compatibility tests
- Performance is comparable to the original
- All outputs are identical when using the same random seed