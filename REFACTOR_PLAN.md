# Pipeline16.py Refactoring Plan

## Current State
- **File Size**: 2,176 lines (too large for maintainability)
- **Mixed Responsibilities**: Contains all pipeline logic in one file
- **Limited Module Usage**: Only uses a few functions from stages modules

## Why Pipeline16.py is Large

The file contains the entire risk model pipeline implementation including:
1. Data validation and preprocessing
2. Variable classification
3. WOE transformation and binning
4. Feature selection algorithms
5. Model training and evaluation
6. Dual pipeline logic (WOE vs Raw)
7. Report generation
8. Excel export logic
9. Calibration
10. Scoring

## Current Module Structure (Partially Used)

The project already has a modular structure in `src/risk_pipeline/`:
- `stages/`: Contains individual pipeline stages
- `model/`: Model-related utilities
- `features/`: Feature engineering (PSI, WOE)
- `reporting/`: Report generation
- `data/`: Data loading
- `config/`: Configuration

However, pipeline16.py duplicates much of this functionality instead of using these modules.

## Proposed Refactoring

### Phase 1: Extract Core Components
Break pipeline16.py into logical components:

```python
# src/risk_pipeline/core/pipeline_base.py
class BasePipeline:
    """Base pipeline with common functionality"""
    
# src/risk_pipeline/core/data_processor.py
class DataProcessor:
    """Data validation, splitting, preprocessing"""
    
# src/risk_pipeline/core/feature_engineer.py
class FeatureEngineer:
    """WOE, PSI, feature selection"""
    
# src/risk_pipeline/core/model_trainer.py
class ModelTrainer:
    """Model training, HPO, evaluation"""
    
# src/risk_pipeline/core/report_generator.py
class ReportGenerator:
    """Report and Excel generation"""
```

### Phase 2: Refactor Pipeline16
```python
# src/risk_pipeline/pipeline16_refactored.py
from .core import BasePipeline, DataProcessor, FeatureEngineer, ModelTrainer, ReportGenerator

class RiskModelPipeline(BasePipeline):
    def __init__(self, config):
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        self.report_generator = ReportGenerator(config)
    
    def run(self, df):
        # Orchestrate components
        pass
```

### Phase 3: Gradual Migration
1. Start by extracting utility functions
2. Move large methods to appropriate modules
3. Keep pipeline16.py as orchestrator only
4. Add comprehensive tests for each module

## Benefits
1. **Maintainability**: Easier to understand and modify
2. **Testability**: Can test components independently
3. **Reusability**: Components can be used in other pipelines
4. **Team Collaboration**: Multiple developers can work on different modules
5. **Code Organization**: Clear separation of concerns

## Implementation Priority
1. Extract report generation (easiest, ~400 lines)
2. Extract model training logic (~300 lines)
3. Extract feature engineering (~400 lines)
4. Extract data processing (~300 lines)
5. Refactor main pipeline as orchestrator

## Backwards Compatibility
- Keep pipeline16.py working during transition
- Create pipeline17.py with refactored version
- Gradually migrate users to new version

## Note
This refactoring should be done incrementally to avoid breaking existing functionality. Each extraction should be accompanied by tests to ensure no regression.