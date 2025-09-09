# Examples

This directory contains example scripts demonstrating how to use the risk-model-pipeline.

## Quick Start

```python
from risk_pipeline import RiskModelPipeline
from risk_pipeline.core.config import Config

# Load your data
df = pd.read_csv('your_data.csv')

# Configure pipeline
config = Config(
    target_col='target',
    id_col='app_id',
    time_col='app_dt'
)

# Run pipeline
pipeline = RiskModelPipeline(config)
pipeline.run(df)
```

See individual example files for more detailed usage.
