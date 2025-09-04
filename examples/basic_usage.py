"""
Basic Usage Example for Risk Model Pipeline
"""

import pandas as pd
import numpy as np
from risk_pipeline import Config, RiskModelPipeline

# Generate sample data
np.random.seed(42)
n_samples = 10000

df = pd.DataFrame({
    'app_id': range(1, n_samples + 1),
    'app_date': pd.date_range(start='2022-01-01', periods=n_samples, freq='H')[:n_samples],
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.exponential(2, n_samples),
    'feature3': np.random.beta(2, 5, n_samples),
    'category1': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.binomial(1, 0.3, n_samples)
})

# Basic configuration
config = Config(
    id_col='app_id',
    time_col='app_date',
    target_col='target',
    output_folder='outputs_example'
)

# Run pipeline
pipeline = RiskModelPipeline(config)
pipeline.run(df)

# Export reports
pipeline.export_reports()
print("Pipeline completed! Check 'outputs_example' folder for results.")