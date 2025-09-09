#!/usr/bin/env python
"""Example: Basic pipeline usage"""

import pandas as pd
from risk_pipeline import RiskModelPipeline
from risk_pipeline.core.config import Config


def run_basic_pipeline():
    """Run a basic risk model pipeline"""
    
    # Load sample data
    df = pd.read_csv('../data/sample_data.csv')
    
    # Configure pipeline
    config = Config(
        target_col='target',
        id_col='app_id',
        time_col='app_dt',
        output_folder='output',
        iv_min=0.02,
        psi_threshold=0.25
    )
    
    # Initialize and run pipeline
    pipeline = RiskModelPipeline(config)
    results = pipeline.run(df)
    
    print(f"Pipeline completed!")
    print(f"Best model: {pipeline.best_model_name_}")
    print(f"AUC: {pipeline.best_auc_:.4f}")
    
    return results


if __name__ == "__main__":
    run_basic_pipeline()
