#!/usr/bin/env python3
"""
Generate sample calibration data for pipeline testing
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_calibration_data(n_samples: int = 1000, output_path: str = "data/calibration.csv"):
    """Generate calibration dataset with realistic score distribution"""
    
    rng = np.random.default_rng(42)
    
    # Generate app_ids
    app_ids = rng.integers(200000, 300000, size=n_samples)
    
    # Generate dates (recent period for calibration)
    dates = pd.date_range('2024-07-01', '2024-08-31', freq='D')
    app_dates = rng.choice(dates, size=n_samples)
    
    # Generate realistic score distribution (0-1000 scale)
    # Higher scores should correlate with higher default probability
    scores = rng.beta(2, 5, size=n_samples) * 1000  # 0-1000 range
    
    # Generate true outcomes based on scores with some noise
    # Higher score = higher probability of default (1)
    logits = (scores - 500) / 200 + rng.normal(0, 0.5, size=n_samples)
    probs = 1 / (1 + np.exp(-logits))
    outcomes = rng.binomial(1, probs, size=n_samples)
    
    # Generate basic features that match main pipeline
    ages = rng.integers(18, 80, size=n_samples)
    incomes = rng.lognormal(mean=10, sigma=0.5, size=n_samples).clip(30000, 500000)
    
    # Create DataFrame with features that pipeline can process
    df = pd.DataFrame({
        'app_id': app_ids,
        'app_dt': app_dates,
        'target': outcomes,
        # Add basic features for WOE transformation
        'age': ages,
        'income': incomes,
        'region': rng.choice(['A', 'B', 'C', 'D', 'E'], size=n_samples),
        'education': rng.choice(['High_School', 'College', 'Graduate'], size=n_samples),
    })
    
    # Sort by date
    df = df.sort_values('app_dt').reset_index(drop=True)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated calibration data:")
    print(f"- Samples: {len(df):,}")
    print(f"- Date range: {df.app_dt.min()} to {df.app_dt.max()}")
    print(f"- Age range: {df.age.min():.0f} to {df.age.max():.0f}")
    print(f"- Income range: {df.income.min():.0f} to {df.income.max():.0f}")
    print(f"- Default rate: {df.target.mean():.3f}")
    print(f"- Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_calibration_data()