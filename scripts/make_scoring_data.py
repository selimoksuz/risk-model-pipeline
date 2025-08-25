#!/usr/bin/env python3
"""
Generate scoring data for pipeline testing - mixed with/without targets
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_scoring_data(n_samples: int = 2000, output_path: str = "data/scoring.csv"):
    """Generate scoring dataset with some target values and some missing"""
    
    rng = np.random.default_rng(123)
    
    # Generate app_ids (different from training data)
    app_ids = rng.integers(400000, 500000, size=n_samples)
    
    # Generate dates (future period for scoring)
    dates = pd.date_range('2024-09-01', '2024-10-31', freq='D')
    app_dates = rng.choice(dates, size=n_samples)
    
    # Generate features similar to main dataset but with some drift
    ages = rng.integers(18, 80, size=n_samples)
    incomes = rng.lognormal(mean=9.8, sigma=0.6, size=n_samples).clip(25000, 800000)  # Slight drift
    debt_ratios = rng.beta(2.2, 4.8, size=n_samples)  # Slight drift
    credit_scores = rng.normal(690, 110, size=n_samples).clip(300, 850)  # Slight drift
    months_employed = rng.integers(0, 240, size=n_samples)
    
    # Categorical variables with some distribution shift
    education_choices = ['High_School', 'College', 'Graduate']
    education_probs = [0.4, 0.4, 0.2]  # Different from training
    education = rng.choice(education_choices, size=n_samples, p=education_probs)
    
    region_choices = ['A', 'B', 'C', 'D', 'E']
    region_probs = [0.25, 0.25, 0.2, 0.15, 0.15]  # Different distribution
    region = rng.choice(region_choices, size=n_samples, p=region_probs)
    
    employment_choices = ['Full_Time', 'Part_Time', 'Self_Employed', 'Unemployed', 'Student']
    employment_probs = [0.5, 0.2, 0.15, 0.1, 0.05]  # Different from training
    employment = rng.choice(employment_choices, size=n_samples, p=employment_probs)
    
    # Additional numeric features with drift
    num1 = rng.normal(105, 35, size=n_samples)  # Drift: mean shifted
    num2 = rng.normal(48, 18, size=n_samples)   # Drift: higher std
    num3 = rng.exponential(25, size=n_samples)  # Similar
    num4 = rng.uniform(0.1, 0.95, size=n_samples)  # Similar
    
    # Binary features
    bin1 = rng.binomial(1, 0.35, size=n_samples)  # Drift: different probability
    bin2 = rng.binomial(1, 0.62, size=n_samples)  # Drift: different probability
    
    # Generate target for only 60% of the data (simulate partial ground truth)
    has_target = rng.binomial(1, 0.6, size=n_samples)
    
    # For records with targets, generate based on features (similar logic but with drift)
    targets = np.full(n_samples, np.nan)
    target_mask = has_target == 1
    
    if target_mask.sum() > 0:
        # Slightly different risk model for target generation (to create some drift)
        risk_scores = (
            -0.3 + 
            0.0002 * incomes[target_mask] +
            -0.015 * ages[target_mask] + 
            0.8 * debt_ratios[target_mask] +
            -0.002 * credit_scores[target_mask] +
            0.003 * months_employed[target_mask] +
            0.01 * num1[target_mask] +
            -0.02 * num2[target_mask] +
            0.15 * bin1[target_mask] +
            0.12 * bin2[target_mask] +
            rng.normal(0, 0.3, size=target_mask.sum())  # More noise
        )
        
        probs = 1 / (1 + np.exp(-risk_scores))
        targets[target_mask] = rng.binomial(1, probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'app_id': app_ids,
        'app_dt': app_dates,
        'age': ages,
        'income': incomes.round(2),
        'debt_ratio': debt_ratios.round(4),
        'credit_score': credit_scores.round(0),
        'months_employed': months_employed,
        'education': education,
        'region': region,
        'employment': employment,
        'num1': num1.round(2),
        'num2': num2.round(2),
        'num3': num3.round(2),
        'num4': num4.round(4),
        'bin1': bin1,
        'bin2': bin2,
        'target': targets  # NaN for records without target
    })
    
    # Sort by date
    df = df.sort_values('app_dt').reset_index(drop=True)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Calculate statistics
    n_with_target = target_mask.sum()
    n_without_target = n_samples - n_with_target
    default_rate = targets[target_mask].mean() if n_with_target > 0 else 0
    
    print(f"Generated scoring data:")
    print(f"- Total samples: {len(df):,}")
    print(f"- With target: {n_with_target:,} ({n_with_target/n_samples*100:.1f}%)")
    print(f"- Without target: {n_without_target:,} ({n_without_target/n_samples*100:.1f}%)")
    print(f"- Date range: {df.app_dt.min()} to {df.app_dt.max()}")
    print(f"- Default rate (with target): {default_rate:.3f}")
    print(f"- Feature drift: Income mean={df.income.mean():.0f}, Age mean={df.age.mean():.1f}")
    print(f"- Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_scoring_data()