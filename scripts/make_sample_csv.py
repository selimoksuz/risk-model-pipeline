import numpy as np
import pandas as pd
from datetime import datetime

rng = np.random.default_rng(42)

# sample size for realistic modeling (larger for better stability)
n = 10000
# app_id sequential
a_pp = np.arange(100000, 100000 + n)

# monthly dates across 18 months, ensuring last 3 months present (more historical data)
months = pd.date_range('2023-01-01', periods=18, freq='MS')
app_dt = rng.choice(months, size=n, replace=True)

# Determine OOT mask early for drift simulation
app_dt_series = pd.Series(app_dt)
oot_mask = app_dt_series >= app_dt_series.max() - pd.offsets.MonthBegin(2)

# numeric features (base)
num1 = rng.normal(0, 1, size=n)
num2 = rng.normal(0, 1.2, size=n)
num3 = rng.gamma(shape=2.0, scale=0.8, size=n)

# add correlated features with different drift patterns
# ~0.95 correlation with num1, but different drift in OOT
num1_corr95 = num1 + rng.normal(0, 0.1, size=n)
# Add slight shift in OOT to create PSI difference
num1_corr95[oot_mask.values] = num1_corr95[oot_mask.values] + rng.normal(0.1, 0.05, size=oot_mask.sum())

# ~0.7 correlation with num1, stable across time
num1_corr70 = 0.7*num1 + rng.normal(0, 0.714, size=n)

# another strong correlation pair with moderate shift
num2_corr92 = num2 + rng.normal(0, 0.15, size=n)
# Small shift in OOT  
num2_corr92[oot_mask.values] = num2_corr92[oot_mask.values] + 0.2

# additional numeric features (more realistic feature count)
income = rng.lognormal(mean=10, sigma=0.5, size=n)  # income-like distribution
age = rng.integers(18, 80, size=n)  # age
debt_ratio = rng.beta(2, 5, size=n)  # debt ratio (0-1)
credit_score = rng.normal(700, 100, size=n).clip(300, 850)  # credit score
months_employed = rng.exponential(24, size=n).clip(0, 240)  # employment months

# categorical with rare categories
cat1_levels = np.array(['A','B','C','RareX'])
cat1_probs  = np.array([0.45,0.35,0.18,0.02])
cat1 = rng.choice(cat1_levels, p=cat1_probs, size=n)

cat2_levels = np.array(['X','Y','Z','RareY'])
cat2_probs  = np.array([0.40,0.40,0.18,0.02])
cat2 = rng.choice(cat2_levels, p=cat2_probs, size=n)

cat3_levels = np.array(['S','T','U','V','RareZ'])
cat3_probs  = np.array([0.30,0.30,0.20,0.18,0.02])
cat3 = rng.choice(cat3_levels, p=cat3_probs, size=n)

# additional categorical features  
education_levels = np.array(['HighSchool', 'Bachelor', 'Master', 'PhD', 'Other'])
education_probs = np.array([0.40, 0.35, 0.15, 0.05, 0.05])
education = rng.choice(education_levels, p=education_probs, size=n)

region_levels = np.array(['North', 'South', 'East', 'West', 'Central', 'Remote'])
region_probs = np.array([0.25, 0.25, 0.20, 0.20, 0.08, 0.02])
region = rng.choice(region_levels, p=region_probs, size=n)

employment_type = np.array(['FullTime', 'PartTime', 'SelfEmployed', 'Unemployed'])
employment_probs = np.array([0.70, 0.15, 0.10, 0.05])
employment = rng.choice(employment_type, p=employment_probs, size=n)

# generate target with meaningful signal from multiple features
z = (
    0.8*num1 - 0.6*num2 + 0.3*num3
    + 0.9*(cat1=='A').astype(float)
    + 0.7*(cat2=='X').astype(float)
    + 0.4*(cat3=='S').astype(float)
    # realistic financial risk factors (normalized scales)
    - 0.00005*income  # higher income = lower risk (scaled for log-normal income)
    - 0.008*age  # older = lower risk  
    + 1.5*debt_ratio  # higher debt ratio = higher risk
    - 0.002*credit_score  # higher credit score = lower risk
    - 0.005*months_employed  # longer employment = lower risk
    + 0.4*(education=='HighSchool').astype(float)  # education effect
    + 0.6*(employment=='Unemployed').astype(float)  # employment effect
    - 0.2  # bias for reasonable class balance
)
z = z + rng.normal(0, 0.5, size=n)
prob = 1/(1 + np.exp(-z))
target = rng.binomial(1, prob, size=n)

# Apply additional MODERATE shifts on selected variables for OOT (last 3 months)
# SUBTLE numeric shift (moderate drift for realistic PSI)
num3_shifted = num3.copy()
# Only 30% mean increase instead of +2.0 (too extreme)
num3_shifted[oot_mask.values] = num3_shifted[oot_mask.values] + 0.3

# SUBTLE categorical distribution shift in OOT  
cat2_shifted = pd.Series(cat2).copy()
mask = oot_mask.values
# Moderate shift: 0.40->0.30, 0.40->0.35, 0.18->0.33 (more realistic)
cat2_shifted.loc[mask] = rng.choice(cat2_levels, p=np.array([0.30,0.35,0.33,0.02]), size=mask.sum())

# Add moderate income drift (economic conditions change)
income_shifted = income.copy()
# 10% income increase in OOT months (economic growth)
income_shifted[oot_mask.values] = income_shifted[oot_mask.values] * 1.1

# introduce missingness per-feature (not all features missing in same row)
miss_rate = 0.08  # slightly lower for larger dataset
mask_num1 = rng.random(n) < miss_rate
mask_num2 = rng.random(n) < miss_rate
mask_num3 = rng.random(n) < miss_rate
mask_income = rng.random(n) < miss_rate*0.5  # income less likely to be missing
mask_credit = rng.random(n) < miss_rate*0.3  # credit score less likely missing
mask_cat1 = rng.random(n) < miss_rate
mask_cat2 = rng.random(n) < miss_rate
mask_cat3 = rng.random(n) < miss_rate
mask_education = rng.random(n) < miss_rate*0.2  # education rarely missing

num1[mask_num1] = np.nan
num2[mask_num2] = np.nan
num3_shifted[mask_num3] = np.nan
income_shifted[mask_income] = np.nan
credit_score[mask_credit] = np.nan

cat1 = pd.Series(cat1)
cat2 = cat2_shifted
cat3 = pd.Series(cat3)
education = pd.Series(education)
region = pd.Series(region)
employment = pd.Series(employment)

cat1[mask_cat1] = pd.NA
cat2[mask_cat2] = pd.NA
cat3[mask_cat3] = pd.NA
education[mask_education] = pd.NA

# ensure no row has all features missing simultaneously
features_df = pd.DataFrame({
    'num1': num1,
    'num2': num2,
    'num3': num3_shifted,
    'num1_corr95': num1_corr95,
    'num1_corr70': num1_corr70,
    'num2_corr92': num2_corr92,
    'income': income_shifted,
    'age': age,
    'debt_ratio': debt_ratio,
    'credit_score': credit_score,
    'months_employed': months_employed,
    'cat1': cat1,
    'cat2': cat2,
    'cat3': cat3,
    'education': education,
    'region': region,
    'employment': employment,
})
all_missing = features_df.isna().sum(axis=1) == features_df.shape[1]
if all_missing.any():
    # for those rare rows, restore num1 from noise
    features_df.loc[all_missing, 'num1'] = rng.normal(0,1, size=int(all_missing.sum()))

# assemble final df
df = pd.DataFrame({
    'app_id': a_pp,
    'app_dt': app_dt,
    'target': target,
}).join(features_df)

# shuffle rows for realism
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# write CSV
out_path = 'data/input.csv'
df.to_csv(out_path, index=False)
print(f'Wrote {len(df)} rows to {out_path}')
