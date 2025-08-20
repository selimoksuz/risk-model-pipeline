import numpy as np
import pandas as pd
from datetime import datetime

rng = np.random.default_rng(42)

# sample size a bit larger for stability
n = 500
# app_id sequential
a_pp = np.arange(100000, 100000 + n)

# monthly dates across 12 months, ensuring last 3 months present
months = pd.date_range('2024-01-01', periods=12, freq='MS')
app_dt = rng.choice(months, size=n, replace=True)

# numeric features (base)
num1 = rng.normal(0, 1, size=n)
num2 = rng.normal(0, 1.2, size=n)
num3 = rng.gamma(shape=2.0, scale=0.8, size=n)

# add correlated features
# ~0.95 correlation with num1
num1_corr95 = num1 + rng.normal(0, 0.1, size=n)
# ~0.7 correlation with num1
num1_corr70 = 0.7*num1 + rng.normal(0, 0.714, size=n)
# another strong correlation pair
num2_corr92 = num2 + rng.normal(0, 0.15, size=n)

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

# generate target with meaningful signal
z = (
    0.8*num1 - 0.6*num2 + 0.3*num3
    + 0.9*(cat1=='A').astype(float)
    + 0.7*(cat2=='X').astype(float)
    + 0.4*(cat3=='S').astype(float)
    - 0.5  # bias for reasonable class balance
)
z = z + rng.normal(0, 0.5, size=n)
prob = 1/(1 + np.exp(-z))
target = rng.binomial(1, prob, size=n)

# induce PSI shifts on selected variables for OOT (last 3 months)
app_dt_series = pd.Series(app_dt)
oot_mask = app_dt_series >= app_dt_series.max() - pd.offsets.MonthBegin(2)
# numeric shift (increase mean in OOT)
num3_shifted = num3.copy()
num3_shifted[oot_mask.values] = num3_shifted[oot_mask.values] + 2.0
# categorical distribution shift in OOT
cat2_shifted = pd.Series(cat2).copy()
mask = oot_mask.values
cat2_shifted.loc[mask] = rng.choice(cat2_levels, p=np.array([0.10,0.20,0.68,0.02]), size=mask.sum())

# introduce missingness per-feature (not all features missing in same row)
miss_rate = 0.10
mask_num1 = rng.random(n) < miss_rate
mask_num2 = rng.random(n) < miss_rate
mask_num3 = rng.random(n) < miss_rate
mask_cat1 = rng.random(n) < miss_rate
mask_cat2 = rng.random(n) < miss_rate
mask_cat3 = rng.random(n) < miss_rate

num1[mask_num1] = np.nan
num2[mask_num2] = np.nan
num3_shifted[mask_num3] = np.nan
cat1 = pd.Series(cat1)
cat2 = cat2_shifted
cat3 = pd.Series(cat3)
cat1[mask_cat1] = pd.NA
cat2[mask_cat2] = pd.NA
cat3[mask_cat3] = pd.NA

# ensure no row has all 6 features missing simultaneously
features_df = pd.DataFrame({
    'num1': num1,
    'num2': num2,
    'num3': num3_shifted,
    'num1_corr95': num1_corr95,
    'num1_corr70': num1_corr70,
    'num2_corr92': num2_corr92,
    'cat1': cat1,
    'cat2': cat2,
    'cat3': cat3,
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
