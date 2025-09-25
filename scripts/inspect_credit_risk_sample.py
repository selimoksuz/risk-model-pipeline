import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

base = Path('examples/data/credit_risk_sample/development.csv')
df = pd.read_csv(base)
auc = roc_auc_score(df['target'], -df['bureau_score'])
print('AUC bureau_score:', round(auc, 4))
print('Gini approx:', round(2 * auc - 1, 4))
print('Correlation utilization-credit_usage:', round(spearmanr(df['utilization_ratio'], df['credit_usage_ratio']).correlation, 3))
print('Monthly default rates:')
print(df.groupby('snapshot_month')['target'].mean())
