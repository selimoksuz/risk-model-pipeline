import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.core.config import Config


def make_data(n_rows: int, start_date: str = '2022-01-01', n_num: int = 20, n_cat: int = 6):
    rng = np.random.default_rng(42)
    base_date = pd.to_datetime(start_date)
    dates = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 540, size=n_rows)]
    df = pd.DataFrame({
        'customer_id': np.arange(1, n_rows + 1),
        'app_dt': dates,
    })
    for i in range(n_num):
        df[f'num_{i}'] = rng.normal(0, 1, size=n_rows)
    for j in range(n_cat):
        df[f'cat_{j}'] = rng.choice(['A', 'B', 'C', None], size=n_rows, p=[0.4, 0.4, 0.19, 0.01])
    # target with signal from num_0 and cat_0
    logit = 0.4 * df['num_0'].fillna(0).to_numpy() + (df['cat_0'] == 'C').to_numpy() * 0.8 + rng.normal(0, 0.3, size=n_rows)
    p = 1 / (1 + np.exp(-logit))
    df['target'] = (rng.uniform(0, 1, size=n_rows) < p).astype(int)
    return df


def main():
    n = 5000
    dev = make_data(n)
    longrun = make_data(4000, start_date='2021-06-01')
    recent = make_data(4000, start_date='2023-02-01')
    ref = make_data(4000, start_date='2022-06-01')
    scoring = make_data(4000, start_date='2023-05-01')
    scoring = scoring.drop(columns=['target'])

    cfg = Config(
        target_column='target',
        id_column='customer_id',
        time_column='app_dt',
        create_test_split=True,
        test_size=0.2,
        oot_size=0.1,
        enable_dual=True,
        enable_scoring=True,
        use_optuna=False,
        algorithms=['logistic', 'lightgbm'],
        selection_steps=['univariate'],
        n_risk_bands=7,
        risk_band_method='pd_constraints',
        risk_band_min_sample_size=1000,
        calculate_shap=False,
        save_reports=False,
    )

    pipe = UnifiedRiskPipeline(cfg)
    results = pipe.fit(
        dev,
        calibration_df=longrun,
        stage2_df=recent,
        risk_band_df=ref,
        score_df=scoring,
    )
    print('Available models:', ', '.join(sorted(pipe.models_.keys())))
    bands = results.get('risk_bands', {}) or {}
    stats = bands.get('band_stats') if isinstance(bands, dict) else None
    print('Risk band stats empty:', (stats is None) or (getattr(stats, 'empty', True)))
    if isinstance(stats, pd.DataFrame) and not stats.empty:
        print(stats.head().to_string(index=False))


if __name__ == '__main__':
    main()
