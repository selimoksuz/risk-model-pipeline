"""Generate synthetic credit risk datasets with desired properties."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240923)

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def calibrate_probabilities(logits: np.ndarray, target_rate: float, iterations: int = 60) -> np.ndarray:
    low, high = -10.0, 10.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        rate = logistic(logits + mid).mean()
        if rate > target_rate:
            high = mid
        else:
            low = mid
    return logistic(logits + 0.5 * (low + high))

def assign_defaults(probs: np.ndarray, target_rate: float) -> np.ndarray:
    n = len(probs)
    n_defaults = int(round(target_rate * n))
    n_defaults = min(max(n_defaults, 1), n - 1)
    order = np.argsort(probs)[::-1]
    targets = np.zeros(n, dtype=int)
    targets[order[:n_defaults]] = 1
    return targets

def make_panel(
    months: Sequence[pd.Timestamp],
    sample_size: int,
    drift: Dict[str, Dict[pd.Timestamp, np.ndarray]],
    target_rate: float,
) -> pd.DataFrame:
    base_ids = np.arange(1, 601)
    bureau_base = RNG.normal(645, 32, size=base_ids.size)
    segment_base = RNG.choice(['Retail', 'SME', 'Corporate'], size=base_ids.size, p=[0.6, 0.3, 0.1])
    region_base = RNG.choice(['North', 'South', 'East', 'West'], size=base_ids.size, p=[0.3, 0.2, 0.25, 0.25])
    employment_base = RNG.choice(['Salaried', 'Self-Employed', 'Contract'], size=base_ids.size, p=[0.55, 0.25, 0.2])
    delinq_base = RNG.binomial(1, 0.16, size=base_ids.size)

    records = []
    per_month = max(80, sample_size // len(months))
    counter = 1
    for month_ts in months:
        month_index = (month_ts.year - 2020) * 12 + month_ts.month
        month_shift = drift.get('intercept', {}).get(month_ts, 0.0)
        channel_probs = drift.get('channel_probs', {}).get(month_ts, np.array([0.45, 0.35, 0.2]))
        psi_flag_prob = drift.get('psi_flag', {}).get(month_ts, 0.05)

        month_records = []
        logits = []

        for _ in range(per_month):
            app_id = RNG.choice(base_ids)
            bureau_score = bureau_base[app_id - 1] + RNG.normal(0, 4)
            utilization_ratio = np.clip(RNG.beta(2.5, 2.0) + delinq_base[app_id - 1] * 0.18 + RNG.normal(0, 0.04), 0, 1)
            credit_usage = np.clip(utilization_ratio * 0.9 + RNG.normal(0, 0.02), 0, 1)
            payment_income_ratio = np.clip(RNG.normal(0.24, 0.04) + (segment_base[app_id - 1] == 'SME') * 0.05, 0.05, 0.55)
            open_trades = max(0, int(RNG.normal(5, 1.8)))
            balance_to_limit = np.clip(utilization_ratio * 0.95 + RNG.normal(0, 0.015), 0, 1)
            monthly_spend = np.maximum(0, RNG.normal(1_250, 180) * (1 + 0.6 * utilization_ratio))

            channel = RNG.choice(['Branch', 'Online', 'Partner'], p=channel_probs)
            psi_flag = RNG.binomial(1, psi_flag_prob)
            low_iv_noise = RNG.uniform(0, 1)

            segment = segment_base[app_id - 1]
            region = region_base[app_id - 1]
            employment = employment_base[app_id - 1]
            delinq = delinq_base[app_id - 1]

            base_logit = (
                -7.2
                + 0.03 * (720 - bureau_score)
                + 2.2 * utilization_ratio
                + 1.8 * payment_income_ratio
                + 1.15 * delinq
                + 0.55 * (segment == 'SME')
                + 1.0 * (segment == 'Corporate')
                + 0.65 * (region == 'South')
                + 0.4 * (employment == 'Contract')
                + 0.95 * (channel == 'Partner')
                + 1.1 * psi_flag
                + month_shift
            )

            month_records.append({
                'app_id': f"A{app_id:04d}{month_index:02d}{counter:03d}",
                'customer_id': app_id,
                'app_dt': month_ts.strftime('%Y-%m-%d'),
                'snapshot_month': month_ts.strftime('%Y-%m'),
                'bureau_score': round(bureau_score, 0),
                'utilization_ratio': round(utilization_ratio, 3),
                'credit_usage_ratio': round(credit_usage, 3),
                'payment_income_ratio': round(payment_income_ratio, 3),
                'balance_to_limit': round(balance_to_limit, 3),
                'monthly_spend': round(monthly_spend, 2),
                'segment': segment,
                'region': region,
                'employment_type': employment,
                'delinquent_last_6m': delinq,
                'open_trades': open_trades,
                'channel_code': channel,
                'promo_flag': psi_flag,
                'noise_feature': round(low_iv_noise, 4),
            })
            logits.append(base_logit)
            counter += 1

        logits_arr = np.array(logits)
        calibrated_probs = calibrate_probabilities(logits_arr, target_rate=target_rate)
        targets = assign_defaults(calibrated_probs, target_rate)

        for rec, target in zip(month_records, targets):
            rec['target'] = int(target)
        records.extend(month_records)

    return pd.DataFrame(records)

def add_dictionary(columns: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for col, desc in columns.items():
        rows.append({'variable': col, 'description': desc, 'category': desc.split('-')[0].strip()})
    return pd.DataFrame(rows)

def main() -> None:
    output_dir = Path('examples/data/credit_risk_sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_months = pd.period_range('2023-01', '2023-09', freq='M').to_timestamp()
    longrun_months = pd.period_range('2022-06', '2022-12', freq='M').to_timestamp()
    recent_months = [pd.Timestamp('2023-10-01')]
    scoring_months = [pd.Timestamp('2023-11-01')]

    drift = {
        'intercept': {pd.Timestamp('2023-10-01'): 0.35, pd.Timestamp('2023-11-01'): 0.45},
        'channel_probs': {
            pd.Timestamp('2023-08-01'): np.array([0.2, 0.5, 0.3]),
            pd.Timestamp('2023-09-01'): np.array([0.2, 0.55, 0.25]),
            pd.Timestamp('2023-10-01'): np.array([0.15, 0.6, 0.25]),
            pd.Timestamp('2023-11-01'): np.array([0.1, 0.65, 0.25]),
        },
        'psi_flag': {
            pd.Timestamp('2023-08-01'): 0.25,
            pd.Timestamp('2023-09-01'): 0.3,
            pd.Timestamp('2023-10-01'): 0.4,
            pd.Timestamp('2023-11-01'): 0.45,
        },
    }

    development = make_panel(dev_months, sample_size=24000, drift=drift, target_rate=0.28)
    calibration = make_panel(longrun_months, sample_size=12000, drift={'intercept': {}}, target_rate=0.26)
    stage2 = make_panel(recent_months, sample_size=5000, drift=drift, target_rate=0.34)
    scoring = make_panel(scoring_months, sample_size=6000, drift=drift, target_rate=0.36)

    for df, name in [
        (development, 'development.csv'),
        (calibration, 'calibration_longrun.csv'),
        (stage2, 'calibration_recent.csv'),
        (scoring, 'scoring_future.csv'),
    ]:
        df.to_csv(output_dir / name, index=False)

    dictionary_columns = {
        'app_id': 'Metadata - Unique application identifier',
        'customer_id': 'Metadata - Underlying customer id for tsfresh grouping',
        'app_dt': 'Metadata - Application date',
        'snapshot_month': 'Metadata - Month bucket for reporting',
        'target': 'Target - Default flag (1 = default)',
        'bureau_score': 'Risk - Bureau credit score (higher is better)',
        'utilization_ratio': 'Risk - Revolving utilization ratio',
        'credit_usage_ratio': 'Risk - Correlated usage ratio',
        'payment_income_ratio': 'Financial - Payment to income ratio',
        'balance_to_limit': 'Risk - Balance to credit limit ratio',
        'monthly_spend': 'Behavioural - Recent monthly spend amount',
        'segment': 'Demographic - Customer segment',
        'region': 'Demographic - Region of the applicant',
        'employment_type': 'Demographic - Employment type',
        'delinquent_last_6m': 'Risk - Delinquency indicator last 6 months',
        'open_trades': 'Risk - Number of open trades',
        'channel_code': 'Operational - Acquisition channel (drift driver)',
        'promo_flag': 'Operational - Promotional campaign flag (high PSI)',
        'noise_feature': 'Diagnostic - Low IV random noise feature',
    }
    data_dictionary = add_dictionary(dictionary_columns)
    data_dictionary.to_csv(output_dir / 'data_dictionary.csv', index=False)

    print(f"Synthetic datasets written to {output_dir}")

if __name__ == '__main__':
    main()
