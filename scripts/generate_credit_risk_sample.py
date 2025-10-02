"""Generate synthetic credit risk datasets with rolling coverage windows."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240923)

WINDOW_MONTHS = 12
BASE_CUSTOMER_COUNT = 4500


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_probabilities(logits: np.ndarray, target_rate: float, iterations: int = 80) -> np.ndarray:
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


def month_sequence(start: str, end: str) -> Sequence[pd.Timestamp]:
    return pd.period_range(start, end, freq="M").to_timestamp()


def clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def make_panel(
    months: Sequence[pd.Timestamp],
    sample_size: int,
    target_rate: float,
    *,
    include_target: bool,
    stage_label: str,
    window_months: int = WINDOW_MONTHS,
) -> pd.DataFrame:
    months = [pd.Timestamp(m) for m in months]
    if not months:
        raise ValueError("months sequence cannot be empty")
    per_month = max(80, int(np.ceil(sample_size / len(months))))

    base_ids = np.arange(1, BASE_CUSTOMER_COUNT + 1)
    bureau_base = RNG.normal(645, 32, size=base_ids.size)
    segment_base = RNG.choice(["Retail", "SME", "Corporate"], size=base_ids.size, p=[0.58, 0.32, 0.10])
    region_base = RNG.choice(["North", "South", "East", "West"], size=base_ids.size, p=[0.3, 0.2, 0.25, 0.25])
    employment_base = RNG.choice(["Salaried", "Self-Employed", "Contract"], size=base_ids.size, p=[0.57, 0.25, 0.18])
    delinq_base = RNG.binomial(1, 0.14, size=base_ids.size)

    history_months: Dict[int, set[int]] = defaultdict(set)
    application_counts: Dict[int, int] = defaultdict(int)

    active_ids: list[int] = []
    new_id_idx = 0

    def pick_customer(month_pos: int) -> int:
        nonlocal new_id_idx
        allow_new = new_id_idx < len(base_ids)
        if not active_ids and allow_new:
            cid = int(base_ids[new_id_idx])
            new_id_idx += 1
            active_ids.append(cid)
            return cid
        exploration_prob = max(0.12, 0.45 * np.exp(-month_pos / 12.0))
        if allow_new and RNG.random() < exploration_prob:
            cid = int(base_ids[new_id_idx])
            new_id_idx += 1
            active_ids.append(cid)
            return cid
        weights = np.array([len(history_months[cid]) + 1 for cid in active_ids], dtype=float)
        if weights.sum() == 0:
            return int(RNG.choice(active_ids))
        weights /= weights.sum()
        return int(RNG.choice(active_ids, p=weights))

    records: list[Dict[str, object]] = []
    logits: list[float] = []

    first_month_ord = months[0].to_period("M").ordinal

    for month_pos, month_ts in enumerate(months):
        month_period = month_ts.to_period("M")
        month_ord = month_period.ordinal
        seasonal = np.sin(2 * np.pi * ((month_period.month - 1) / 12.0))
        trend = (month_pos + 1) / len(months)
        month_shift = -0.3 + 0.45 * seasonal + 0.35 * trend

        channel_base = np.array([0.48, 0.37, 0.15])
        channel_base += np.array([0.05 * seasonal, 0.07 * trend, -0.12 * trend])
        channel_probs = channel_base.clip(0.05, 0.9)
        channel_probs /= channel_probs.sum()

        psi_flag_prob = clip_probability(0.08 + 0.3 * trend + 0.12 * seasonal)

        for _ in range(per_month):
            customer_id = pick_customer(month_pos)
            base_index = customer_id - 1

            history_set = history_months[customer_id]
            recent_cutoff = month_ord - (window_months - 1)
            recent_months = {m for m in history_set if m >= recent_cutoff}
            months_observed = len(recent_months)
            months_missing = max(0, window_months - months_observed)
            coverage_ratio = months_observed / window_months if window_months else 1.0
            partial_flag = int(months_observed < window_months)
            span_months = month_ord - min(history_set) if history_set else 0
            recent_activity = sum(1 for m in history_set if m >= month_ord - 2)

            bureau_score = bureau_base[base_index] + RNG.normal(0, 5)
            utilization_ratio = np.clip(
                RNG.beta(2.8, 1.9) + delinq_base[base_index] * 0.22 + RNG.normal(0, 0.05),
                0,
                1,
            )
            credit_usage = np.clip(utilization_ratio * 0.92 + RNG.normal(0, 0.025), 0, 1)
            payment_income = np.clip(
                RNG.normal(0.24, 0.045) + (segment_base[base_index] == "SME") * 0.05,
                0.05,
                0.6,
            )
            balance_to_limit = np.clip(utilization_ratio * 0.96 + RNG.normal(0, 0.02), 0, 1)
            monthly_spend = np.maximum(0, RNG.normal(1325, 210) * (1 + 0.55 * utilization_ratio))

            channel = RNG.choice(["Branch", "Online", "Partner"], p=channel_probs)
            promo_flag = RNG.binomial(1, psi_flag_prob)
            noise_feature = RNG.uniform(0, 1)

            segment = segment_base[base_index]
            region = region_base[base_index]
            employment = employment_base[base_index]
            delinquent = delinq_base[base_index]
            open_trades = max(0, int(RNG.normal(6, 2.1)))

            raw_logit = (
                -7.5
                + 0.028 * (720 - bureau_score)
                + 2.4 * utilization_ratio
                + 1.85 * payment_income
                + 1.2 * delinquent
                + 0.6 * (segment == "SME")
                + 1.05 * (segment == "Corporate")
                + 0.55 * (region == "South")
                + 0.42 * (employment == "Contract")
                + 0.75 * (channel == "Partner")
                + 1.25 * promo_flag
                + 0.9 * (1 - coverage_ratio)
                + 0.35 * partial_flag
                + 0.2 * recent_activity
                + month_shift
            )

            app_counter = application_counts[customer_id] + 1
            application_counts[customer_id] = app_counter

            records.append(
                {
                    "app_id": f"A{customer_id:05d}{month_ord:04d}{app_counter:04d}",
                    "customer_id": customer_id,
                    "app_dt": month_ts.strftime("%Y-%m-%d"),
                    "snapshot_month": month_ts.strftime("%Y-%m"),
                    "bureau_score": round(bureau_score, 0),
                    "utilization_ratio": round(utilization_ratio, 3),
                    "credit_usage_ratio": round(credit_usage, 3),
                    "payment_income_ratio": round(payment_income, 3),
                    "balance_to_limit": round(balance_to_limit, 3),
                    "monthly_spend": round(monthly_spend, 2),
                    "segment": segment,
                    "region": region,
                    "employment_type": employment,
                    "delinquent_last_6m": delinquent,
                    "open_trades": open_trades,
                    "channel_code": channel,
                    "promo_flag": promo_flag,
                    "noise_feature": round(noise_feature, 4),
                    "history_months_observed": months_observed,
                    "history_months_missing": months_missing,
                    "history_coverage_ratio": round(coverage_ratio, 3),
                    "history_partial_flag": partial_flag,
                    "recent_app_count_3m": recent_activity,
                    "history_span_months": span_months,
                    "vintage_month_index": month_ord - first_month_ord,
                    "stage_label": stage_label,
                }
            )
            logits.append(raw_logit)

            history_set.add(month_ord)

    df = pd.DataFrame(records)
    if len(df) > sample_size:
        df = df.iloc[:sample_size].copy()
        logits = logits[:sample_size]

    if include_target:
        probabilities = calibrate_probabilities(np.array(logits), target_rate=target_rate)
        targets = assign_defaults(probabilities, target_rate)
        df.insert(4, "target", targets)

    return df.reset_index(drop=True)


def build_data_dictionary() -> pd.DataFrame:
    dictionary_columns = {
        "app_id": "Metadata - Unique application identifier",
        "customer_id": "Metadata - Underlying customer id for tsfresh grouping",
        "app_dt": "Metadata - Application date",
        "snapshot_month": "Metadata - Month bucket for reporting",
        "target": "Target - Default flag (1 = default)",
        "bureau_score": "Risk - Bureau credit score (higher is better)",
        "utilization_ratio": "Risk - Revolving utilisation ratio",
        "credit_usage_ratio": "Risk - Correlated utilisation proxy",
        "payment_income_ratio": "Financial - Payment to income ratio",
        "balance_to_limit": "Risk - Balance to credit limit ratio",
        "monthly_spend": "Behavioural - Recent monthly spend amount",
        "segment": "Demographic - Customer segment",
        "region": "Demographic - Region of the applicant",
        "employment_type": "Demographic - Employment type",
        "delinquent_last_6m": "Risk - Delinquency indicator last 6 months",
        "open_trades": "Risk - Number of open trades",
        "channel_code": "Operational - Acquisition channel (drift driver)",
        "promo_flag": "Operational - Promotional campaign flag (high PSI feature)",
        "noise_feature": "Diagnostic - Low predictive noise variable",
        "history_months_observed": "Derived - Unique months observed in trailing 12 month window",
        "history_months_missing": "Derived - Missing months within trailing 12 month window",
        "history_coverage_ratio": "Derived - Coverage ratio over trailing 12 month window",
        "history_partial_flag": "Derived - Indicator of incomplete 12 month window",
        "recent_app_count_3m": "Derived - Application count in trailing 3 months",
        "history_span_months": "Derived - Months since customer first observed",
        "vintage_month_index": "Temporal - Month index relative to panel start",
        "stage_label": "Metadata - Dataset source label",
    }
    rows = []
    for variable, description in dictionary_columns.items():
        category = description.split("-")[0].strip() if "-" in description else "Other"
        rows.append({"variable": variable, "description": description, "category": category})
    return pd.DataFrame(rows)


def build_datasets() -> Dict[str, pd.DataFrame]:
    development_months = month_sequence("2021-01", "2023-12")
    calibration_months = month_sequence("2023-01", "2023-12")
    recent_months = month_sequence("2024-01", "2024-12")
    scoring_months = month_sequence("2025-01", "2025-12")

    development = make_panel(
        development_months,
        sample_size=72_000,
        target_rate=0.24,
        include_target=True,
        stage_label="development",
    )
    calibration = make_panel(
        calibration_months,
        sample_size=54_000,
        target_rate=0.26,
        include_target=True,
        stage_label="calibration_longrun",
    )
    stage2 = make_panel(
        recent_months,
        sample_size=54_000,
        target_rate=0.28,
        include_target=False,
        stage_label="calibration_recent",
    )
    scoring = make_panel(
        scoring_months,
        sample_size=54_000,
        target_rate=0.30,
        include_target=False,
        stage_label="scoring_future",
    )

    return {
        "development": development,
        "calibration_longrun": calibration,
        "calibration_recent": stage2,
        "scoring_future": scoring,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dirs = [
        root / "examples" / "data" / "credit_risk_sample",
        root / "src" / "risk_pipeline" / "data" / "sample" / "credit_risk",
    ]

    datasets = build_datasets()
    dictionary = build_data_dictionary()

    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in datasets.items():
            frame.to_csv(directory / f"{name}.csv", index=False)
        dictionary.to_csv(directory / "data_dictionary.csv", index=False)

    print("Synthetic datasets written to:")
    for directory in output_dirs:
        print(f" - {directory.resolve()}")


if __name__ == "__main__":
    main()
