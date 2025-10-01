"""Packaged sample datasets for risk_pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:  # Python >=3.9
    from importlib import resources  # type: ignore

    files = resources.files  # type: ignore[attr-defined]
    as_file = resources.as_file  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - fallback for Python <3.9
    import importlib_resources as resources  # type: ignore

    files = resources.files  # type: ignore[attr-defined]
    as_file = resources.as_file  # type: ignore[attr-defined]

__all__ = [
    "CreditRiskSample",
    "load_credit_risk_sample",
    "copy_credit_risk_sample",
]


DERIVED_FEATURES = [
    {
        "variable": "utilization_ratio_twin",
        "description": "Derived - Highly correlated variant of utilization_ratio for multicollinearity diagnostics",
        "category": "Derived",
    },
    {
        "variable": "random_noise_signal",
        "description": "Derived - Synthetic low-predictive variable for univariate Gini demonstrations",
        "category": "Derived",
    },
    {
        "variable": "portfolio_shift_score",
        "description": "Derived - Synthetic drift driver with deliberate PSI shift across datasets",
        "category": "Derived",
    },
]


def _credit_risk_root():
    return files(__name__).joinpath('credit_risk')


def _read_csv(name: str) -> pd.DataFrame:
    resource = _credit_risk_root().joinpath(name)
    with as_file(resource) as path:
        return pd.read_csv(path)


def _ensure_min_rows(frame: pd.DataFrame, target: int, *, seed: int = 42) -> pd.DataFrame:
    if frame is None or target is None or target <= 0:
        return frame
    frame = frame.copy().reset_index(drop=True)
    current = len(frame)
    if current >= target:
        return frame
    multiplier, remainder = divmod(target, current)
    pieces = [frame.copy() for _ in range(max(multiplier - 1, 0))]
    if remainder:
        pieces.append(frame.sample(n=remainder, replace=True, random_state=seed).reset_index(drop=True))
    if pieces:
        frame = pd.concat([frame, *pieces], ignore_index=True)
    return frame.iloc[:target].reset_index(drop=True)


def _add_demo_features(df: pd.DataFrame, *, rng: np.random.Generator, shift: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    if 'utilization_ratio_twin' not in df.columns:
        if 'utilization_ratio' in df.columns:
            base = df['utilization_ratio'].to_numpy(dtype=float, copy=True)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            base = df[numeric_cols[0]].to_numpy(dtype=float, copy=True) if len(numeric_cols) else np.zeros(len(df))
        base_std = float(np.nanstd(base))
        if not np.isfinite(base_std) or base_std <= 0:
            base_std = 1.0
        correlated = base * 0.97 + rng.normal(loc=0.0, scale=0.01 * base_std, size=len(df))
        if np.isnan(base).any():
            correlated = np.where(np.isnan(base), np.nan, correlated)
        df['utilization_ratio_twin'] = correlated

    if 'random_noise_signal' not in df.columns:
        df['random_noise_signal'] = rng.normal(loc=0.0, scale=1.0, size=len(df))

    if 'portfolio_shift_score' not in df.columns:
        df['portfolio_shift_score'] = rng.normal(loc=shift, scale=1.0, size=len(df))

    return df


def _augment_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['variable', 'description', 'category']
    if dictionary is None or dictionary.empty:
        dictionary = pd.DataFrame(columns=required_cols)
    else:
        dictionary = dictionary.copy()
        for col in required_cols:
            if col not in dictionary.columns:
                dictionary[col] = ''

    existing = set(dictionary['variable'].astype(str)) if 'variable' in dictionary.columns else set()
    additions = [entry for entry in DERIVED_FEATURES if entry['variable'] not in existing]
    if additions:
        dictionary = pd.concat([dictionary, pd.DataFrame(additions)], ignore_index=True)
    return dictionary


def _augment_credit_risk_sample(sample: "CreditRiskSample", *, min_development_rows: int, random_state: int) -> "CreditRiskSample":
    dev = _ensure_min_rows(sample.development, min_development_rows, seed=random_state)
    cal_long = sample.calibration_longrun.copy()
    cal_recent = sample.calibration_recent.copy()
    scoring = sample.scoring_future.copy()

    dev = _add_demo_features(dev, rng=np.random.default_rng(random_state), shift=0.0)
    cal_long = _add_demo_features(cal_long, rng=np.random.default_rng(random_state + 1), shift=0.5)
    cal_recent = _add_demo_features(cal_recent, rng=np.random.default_rng(random_state + 2), shift=1.0)
    scoring = _add_demo_features(scoring, rng=np.random.default_rng(random_state + 3), shift=1.8)

    dictionary = _augment_dictionary(sample.data_dictionary)

    return CreditRiskSample(
        development=dev,
        calibration_longrun=cal_long,
        calibration_recent=cal_recent,
        scoring_future=scoring,
        data_dictionary=dictionary,
    )


@dataclass(frozen=True)
class CreditRiskSample:
    development: pd.DataFrame
    calibration_longrun: pd.DataFrame
    calibration_recent: pd.DataFrame
    scoring_future: pd.DataFrame
    data_dictionary: pd.DataFrame

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            'development': self.development,
            'calibration_longrun': self.calibration_longrun,
            'calibration_recent': self.calibration_recent,
            'scoring_future': self.scoring_future,
            'data_dictionary': self.data_dictionary,
        }


def load_credit_risk_sample(*, min_development_rows: int = 50000, random_state: int = 42) -> CreditRiskSample:
    calibration_recent = _read_csv('calibration_recent.csv')
    for target_col in ('target', 'bad_flag'):
        if target_col in calibration_recent.columns:
            calibration_recent = calibration_recent.drop(columns=target_col)

    raw_sample = CreditRiskSample(
        development=_read_csv('development.csv'),
        calibration_longrun=_read_csv('calibration_longrun.csv'),
        calibration_recent=calibration_recent,
        scoring_future=_read_csv('scoring_future.csv'),
        data_dictionary=_read_csv('data_dictionary.csv'),
    )

    return _augment_credit_risk_sample(
        raw_sample,
        min_development_rows=min_development_rows,
        random_state=random_state,
    )


def copy_credit_risk_sample(destination: Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    root = _credit_risk_root()
    for resource in root.iterdir():
        if resource.name.endswith('.csv'):
            with as_file(resource) as path:
                target = destination / resource.name
                target.write_bytes(Path(path).read_bytes())

    return destination
