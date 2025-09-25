"""Packaged sample datasets for risk_pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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


def _credit_risk_root():
    return files(__name__).joinpath('credit_risk')


def _read_csv(name: str) -> pd.DataFrame:
    resource = _credit_risk_root().joinpath(name)
    with as_file(resource) as path:
        return pd.read_csv(path)


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


def load_credit_risk_sample() -> CreditRiskSample:
    return CreditRiskSample(
        development=_read_csv('development.csv'),
        calibration_longrun=_read_csv('calibration_longrun.csv'),
        calibration_recent=_read_csv('calibration_recent.csv'),
        scoring_future=_read_csv('scoring_future.csv'),
        data_dictionary=_read_csv('data_dictionary.csv'),
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
