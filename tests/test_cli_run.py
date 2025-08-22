import pandas as pd
from pathlib import Path

from risk_pipeline.cli import run as cli_run


def test_cli_run_creates_metrics(tmp_path: Path):
    # minimal numeric-only dataset
    df = pd.DataFrame({
        'app_id': [1, 2, 3, 4, 5, 6],
        'app_dt': ['2024-01-01']*6,
        'target': [0, 1, 0, 1, 0, 1],
        'x1': [0.1, 0.3, -0.2, 1.1, 0.0, -0.5],
        'x2': [1.0, 0.0, 1.0, 0.5, 0.2, 0.8],
    })
    csv_path = tmp_path / 'in.csv'
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / 'artifacts'
    cli_run.callback(
        config_json=None,
        input_csv=str(csv_path),
        target_col='target',
        artifacts_dir=str(out_dir),
    )

    # metrics.json should be created by save_metrics
    assert (out_dir / 'metrics.json').exists()
