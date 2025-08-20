"""Unified pipeline runner writing logs to a file.

This script reads the sample CSV under ``data/input.csv`` and executes the
16-step risk model pipeline.  All logs produced by the pipeline are written to
``outputs/pipeline.log`` in addition to being printed to the console.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

# Allow running without installing the package by adding ``src`` to ``sys.path``
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from risk_pipeline.pipeline16 import Config, RiskModelPipeline, Orchestrator


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Clean previous outputs to avoid accumulation
    for f in out_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    df = pd.read_csv(base_dir / "data" / "input.csv")

    cfg = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        use_test_split=True,            # enable internal TEST split (80/20 by months)
        oot_window_months=3,            # last 3 months as OOT
        output_folder=str(out_dir),
        output_excel_path="model_report.xlsx",
        psi_verbose=True,
        write_parquet=False,
        write_csv=False,
        run_id="latest",                # overwrite artifacts based on run_id-stamped names
        log_file=str(out_dir / "pipeline.log"),
        orchestrator=Orchestrator(),
    )

    pipe = RiskModelPipeline(cfg)
    pipe.run(df)
    print(
        f"Done. Best={pipe.best_model_name_} | Reports -> {cfg.output_folder} | Log -> {cfg.log_file}"
    )


if __name__ == "__main__":
    main()

