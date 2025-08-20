# risk-model-pipeline

Production-ready scaffold for a modular risk modelling pipeline (WOE → PSI → FS → Model → Calibration → Report).
This repo is structured for incremental development and easy Git usage.

The reference pipeline trains Logistic Regression, XGBoost, LightGBM and GAM models
with automatic hyper-parameter tuning (≤20 minutes per model) and performs feature
selection via the Boruta algorithm.

## Quickstart

```bash
# create and activate venv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# install
pip install -U pip
pip install -e .  # editable install

# run CLI
risk-pipeline --help

# or run the bundled executor (logs -> outputs/pipeline.log)
python scripts/executor.py
```

## Project layout
```
src/risk_pipeline/        # library code
  config/                 # pydantic schemas / defaults
  data/                   # loading / IO connectors
  features/               # WOE/PSI/feature engineering
  model/                  # training / calibration
  reporting/              # reports & artifacts
  cli.py                  # Typer CLI entry point
tests/                    # pytest-based tests
scripts/                  # runnable scripts (optional)
```

## Next steps
- Drop your existing modules into `src/risk_pipeline/` under the appropriate subpackage.
- Wire functions into the CLI commands in `cli.py`.
- Add tests in `tests/` as you go.


## Run the 16-parcel pipeline from CSV
```bash
risk-pipeline run16 --input-csv data/input.csv --id-col app_id --time-col app_dt --target-col target   --oot-months 3 --use-test-split False --output-folder outputs --output-excel model_report.xlsx
```
Artifacts (Excel + parquet/CSV) will be written under `outputs/`.
