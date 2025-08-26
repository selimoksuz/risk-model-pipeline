# Risk Model Pipeline

A comprehensive risk modeling pipeline for credit scoring with WOE transformation, model calibration, and advanced reporting capabilities.

## ✨ Key Features

- **Multiple Input Methods**: CSV/Parquet files or pandas DataFrames
- **6 ML Algorithms**: Logistic Regression, RandomForest, XGBoost, LightGBM, ExtraTrees, GAM
- **Advanced Preprocessing**: WOE transformation, feature selection, correlation clustering
- **Model Calibration**: Isotonic/sigmoid calibration support
- **Mixed Target Scoring**: Handles records with/without targets separately
- **Rich Reporting**: Excel reports with 15+ sheets including performance metrics, SHAP values
- **Production Ready**: CLI interface, comprehensive logging, PSI monitoring

## Installation

### Option 1: Install from GitHub
```bash
pip install git+https://github.com/selimoksuz/risk-model-pipeline.git
```

### Option 2: Clone and Install Locally
```bash
git clone https://github.com/selimoksuz/risk-model-pipeline.git
cd risk-model-pipeline

# Create virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS  
source .venv/bin/activate

# Install package
pip install -e .
```

## Commands

### run16
Train full pipeline from a CSV.

Key options:
- --input-csv: Path to input CSV (must include ID, time, target + features)
- --id-col: ID column (default: app_id)
- --time-col: Time column (default: app_dt)
- --target-col: Binary target {0,1} (default: target)
- --oot-months: OOT window in months (default: 3)
- --use-test-split/--no-use-test-split: make internal TEST from pre-OOT months
- --output-folder: output folder (default: outputs)
- --output-excel: Excel report name (default: model_report.xlsx)
- --log-file: optional custom log path; default is outputs/pipeline.log
- --calibration-data: external CSV/Parquet for calibration
- --calibration-method: isotonic|sigmoid (default: isotonic)
- --score-data: external CSV/Parquet for scoring (PSI + metrics if target exists)
- PSI/IV/Corr/FS knobs: --psi-threshold-feature, --psi-threshold-score, --rho-threshold, --vif-threshold, --iv-min, etc.

Example:
```bash
.venv\Scripts\risk-pipeline run16 ^
  --input-csv data\input.csv ^
  --use-test-split ^
  --calibration-data data\calibration.csv ^
  --score-data data\score.csv ^
  --output-folder outputs ^
  --output-excel model_report.xlsx
```

## Python/Notebook Usage

### Quick Start with DataFrames

```python
import pandas as pd
from risk_pipeline.utils.pipeline_runner import run_pipeline_from_dataframe

# Prepare your data
train_df = pd.DataFrame({
    'app_id': range(1000),
    'app_dt': pd.date_range('2024-01-01', periods=1000),
    'target': np.random.binomial(1, 0.2, 1000),  # Binary target
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.lognormal(10, 0.5, 1000),
    # ... more features
})

# Optional: Calibration data as DataFrame (no CSV needed!)
calibration_df = pd.DataFrame({
    'app_id': range(2000, 2200),
    'app_dt': pd.date_range('2024-06-01', periods=200),
    'target': np.random.binomial(1, 0.25, 200),
    # ... same features as training
})

# Run pipeline
results = run_pipeline_from_dataframe(
    df=train_df,
    calibration_df=calibration_df,  # Optional DataFrame calibration
    id_col="app_id",
    time_col="app_dt",
    target_col="target",
    hpo_trials=30,
    hpo_timeout_sec=300,
    output_folder="outputs",
    output_excel="model_report.xlsx"
)

print(f"Best Model: {results['best_model']}")
print(f"Selected Features: {results['final_features']}")
```

### Scoring New Data

```python
from risk_pipeline.utils.scoring import score_data
import joblib, json

# Load model artifacts
model = joblib.load('outputs/best_model_xxx.joblib')
woe_mapping = json.load(open('outputs/woe_mapping_xxx.json'))
final_features = json.load(open('outputs/final_vars_xxx.json'))

# Prepare scoring data (can have mixed targets)
scoring_df = pd.DataFrame({
    'app_id': range(5000, 5500),
    'app_dt': pd.date_range('2024-08-01', periods=500),
    'target': [np.nan] * 300 + list(np.random.binomial(1, 0.3, 200)),  # 60% without target
    # ... features
})

# Score
results = score_data(
    scoring_df=scoring_df,
    model=model,
    final_features=final_features,
    woe_mapping=woe_mapping,
    calibrator=None,  # Optional
    training_scores=None  # Optional for PSI
)

print(f"Scored: {results['n_total']} records")
print(f"With target: {results['n_with_target']} (performance metrics calculated)")
print(f"Without target: {results['n_without_target']} (scores only)")
if 'with_target' in results:
    print(f"AUC: {results['with_target']['auc']:.3f}")
```

### Advanced Examples

See [notebooks/usage_example.ipynb](notebooks/usage_example.ipynb) for:
- Model training without calibration
- Model training with calibration
- Scoring with pre-trained models
- Adding calibration to existing models
- Batch scoring for large datasets
- PSI monitoring over time
- Custom model integration

### score
Score an external dataset using the WOE mapping + final vars + best model. Reads F1 threshold from report if provided.

Options:
- --input-csv: CSV with features
- --woe-mapping: outputs/woe_mapping_<run_id>.json
- --final-vars-json: outputs/final_vars_<run_id>.json
- --model-path: outputs/best_model_<run_id>.joblib
- --report-xlsx: outputs/model_report.xlsx (to read thresholds F1)
- --output-csv: output path for scores.csv
- --calibrator-path: optional pickled calibrator

Example:
```bash
.venv\Scripts\risk-pipeline score ^
  --input-csv data\score.csv ^
  --woe-mapping outputs\woe_mapping_<run_id>.json ^
  --final-vars-json outputs\final_vars_<run_id>.json ^
  --model-path outputs\best_model_<run_id>.joblib ^
  --report-xlsx outputs\model_report.xlsx ^
  --output-csv outputs\scores.csv
```

## Report (Excel) sheets
- models_summary: model family metrics across splits
- best_model: selected best model row; best_name sheet has the name
- psi_summary, psi_dropped_features: PSI screening results
- ks_info_traincv, ks_info_test, ks_info_oot: KS bands/metrics
- final_vars, best_model_vars_df, best_model_woe_df: selections and WOE details
- woe_mapping: flattened WOE bins/groups
- thresholds: OOT-based F1-optimal threshold (threshold, precision, recall, f1)
- woe_bin_counts: per-variable WOE bin/group counts
- external_scores: scored external dataset (if provided)
- external_psi_features: Train vs external WOE-PSI per feature (if provided)
- external_psi_score/external_metrics: score PSI and AUC/Gini/KS (if target exists)
- run_meta: run configuration and meta

## Logs
- outputs/pipeline.log (UTF-8 BOM), Türkçe karakterler okunaklı; konsolda ASCII işaretler (>>, **, -).
- Özet loglar: WOE bin/grup sayıları, corr cluster temsilcisi, FS sonrası sayı, F1-eşik bilgisi.

## Notebook
Bkz: `notebooks/GettingStarted.ipynb` — örnek uçtan uca kullanım (veri üret, pipeline, sheet’leri inceleme, skor üretme).

## Sample data
```bash
.venv\Scripts\python scripts\make_sample_csv.py
# writes: data/input.csv, data/calibration.csv, data/score.csv
```

## Quickstart

Windows:
- `python -m venv .venv`
- `.venv\Scripts\python -m pip install -U pip`
- `.venv\Scripts\pip install -e .[dev]`
- `.venv\Scripts\python scripts\make_sample_csv.py`
- `.venv\Scripts\risk-pipeline run16 --input-csv data\input.csv --use-test-split --output-folder outputs --output-excel model_report.xlsx`

Linux/macOS:
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -U pip && pip install -e .[dev]`
- `python scripts/make_sample_csv.py`
- `risk-pipeline run16 --input-csv data/input.csv --use-test-split --output-folder outputs --output-excel model_report.xlsx`

Notes:
- Generated outputs under `outputs/` are git-ignored. Avoid committing artifacts.
- For faster runs, tune `--hpo-trials`, `--hpo-timeout-sec`, `--shap-sample`.

## Development
- Tests: `pytest`
- Lint/format: `flake8`, `black`, `isort`
- Pre-commit hooks: `pre-commit install`

## CI/CD
- A minimal GitHub Actions workflow is provided under `.github/workflows/ci.yml` to run lint and tests on push/PR.
