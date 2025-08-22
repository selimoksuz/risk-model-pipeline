# risk-model-pipeline

Production-ready scaffold for modular risk modelling (WOE -> PSI -> FS -> Model -> Calibration -> Report). Turkish-friendly logs and Excel report with rich sheets.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -e .
# Linux/macOS
source .venv/bin/activate
pip install -U pip
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

### monitor
Monitor production drift via WOE transformation, model scoring and PSI metrics.

Options:
- `baseline`: Reference CSV of past production scores/features
- `new`: New CSV to compare
- `mapping`: WOE mapping JSON path
- `final_vars`: Comma separated list of model variables
- `model`: Trained model path
- `--calibrator`: Optional calibrator path
- `--expected-model-type`: Fail if loaded model class mismatches

Example:
```bash
.venv\Scripts\python scripts/monitor_cli.py ^
  baseline.csv new.csv mapping.json "var1,var2" model.joblib ^
  --calibrator calib.pkl --expected-model-type LogisticRegression
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
