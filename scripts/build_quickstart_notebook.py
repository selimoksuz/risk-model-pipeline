from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Credit Risk Pipeline Quickstart

This notebook exercises the **Unified Risk Pipeline** end-to-end using the bundled synthetic dataset.
The sample includes stratified monthly observations, calibration hold-outs, stage-2 data and a future
scoring batch so that every major pipeline stage can be validated quickly."""))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Imports and paths

All sample inputs live under `examples/data/credit_risk_sample`."""))

cells.append(nbf.v4.new_code_cell(
"from pathlib import Path\nimport pandas as pd\n\nfrom risk_pipeline.core.config import Config\nfrom risk_pipeline.unified_pipeline import UnifiedRiskPipeline\n\nBASE_DIR = Path('examples/data/credit_risk_sample')\nDEV_PATH = BASE_DIR / 'development.csv'\nCAL_LONG_PATH = BASE_DIR / 'calibration_longrun.csv'\nCAL_RECENT_PATH = BASE_DIR / 'calibration_recent.csv'\nSCORE_PATH = BASE_DIR / 'scoring_future.csv'\nDICT_PATH = BASE_DIR / 'data_dictionary.csv'\nOUTPUT_DIR = Path('output/credit_risk_sample_notebook')\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\ndev_df = pd.read_csv(DEV_PATH)\ncal_long_df = pd.read_csv(CAL_LONG_PATH)\ncal_recent_df = pd.read_csv(CAL_RECENT_PATH)\nscore_df = pd.read_csv(SCORE_PATH)\ndata_dictionary = pd.read_csv(DICT_PATH)\n\ndev_df.head()"))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Quick sanity checks"""))

cells.append(nbf.v4.new_code_cell(
"dev_df['target'].value_counts(normalize=True).rename('default_rate')"))

cells.append(nbf.v4.new_code_cell(
"dev_df.groupby('snapshot_month')['target'].mean().rename('monthly_default_rate')"))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. Configure the pipeline

The configuration below enables tsfresh feature generation, dual modelling flow (WOE + raw),
calibration stages and risk band optimisation while remaining light enough for a laptop."""))

config_source = "cfg = Config(\n    target_column='target',\n    id_column='customer_id',\n    time_column='app_dt',\n    create_test_split=True,\n    stratify_test=True,\n    oot_months=2,\n    enable_dual=True,\n    enable_tsfresh_features=True,\n    enable_scoring=True,\n    enable_stage2_calibration=True,\n    output_folder=str(OUTPUT_DIR),\n    n_risk_bands=6,\n    risk_band_method='quantile',\n    max_psi=0.6,\n    selection_steps=['psi', 'univariate', 'iv', 'correlation', 'stepwise'],\n    algorithms=['logistic', 'lightgbm'],\n    use_optuna=False,\n    calculate_shap=False,\n    use_noise_sentinel=False,\n    random_state=42,\n)\ncfg.model_type = ['LogisticRegression', 'LightGBM']"

cells.append(nbf.v4.new_code_cell(config_source))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. Run the unified pipeline"""))

run_source = "pipe = UnifiedRiskPipeline(cfg)\nresults = pipe.fit(\n    dev_df,\n    data_dictionary=data_dictionary,\n    calibration_df=cal_long_df,\n    stage2_df=cal_recent_df,\n    score_df=score_df,\n)"

cells.append(nbf.v4.new_code_cell(run_source))

cells.append(nbf.v4.new_markdown_cell(
"""## 5. Inspect key outputs"""))

cells.append(nbf.v4.new_code_cell(
"best_model = results.get('best_model_name')\nmodel_scores = results.get('model_results', {}).get('scores', {})\nprint(f'Best model: {best_model}')\npd.DataFrame(model_scores).T"))

cells.append(nbf.v4.new_code_cell(
"feature_report = pipe.reporter.reports_.get('features')\nfeature_report.head() if feature_report is not None else 'No feature report available.'"))

cells.append(nbf.v4.new_code_cell(
"calibration_report = pipe.reporter.reports_.get('calibration')\ncalibration_report"))

cells.append(nbf.v4.new_code_cell(
"risk_bands = pipe.reporter.reports_.get('risk_bands_summary', {})\nrisk_bands"))

cells.append(nbf.v4.new_markdown_cell(
"""## 6. Generated files"""))

cells.append(nbf.v4.new_code_cell(
"sorted(p.relative_to(OUTPUT_DIR.parent) for p in OUTPUT_DIR.glob('**/*') if p.is_file())"))

cells.append(nbf.v4.new_markdown_cell(
"""## 7. Automating via script

`examples/quickstart_demo.py` mirrors the steps above so the flow can be validated headless
(e.g. in CI pipelines)."""))

nb['cells'] = cells

notebook_path = Path('notebooks') / 'risk_pipeline_quickstart.ipynb'
nbf.write(nb, notebook_path)
