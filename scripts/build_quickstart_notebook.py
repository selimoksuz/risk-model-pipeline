from pathlib import Path
from textwrap import dedent

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Credit Risk Pipeline Quickstart

This notebook exercises the **Unified Risk Pipeline** end-to-end using the bundled synthetic dataset.
The sample includes stratified monthly observations, calibration hold-outs, stage-2 data and a future
scoring batch so that every major pipeline step can be validated quickly."""))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Imports and sample loader

The dataset ships with the package under `risk_pipeline.data.sample`."""))

cells.append(nbf.v4.new_code_cell(
"from pathlib import Path\nimport pandas as pd\n\nfrom risk_pipeline.core.config import Config\nfrom risk_pipeline.unified_pipeline import UnifiedRiskPipeline\nfrom risk_pipeline.data.sample import load_credit_risk_sample\n\nsample = load_credit_risk_sample()\nOUTPUT_DIR = Path('output/credit_risk_sample_notebook')\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\ndev_df = sample.development\ncal_long_df = sample.calibration_longrun\ncal_recent_df = sample.calibration_recent\nscore_df = sample.scoring_future\ndata_dictionary = sample.data_dictionary\n\ndev_df.head()"))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Quick sanity checks"""))

cells.append(nbf.v4.new_code_cell(
"dev_df['target'].value_counts(normalize=True).rename('default_rate')"))

cells.append(nbf.v4.new_code_cell(
"dev_df.groupby('snapshot_month')['target'].mean().rename('monthly_default_rate')"))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. Configure the pipeline

The configuration below enables dual modelling (raw + WoE), Optuna (single rapid trial), balanced model selection with stability guard rails,
noise sentinel monitoring, SHAP explainability, the WoE-LI and Shao logistic challengers, and the PD-constrained risk band optimizer.
Train/Test/OOT ratios and all threshold knobs (PSI/IV/Gini/Correlation) are explicit so the notebook mirrors production-ready configuration files."""))

config_source = dedent("""\
cfg = Config(
    target_column='target',
    id_column='customer_id',
    time_column='app_dt',
    create_test_split=True,
    use_test_split=True,
    train_ratio=0.6,
    test_ratio=0.2,
    oot_ratio=0.2,
    stratify_test=True,
    oot_months=2,
    enable_dual=True,
    enable_tsfresh_features=True,
    enable_scoring=True,
    enable_stage2_calibration=True,
    output_folder=str(OUTPUT_DIR),
    selection_steps=['psi', 'univariate', 'iv', 'correlation', 'boruta', 'stepwise'],
    algorithms=[
        'logistic', 'gam', 'catboost', 'lightgbm', 'xgboost',
        'randomforest', 'extratrees', 'woe_boost', 'woe_li', 'shao', 'xbooster',
    ],
    model_selection_method='balanced',
    model_stability_weight=0.25,
    min_gini_threshold=0.45,
    max_train_oot_gap=0.08,
    psi_threshold=0.25,
    iv_threshold=0.02,
    univariate_gini_threshold=0.05,
    correlation_threshold=0.95,
    vif_threshold=5.0,
    woe_binning_strategy='iv_optimal',
    use_optuna=True,
    n_trials=1,
    optuna_timeout=120,
    hpo_method='optuna',
    hpo_trials=1,
    hpo_timeout_sec=120,
    use_noise_sentinel=True,
    calculate_shap=True,
    shap_sample_size=500,
    risk_band_method='pd_constraints',
    n_risk_bands=8,
    risk_band_min_bins=7,
    risk_band_max_bins=10,
    risk_band_micro_bins=1000,
    risk_band_min_weight=0.05,
    risk_band_max_weight=0.30,
    risk_band_hhi_threshold=0.15,
    risk_band_binomial_pass_weight=0.85,
    risk_band_alpha=0.05,
    risk_band_pd_dr_tolerance=1e-4,
    risk_band_max_iterations=100,
    risk_band_max_phase_iterations=50,
    risk_band_early_stop_rounds=10,
    calibration_stage1_method='isotonic',
    calibration_stage2_method='lower_mean',
    random_state=42,
)
cfg.model_type = 'all'
""")

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
"""## 7. XBooster scorecard"""))

scorecard_code = "xbooster_artifacts = results.get('model_results', {}).get('interpretability', {}).get('XBooster', {})\nif isinstance(xbooster_artifacts, dict):\n    scorecard_df = xbooster_artifacts.get('scorecard_points')\n    warnings = xbooster_artifacts.get('warnings')\n    display_obj = scorecard_df.head() if hasattr(scorecard_df, 'head') else xbooster_artifacts\nelse:\n    warnings = None\n    display_obj = 'No XBooster artifacts available.'\nprint('Warnings:', warnings if warnings else 'None')\ndisplay_obj"

cells.append(nbf.v4.new_code_cell(scorecard_code))

cells.append(nbf.v4.new_markdown_cell(
"""## 8. Automating via script

`examples/quickstart_demo.py` mirrors the steps above so the flow can be validated headless
(e.g. in CI pipelines)."""))

nb['cells'] = cells

notebook_path = Path('notebooks') / 'risk_pipeline_quickstart.ipynb'
nbf.write(nb, notebook_path)
