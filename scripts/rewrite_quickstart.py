import json
from pathlib import Path


NB_PATH = Path('notebooks/risk_pipeline_quickstart.ipynb')


def nb_cell_markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def nb_cell_code(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    # Title
    cells.append(nb_cell_markdown(
        "# Risk Pipeline Quickstart (Simplified)\n\n"
        "Minimal, remote-only setup. Config is separate; pipeline runs once;\n"
        "then we display key summaries (models, risk bands, data layers, scored data).\n"
    ))

    # Remote-only robust import (auto clean reinstall on IndentationError)
    cells.append(nb_cell_code([
        "import sys, importlib, importlib.util, subprocess, site, shutil, pathlib\n",
        "\n",
        "def _force_reinstall_from_dev():\n",
        "    py = sys.executable\n",
        "    for name in ('risk-pipeline','risk_pipeline'):\n",
        "        try:\n",
        "            subprocess.run([py,'-m','pip','uninstall','-y',name], check=False, text=True)\n",
        "        except Exception:\n",
        "            pass\n",
        "    try:\n",
        "        for sp in site.getsitepackages()+[site.getusersitepackages()]:\n",
        "            p = pathlib.Path(sp)/'risk_pipeline'\n",
        "            if p.exists():\n",
        "                shutil.rmtree(p, ignore_errors=True)\n",
        "    except Exception:\n",
        "        pass\n",
        "    try:\n",
        "        subprocess.run([py,'-m','pip','cache','purge'], check=False, text=True)\n",
        "    except Exception:\n",
        "        pass\n",
        "    url = 'git+https://github.com/selimoksuz/risk-model-pipeline.git@development'\n",
        "    subprocess.run([py,'-m','pip','install','--no-cache-dir','--force-reinstall','-U',url], check=True, text=True)\n",
        "\n",
        "def _import_risk_pipeline():\n",
        "    try:\n",
        "        import risk_pipeline as rp\n",
        "        return rp\n",
        "    except Exception as e:\n",
        "        msg = repr(e)\n",
        "        print('risk_pipeline import failed:', msg)\n",
        "        if 'IndentationError' in msg or 'data_processor.py' in msg:\n",
        "            print('Attempting clean reinstall from development branch...')\n",
        "            _force_reinstall_from_dev()\n",
        "            import importlib as _il\n",
        "            return _il.import_module('risk_pipeline')\n",
        "        print('Install from development branch:')\n",
        "        print('  pip install -U \"git+https://github.com/selimoksuz/risk-model-pipeline.git@development\"')\n",
        "        raise\n",
        "\n",
        "TSFRESH_AVAILABLE = importlib.util.find_spec('tsfresh') is not None\n",
        "print('tsfresh available' if TSFRESH_AVAILABLE else 'tsfresh is not installed (fallback to aggregates).')\n",
        "risk_pipeline_module = _import_risk_pipeline()\n",
        "NOTEBOOK_CONTEXT = globals().setdefault('_NOTEBOOK_CONTEXT', {'data': {}, 'artifacts': {}, 'paths': {}, 'options': {}})\n",
        "\n",
    ]))

    # Data load and sample prep
    cells.append(nb_cell_code([
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "from risk_pipeline.data.sample import load_credit_risk_sample\n",
        "\n",
        "OUTPUT_DIR = Path('outputs/credit_risk_sample_notebook')\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "NOTEBOOK_CONTEXT['paths']['output'] = OUTPUT_DIR\n",
        "\n",
        "sample = load_credit_risk_sample()\n",
        "MIN_SAMPLE_SIZE = 50000\n",
        "CALIBRATION_SAMPLE_SIZE = 50000\n",
        "STAGE2_SAMPLE_SIZE = 50000\n",
        "RISK_BAND_SAMPLE_SIZE = 50000\n",
        "\n",
        "def _ensure_min_rows(frame: pd.DataFrame, target: int, seed: int = 42) -> pd.DataFrame:\n",
        "    if frame is None or target is None:\n",
        "        return frame\n",
        "    frame = frame.copy()\n",
        "    current = len(frame)\n",
        "    if current >= target:\n",
        "        return frame\n",
        "    multiplier, remainder = divmod(target, current)\n",
        "    pieces = [frame.copy() for _ in range(max(multiplier - 1, 0))]\n",
        "    if remainder:\n",
        "        pieces.append(frame.sample(remainder, replace=True, random_state=seed).reset_index(drop=True))\n",
        "    if pieces:\n",
        "        frame = pd.concat([frame, *pieces], ignore_index=True)\n",
        "    return frame\n",
        "\n",
        "def _harmonize_snapshot_month(frame: pd.DataFrame) -> pd.DataFrame:\n",
        "    if frame is not None and 'snapshot_month' in frame.columns:\n",
        "        try:\n",
        "            frame['snapshot_month'] = pd.to_datetime(frame['snapshot_month']).dt.to_period('M').dt.to_timestamp()\n",
        "        except Exception:\n",
        "            pass\n",
        "    return frame\n",
        "\n",
        "def _prep(frame: pd.DataFrame, target: int) -> pd.DataFrame:\n",
        "    return _harmonize_snapshot_month(_ensure_min_rows(frame, target))\n",
        "\n",
        "dev_df = _prep(sample.development, MIN_SAMPLE_SIZE)\n",
        "cal_long_df = _prep(sample.calibration_longrun, CALIBRATION_SAMPLE_SIZE)\n",
        "cal_recent_df = _prep(sample.calibration_recent, STAGE2_SAMPLE_SIZE)\n",
        "risk_band_df = _prep(sample.calibration_longrun, RISK_BAND_SAMPLE_SIZE)\n",
        "score_df = sample.scoring_future.copy()\n",
        "NOTEBOOK_CONTEXT['data'].update(dict(development=dev_df, calibration_longrun=cal_long_df, calibration_recent=cal_recent_df, risk_band_reference=risk_band_df, scoring=score_df))\n",
    ]))

    # Configure pipeline
    cells.append(nb_cell_markdown("## Configure Pipeline\n"))
    cells.append(nb_cell_code([
        "from risk_pipeline.core.config import Config\n",
        "from risk_pipeline.unified_pipeline import UnifiedRiskPipeline\n",
        "\n",
        "cfg_params = {\n",
        "    'target_column': 'target',\n",
        "    'id_column': 'customer_id',\n",
        "    'time_column': 'app_dt',\n",
        "    'create_test_split': True, 'group_split_by_id': True, 'stratify_test': True,\n",
        "    'train_ratio': 0.8, 'test_ratio': 0.2, 'oot_ratio': 0.0, 'oot_months': 3,\n",
        "    'output_folder': str(NOTEBOOK_CONTEXT['paths']['output']),\n",
        "    'output_excel_path': str(NOTEBOOK_CONTEXT['paths']['output'] / 'risk_pipeline_report.xlsx'),\n",
        "    'enable_tsfresh_features': False,\n",
        "    'selection_steps': ['univariate','psi','vif','correlation','iv','boruta','stepwise'],\n",
        "    'min_univariate_gini': 0.05, 'psi_threshold': 0.25, 'monthly_psi_threshold': 0.15, 'oot_psi_threshold': 0.25, 'test_psi_threshold': 0.25, 'psi_bucketing_mode_woe': 'woe_bucket', 'psi_bucketing_mode_raw': 'quantile', 'psi_bins': 10, 'psi_compare_axes': ['monthly','oot','test'], 'psi_decision': 'any', 'vif_threshold': 5.0, 'correlation_threshold': 0.9, 'iv_threshold': 0.02, 'stepwise_method': 'forward', 'stepwise_max_features': 25,\n",
        "    'algorithms': ['logistic','lightgbm','xgboost','catboost','randomforest','extratrees','woe_boost','woe_li','shao','xbooster'],\n",
        "    'model_selection_method': 'gini_oot', 'model_stability_weight': 0.2, 'min_gini_threshold': 0.5, 'max_train_oot_gap': 0.03, 'use_optuna': True, 'hpo_trials': 1, 'hpo_timeout_sec': 1800,\n",
        "    'use_noise_sentinel': True, 'enable_dual': True, 'enable_woe_boost_scorecard': True, 'calculate_shap': True, 'enable_scoring': True, 'score_model_name': 'best', 'enable_stage2_calibration': True, 'stage2_adjustment': 'lower',\n",
        "    'n_risk_bands': 10, 'risk_band_method': 'pd_constraints', 'risk_band_min_bins': 7, 'risk_band_max_bins': 10, 'risk_band_hhi_threshold': 0.15, 'risk_band_binomial_pass_weight': 0.85,\n",
        "    'random_state': 42, 'n_jobs': -1, 'cv_enable': True, 'cv_folds': 5, 'early_stopping_rounds': 200,\n",
        "}\n",
        "cfg_field_names = set(Config.__dataclass_fields__.keys())\n",
        "supported_params = {k: v for k, v in cfg_params.items() if k in cfg_field_names}\n",
        "unsupported = sorted(set(cfg_params.keys()) - set(supported_params.keys()))\n",
        "if unsupported: print('[WARN] Ignored config params:', unsupported)\n",
        "cfg = Config(**supported_params)\n",
        "pipe = UnifiedRiskPipeline(cfg)\n",
        "results = {}\n",
    ]))

    # Run pipeline end-to-end
    cells.append(nb_cell_markdown("## Run Pipeline\n"))
    cells.append(nb_cell_code([
        "datasets = NOTEBOOK_CONTEXT['data']\n",
        "results = pipe.fit(\n",
        "    datasets['development'],\n",
        "    data_dictionary=getattr(load_credit_risk_sample(), 'data_dictionary', None),\n",
        "    calibration_df=datasets['calibration_longrun'],\n",
        "    stage2_df=datasets['calibration_recent'],\n",
        "    risk_band_df=datasets['risk_band_reference'],\n",
        "    score_df=datasets['scoring'],\n",
        ")\n",
        "reports = pipe.run_reporting(force=True)\n",
        "print('Excel path:', reports.get('excel_path'))\n",
    ]))

    # Model summary
    cells.append(nb_cell_markdown("## Model Summary\n"))
    cells.append(nb_cell_code([
        "from IPython.display import display\n",
        "ms = reports.get('models_summary')\n",
        "bm = reports.get('best_model')\n",
        "display(ms if ms is not None else 'models_summary not available')\n",
        "display(bm if bm is not None else 'best_model not available')\n",
    ]))

    # Risk bands summary
    cells.append(nb_cell_markdown("## Risk Bands Summary\n"))
    cells.append(nb_cell_code([
        "display(reports.get('risk_bands'))\n",
        "display(reports.get('band_metrics'))\n",
        "display(reports.get('risk_bands_summary'))\n",
    ]))

    # Data layers and RAW preprocess summary
    cells.append(nb_cell_markdown("## Data Layers Overview & RAW Preprocessing Summary\n"))
    cells.append(nb_cell_code([
        "display(pipe.reporter.reports_.get('data_layers_overview'))\n",
        "display(pipe.reporter.reports_.get('raw_preprocessing_summary'))\n",
    ]))

    # Scored data preview
    cells.append(nb_cell_markdown("## Scored Data Preview\n"))
    cells.append(nb_cell_code([
        "scored = reports.get('scoring').get('dataframe') if isinstance(reports.get('scoring'), dict) else None\n",
        "if scored is not None:\n",
        "    display(scored.head(10))\n",
        "else:\n",
        "    print('Scored data not available.')\n",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main():
    nb = build_notebook()
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Notebook rewritten with simplified design.')


if __name__ == '__main__':
    main()

