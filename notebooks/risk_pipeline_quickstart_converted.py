# Auto-generated from notebook by simple converter
try:
    from IPython import get_ipython
except Exception:
    def get_ipython():
        class Dummy:
            def run_line_magic(self, *a, **k):
                return None
        return Dummy()

# In [3]

import importlib
import importlib.util
import sys
from pathlib import Path


def _locate_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / 'src' / 'risk_pipeline').exists():
        return cwd
    candidate = cwd / 'risk-model-pipeline-dev'
    if (candidate / 'src' / 'risk_pipeline').exists():
        return candidate
    for parent in cwd.parents:
        maybe = parent / 'risk-model-pipeline-dev'
        if (maybe / 'src' / 'risk_pipeline').exists():
            return maybe
    return cwd


PROJECT_ROOT = _locate_project_root()
SRC_PATH = PROJECT_ROOT / 'src'
PACKAGE_PATH = SRC_PATH / 'risk_pipeline'
MODULE_INIT = PACKAGE_PATH / '__init__.py'
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _load_local_package():
    if not MODULE_INIT.exists():
        return None
    spec = importlib.util.spec_from_file_location('risk_pipeline', MODULE_INIT)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules['risk_pipeline'] = module
        spec.loader.exec_module(module)
        return module
    return None


def ensure_risk_pipeline():
    module = _load_local_package()
    if module is None:
        module = importlib.import_module('risk_pipeline')
    version = getattr(module, '__version__', 'local-dev')
    location = Path(getattr(module, '__file__', 'unknown')).resolve()
    print(f'risk-pipeline loaded (version {version}, path={location})')
    return module


TSFRESH_AVAILABLE = importlib.util.find_spec('tsfresh') is not None
if TSFRESH_AVAILABLE:
    print('tsfresh available (advanced time-series features can be enabled via config).')
else:
    print('tsfresh is not installed; pipeline will fall back to lightweight aggregate features when needed.')

risk_pipeline_module = ensure_risk_pipeline()
NOTEBOOK_FLAGS = globals().setdefault('_NOTEBOOK_FLAGS', {})
NOTEBOOK_FLAGS['tsfresh_available'] = TSFRESH_AVAILABLE
NOTEBOOK_FLAGS['project_root'] = PROJECT_ROOT
NOTEBOOK_FLAGS['src_path'] = SRC_PATH

# In [4]

from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.data.sample import load_credit_risk_sample

NOTEBOOK_CONTEXT = globals().setdefault('_NOTEBOOK_CONTEXT', {'data': {}, 'artifacts': {}, 'paths': {}, 'options': {}})

# ensure pipeline placeholders exist for diagnostic cells during step-by-step execution
if 'pipe' not in globals():
    pipe = None
if 'results' not in globals():
    results = {}
if 'full_results' not in globals():
    full_results = {}

sample = load_credit_risk_sample()
OUTPUT_DIR = Path('output/credit_risk_sample_notebook')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLE_SIZE = 50000
CALIBRATION_SAMPLE_SIZE = 50000
STAGE2_SAMPLE_SIZE = 50000
RISK_BAND_SAMPLE_SIZE = 50000
random_seed = 42


def _ensure_min_rows(frame: pd.DataFrame, target: int, *, seed: int = 42) -> pd.DataFrame:
    if frame is None or target is None:
        return frame
    frame = frame.copy()
    current = len(frame)
    if current >= target:
        return frame
    multiplier, remainder = divmod(target, current)
    pieces = [frame.copy() for _ in range(max(multiplier - 1, 0))]
    if remainder:
        pieces.append(frame.sample(remainder, replace=True, random_state=seed).reset_index(drop=True))
    if pieces:
        frame = pd.concat([frame, *pieces], ignore_index=True)
    return frame


def _harmonize_snapshot_month(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is not None and 'snapshot_month' in frame.columns:
        try:
            frame['snapshot_month'] = pd.to_datetime(frame['snapshot_month']).dt.to_period('M').dt.to_timestamp()
        except Exception:
            pass
    return frame


def _prepare_with_target_size(frame: pd.DataFrame, target: int) -> pd.DataFrame:
    adjusted = _ensure_min_rows(frame, target, seed=random_seed)
    return _harmonize_snapshot_month(adjusted)


dev_df = _prepare_with_target_size(sample.development, MIN_SAMPLE_SIZE)
cal_long_df = _prepare_with_target_size(sample.calibration_longrun, CALIBRATION_SAMPLE_SIZE)
cal_recent_df = _prepare_with_target_size(sample.calibration_recent, STAGE2_SAMPLE_SIZE)
risk_band_df = _prepare_with_target_size(sample.calibration_longrun, RISK_BAND_SAMPLE_SIZE)
score_df = sample.scoring_future.copy()
data_dictionary = sample.data_dictionary.copy() if hasattr(sample.data_dictionary, 'copy') else sample.data_dictionary

datasets = {
    'development': dev_df,
    'calibration_longrun': cal_long_df,
    'calibration_recent': cal_recent_df,
    'risk_band_reference': risk_band_df,
    'scoring': score_df,
    'dictionary': data_dictionary,
}
NOTEBOOK_CONTEXT['data'].update(datasets)
NOTEBOOK_CONTEXT['paths']['output'] = OUTPUT_DIR

# ensure demo missingness as before
_exclusion_cols = {'target', 'snapshot_month', 'customer_id', 'app_id', 'application_id', 'app_dt', 'decision_dt'}
_rng = np.random.default_rng(random_seed)


def _inject_demo_missing(frame, rate=0.01, max_features=5):
    if frame.isna().sum().sum() > 0:
        return frame
    numeric_candidates = [
        col for col in frame.select_dtypes(include=['number']).columns
        if col.lower() not in _exclusion_cols and not col.lower().endswith('_id')
    ]
    if not numeric_candidates:
        return frame
    for col in numeric_candidates[:max_features]:
        mask = _rng.random(len(frame)) < rate
        if mask.any():
            frame.loc[mask, col] = np.nan
    return frame

for key in ('development', 'calibration_longrun', 'calibration_recent', 'risk_band_reference', 'scoring'):
    _inject_demo_missing(datasets[key])

dataset_overview = []
for name in ('development', 'calibration_longrun', 'calibration_recent', 'risk_band_reference', 'scoring'):
    df = NOTEBOOK_CONTEXT['data'].get(name)
    if isinstance(df, pd.DataFrame):
        overview = {
            'dataset': name,
            'rows': len(df),
            'target_non_null': int(df['target'].notna().sum()) if 'target' in df.columns else None,
            'unique_customers': df['customer_id'].nunique() if 'customer_id' in df.columns else None,
        }
        dataset_overview.append(overview)
if dataset_overview:
    display(pd.DataFrame(dataset_overview))

dev_df.head()

# In [5]
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.data.sample import load_credit_risk_sample

NOTEBOOK_CONTEXT = globals().setdefault('_NOTEBOOK_CONTEXT', {'data': {}, 'artifacts': {}, 'paths': {}, 'options': {}})
def _current_config():
    for name in ('pipe', 'full_pipe', 'raw_pipe'):
        candidate = globals().get(name)
        cfg = getattr(candidate, 'config', None) if candidate is not None else None
        if cfg is not None:
            return cfg
    return globals().get('cfg')

def _config_flag(name, default=False):
    cfg_obj = _current_config()
    if cfg_obj is None:
        return bool(default)
    return bool(getattr(cfg_obj, name, default))

def _get_pipeline_context():
    context = globals().get('_NOTEBOOK_CONTEXT') or {}
    pipe_candidates = [
        globals().get('pipe'),
        globals().get('full_pipe'),
        globals().get('raw_pipe'),
    ]
    pipe = next((p for p in pipe_candidates if p is not None), None)
    results = globals().get('results')
    if not isinstance(results, dict):
        results = {}
    if not results and pipe is not None:
        results = getattr(pipe, 'results_', {}) or {}
    if not results and isinstance(globals().get('full_results'), dict):
        results = globals()['full_results']
    if pipe is not None:
        signature = _config_signature(getattr(pipe, 'config', None))
        options = context.setdefault('options', {})
        previous = options.get('config_signature')
        if signature != previous:
            options['config_signature'] = signature
            context['artifacts'] = {}
            results = {}
            globals()['results'] = results
            if hasattr(pipe, 'results_'):
                pipe.results_ = {}
    globals()['results'] = results
    return pipe, results if isinstance(results, dict) else {}

def _store_artifact(name, value):
    context = globals().get('_NOTEBOOK_CONTEXT') or {}
    context.setdefault('artifacts', {})[name] = value
    return value

def _artifact_available(value):
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True




def _materialize_model_registry(model_results=None, force=False):
    context = globals().get('_NOTEBOOK_CONTEXT') or {}
    artifacts = context.setdefault('artifacts', {})
    if not force:
        cached = artifacts.get('model_registry_df')
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
    if model_results is None:
        stored = artifacts.get('model_results')
        if not isinstance(stored, dict):
            stored = globals().get('results', {}).get('model_results') if isinstance(globals().get('results'), dict) else None
        model_results = stored or {}
    if not isinstance(model_results, dict):
        model_results = {}

    def _gather(key):
        sources = [model_results, globals().get('results'), artifacts, context.get('artifacts', {})]
        for source in sources:
            if isinstance(source, dict) and source.get(key):
                return source.get(key)
        return None

    registry_df = pd.DataFrame()
    registry = model_results.get('model_registry') if isinstance(model_results, dict) else None
    if isinstance(registry, pd.DataFrame):
        registry_df = registry.copy()
    elif registry:
        try:
            registry_df = pd.DataFrame(registry)
        except Exception:
            registry_df = pd.DataFrame()

    flow_scores = {}
    scores_registry = _gather('model_scores_registry')
    if isinstance(scores_registry, dict):
        for flow_label, score_map in scores_registry.items():
            if isinstance(score_map, dict):
                existing = flow_scores.setdefault(flow_label, {})
                for model_name, metrics in score_map.items():
                    if isinstance(model_name, str):
                        existing[model_name] = metrics

    base_scores = model_results.get('scores') if isinstance(model_results, dict) else {}
    if isinstance(base_scores, dict) and base_scores:
        flow_scores.setdefault('active', {}).update(base_scores)

    flow_rows = []
    for flow_label, score_map in flow_scores.items():
        for model_name, metrics in (score_map or {}).items():
            row = {'model_name': model_name, 'flow': flow_label}
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(key, str):
                        row[key] = value
            flow_rows.append(row)

    if flow_rows:
        flow_df = pd.DataFrame(flow_rows)
        if registry_df.empty:
            registry_df = flow_df
        else:
            registry_df = pd.concat([registry_df, flow_df], ignore_index=True, sort=False)

    if registry_df.empty:
        registry_df = pd.DataFrame(columns=['model_name'])
    else:
        if 'model_name' not in registry_df.columns:
            if 'name' in registry_df.columns:
                registry_df = registry_df.rename(columns={'name': 'model_name'})
            else:
                registry_df = registry_df.reset_index().rename(columns={'index': 'model_name'})
        mode_hint = None
        if isinstance(model_results, dict):
            for key in ('mode', 'best_model_mode', 'active_model_mode', 'best_mode', 'chosen_flow_mode'):
                value = model_results.get(key)
                if value:
                    mode_hint = value
                    break
        if 'mode' not in registry_df.columns:
            registry_df['mode'] = mode_hint
        else:
            registry_df['mode'] = registry_df['mode'].fillna(mode_hint)
        if 'flow' not in registry_df.columns:
            registry_df['flow'] = mode_hint or 'active'
        if isinstance(flow_scores, dict) and flow_scores:
            registry_df['flow'] = registry_df['flow'].fillna(registry_df['mode'])

        def _score_lookup(model_name, key):
            entry = base_scores.get(model_name) if isinstance(base_scores, dict) else None
            if entry is None and isinstance(flow_scores, dict):
                for score_map in flow_scores.values():
                    if isinstance(score_map, dict) and model_name in score_map:
                        cand = score_map[model_name]
                        if isinstance(cand, dict):
                            entry = cand
                            break
            return entry.get(key) if isinstance(entry, dict) else None

        for key in ('oot_auc', 'test_auc', 'train_auc', 'oot_gini', 'test_gini', 'train_gini'):
            if key not in registry_df.columns:
                registry_df[key] = registry_df['model_name'].map(lambda name: _score_lookup(name, key))
        sort_cols = [col for col in ('flow', 'mode', 'oot_auc', 'test_auc', 'train_auc') if col in registry_df.columns]
        if sort_cols:
            ascending = [True] + [True] + [False] * (len(sort_cols) - 2)
            registry_df = registry_df.sort_values(sort_cols, ascending=ascending, ignore_index=True)
        registry_df = registry_df.reset_index(drop=True)

    artifacts['model_registry_df'] = registry_df
    globals().setdefault('results', {}).setdefault('model_registry_df', registry_df)
    return registry_df


def _config_signature(cfg):
    if cfg is None:
        return None
    watched = {
        'enable_tsfresh_features': getattr(cfg, 'enable_tsfresh_features', None),
        'enable_tsfresh_rolling': getattr(cfg, 'enable_tsfresh_rolling', None),
        'tsfresh_window_months': getattr(cfg, 'tsfresh_window_months', None),
        'tsfresh_min_events': getattr(cfg, 'tsfresh_min_events', None),
        'tsfresh_min_unique_months': getattr(cfg, 'tsfresh_min_unique_months', None),
        'tsfresh_min_coverage_ratio': getattr(cfg, 'tsfresh_min_coverage_ratio', None),
        'tsfresh_include_current_record': getattr(cfg, 'tsfresh_include_current_record', None),
        'tsfresh_feature_set': getattr(cfg, 'tsfresh_feature_set', None),
        'tsfresh_custom_fc_parameters': bool(getattr(cfg, 'tsfresh_custom_fc_parameters', None)),
        'tsfresh_n_jobs': getattr(cfg, 'tsfresh_n_jobs', None),
        'tsfresh_use_multiprocessing': getattr(cfg, 'tsfresh_use_multiprocessing', None),
        'enable_stage2_calibration': getattr(cfg, 'enable_stage2_calibration', None),
        'enable_scoring': getattr(cfg, 'enable_scoring', None),
        'optimize_risk_bands': getattr(cfg, 'optimize_risk_bands', None),
    }
    return tuple(sorted(watched.items()))

def _update_results(results_ref, **artifacts):
    if not isinstance(results_ref, dict):
        return results_ref
    results_ref.update(artifacts)
    globals()['results'] = results_ref
    return results_ref

def _ensure_dev_df():
    context = globals().get('_NOTEBOOK_CONTEXT') or {}
    data = context.get('data', {})
    df = data.get('development')
    if df is None:
        df = globals().get('dev_df')
    return df

def _ensure_processed(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('processed_data')
        if cached is None:
            cached = pipe.results_.get('processed_data')
        if cached is not None:
            _store_artifact('processed_data', cached)
            return cached
    source_df = _ensure_dev_df()
    if source_df is None:
        raise RuntimeError('Development dataframe is not loaded yet. Run the data preparation cell first.')
    processed = pipe.run_process(source_df, create_map=True, include_noise=False, force=bool(force))
    _update_results(results_ref, processed_data=processed)
    _store_artifact('processed_data', processed)
    return processed

def _ensure_splits(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('splits')
        if cached is None:
            cached = pipe.results_.get('splits')
        if cached is not None:
            _store_artifact('splits', cached)
            return cached
    processed = _ensure_processed(force=False)
    splits = pipe.run_split(processed, force=True)
    _update_results(results_ref, splits=splits)
    _store_artifact('splits', splits)
    return splits

def _ensure_woe(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('woe_results')
        if cached is None:
            cached = pipe.results_.get('woe_results')
        if cached is not None:
            _store_artifact('woe_results', cached)
            return cached
    splits = _ensure_splits(force=False)
    if splits is None:
        raise RuntimeError('Splits are unavailable; run the split cell first.')
    woe_results = pipe.run_woe(splits=splits, force=True)
    _update_results(results_ref, woe_results=woe_results)
    _store_artifact('woe_results', woe_results)
    return woe_results

def _ensure_selection(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('selection_results')
        if cached is None:
            cached = pipe.results_.get('selection_results')
        if _artifact_available(cached):
            _store_artifact('selection_results', cached)
            return cached
    splits = _ensure_splits(force=False)
    woe_results = _ensure_woe(force=False)
    selection_mode = 'WOE' if getattr(pipe.config, 'enable_woe', True) else 'RAW'
    selection_results = pipe.run_selection(
        mode=selection_mode,
        splits=splits,
        woe_results=woe_results,
        force=True,
    )
    selected = selection_results.get('selected_features', [])
    _update_results(results_ref, selection_results=selection_results, selected_features=selected)
    _store_artifact('selection_results', selection_results)
    return selection_results

def _ensure_model_results(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('model_results')
        if cached is None:
            cached = pipe.results_.get('model_results')
        if _artifact_available(cached):
            pipe.selected_features_ = cached.get('selected_features', getattr(pipe, 'selected_features_', []))
            _store_artifact('model_results', cached)
            registry_df = _materialize_model_registry(cached, force=True)
            _update_results(results_ref, model_registry_df=registry_df)
            return cached
    selection_results = _ensure_selection(force=False) or {}
    splits = _ensure_splits(force=False)
    dual_enabled = getattr(pipe.config, 'enable_dual_pipeline', getattr(pipe.config, 'enable_dual', False))

    def _best_auc(result):
        if not isinstance(result, dict):
            return float('-inf')
        name = result.get('best_model_name')
        scores = result.get('scores') or {}
        entry = scores.get(name) if name else None
        if not isinstance(entry, dict):
            return float('-inf')
        return entry.get('oot_auc') or entry.get('test_auc') or entry.get('train_auc') or float('-inf')

    if dual_enabled:
        woe_results = pipe.run_modeling(
            mode='WOE',
            splits=splits,
            selection_results=selection_results if (selection_results.get('mode') if isinstance(selection_results, dict) else 'WOE') == 'WOE' else None,
            force=True,
        )
        raw_results = pipe.run_modeling(
            mode='RAW',
            splits=splits,
            selection_results=None,
            force=True,
        )
        flows = {'WOE': woe_results, 'RAW': raw_results}
        best_mode = max(flows.keys(), key=lambda mode: _best_auc(flows[mode]))
        best_results = pipe.run_modeling(mode=best_mode, splits=splits, selection_results=None, force=False)
        dual_registry = {
            mode: {
                'best_model_name': res.get('best_model_name'),
                'best_auc': _best_auc(res),
                'scores': res.get('scores', {}),
            }
            for mode, res in flows.items()
        }
        model_results = dict(best_results)
        model_results['dual_registry'] = dual_registry
        model_results['best_mode'] = best_mode
        model_results['best_model_mode'] = best_results.get('mode', best_mode)
        model_results['best_auc'] = _best_auc(best_results)
        results_ref['model_results_WOE'] = woe_results
        results_ref['model_results_RAW'] = raw_results
    else:
        selection_mode = selection_results.get('mode') if isinstance(selection_results, dict) else 'WOE'
        selection_mode = selection_mode or ('WOE' if getattr(pipe.config, 'enable_woe', True) else 'RAW')
        model_results = pipe.run_modeling(
            mode=selection_mode,
            splits=splits,
            selection_results=selection_results,
            force=True,
        )

    best_selection = pipe.results_.get('selection_results')
    if isinstance(best_selection, dict):
        _update_results(results_ref, selection_results=best_selection)
    registry_df = _materialize_model_registry(model_results, force=True)
    _update_results(results_ref, model_results=model_results, model_registry_df=registry_df)
    _store_artifact('model_results', model_results)
    return model_results
def _ensure_stage1(force=False):
    if not _config_flag('enable_stage2_calibration', False):
        return None
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('calibration_stage1')
        if cached is None:
            cached = pipe.results_.get('calibration_stage1')
        if _artifact_available(cached):
            _store_artifact('calibration_stage1', cached)
            return cached
    model_results = _ensure_model_results(force=False)
    calibration_df = NOTEBOOK_CONTEXT.get('data', {}).get('calibration_longrun')
    if calibration_df is None:
        calibration_df = NOTEBOOK_CONTEXT.get('data', {}).get('development')
    stage1 = pipe.run_stage1_calibration(model_results=model_results, calibration_df=calibration_df, force=True)
    _update_results(results_ref, calibration_stage1=stage1)
    _store_artifact('calibration_stage1', stage1)
    return stage1

def _ensure_stage2(force=False):
    if not _config_flag('enable_stage2_calibration', False):
        return None
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('calibration_stage2')
        if cached is None:
            cached = pipe.results_.get('calibration_stage2')
        if _artifact_available(cached):
            _store_artifact('calibration_stage2', cached)
            return cached
    stage1 = _ensure_stage1(force=False)
    if not stage1:
        print('Stage-2 calibration skipped: Stage-1 results unavailable.')
        return None
    recent_df = NOTEBOOK_CONTEXT.get('data', {}).get('calibration_recent')
    if recent_df is None:
        print('Stage-2 calibration skipped: recent dataset not loaded.')
        return None
    try:
        stage2 = pipe.run_stage2_calibration(stage1_results=stage1, recent_df=recent_df, force=True)
    except Exception as exc:
        print(f'Stage-2 calibration failed: {exc}')
        return None
    _update_results(results_ref, calibration_stage2=stage2)
    _store_artifact('calibration_stage2', stage2)
    return stage2
def _ensure_risk_bands(force=False):
    if not _config_flag('optimize_risk_bands', True):
        return {}
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('risk_bands')
        if cached is None:
            cached = pipe.results_.get('risk_bands')
        if _artifact_available(cached):
            _store_artifact('risk_bands', cached)
            return cached
    stage1 = _ensure_stage1(force=False)
    stage2 = _ensure_stage2(force=False)
    if not stage2 and stage1:
        print('Risk band optimisation: Stage-2 results unavailable; using Stage-1 calibration output.')
        stage2 = stage1
    if not stage2:
        print('Risk band optimisation skipped: calibration results unavailable.')
        return {}
    splits = _ensure_splits(force=False)
    raw_override = NOTEBOOK_CONTEXT.get('data', {}).get('risk_band_reference')
    processed_override = pipe.data_.get('risk_band_reference') if hasattr(pipe, 'data_') else None
    if processed_override is None and raw_override is not None:
        try:
            processed_override = pipe._process_data(raw_override, create_map=False, include_noise=False)
        except Exception:
            processed_override = raw_override
    if processed_override is None:
        fallback_raw = NOTEBOOK_CONTEXT.get('data', {}).get('development')
        if fallback_raw is not None:
            try:
                processed_override = pipe._process_data(fallback_raw, create_map=False, include_noise=False)
            except Exception:
                processed_override = fallback_raw
    if processed_override is not None:
        override_rows = len(processed_override)
    else:
        override_rows = 0
    risk_band_df = processed_override
    if risk_band_df is None:
        risk_band_df = NOTEBOOK_CONTEXT.get('data', {}).get('development')
    bands = pipe.run_risk_bands(stage1_results=stage1, stage2_results=stage2, splits=splits, data_override=risk_band_df, force=True)
    if isinstance(bands, dict):
        bands.setdefault('override_rows', override_rows)
    _update_results(results_ref, risk_bands=bands)
    _store_artifact('risk_bands', bands)
    return bands
def _ensure_scoring(force=False):
    if not _config_flag('enable_scoring', False):
        return {}
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('scoring_output')
        if cached is None:
            cached = pipe.results_.get('scoring_output')
        if _artifact_available(cached):
            _store_artifact('scoring_output', cached)
            return cached
    score_df = NOTEBOOK_CONTEXT.get('data', {}).get('scoring')
    if score_df is None:
        score_df = globals().get('score_df')
    if score_df is None:
        raise RuntimeError('Scoring dataset is not loaded.')
    stage2 = _ensure_stage2(force=False)
    selection = _ensure_selection(force=False)
    woe_results = _ensure_woe(force=False)
    model_results = _ensure_model_results(force=False)
    splits = _ensure_splits(force=False)
    scoring_output = pipe.run_scoring(
        score_df,
        stage2_results=stage2,
        selection_results=selection,
        woe_results=woe_results,
        model_results=model_results,
        splits=splits,
        force=True,
    )
    _update_results(results_ref, scoring_output=scoring_output)
    _store_artifact('scoring_output', scoring_output)
    return scoring_output

def _ensure_reports(force=False):
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        return None
    if not force:
        cached = results_ref.get('reports')
        if cached is None:
            cached = pipe.results_.get('reports')
        if _artifact_available(cached):
            _store_artifact('reports', cached)
            return cached
    reports = pipe.run_reporting(force=True)
    _update_results(results_ref, reports=reports)
    _store_artifact('reports', reports)
    return reports

# In [7]

import importlib

import risk_pipeline.core.feature_selector_enhanced as fs_module
import risk_pipeline.core.config as config_module
import risk_pipeline.unified_pipeline as pipeline_module

AdvancedFeatureSelector = importlib.reload(fs_module).AdvancedFeatureSelector
Config = importlib.reload(config_module).Config
UnifiedRiskPipeline = importlib.reload(pipeline_module).UnifiedRiskPipeline

cfg_params = {
    # Core identifiers
    'target_column': 'target',
    'id_column': 'customer_id',
    'time_column': 'app_dt',

    # Split configuration
    'create_test_split': True,
    'stratify_test': True,
    'train_ratio': 0.8,
    'test_ratio': 0.2,
    'oot_ratio': 0.0,
    'oot_months': 3,

    # Output controls
    'output_folder': str(NOTEBOOK_CONTEXT['paths']['output']),
    'output_excel_path': str(NOTEBOOK_CONTEXT['paths']['output'] / 'risk_pipeline_report.xlsx'),

    # TSFresh controls (auto-disabled if package missing)
    'enable_tsfresh_features': False,
    'tsfresh_feature_set': 'efficient',
    'tsfresh_n_jobs': 4,
    'enable_tsfresh_rolling': False,
    'tsfresh_window_months': 12,
    'tsfresh_min_events': 1,
    'tsfresh_min_unique_months': 1,
    'tsfresh_min_coverage_ratio': 1.0,
    'tsfresh_include_current_record': False,

    # Feature selection strategy
    'selection_steps': [
        'univariate',
        'psi',
        'vif',
        'correlation',
        'iv',
        'boruta',
        'stepwise',
    ],
    'min_univariate_gini': 0.05,
    'psi_threshold': 0.25,
    'monthly_psi_threshold': 0.15,
    'oot_psi_threshold': 0.25,
    'vif_threshold': 5.0,
    'correlation_threshold': 0.9,
    'iv_threshold': 0.02,
    'stepwise_method': 'forward',
    'stepwise_max_features': 25,

    # Model training preferences
    'algorithms': [
        'logistic',
        'lightgbm',
        'xgboost',
        'catboost',
        'randomforest',
        'extratrees',
        'woe_boost',
        'woe_li',
        'shao',
        'xbooster',
    ],
    'model_selection_method': 'gini_oot',
    'model_stability_weight': 0.2,
    'min_gini_threshold': 0.5,
    'max_train_oot_gap': 0.03,
    'use_optuna': True,
    'hpo_trials': 1,
    'hpo_timeout_sec': 1800,

    # Diagnostics & toggles
    'use_noise_sentinel': True,
    'enable_dual': True,
    'enable_woe_boost_scorecard': True,
    'calculate_shap': True,
    'enable_scoring': True,
    'score_model_name': 'best',
    'enable_stage2_calibration': True,

    # Risk band settings
    'n_risk_bands': 10,
    'risk_band_method': 'pd_constraints',
    'risk_band_min_bins': 7,
    'risk_band_max_bins': 10,
    'risk_band_hhi_threshold': 0.15,
    'risk_band_binomial_pass_weight': 0.85,

    # Runtime controls
    'random_state': 42,
    'n_jobs': -1,
}

if not NOTEBOOK_FLAGS.get('tsfresh_available', False):
    print('Notebook config: tsfresh features disabled automatically (package not detected).')

cfg_field_names = set(Config.__dataclass_fields__.keys())
supported_params = {k: v for k, v in cfg_params.items() if k in cfg_field_names}
unsupported = sorted(set(cfg_params.keys()) - set(supported_params.keys()))
if unsupported:
    print(f"[WARN] Config ignores unsupported parameters: {unsupported}")

cfg = Config(**supported_params)
pipe = UnifiedRiskPipeline(cfg)
results = {}

# In [9]
pipe, results_ref = _get_pipeline_context()
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'enable_tsfresh_features', False):
    print('TSFresh feature mining disabled via config; skipping processing cell.')
else:
    if pipe is None:
        raise RuntimeError('Pipeline instance is not initialized yet. Run the configuration cell first.')
    processed = _ensure_processed(force=False)
    _update_results(results_ref, processed_data=processed)
    results = results_ref
    print(f"Processed feature space: {processed.shape[1]} columns")

    tsfresh_meta = pipe.data_.get('tsfresh_metadata') if pipe is not None else None
    if isinstance(tsfresh_meta, pd.DataFrame) and not tsfresh_meta.empty:
        display(tsfresh_meta.head())
    else:
        flag = 'disabled via config' if cfg_local is not None and not getattr(cfg_local, 'enable_tsfresh_features', False) else 'not generated'
        print(f'No TSFresh features were generated ({flag}).')

# In [11]

pipe, results_ref = _get_pipeline_context()
if pipe is None:
    raise RuntimeError('Pipeline instance is not initialized yet. Run the configuration cell first.')
processed = _ensure_processed(force=False)
splits = _ensure_splits(force=True)
_update_results(results_ref, processed_data=processed, splits=splits)
results = results_ref

raw_layers = pipe.results_.get('raw_numeric_layers', {})
print(f"Identified numeric features: {len(pipe.data_.get('numeric_features', []))}")
if raw_layers:
    train_raw = raw_layers.get('train_raw_prepped')
    if train_raw is not None:
        display(train_raw[pipe.data_.get('numeric_features', [])].head())
else:
    print('No numeric preprocessing layer was created.')

impute_stats = getattr(pipe.data_processor, 'imputation_stats_', {})
if impute_stats:
    display(pd.DataFrame(impute_stats).T.head())

# Summarise configuration choices for quick inspection
config_summary = pd.DataFrame([
    ("Target column", cfg.target_column),
    ("ID column", cfg.id_column),
    ("Time column", cfg.time_column),
    ("Train/Test/OOT split", f"{cfg.train_ratio:.0%}/{cfg.test_ratio:.0%}/{cfg.oot_ratio:.0%}"),
    ("OOT holdout months", cfg.oot_months),
    ("Risk bands", f"{cfg.n_risk_bands} (method={cfg.risk_band_method})"),
    ("Calibration chain", f"{cfg.calibration_stage1_method} -> {cfg.calibration_stage2_method}"),
], columns=["Parameter", "Configured value"])
display(config_summary)

flag_toggles = pd.DataFrame({
    "Feature": [
        "Dual RAW+WOE flow",
        "TSFresh feature mining",
        "Scoring on hold-out data",
        "Stage 2 calibration",
        "Optuna HPO",
        "Noise sentinel",
        "SHAP importance",
    ],
    "Enabled": [
        getattr(cfg, 'enable_dual', False),
        getattr(cfg, 'enable_tsfresh_features', False),
        getattr(cfg, 'enable_scoring', False),
        getattr(cfg, 'enable_stage2_calibration', False),
        getattr(cfg, 'use_optuna', False),
        getattr(cfg, 'use_noise_sentinel', False),
        getattr(cfg, 'calculate_shap', False),
    ],
})
flag_toggles['Enabled'] = flag_toggles['Enabled'].map({True: 'Yes', False: 'No'})
display(flag_toggles)

thresholds = pd.DataFrame({
    "Threshold": [
        "PSI",
        "IV",
        "Univariate Gini",
        "Correlation ceiling",
        "VIF ceiling",
        "|Train-OOT| Gini gap",
    ],
    "Value": [
        cfg.psi_threshold,
        cfg.iv_threshold,
        cfg.min_univariate_gini,
        cfg.correlation_threshold,
        cfg.vif_threshold,
        cfg.max_train_oot_gap,
    ],
})
display(thresholds)

selection_order = pd.DataFrame({"Selection step": cfg.selection_steps})
selection_order.index = selection_order.index + 1
display(selection_order)

algorithms_df = pd.DataFrame({"Algorithm": cfg.algorithms})
algorithms_df.index = algorithms_df.index + 1
display(algorithms_df)

# In [13]

pipe, results_ref = _get_pipeline_context()
if pipe is None:
    print('WOE transformation skipped: pipeline instance not available yet.')
else:
    splits = _ensure_splits(force=False)
    woe_results = pipe.run_woe(splits=splits, force=True)
    _update_results(results_ref, splits=splits, woe_results=woe_results)
    results = results_ref
    woe_values = woe_results.get('woe_values', {})
    print(f"WOE computed for {len(woe_values)} features.")

# In [15]

pipe, results_ref = _get_pipeline_context()
if pipe is None:
    print('Feature selection skipped: pipeline instance not available yet.')
else:
    splits = _ensure_splits(force=False)
    woe_results = _ensure_woe(force=False)
    selection_mode = 'WOE' if getattr(pipe.config, 'enable_woe', True) else 'RAW'
    selection_results = pipe.run_selection(
        mode=selection_mode,
        splits=splits,
        woe_results=woe_results,
        force=True,
    )
    selected = selection_results.get('selected_features', [])
    _update_results(results_ref, selection_results=selection_results, selected_features=selected)
    results = results_ref
    print(f"Selected {len(selected)} features using {selection_mode} flow.")

# In [18]

pipe, results_ref = _get_pipeline_context()
processed_df = getattr(pipe, 'data_', {}).get('processed') if pipe is not None else None
if processed_df is None or processed_df.empty:
    processed_df = _ensure_processed(force=False)
if processed_df is None or processed_df.empty:
    print("Processed dataset snapshot is not available.")
else:
    numeric_cols = (
        dev_df.select_dtypes(include=['number'])
        .columns.difference([cfg.target_column])
    )
    diagnostics = []
    for col in numeric_cols:
        raw_series = dev_df[col]
        proc_series = processed_df[col]
        diagnostics.append({
            'feature': col,
            'raw_missing': int(raw_series.isna().sum()),
            'processed_missing': int(proc_series.isna().sum()),
            'raw_mean': float(raw_series.mean()),
            'processed_mean': float(proc_series.mean()),
        })
    diag_df = pd.DataFrame(diagnostics)
    if diag_df.empty:
        print("No numeric columns found for diagnostics.")
    else:
        diag_df['missing_delta'] = diag_df['raw_missing'] - diag_df['processed_missing']
        diag_df['mean_shift'] = diag_df['processed_mean'] - diag_df['raw_mean']
        display(diag_df.sort_values(['missing_delta', 'mean_shift'], ascending=[False, False]).head(12))
        top_cols = diag_df.sort_values(['missing_delta', 'mean_shift'], ascending=[False, False])['feature'].head(4).tolist()
        if top_cols:
            comparison = pd.concat({'raw': dev_df[top_cols], 'prepped': processed_df[top_cols]}, axis=1)
            display(comparison.head(5))

# In [20]
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'enable_tsfresh_features', False):
    print('TSFresh reports skipped (disabled via config).')
else:
    pipe, results_ref = _get_pipeline_context()
    selection_results = results_ref.get('selection_results')
    if not isinstance(selection_results, dict):
        selection_results = None

    tsfresh_meta = results_ref.get('tsfresh_metadata')
    if tsfresh_meta is None and selection_results is not None:
        tsfresh_meta = selection_results.get('tsfresh_metadata')
    if tsfresh_meta is None and pipe is not None:
        tsfresh_meta = pipe.data_.get('tsfresh_metadata')

    if isinstance(tsfresh_meta, pd.DataFrame) and not tsfresh_meta.empty:
        display(tsfresh_meta.head())
    else:
        print('TSFresh metadata is empty.')

# In [22]

pipe, results_ref = _get_pipeline_context()
selection_results = _ensure_selection(force=False) or {}
woe_results = _ensure_woe(force=False) or {}
if not woe_results:
    print("WOE diagnostics skipped: run the WOE transformation cell first.")
else:
    woe_values = woe_results.get('woe_values', {})
    feature = next(iter(selection_results.get('selected_features', woe_values.keys())), None)
    if feature is None:
        print('No features available for WOE diagnostic display.')
    else:
        info = woe_values.get(feature, {})
        print(f'Details for feature: {feature}')
        if isinstance(info, dict) and info.get('stats'):
            display(pd.DataFrame(info['stats']).head())
        else:
            print('  WOE stats not available for this feature.')

# In [24]

pipe, results_ref = _get_pipeline_context()
selection_results = _ensure_selection(force=False)
if not selection_results:
    print("Selection history is not available yet.")
else:
    history = selection_results.get('selection_history')
    if not history:
        print('Selection history is empty.')
    else:
        rows = []
        for step in history:
            if not isinstance(step, dict):
                continue
            rows.append({
                'method': step.get('method'),
                'before': step.get('before'),
                'after': step.get('after'),
                'removed': ', '.join(sorted(step.get('removed', []))) if step.get('removed') else '',
            })
        display(pd.DataFrame(rows))

# In [26]
pipe, results_ref = _get_pipeline_context()
model_results = _ensure_model_results(force=False) or {}
registry_df = _materialize_model_registry(model_results=model_results, force=False)
globals()['MODEL_REGISTRY_DF'] = registry_df
if isinstance(registry_df, pd.DataFrame) and not registry_df.empty:
    display(registry_df)
    if 'model_name' in registry_df.columns:
        model_names = registry_df['model_name'].dropna().astype(str).unique().tolist()
        if model_names:
            print(f"Available model names: {', '.join(model_names)}")
else:
    print('Model registry is empty.')
active_name = model_results.get('active_model_name') or model_results.get('best_model_name')
if active_name:
    print(f'Active model: {active_name}')

# In [29]
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'enable_stage2_calibration', False):
    print('Stage-1/Stage-2 calibration disabled via config; skipping metrics cell.')
else:
    pipe, results_ref = _get_pipeline_context()
    stage1 = _ensure_stage1(force=True)
    stage2 = _ensure_stage2(force=False)
    if isinstance(stage1, dict) and stage1:
        stage1_metrics = stage1.get('calibration_metrics', {})
        print('Stage-1 calibration metrics:')
        if stage1_metrics:
            display(pd.DataFrame([stage1_metrics]))
        else:
            print('  Metrics unavailable.')
    else:
        print('Stage-1 calibration metrics are unavailable.')
    if isinstance(stage2, dict) and stage2:
        stage2_metrics = stage2.get('stage2_metrics', {})
        print('Stage-2 calibration metrics:')
        if stage2_metrics:
            display(pd.DataFrame([stage2_metrics]))
        else:
            print('  Metrics unavailable.')
    else:
        print('Stage-2 calibration metrics are unavailable.')

# In [31]
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'enable_scoring', False):
    print('Scoring disabled via config; skipping.')
else:

    pipe, results_ref = _get_pipeline_context()
    scoring_output = _ensure_scoring(force=False)
    scoring_metrics = scoring_output.get('metrics') if isinstance(scoring_output, dict) else None
    if scoring_metrics:
        display(pd.DataFrame([scoring_metrics]))
    else:
        print('Scoring metrics not available.')

# In [33]
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'enable_stage2_calibration', False):
    print('Calibration pipeline disabled via config; skipping execution.')
else:
    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        print('Calibration skipped: pipeline instance not available yet.')
    else:
        stage1 = _ensure_stage1(force=True)
        stage2 = _ensure_stage2(force=True)
        _update_results(results_ref, calibration_stage1=stage1, calibration_stage2=stage2)
        results = results_ref
        print('Calibration refreshed.')

# In [36]
pipe, results_ref = _get_pipeline_context()
if pipe is None:
    print('Risk band optimisation skipped: pipeline instance not available yet.')
else:
    stage1 = _ensure_stage1(force=False)
    stage2 = _ensure_stage2(force=False)
    override_df = pipe.data_.get('risk_band_reference') or NOTEBOOK_CONTEXT.get('data', {}).get('risk_band_reference')
    bands = pipe.run_risk_bands(stage1_results=stage1, stage2_results=stage2, splits=pipe.results_.get('splits'), data_override=override_df, force=True)
    _update_results(results_ref, risk_bands=bands)
    results = results_ref
    band_stats = None
    if isinstance(bands, dict):
        band_stats = bands.get('band_stats')
        if (band_stats is None or (isinstance(band_stats, pd.DataFrame) and band_stats.empty)):
            alt = bands.get('bands')
            if isinstance(alt, pd.DataFrame) and not alt.empty:
                band_stats = alt
    if isinstance(band_stats, pd.DataFrame) and not band_stats.empty:
        display(band_stats)
    else:
        print('Risk band statistics dataframe is empty.')
    print('Risk bands recomputed.')

# In [38]
cfg_local = _current_config()
if cfg_local is not None and not getattr(cfg_local, 'optimize_risk_bands', False):
    print('Risk band optimizer disabled via config; skipping.')
else:

    pipe, results_ref = _get_pipeline_context()
    risk_band_results = _ensure_risk_bands(force=False)
    if not risk_band_results:
        print("Risk band optimizer did not produce results yet.")
    else:
        band_stats = risk_band_results.get('band_stats')
        if band_stats is None or (isinstance(band_stats, pd.DataFrame) and band_stats.empty):
            bands_alt = risk_band_results.get('bands')
            if isinstance(bands_alt, pd.DataFrame) and not bands_alt.empty:
                band_stats = bands_alt
        if isinstance(band_stats, pd.DataFrame) and not band_stats.empty:
            display(band_stats)
        else:
            print('Risk band statistics dataframe is empty.')

# In [40]
RUN_FULL_PIPELINE = False

if RUN_FULL_PIPELINE:
    print('Running full pipeline to validate reproducibility...')

    pipe, results_ref = _get_pipeline_context()
    if pipe is None:
        pipe = UnifiedRiskPipeline(cfg)
        globals()['pipe'] = pipe
    full_results = pipe.fit(
        NOTEBOOK_CONTEXT['data']['development'],
        data_dictionary=NOTEBOOK_CONTEXT['data']['dictionary'],
        calibration_df=NOTEBOOK_CONTEXT['data']['calibration_longrun'],
        stage2_df=NOTEBOOK_CONTEXT['data']['calibration_recent'],
        risk_band_df=NOTEBOOK_CONTEXT['data']['risk_band_reference'],
        score_df=NOTEBOOK_CONTEXT['data']['scoring'],
    )
    results = full_results
    NOTEBOOK_CONTEXT['artifacts'].clear()
    NOTEBOOK_CONTEXT['artifacts'].update(full_results)
    print(f"Best mode: {full_results.get('best_model_mode')} | Best model: {full_results.get('best_model_name')}")
    print('Model registry (top rows):')
    model_registry = pd.DataFrame(full_results.get('model_registry', []))
    if not model_registry.empty:
        sort_columns = [col for col in ['mode', 'oot_auc', 'test_auc', 'train_auc'] if col in model_registry.columns]
        if sort_columns:
            asc_flags = [True] + [False] * (len(sort_columns) - 1)
            display(model_registry.sort_values(sort_columns, ascending=asc_flags).head())
        else:
            display(model_registry.head())
    else:
        print('Model registry is empty.')
else:
    print('Skip: set RUN_FULL_PIPELINE = True to rerun the entire pipeline at once.')

# In [42]

pipe, results_ref = _get_pipeline_context()
if pipe is None:
    print('Scoring skipped: pipeline instance not available yet.')
else:
    scoring_output = pipe.run_scoring(NOTEBOOK_CONTEXT['data']['scoring'], force=True)
    _update_results(results_ref, scoring_output=scoring_output)
    results = results_ref
    reports = _ensure_reports(force=True)
    excel_path = reports.get('excel_path') if isinstance(reports, dict) else None
    if excel_path:
        print(f"Latest reporting workbook: {excel_path}")

# In [43]

pipe, results_ref = _get_pipeline_context()
reports = _ensure_reports(force=True)
if not reports:
    print('Reporting artifacts are not available yet.')
else:
    excel_path = reports.get('excel_path')
    if excel_path:
        print(f"Excel workbook generated: {excel_path}")
    available_keys = sorted(reports.keys())
    display(pd.DataFrame({'report_key': available_keys}))

# In [46]
# Quick config + model overview
import pandas as pd
cfg = pipe.config if 'pipe' in globals() else None
mr = results.get('model_results', {}) if 'results' in globals() else {}
reports_present = isinstance(reports.get('models_summary'), pd.DataFrame) if 'reports' in globals() else False
print('enable_dual:', getattr(cfg,'enable_dual', None), ' enable_dual_pipeline:', getattr(cfg,'enable_dual_pipeline', None))
print('Active model:', mr.get('active_model_name'))
print('Selected feature count:', len(mr.get('selected_features', []) or []))
avail = []
if 'pipe' in globals():
    avail = list((getattr(pipe, 'models_', {}) or {}).keys())
if not avail and 'results' in globals():
    mr_models = (results.get('model_results', {}) or {}).get('models', {})
    if isinstance(mr_models, dict) and mr_models:
        avail = list(mr_models.keys())
    else:
        reg = results.get('model_object_registry', {})
        if isinstance(reg, dict) and reg:
            names = []
            for mode_map in reg.values():
                if isinstance(mode_map, dict):
                    names.extend(list(mode_map.keys()))
            seen = {}
            avail = [seen.setdefault(n, n) for n in names if n not in seen]
print('Available models:', ', '.join(avail) if avail else '(none)')
print('models_summary available in reports:', reports_present)

# In [47]
# X_eval reconstruction and feature alignment check
import numpy as np, pandas as pd
cfg = pipe.config if 'pipe' in globals() else None
splits = results.get('splits', {}) if 'results' in globals() else {}
mr = results.get('model_results', {}) if 'results' in globals() else {}
selected = list(mr.get('selected_features', []) or [])
def _guess_eval(splits, cfg, selected):
    if cfg is None: return None, None
    if getattr(cfg, 'enable_woe', False):
        X = splits.get('test_woe')
        if X is None or getattr(X, 'empty', False):
            X = splits.get('train_woe')
    else:
        X = splits.get('test_raw_prepped')
        if X is None or getattr(X, 'empty', False):
            X = splits.get('train_raw_prepped')
    base = splits.get('test')
    if base is None or getattr(base, 'empty', False):
        base = splits.get('train')
    y = base[cfg.target_col] if isinstance(base, pd.DataFrame) and cfg.target_col in base.columns else None
    if X is not None and selected:
        cols = [c for c in selected if c in X.columns]
        X = X[cols].copy()
    return X, y
X_eval, y_eval = _guess_eval(splits, cfg, selected)
mdl = mr.get('active_model')
names = getattr(mdl, 'feature_names_in_', None)
print('X_eval shape:', None if X_eval is None else X_eval.shape)
print('model has feature_names_in_:', names is not None)
if names is not None and X_eval is not None:
    names = list(names)
    present = [c for c in names if c in X_eval.columns]
    missing = [c for c in names if c not in X_eval.columns]
    print('present/expected:', len(present), '/', len(names), ' missing:', len(missing))
    if missing: print('missing sample:', missing[:10])

# In [48]
# Try quick probability sample from active model (safe)
import numpy as np
def _safe_proba(m, X, limit=2000):
    if X is None or m is None: return None
    Xc = X.copy()
    names = getattr(m, 'feature_names_in_', None)
    if names is not None:
        names = list(names)
        for c in names:
            if c not in Xc.columns: Xc[c] = 0.0
        Xc = Xc[names]
    Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0)
    try:
        proba = getattr(m, 'predict_proba', None)
        if callable(proba):
            p = proba(Xc[:limit])
            p = np.asarray(p)
            return p[:,1] if p.ndim==2 else p.ravel()
    except Exception as e:
        print('predict_proba failed:', e)
    dec = getattr(m, 'decision_function', None)
    if callable(dec):
        s = dec(Xc[:limit])
        s = np.asarray(s)
        if s.ndim==1:
            try:
                from scipy.special import expit
                return expit(s)
            except Exception:
                return s
        else:
            return s[:,-1]
    return None
p = _safe_proba(mdl, X_eval)
print('proba sample range:', None if p is None else (float(np.nanmin(p)), float(np.nanmax(p))))
