from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from examples import quickstart_demo
from examples.quickstart_demo import run_quickstart


def test_quickstart_demo_runs(tmp_path):
    original_builder = quickstart_demo.build_quickstart_config

    def lightweight_config(output_dir):
        cfg = original_builder(output_dir)
        cfg.algorithms = ["logistic"]
        cfg.enable_dual = False
        cfg.enable_tsfresh_features = False
        cfg.enable_scoring = False
        cfg.selection_steps = ["univariate"]
        cfg.use_optuna = False
        cfg.n_trials = 1
        return cfg

    quickstart_demo.build_quickstart_config = lightweight_config
    try:
        results = run_quickstart(tmp_path)
    finally:
        quickstart_demo.build_quickstart_config = original_builder

    assert results.get('best_model_name') is not None
    assert results.get('calibration_stage1') is not None
    assert results.get('calibration_stage2') is not None
    assert results.get('calibration_stage1_curve') is not None
    assert results.get('calibration_stage2_curve') is not None
    assert isinstance(results.get('feature_name_map'), dict)
    assert isinstance(results.get('imputation_stats'), dict)
    band_info = results.get('risk_bands') or {}
    assert isinstance(band_info, dict)
    assert results.get('risk_band_source') in (None, 'override', 'split', 'override_reference')
    output_files = list(Path(tmp_path).glob('**/*'))
    assert output_files, 'Expected artefacts in output directory'
