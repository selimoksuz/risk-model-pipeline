from pathlib import Path

from examples.quickstart_demo import run_quickstart


def test_quickstart_demo_runs(tmp_path):
    results = run_quickstart(tmp_path)
    assert results.get('best_model_name') is not None
    assert results.get('calibration_stage1') is not None
    assert results.get('calibration_stage2') is not None
    output_files = list(Path(tmp_path).glob('**/*'))
    assert output_files, 'Expected artefacts in output directory'
