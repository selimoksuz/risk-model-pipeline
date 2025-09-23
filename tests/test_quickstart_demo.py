from pathlib import Path

from examples.quickstart_demo import run_quickstart


def test_quickstart_demo_runs(tmp_path):
    results = run_quickstart(tmp_path)
    assert results.get('best_model_name') is not None
    generated = sorted(p.name for p in Path(tmp_path).glob('*'))
    assert generated, 'Expected artefacts in output directory'
