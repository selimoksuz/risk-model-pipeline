import json
from pathlib import Path


TARGET = Path('risk-model-pipeline-dev/notebooks/risk_pipeline_quickstart.ipynb')


NEW_SOURCE = [
    "import importlib\n",
    "import importlib.util\n",
    "from pathlib import Path\n",
    "\n",
    "# Remote package import (no local path hacks)\n",
    "def _import_risk_pipeline():\n",
    "    try:\n",
    "        import risk_pipeline as rp\n",
    "        return rp\n",
    "    except ImportError:\n",
    "        print('risk_pipeline is not installed. Install from development branch:')\n",
    "        print('  pip install -U \"git+https://github.com/selimoksuz/risk-model-pipeline.git@development\"')\n",
    "        raise\n",
    "\n",
    "TSFRESH_AVAILABLE = importlib.util.find_spec('tsfresh') is not None\n",
    "if TSFRESH_AVAILABLE:\n",
    "    print('tsfresh available (advanced time-series features can be enabled via config).')\n",
    "else:\n",
    "    print('tsfresh is not installed; pipeline will fall back to lightweight aggregate features when needed.')\n",
    "\n",
    "risk_pipeline_module = _import_risk_pipeline()\n",
    "NOTEBOOK_FLAGS = globals().setdefault('_NOTEBOOK_FLAGS', {})\n",
    "NOTEBOOK_FLAGS['tsfresh_available'] = TSFRESH_AVAILABLE\n",
]


def main() -> None:
    nb = json.loads(TARGET.read_text(encoding='utf-8'))
    cells = nb.get('cells') or []
    changed = False
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source') or [])
        if '_locate_project_root' in src or 'ensure_risk_pipeline' in src and 'spec_from_file_location' in src:
            cell['source'] = NEW_SOURCE
            changed = True
            break
    if changed:
        TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print('Notebook imports cleaned (remote-only).')
    else:
        print('No matching local import cell found; no changes applied.')


if __name__ == '__main__':
    main()

