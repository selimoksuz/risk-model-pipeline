import json
from pathlib import Path


TARGET = Path('notebooks/risk_pipeline_quickstart.ipynb')


NEW_SOURCE = [
    "import sys\n",
    "import importlib\n",
    "import importlib.util\n",
    "import subprocess, site, shutil, pathlib\n",
    "\n",
    "def _force_reinstall_from_dev():\n",
    "    py = sys.executable\n",
    "    for name in ('risk-pipeline','risk_pipeline'):\n",
    "        try:\n",
    "            subprocess.run([py,'-m','pip','uninstall','-y',name], check=False)\n",
    "        except Exception:\n",
    "            pass\n",
    "    # Remove leftovers\n",
    "    try:\n",
    "        for sp in site.getsitepackages()+[site.getusersitepackages()]:\n",
    "            p = pathlib.Path(sp)/'risk_pipeline'\n",
    "            if p.exists():\n",
    "                shutil.rmtree(p, ignore_errors=True)\n",
    "    except Exception:\n",
    "        pass\n",
    "    # Purge cache and install development branch\n",
    "    try:\n",
    "        subprocess.run([py,'-m','pip','cache','purge'], check=False)\n",
    "    except Exception:\n",
    "        pass\n",
    "    url = 'git+https://github.com/selimoksuz/risk-model-pipeline.git@development'\n",
    "    subprocess.run([py,'-m','pip','install','--no-cache-dir','--force-reinstall','-U',url], check=True)\n",
    "\n",
    "# Remote package import (no local path hacks)\n",
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
        # Replace either old local-loader import cell or the simple remote-only variant
        if (
            '_locate_project_root' in src
            or 'spec_from_file_location' in src
            or 'def _import_risk_pipeline()' in src
            or '# Remote package import (no local path hacks)' in src
        ):
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
