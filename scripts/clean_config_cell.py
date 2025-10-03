import json
from pathlib import Path


NB_PATH = Path('risk-model-pipeline-dev/notebooks/risk_pipeline_quickstart.ipynb')


NEW_HEADER = [
    "from risk_pipeline.core.config import Config\n",
    "from risk_pipeline.unified_pipeline import UnifiedRiskPipeline\n",
    "\n",
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cells = nb.get('cells') or []
    changed = False
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        src_list = cell.get('source') or []
        src = ''.join(src_list)
        # Heuristic: find config cell by presence of cfg_params dict and importlib.reload usage
        if 'cfg_params' in src and ('importlib.reload' in src or 'AdvancedFeatureSelector' in src):
            # Split source at the beginning of cfg_params to preserve dictionary and below
            idx = src.find('cfg_params')
            if idx > 0:
                tail = src[idx:]
                cell['source'] = NEW_HEADER + [tail]
                changed = True
                break
    if changed:
        NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print('Config cell cleaned (no reloads, remote-only imports).')
    else:
        print('Config cell not found or already clean; no changes made.')


if __name__ == '__main__':
    main()

