import json
from pathlib import Path


SRC_NB = Path('risk-model-pipeline-dev/notebooks/risk_pipeline_quickstart.ipynb')
OUT_DIR = Path('risk-model-pipeline-dev/notebooks/_exec')
DST_NB = OUT_DIR / 'risk_pipeline_quickstart_light.ipynb'


def tweak_cfg_cell(src: list[str]) -> list[str]:
    text = ''.join(src)
    if 'cfg_params' not in text:
        return src

    lines = text.splitlines(keepends=True)
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "'algorithms': [" in ln:
            # write a compact reduced algorithms list and skip original block until its closing ],
            new_lines.append("    'algorithms': [\n")
            new_lines.append("        'logistic',\n")
            new_lines.append("        'lightgbm',\n")
            new_lines.append("    ],\n")
            i += 1
            # skip until we see a line ending the original list (contains '],')
            while i < len(lines):
                if '],' in lines[i]:
                    i += 1
                    break
                i += 1
            continue
        # simple boolean switches
        if "'calculate_shap':" in ln:
            new_lines.append(ln.replace('True', 'False'))
            i += 1
            continue
        if "'use_optuna':" in ln:
            new_lines.append(ln.replace('True', 'False'))
            i += 1
            continue
        if "'n_jobs':" in ln:
            new_lines.append("    'n_jobs': 2,\n")
            i += 1
            continue
        new_lines.append(ln)
        i += 1
    return new_lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(SRC_NB.read_text(encoding='utf-8'))
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code' and isinstance(cell.get('source'), list):
            src = cell['source']
            joined = ''.join(src)
            if 'cfg_params' in joined and 'UnifiedRiskPipeline' in joined:
                cell['source'] = tweak_cfg_cell(src)
                break
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote tweaked notebook to {DST_NB}')


if __name__ == '__main__':
    main()
