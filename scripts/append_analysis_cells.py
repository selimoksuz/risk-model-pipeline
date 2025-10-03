import json
from pathlib import Path


NB_PATH = Path('notebooks/risk_pipeline_quickstart.ipynb')


def code_cell(lines):
    return {
        "cell_type": "code",
        "metadata": {"tags": ["analysis-cells"]},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["analysis-cells"]},
        "source": [text],
    }


def append_cells():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cells = nb.get('cells') or []

    # Avoid duplicates: drop existing analysis-cells
    filtered = []
    for c in cells:
        tags = (c.get('metadata') or {}).get('tags') or []
        if isinstance(tags, list) and 'analysis-cells' in tags:
            continue
        filtered.append(c)
    cells = filtered

    cells.append(md_cell("## 9.1 Recompute Risk Bands (Revised)\n"))
    cells.append(code_cell([
        "# Recompute bands only (no reporting here)\n",
        "pipe, results_ref = _get_pipeline_context()\n",
        "if pipe is None:\n",
        "    raise RuntimeError('Pipeline instance is not initialized yet. Run the configuration cell first.')\n",
        "bands = pipe.run_risk_bands(force=True)\n",
        "_update_results(results_ref, risk_bands=bands)\n",
        "results = results_ref\n",
        "print('Risk bands recomputed.')\n",
    ]))

    cells.append(md_cell("## 9.2 Risk Band Optimization Summary (Revised)\n"))
    cells.append(code_cell([
        "# Summarize risk band outputs without recomputing\n",
        "reports = _ensure_reports(force=True) or {}\n",
        "from IPython.display import display\n",
        "for k in ['risk_bands', 'band_metrics', 'risk_bands_summary']:\n",
        "    obj = reports.get(k)\n",
        "    if obj is None:\n",
        "        print(f'{k}: not available')\n",
        "    else:\n",
        "        display(obj)\n",
    ]))

    cells.append(md_cell("## Data Layers Overview and RAW Preprocessing Summary\n"))
    cells.append(code_cell([
        "# Display data layers and RAW preprocessing summary\n",
        "reports = _ensure_reports(force=True) or {}\n",
        "from IPython.display import display\n",
        "layers = pipe.reporter.reports_.get('data_layers_overview') if pipe is not None else None\n",
        "raw_prep = pipe.reporter.reports_.get('raw_preprocessing_summary') if pipe is not None else None\n",
        "display(layers if layers is not None else reports.get('data_layers_overview'))\n",
        "display(raw_prep if raw_prep is not None else reports.get('raw_preprocessing_summary'))\n",
    ]))

    cells.append(md_cell("## Scored Data Preview\n"))
    cells.append(code_cell([
        "# Show head of scored data (implementation result)\n",
        "from IPython.display import display\n",
        "scoring_output = results.get('scoring_output') if 'results' in globals() else None\n",
        "df = None\n",
        "if isinstance(scoring_output, dict):\n",
        "    df = scoring_output.get('dataframe')\n",
        "if df is None and pipe is not None:\n",
        "    df = pipe.reporter.reports_.get('scored_data')\n",
        "if df is None:\n",
        "    print('Scored data not available. Ensure scoring step has run.')\n",
        "else:\n",
        "    display(df.head(10))\n",
    ]))

    nb['cells'] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Analysis cells appended.')


if __name__ == '__main__':
    append_cells()
