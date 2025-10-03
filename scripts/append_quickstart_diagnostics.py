import json
from pathlib import Path


def append_diagnostics(nb_path: Path) -> None:
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells')
    if not isinstance(cells, list):
        raise SystemExit('Notebook has no cells list')

    # Remove existing diagnostic cells to refresh content
    filtered = []
    for c in cells:
        md = c.get('metadata') if isinstance(c, dict) else None
        tags = (md or {}).get('tags') if isinstance(md, dict) else None
        if isinstance(tags, list) and 'diag-cells' in tags:
            continue
        filtered.append(c)
    cells[:] = filtered

    def md_cell(text: str) -> dict:
        return {
            "cell_type": "markdown",
            "metadata": {"tags": ["diag-cells"]},
            "source": [text],
        }

    def code_cell(lines):
        return {
            "cell_type": "code",
            "metadata": {"tags": ["diag-cells"]},
            "execution_count": None,
            "outputs": [],
            "source": lines,
        }

    md1 = md_cell(
        "## Diagnostics: Model, Dual Flow, Risk-Bands\n"
        "Bu bölüm, dual (RAW/WOE) akışı, aktif modeli ve risk bandı skorlama uyumunu hızlıca doğrulamak içindir.\n"
    )

    code1 = code_cell([
        "# Quick config + model overview\n",
        "import pandas as pd\n",
        "cfg = pipe.config if 'pipe' in globals() else None\n",
        "mr = results.get('model_results', {}) if 'results' in globals() else {}\n",
        "reports_present = isinstance(reports.get('models_summary'), pd.DataFrame) if 'reports' in globals() else False\n",
        "print('enable_dual:', getattr(cfg,'enable_dual', None), ' enable_dual_pipeline:', getattr(cfg,'enable_dual_pipeline', None))\n",
        "print('Active model:', mr.get('active_model_name'))\n",
        "print('Selected feature count:', len(mr.get('selected_features', []) or []))\n",
        "avail = []\n",
        "if 'pipe' in globals():\n",
        "    avail = list((getattr(pipe, 'models_', {}) or {}).keys())\n",
        "if not avail and 'results' in globals():\n",
        "    mr_models = (results.get('model_results', {}) or {}).get('models', {})\n",
        "    if isinstance(mr_models, dict) and mr_models:\n",
        "        avail = list(mr_models.keys())\n",
        "    else:\n",
        "        reg = results.get('model_object_registry', {})\n",
        "        if isinstance(reg, dict) and reg:\n",
        "            names = []\n",
        "            for mode_map in reg.values():\n",
        "                if isinstance(mode_map, dict):\n",
        "                    names.extend(list(mode_map.keys()))\n",
        "            seen = {}\n",
        "            avail = [seen.setdefault(n, n) for n in names if n not in seen]\n",
        "print('Available models:', ', '.join(avail) if avail else '(none)')\n",
        "print('models_summary available in reports:', reports_present)\n",
    ])

    code2 = code_cell([
        "# X_eval reconstruction and feature alignment check\n",
        "import numpy as np, pandas as pd\n",
        "cfg = pipe.config if 'pipe' in globals() else None\n",
        "splits = results.get('splits', {}) if 'results' in globals() else {}\n",
        "mr = results.get('model_results', {}) if 'results' in globals() else {}\n",
        "selected = list(mr.get('selected_features', []) or [])\n",
        "def _guess_eval(splits, cfg, selected):\n",
        "    if cfg is None: return None, None\n",
        "    if getattr(cfg, 'enable_woe', False):\n",
        "        X = splits.get('test_woe')\n",
        "        if X is None or getattr(X, 'empty', False):\n",
        "            X = splits.get('train_woe')\n",
        "    else:\n",
        "        X = splits.get('test_raw_prepped')\n",
        "        if X is None or getattr(X, 'empty', False):\n",
        "            X = splits.get('train_raw_prepped')\n",
        "    base = splits.get('test')\n",
        "    if base is None or getattr(base, 'empty', False):\n",
        "        base = splits.get('train')\n",
        "    y = base[cfg.target_col] if isinstance(base, pd.DataFrame) and cfg.target_col in base.columns else None\n",
        "    if X is not None and selected:\n",
        "        cols = [c for c in selected if c in X.columns]\n",
        "        X = X[cols].copy()\n",
        "    return X, y\n",
        "X_eval, y_eval = _guess_eval(splits, cfg, selected)\n",
        "mdl = mr.get('active_model')\n",
        "names = getattr(mdl, 'feature_names_in_', None)\n",
        "print('X_eval shape:', None if X_eval is None else X_eval.shape)\n",
        "print('model has feature_names_in_:', names is not None)\n",
        "if names is not None and X_eval is not None:\n",
        "    names = list(names)\n",
        "    present = [c for c in names if c in X_eval.columns]\n",
        "    missing = [c for c in names if c not in X_eval.columns]\n",
        "    print('present/expected:', len(present), '/', len(names), ' missing:', len(missing))\n",
        "    if missing: print('missing sample:', missing[:10])\n",
    ])

    code3 = code_cell([
        "# Try quick probability sample from active model (safe)\n",
        "import numpy as np\n",
        "def _safe_proba(m, X, limit=2000):\n",
        "    if X is None or m is None: return None\n",
        "    Xc = X.copy()\n",
        "    names = getattr(m, 'feature_names_in_', None)\n",
        "    if names is not None:\n",
        "        names = list(names)\n",
        "        for c in names:\n",
        "            if c not in Xc.columns: Xc[c] = 0.0\n",
        "        Xc = Xc[names]\n",
        "    Xc = Xc.apply(pd.to_numeric, errors='coerce').fillna(0)\n",
        "    try:\n",
        "        proba = getattr(m, 'predict_proba', None)\n",
        "        if callable(proba):\n",
        "            p = proba(Xc[:limit])\n",
        "            p = np.asarray(p)\n",
        "            return p[:,1] if p.ndim==2 else p.ravel()\n",
        "    except Exception as e:\n",
        "        print('predict_proba failed:', e)\n",
        "    dec = getattr(m, 'decision_function', None)\n",
        "    if callable(dec):\n",
        "        s = dec(Xc[:limit])\n",
        "        s = np.asarray(s)\n",
        "        if s.ndim==1:\n",
        "            try:\n",
        "                from scipy.special import expit\n",
        "                return expit(s)\n",
        "            except Exception:\n",
        "                return s\n",
        "        else:\n",
        "            return s[:,-1]\n",
        "    return None\n",
        "p = _safe_proba(mdl, X_eval)\n",
        "print('proba sample range:', None if p is None else (float(np.nanmin(p)), float(np.nanmax(p))))\n",
    ])

    cells.extend([md1, code1, code2, code3])
    with nb_path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Diagnostics cells appended to quickstart notebook.')


if __name__ == '__main__':
    path = Path('risk-model-pipeline-dev/notebooks/risk_pipeline_quickstart.ipynb')
    append_diagnostics(path)

