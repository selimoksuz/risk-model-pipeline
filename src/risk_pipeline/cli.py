import typer
from pathlib import Path
import pandas as pd

from .config.schema import Config
from .data.load import load_csv
from .model.train import train_logreg
from .reporting.report import save_metrics
import json
import joblib
import pandas as pd

app = typer.Typer(help="Risk Model Pipeline CLI")

@app.command()
def run(
    config_json: str = typer.Option(None, help="Inline JSON for config (overrides defaults)"),
    input_csv: str = typer.Option(..., help="Path to input CSV with features + target"),
    target_col: str = typer.Option("target", help="Target column name"),
    artifacts_dir: str = typer.Option("artifacts", help="Artifacts output directory"),
):
    cfg = Config()
    if config_json:
        cfg = cfg.model_validate_json(config_json)

    df = load_csv(input_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    res = train_logreg(X, y, seed=cfg.run.seed)
    save_metrics({"auc": res["auc"]}, artifacts_dir)
    typer.echo(f"Done. AUC={res['auc']:.4f}. Artifacts -> {artifacts_dir}")

if __name__ == "__main__":
    app()


from .pipeline16 import Config as Config16, RiskModelPipeline as RiskModelPipeline16

@app.command()
def run16(
    input_csv: str = typer.Option(..., help="Path to input CSV with features + target"),
    id_col: str = typer.Option("app_id", help="ID column"),
    time_col: str = typer.Option("app_dt", help="Time column (date-like)"),
    target_col: str = typer.Option("target", help="Binary target column {0,1}"),
    oot_months: int = typer.Option(3, help="Number of months for OOT window"),
    use_test_split: bool = typer.Option(False, help="Create an internal TEST split from pre-OOT months"),
    output_folder: str = typer.Option("outputs", help="Report/artifact output folder"),
    output_excel: str = typer.Option("model_report.xlsx", help="Excel file name inside output folder"),
    dictionary_path: str = typer.Option(None, help="Optional Excel for dictionary enrichment"),
    psi_verbose: bool = typer.Option(True, help="Verbose PSI logging"),
):
    df = pd.read_csv(input_csv)
    cfg = Config16(
        id_col=id_col, time_col=time_col, target_col=target_col,
        use_test_split=use_test_split, oot_window_months=oot_months,
        output_excel_path=output_excel, output_folder=output_folder,
        dictionary_path=dictionary_path, psi_verbose=psi_verbose
    )
    pipe = RiskModelPipeline16(cfg)
    pipe.run(df)
    typer.echo(f"Done. Best={pipe.best_model_name_} | Reports -> {output_folder}")


@app.command()
def score(
    input_csv: str = typer.Option(..., help="Path to input CSV with features"),
    woe_mapping: str = typer.Option(..., help="Path to WOE mapping JSON (woe_mapping_<run_id>.json)"),
    final_vars_json: str = typer.Option(..., help="Path to final vars JSON (final_vars_<run_id>.json)"),
    model_path: str = typer.Option(..., help="Path to best model file (.joblib or .pkl)"),
    id_col: str = typer.Option("app_id", help="ID column to keep in output"),
    output_csv: str = typer.Option("scores.csv", help="Output CSV path for scores"),
):
    df = pd.read_csv(input_csv)
    with open(woe_mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(final_vars_json, "r", encoding="utf-8") as f:
        final_vars = json.load(f).get("final_vars", [])
    try:
        mdl = joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            mdl = pickle.load(f)

    # Apply WOE mapping to new data
    def apply_woe(df_in: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        out = {}
        for v, info in mapping.get("variables", {}).items():
            if info.get("type") == "numeric":
                s = df_in[v]
                w = pd.Series(index=s.index, dtype="float32")
                miss = s.isna()
                # Missing bin: any bin with NaN edges
                miss_woe = 0.0
                for b in info.get("bins", []):
                    left = b.get("left"); right = b.get("right"); woe = b.get("woe", 0.0)
                    if left is None or right is None or (pd.isna(left) and pd.isna(right)):
                        miss_woe = float(woe)
                        continue
                    m = (~miss) & (s >= left) & (s <= right)
                    w.loc[m] = float(woe)
                w.loc[miss] = float(miss_woe)
                out[v] = w.values
            else:
                s = df_in[v].astype(object)
                w = pd.Series(index=s.index, dtype="float32")
                miss = s.isna()
                assigned = miss.copy()
                miss_woe = 0.0
                other_woe = 0.0
                groups = info.get("groups", [])
                for g in groups:
                    lab = g.get("label")
                    woe = float(g.get("woe", 0.0))
                    if lab == "MISSING":
                        miss_woe = woe
                        continue
                    if lab == "OTHER":
                        other_woe = woe
                        continue
                    members = set(map(str, g.get("members", [])))
                    m = (~miss) & (s.astype(str).isin(members))
                    w.loc[m] = woe
                    assigned |= m
                w.loc[miss] = float(miss_woe)
                w.loc[~assigned] = float(other_woe)
                out[v] = w.values
        return pd.DataFrame(out, index=df_in.index)

    X = apply_woe(df, mapping)
    cols = [c for c in final_vars if c in X.columns]
    if not cols:
        raise typer.BadParameter("No overlap between final_vars and available columns after WOE transform.")
    import numpy as np
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X[cols])
        proba = np.asarray(proba)
        if proba.ndim == 1:
            score = proba
        elif proba.shape[1] >= 2:
            score = proba[:, 1]
        else:
            score = proba[:, 0]
    else:
        score = mdl.predict(X[cols])
    out = pd.DataFrame({id_col: df[id_col] if id_col in df.columns else range(len(df)), "score": score})
    out.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(out)} scores to {output_csv}")
