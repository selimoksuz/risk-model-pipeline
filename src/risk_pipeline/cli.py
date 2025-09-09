"""Command Line Interface for Risk Model Pipeline"""
import os as _os_utf8
import sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8', '1')
try:
    if hasattr(_sys_utf8.stdout, 'reconfigure'):
        _sys_utf8.stdout.reconfigure(encoding='utf-8')
        _sys_utf8.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import json
import pickle
import joblib
import pandas as pd
import typer
from openpyxl import load_workbook

from .core.config import Config
from .pipeline import RiskModelPipeline, DualPipeline
from .stages.scoring import build_scored_frame


app = typer.Typer(help="Risk Model Pipeline CLI")


@app.command()
def run(
    input_csv: str = typer.Option(..., help="Path to input CSV with features + target"),
    target_col: str = typer.Option("target", help="Target column name"),
    id_col: str = typer.Option("app_id", help="ID column"),
    time_col: str = typer.Option("app_dt", help="Time column for splitting"),
    output_folder: str = typer.Option("output", help="Output folder for reports"),
    dual_pipeline: bool = typer.Option(True, help="Run dual pipeline (WOE + RAW)"),
    iv_min: float = typer.Option(0.02, help="Minimum IV threshold"),
    psi_threshold: float = typer.Option(0.25, help="PSI threshold"),
    model_type: str = typer.Option("lightgbm", help="Model type: lightgbm, xgboost, catboost"),
):
    """Run the main risk model pipeline"""
    df = pd.read_csv(input_csv)

    cfg = Config(
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        output_folder=output_folder,
        enable_dual_pipeline=dual_pipeline,
        iv_min=iv_min,
        psi_threshold=psi_threshold,
        model_type=model_type
    )

    if dual_pipeline:
        pipe = DualPipeline(cfg)
    else:
        pipe = RiskModelPipeline(cfg)

    pipe.run(df)
    typer.echo(f"Done. Best model: {pipe.best_model_name_} | Reports -> {output_folder}")


@app.command()
def run_advanced(
    input_csv: str = typer.Option(..., help="Path to input CSV with features + target"),
    id_col: str = typer.Option("app_id", help="ID column"),
    time_col: str = typer.Option("app_dt", help="Time column (date-like)"),
    target_col: str = typer.Option("target", help="Binary target column {0, 1}"),
    oot_months: int = typer.Option(3, help="Number of months for OOT window"),
    use_test_split: bool = typer.Option(False, help="Create an internal TEST split from pre-OOT months"),
    output_folder: str = typer.Option("outputs", help="Report/artifact output folder"),
    output_excel: str = typer.Option("model_report.xlsx", help="Excel file name inside output folder"),
    dictionary_path: str = typer.Option(None, help="Optional Excel for dictionary enrichment"),
    psi_verbose: bool = typer.Option(True, help="Verbose PSI logging"),
    calibration_data: str = typer.Option(None, "--calibration-data", help="Calibration data path"),
    calibration_method: str = typer.Option("isotonic", help="Calibration method"),
    cluster_top_k: int = typer.Option(2, help="Variables per correlation cluster"),
    rho_threshold: float = typer.Option(0.8, help="Final correlation Spearman threshold"),
    vif_threshold: float = typer.Option(5.0, help="VIF threshold"),
    iv_min: float = typer.Option(0.02, help="Minimum IV to keep variable"),
    iv_high_flag: float = typer.Option(0.50, help="High IV flag threshold"),
    psi_threshold_feature: float = typer.Option(0.25, help="PSI threshold for features"),
    psi_threshold_score: float = typer.Option(0.10, help="PSI threshold for score"),
    shap_sample: int = typer.Option(25000, help="Sample size for SHAP"),
    ensemble: bool = typer.Option(False, "--ensemble/--no-ensemble", help="Enable ensemble of top models"),
    ensemble_top_k: int = typer.Option(3, help="Top-k models for ensemble"),
    try_mlp: bool = typer.Option(False, "--try-mlp/--no-try-mlp", help="Train simple MLP challenger"),
    hpo_method: str = typer.Option("random", help="HPO method"),
    hpo_timeout_sec: int = typer.Option(1200, help="HPO timeout seconds"),
    hpo_trials: int = typer.Option(60, help="HPO trial count"),
    log_file: str = typer.Option(None, help="Optional log file path (default: <output_folder>/pipeline.log)"),
):
    """Run the risk model pipeline with advanced configuration"""
    df = pd.read_csv(input_csv)

    cfg = Config(
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        oot_months=oot_months,
        output_excel_path=output_excel,
        output_folder=output_folder,
        cluster_top_k=cluster_top_k,
        rho_threshold=rho_threshold,
        vif_threshold=vif_threshold,
        iv_min=iv_min,
        iv_high_threshold=iv_high_flag,
        psi_threshold=psi_threshold_feature,
        use_optuna=(hpo_method != 'none'),
        n_trials=hpo_trials,
        optuna_timeout=hpo_timeout_sec,
        enable_dual_pipeline=ensemble
    )

    if ensemble:
        pipe = DualPipeline(cfg)
    else:
        pipe = RiskModelPipeline(cfg)

    pipe.run(df)
    typer.echo(f"Done. Best model: {pipe.best_model_name_} | Reports -> {output_folder}")


@app.command()
def score(
    input_csv: str = typer.Option(..., help="Path to input CSV with features"),
    woe_mapping: str = typer.Option(..., help="Path to WOE mapping JSON file"),
    final_vars_json: str = typer.Option(..., help="Path to final vars JSON file"),
    model_path: str = typer.Option(..., help="Path to best model file (.joblib or .pkl)"),
    id_col: str = typer.Option("app_id", help="ID column to keep in output"),
    output_csv: str = typer.Option(None, help="Optional CSV path for scores"),
    output_xlsx: str = typer.Option(None, help="Optional Excel output path (writes combined raw+woe+preds)"),
    calibrator_path: str = typer.Option(None, help="Optional calibrator pickle path"),
    report_xlsx: str = typer.Option(None, "--report-xlsx", help="Optional report path for external scores"),
):
    """Score new data using a trained model"""
    df = pd.read_csv(input_csv)
    with open(woe_mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(final_vars_json, "r", encoding="utf-8") as f:
        final_vars = json.load(f).get("final_vars", [])
    try:
        mdl = joblib.load(model_path)
    except Exception:
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
                    left = b.get("left")
                    right = b.get("right")
                    woe = b.get("woe", 0.0)
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
        raise ValueError("No overlap between final_vars and available columns after WOE transform.")

    # Load calibrator if provided
    calib = None
    if calibrator_path and os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as f:
            calib = pickle.load(f)

    combined = build_scored_frame(df, mapping=mapping, final_vars=final_vars,
                                  model=mdl, id_col=id_col, calibrator=calib)

    if output_csv:
        combined.to_csv(output_csv, index=False)
        typer.echo(f"Scores saved to {output_csv}")

    if output_xlsx:
        with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as w:
            combined.to_excel(w, sheet_name="combined_scores", index=False)
        typer.echo(f"Excel saved to {output_xlsx}")

    # Append to existing report if requested
    if report_xlsx and os.path.exists(report_xlsx):
        book = load_workbook(report_xlsx)
        with pd.ExcelWriter(report_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            writer.book = book
            combined.to_excel(writer, sheet_name="external_scores", index=False)
        typer.echo(f"Added 'external_scores' sheet to {report_xlsx}")


if __name__ == "__main__":
    app()
