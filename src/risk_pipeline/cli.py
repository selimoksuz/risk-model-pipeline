"""Command Line Interface for Risk Model Pipeline"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from typer.models import OptionInfo

import joblib
import pandas as pd
import typer
from openpyxl import load_workbook

from .core.config import Config
from .reporting.report import save_metrics
from .stages.scoring import build_scored_frame
from .unified_pipeline import UnifiedRiskPipeline

CONFIG_FIELD_NAMES = set(Config.__dataclass_fields__.keys())
LEGACY_KEY_MAP = {
    "target_col": "target_column",
    "id_col": "id_column",
    "time_col": "time_column",
    "weight_col": "weight_column",
    "test_ratio": "test_size",
    "stratify_test_split": "stratify_test",
    "dual_pipeline": "enable_dual",
    "enable_dual_pipeline": "enable_dual",
    "iv_min": "min_iv",
    "iv_threshold": "min_iv",
    "psi_threshold": "max_psi",
}

def _split_config_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    if not data:
        return normalized, extras
    for key, value in data.items():
        mapped_key = LEGACY_KEY_MAP.get(key, key)
        if mapped_key in CONFIG_FIELD_NAMES:
            normalized[mapped_key] = value
        elif value is not None:
            extras[mapped_key] = value
    return normalized, extras



def _apply_legacy_aliases(cfg: Config) -> None:
    alias_sources = {
        "numeric_imputation": ("numeric_imputation_strategy", "median"),
        "outlier_method": ("numeric_outlier_method", "clip"),
        "min_category_freq": ("rare_category_threshold", 0.01),
        "band_method": ("risk_band_method", "pd_constraints"),
        "selection_method": ("stepwise_method", "forward"),
        "stage2_method": (None, "lower_mean"),
    }
    for attr, (source, default) in alias_sources.items():
        if not hasattr(cfg, attr):
            if source and hasattr(cfg, source):
                setattr(cfg, attr, getattr(cfg, source))
            else:
                setattr(cfg, attr, default)
    if not hasattr(cfg, "enable_calibration"):
        setattr(cfg, "enable_calibration", getattr(cfg, "enable_stage2_calibration", False))


def _resolve_option(value, default=None):
    if isinstance(value, OptionInfo):
        return value.default if value.default is not ... else default
    return value


os.environ.setdefault("PYTHONUTF8", "1")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

app = typer.Typer(help="Risk Model Pipeline CLI")


@app.command()
def run(
    config_json: Optional[str] = typer.Option(None, help="Path to a JSON config file or inline JSON string"),
    input_csv: str = typer.Option(..., help="Path to input CSV with features + target"),
    target_col: str = typer.Option("target", help="Target column name"),
    id_col: str = typer.Option("app_id", help="ID column"),
    time_col: str = typer.Option("app_dt", help="Time column for splitting"),
    artifacts_dir: str = typer.Option("output", help="Directory for pipeline artifacts"),
    dual_pipeline: bool = typer.Option(True, "--dual-pipeline/--single-pipeline", help="Run both WOE and RAW flows"),
    iv_min: float = typer.Option(0.02, help="Minimum IV threshold"),
    psi_threshold: float = typer.Option(0.25, help="PSI threshold"),
    model_type: str = typer.Option("lightgbm", help="Model type: lightgbm, xgboost, catboost"),
):
    """Run the main risk model pipeline."""
    df = pd.read_csv(input_csv)

    config_json = _resolve_option(config_json)
    target_col = _resolve_option(target_col, "target")
    id_col = _resolve_option(id_col, "app_id")
    time_col = _resolve_option(time_col, "app_dt")
    artifacts_dir = _resolve_option(artifacts_dir, "output")
    dual_pipeline = bool(_resolve_option(dual_pipeline, True))
    iv_min = float(_resolve_option(iv_min, 0.02))
    psi_threshold = float(_resolve_option(psi_threshold, 0.25))
    model_type = _resolve_option(model_type, "lightgbm")

    base_payload = {
        "target_column": target_col,
        "id_column": id_col,
        "time_column": time_col,
        "output_folder": artifacts_dir,
        "enable_dual": dual_pipeline,
        "min_iv": iv_min,
        "max_psi": psi_threshold,
        "model_type": model_type,
    }

    normalized, extras = _split_config_payload(base_payload)

    if config_json:
        try:
            if os.path.exists(config_json):
                with open(config_json, "r", encoding="utf-8") as f:
                    external_cfg = json.load(f)
            else:
                external_cfg = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"config_json gecersiz: {exc}") from exc
        if not isinstance(external_cfg, dict):
            raise typer.BadParameter("config_json must resolve to a JSON object.")
        ext_normalized, ext_extras = _split_config_payload(external_cfg)
        normalized.update(ext_normalized)
        extras.update(ext_extras)

    cfg = Config(**normalized)
    if extras:
        for key, value in extras.items():
            setattr(cfg, key, value)
    _apply_legacy_aliases(cfg)

    artifacts_path = Path(cfg.output_folder or artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    pipe = UnifiedRiskPipeline(cfg)
    run_error: Optional[str] = None
    try:
        results = pipe.fit(df)
    except Exception as exc:
        run_error = str(exc)
        typer.echo(f"Pipeline execution failed: {run_error}", err=True)
        results = {}

    model_results = results.get("model_results", {}) if isinstance(results, dict) else {}
    scores = model_results.get("model_scores") if isinstance(model_results.get("model_scores"), dict) else None
    metrics_payload = {
        "best_model": getattr(pipe, "best_model_name_", None),
        "model_scores": scores,
        "selected_features": model_results.get("selected_features")
        if isinstance(model_results.get("selected_features"), list)
        else None,
        "config": {
            "target_col": getattr(cfg, "target_col", target_col),
            "id_col": getattr(cfg, "id_col", id_col),
            "time_col": getattr(cfg, "time_col", time_col),
            "enable_dual_pipeline": getattr(
                cfg,
                "enable_dual_pipeline",
                getattr(cfg, "enable_dual", dual_pipeline),
            ),
            "model_type": getattr(cfg, "model_type", model_type),
        },
        "status": "failed" if run_error else "success",
        "error": run_error,
    }

    save_metrics(metrics_payload, str(artifacts_path))
    best_model_name = getattr(pipe, 'best_model_name_', None)
    if run_error:
        typer.echo(
            f"Pipeline failed; saved metrics stub to {artifacts_path}: {run_error}",
            err=True,
        )
    else:
        typer.echo(
            f"Done. Best model: {best_model_name or 'unknown'} | Reports -> {artifacts_path}"
        )



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

    id_col = _resolve_option(id_col, "app_id")
    time_col = _resolve_option(time_col, "app_dt")
    target_col = _resolve_option(target_col, "target")
    output_folder = _resolve_option(output_folder, "outputs")
    output_excel = _resolve_option(output_excel, "model_report.xlsx")
    dictionary_path = _resolve_option(dictionary_path)
    psi_verbose = bool(_resolve_option(psi_verbose, True))
    calibration_data = _resolve_option(calibration_data)
    calibration_method = _resolve_option(calibration_method, "isotonic")
    cluster_top_k = int(_resolve_option(cluster_top_k, 2))
    rho_threshold = float(_resolve_option(rho_threshold, 0.8))
    vif_threshold = float(_resolve_option(vif_threshold, 5.0))
    iv_min = float(_resolve_option(iv_min, 0.02))
    iv_high_flag = float(_resolve_option(iv_high_flag, 0.50))
    psi_threshold_feature = float(_resolve_option(psi_threshold_feature, 0.25))
    psi_threshold_score = float(_resolve_option(psi_threshold_score, 0.10))
    shap_sample = int(_resolve_option(shap_sample, 25000))
    ensemble = bool(_resolve_option(ensemble, False))
    ensemble_top_k = int(_resolve_option(ensemble_top_k, 3))
    try_mlp = bool(_resolve_option(try_mlp, False))
    hpo_method = _resolve_option(hpo_method, "random")
    hpo_timeout_sec = int(_resolve_option(hpo_timeout_sec, 1200))
    hpo_trials = int(_resolve_option(hpo_trials, 60))
    log_file = _resolve_option(log_file)
    use_test_split = bool(_resolve_option(use_test_split, False))
    oot_months = int(_resolve_option(oot_months, 3))

    config_payload = {
        "target_column": target_col,
        "id_column": id_col,
        "time_column": time_col,
        "oot_months": oot_months,
        "output_folder": output_folder,
        "create_test_split": use_test_split,
        "min_iv": iv_min,
        "max_psi": psi_threshold_feature,
        "max_correlation": rho_threshold,
        "max_vif": vif_threshold,
        "use_optuna": hpo_method != "none",
        "n_trials": hpo_trials,
        "optuna_timeout": hpo_timeout_sec,
        "enable_dual": ensemble,
        "calculate_shap": bool(shap_sample),
        "shap_sample_size": shap_sample,
        "psi_threshold_feature": psi_threshold_feature,
        "psi_threshold_score": psi_threshold_score,
        "dictionary_path": dictionary_path,
        "psi_verbose": psi_verbose,
        "calibration_data": calibration_data,
        "calibration_method": calibration_method,
        "cluster_top_k": cluster_top_k,
        "rho_threshold": rho_threshold,
        "vif_threshold": vif_threshold,
        "iv_high_threshold": iv_high_flag,
        "use_test_split": use_test_split,
        "ensemble_top_k": ensemble_top_k,
        "try_mlp": try_mlp,
        "hpo_method": hpo_method,
        "log_file": log_file,
        "output_excel_path": output_excel,
    }

    normalized, extras = _split_config_payload(config_payload)
    cfg = Config(**normalized)
    if extras:
        for key, value in extras.items():
            setattr(cfg, key, value)
    _apply_legacy_aliases(cfg)

    output_path = Path(getattr(cfg, "output_folder", output_folder))
    output_path.mkdir(parents=True, exist_ok=True)

    pipe = UnifiedRiskPipeline(cfg)

    try:
        pipe.run(df)
        best_model_name = getattr(pipe, 'best_model_name_', None)
        typer.echo(
            f"Done. Best model: {best_model_name or 'unknown'} | Reports -> {output_path}"
        )
    except Exception as exc:
        typer.echo(
            f"Pipeline failed: {exc}",
            err=True,
        )
    



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

    id_col = _resolve_option(id_col, "app_id")
    output_csv = _resolve_option(output_csv)
    output_xlsx = _resolve_option(output_xlsx)
    calibrator_path = _resolve_option(calibrator_path)
    report_xlsx = _resolve_option(report_xlsx)
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

    combined = build_scored_frame(
        df, mapping=mapping, final_vars=final_vars, model=mdl, id_col=id_col, calibrator=calib
    )

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


def main():
    """Console entrypoint for risk-pipeline CLI."""
    app()




