import typer
from pathlib import Path
import pandas as pd

from .config.schema import Config
from .data.load import load_csv
from .model.train import train_logreg
from .reporting.report import save_metrics

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
