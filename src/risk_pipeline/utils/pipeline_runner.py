"""
Pipeline runner utilities for DataFrame inputs
"""

import pandas as pd
from pathlib import Path
from ..pipeline import RiskModelPipeline, Config
from ..orchestrator import Orchestrator


def run_pipeline_from_dataframe(
    df: pd.DataFrame,
    id_col: str = "app_id",
    time_col: str = "app_dt",
    target_col: str = "target",
    output_folder: str = "outputs",
    output_excel: str = "model_report.xlsx",
    use_test_split: bool = True,
    oot_months: int = 3,
    calibration_df: pd.DataFrame = None,  # NEW: DataFrame calibration support
    calibration_data_path: str = None,  # Keep file path support too
    data_dictionary_df: pd.DataFrame = None,  # DataFrame with alan_adi, alan_aciklamasi
    data_dictionary_path: str = None,  # Excel file with variable descriptions
    **kwargs
) -> dict:
    """
    Run the risk model pipeline directly from a pandas DataFrame

    Args:
        df: Input DataFrame with features, target, id, and time columns
        id_col: Name of ID column
        time_col: Name of time/date column
        target_col: Name of binary target column {0, 1}
        output_folder: Output folder for reports and artifacts
        output_excel: Excel report filename
        use_test_split: Whether to create internal TEST split
        oot_months: Number of months for OOT window
        calibration_df: Optional calibration DataFrame
        calibration_data_path: Optional calibration file path (CSV/parquet)
        data_dictionary_df: Optional DataFrame with alan_adi and alan_aciklamasi columns
        data_dictionary_path: Optional Excel file with variable descriptions
        **kwargs: Additional config parameters

    Returns:
        dict: Pipeline results including best model name, performance metrics
    """

    # Create config
    cfg = Config(
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        use_test_split=use_test_split,
        oot_window_months=oot_months,
        output_folder=output_folder,
        output_excel_path=output_excel,
        calibration_df=calibration_df,  # Pass DataFrame if provided
        calibration_data_path=calibration_data_path,  # Pass file path if provided
        data_dictionary_df=data_dictionary_df,  # Pass data dictionary DataFrame
        data_dictionary_path=data_dictionary_path,  # Pass data dictionary path
        **kwargs
    )

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipe = RiskModelPipeline(cfg)
    pipe.run(df)

    # Return results
    results = {
        "best_model": pipe.best_model_name_,
        "final_features": pipe.final_vars_,
        "run_id": pipe.cfg.run_id,
        "output_folder": output_folder,
        "models_performance": getattr(pipe, 'models_summary_', None)
    }

    if hasattr(pipe, 'models_'):
        results["best_model_object"] = pipe.models_.get(pipe.best_model_name_)

    return results


def run_pipeline_from_csv(
    csv_path: str,
    **kwargs
) -> dict:
    """
    Convenience function to run pipeline from CSV file

    Args:
        csv_path: Path to input CSV file
        **kwargs: Arguments passed to run_pipeline_from_dataframe

    Returns:
        dict: Pipeline results
    """
    df = pd.read_csv(csv_path)
    return run_pipeline_from_dataframe(df, **kwargs)


# Enable all pipeline stages by default
def get_full_config(**overrides) -> Config:
    """
    Get a configuration with all pipeline stages enabled

    Args:
        **overrides: Configuration overrides

    Returns:
        Config: Full configuration object
    """

    # Create orchestrator with all stages enabled
    orchestrator = Orchestrator(
        enable_validate=True,
        enable_classify=True,
        enable_missing_policy=True,
        enable_split=True,
        enable_woe=True,
        enable_psi=True,
        enable_transform=True,
        enable_corr_cluster=True,
        enable_fs=True,
        enable_final_corr=True,
        enable_noise=True,
        enable_model=True,
        enable_best_select=True,
        enable_report=True,
        enable_dictionary=True,  # Enable dictionary support
    )

    # Default full config
    config = Config(
        # Core settings
        use_test_split=True,
        oot_window_months=3,

        # Feature engineering
        psi_threshold=0.25,
        psi_threshold_feature=0.25,
        psi_threshold_score=0.10,
        iv_min=0.02,
        rho_threshold=0.8,
        vif_threshold=5.0,

        # Model settings
        cv_folds=5,
        hpo_trials=30,
        hpo_timeout_sec=300,
        shap_sample=25000,
        try_mlp=True,  # Enable MLP
        ensemble=True,  # Enable ensemble
        ensemble_top_k=3,

        # Calibration
        calibration_method="isotonic",

        # Outputs
        write_parquet=True,
        write_csv=True,

        # Orchestrator - all stages enabled
        orchestrator=orchestrator
    )

    # Apply overrides using setattr
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
