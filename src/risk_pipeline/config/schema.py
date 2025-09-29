from typing import List, Optional

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    experiment_name: str = Field(default="default_experiment")
    seed: int = 42
    target: str = "target"
    id_col: str = "app_id"
    date_col: str = "app_dt"
    oot_start: Optional[str] = None
    oot_end: Optional[str] = None
    enable_psi: bool = True
    enable_fs: bool = True
    enable_model: bool = True
    enable_calibration: bool = True
    enable_report: bool = True


class PathConfig(BaseModel):
    data_input: str = "data/input.csv"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


class SelectionConfig(BaseModel):
    steps: List[str] = Field(
        default_factory=lambda: [
            "psi",
            "univariate",
            "iv",
            "correlation",
            "boruta",
            "stepwise",
        ]
    )
    psi_threshold: float = 0.25
    iv_threshold: float = 0.02
    univariate_gini_threshold: float = 0.05
    correlation_threshold: float = 0.95
    vif_threshold: float = 5.0


class CalibrationConfig(BaseModel):
    stage1_method: str = "isotonic"
    stage1_cv_folds: int = 3
    enable_stage2: bool = True
    stage2_method: str = "lower_mean"
    stage2_lower_bound: float = 0.8
    stage2_upper_bound: float = 1.2
    stage2_confidence_level: float = 0.95


class RiskBandConfig(BaseModel):
    method: str = "pd_constraints"
    n_bands: int = 10
    min_bins: int = 7
    max_bins: int = 10
    micro_bins: int = 1000
    min_sample_size: int = 20000
    min_weight: float = 0.05
    max_weight: float = 0.30
    hhi_threshold: float = 0.15
    binomial_pass_weight: float = 0.85
    alpha: float = 0.05
    pd_dr_tolerance: float = 1e-4
    max_iterations: int = 100
    max_phase_iterations: int = 50
    early_stop_rounds: int = 10


class ModelingConfig(BaseModel):
    model_type: str = "all"
    algorithms: List[str] = Field(
        default_factory=lambda: [
            "logistic",
            "gam",
            "catboost",
            "lightgbm",
            "xgboost",
            "randomforest",
            "extratrees",
            "woe_boost",
            "xbooster",
        ]
    )
    use_optuna: bool = True
    n_trials: int = 100
    optuna_timeout: int = 3600
    hpo_method: str = "optuna"
    hpo_trials: int = 100
    hpo_timeout_sec: Optional[int] = None
    max_train_oot_gap: Optional[float] = None
    model_selection_method: str = "gini_oot"
    model_stability_weight: float = 0.0
    min_gini_threshold: float = 0.5
    try_mlp: bool = False
    enable_noise_sentinel: bool = True


class Config(BaseModel):
    run: RunConfig = RunConfig()
    path: PathConfig = PathConfig()
    selection: SelectionConfig = SelectionConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    risk_bands: RiskBandConfig = RiskBandConfig()
    modeling: ModelingConfig = ModelingConfig()
