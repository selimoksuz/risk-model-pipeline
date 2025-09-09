"""Configuration module for Risk Model Pipeline"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class OrchestratorConfig:
    """Control which steps to enable in the pipeline"""
    enable_validate: bool = True
    enable_classify: bool = True
    enable_split: bool = True
    enable_woe: bool = True
    enable_psi: bool = True
    enable_transform: bool = True
    enable_corr: bool = True
    enable_selection: bool = True
    enable_final_corr: bool = True
    enable_noise: bool = True
    enable_model: bool = True
    enable_best_select: bool = True
    enable_report: bool = True


@dataclass
class Config:
    """Main configuration for Risk Model Pipeline"""

    # Basic settings
    target_col: str = 'target'
    id_col: str = 'app_id'
    time_col: str = 'app_dt'
    random_state: int = 42

    # Feature selection parameters
    iv_min: float = 0.02
    iv_high_threshold: float = 0.5
    psi_threshold: float = 0.25
    rho_threshold: float = 0.90
    vif_threshold: float = 5.0
    rare_threshold: float = 0.01
    cluster_top_k: int = 2

    # WOE settings
    n_bins: int = 10
    min_bin_size: float = 0.05
    woe_monotonic: bool = False
    max_abs_woe: Optional[float] = None
    handle_missing: str = 'as_category'

    # Model training settings
    use_optuna: bool = True
    n_trials: int = field(default=100)
    optuna_timeout: Optional[int] = field(default=None)
    cv_folds: int = 5

    # Model selection criteria
    model_selection_method: str = "gini_oot"  # gini_oot, stable, balanced, conservative
    max_train_oot_gap: Optional[float] = None
    model_stability_weight: float = 0.0
    min_gini_threshold: float = 0.5

    # Feature selection settings
    use_boruta: bool = True
    forward_1se: bool = True
    max_features: int = 20
    min_features: int = 3
    use_noise_sentinel: bool = True

    # Imputation settings
    imputation_strategy: str = "median"  # median, mean, forward_fill, target_mean, multiple

    # RAW pipeline settings
    raw_outlier_method: str = "none"  # none, clip, winsorize, remove
    raw_outlier_threshold: float = 3.0  # Number of standard deviations
    raw_scaler_type: str = "standard"  # standard, minmax, robust

    # Dual pipeline settings
    enable_dual_pipeline: bool = True

    # Output settings
    output_folder: str = 'output'
    output_excel_path: Optional[str] = None
    write_csv: bool = False
    run_id: Optional[str] = None

    # Data splitting settings
    use_test_split: bool = True  # Whether to create a test split (if False, all pre-OOT goes to train)
    train_ratio: float = 0.60
    test_ratio: float = 0.20  # Ratio of test within pre-OOT data (if use_test_split = True)
    oot_ratio: float = 0.20  # Ratio for OOT (time-based split)
    oot_months: Optional[int] = None  # If set, use last N months for OOT instead of ratio
    min_oot_size: int = 50  # Minimum OOT samples

    # Orchestrator config
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Create output folder if it doesn't exist
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        # Validate ratios sum to 1
        total_ratio = self.train_ratio + self.test_ratio + self.oot_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Train, test and OOT ratios must sum to 1.0, got {total_ratio}")

        # Set run_id if not provided
        if self.run_id is None:
            from datetime import datetime
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create HPO aliases for backward compatibility
        self.hpo_trials = self.n_trials
        self.hpo_timeout_sec = self.optuna_timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'orchestrator':
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        orchestrator_dict = config_dict.pop('orchestrator', {})
        orchestrator = OrchestratorConfig(**orchestrator_dict)
        return cls(orchestrator=orchestrator, **config_dict)
