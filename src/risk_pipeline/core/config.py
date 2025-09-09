"""Configuration module for Risk Model Pipeline - Fixed version"""
from dataclasses import dataclass
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


class Config:
    """Main configuration for Risk Model Pipeline with proper parameter override support"""

    def __init__(self, **kwargs):
        """Initialize config with keyword arguments"""
        # Basic settings
        self.target_col = kwargs.get('target_col', 'target')
        self.id_col = kwargs.get('id_col', 'app_id')
        self.time_col = kwargs.get('time_col', 'app_dt')
        self.random_state = kwargs.get('random_state', 42)

        # Feature selection parameters
        self.iv_min = kwargs.get('iv_min', 0.02)
        self.iv_high_threshold = kwargs.get('iv_high_threshold', 0.5)
        self.psi_threshold = kwargs.get('psi_threshold', 0.25)
        self.rho_threshold = kwargs.get('rho_threshold', 0.90)
        self.vif_threshold = kwargs.get('vif_threshold', 5.0)
        self.rare_threshold = kwargs.get('rare_threshold', 0.01)
        self.cluster_top_k = kwargs.get('cluster_top_k', 2)

        # WOE settings
        self.n_bins = kwargs.get('n_bins', 10)
        self.min_bin_size = kwargs.get('min_bin_size', 0.05)
        self.woe_monotonic = kwargs.get('woe_monotonic', False)
        self.max_abs_woe = kwargs.get('max_abs_woe', None)
        self.handle_missing = kwargs.get('handle_missing', 'as_category')

        # Model training settings - FIXED n_trials
        self.use_optuna = kwargs.get('use_optuna', True)
        self.n_trials = kwargs.get('n_trials', 100)  # This will work properly now
        self.optuna_timeout = kwargs.get('optuna_timeout', None)
        self.cv_folds = kwargs.get('cv_folds', 5)

        # Model selection criteria
        self.model_selection_method = kwargs.get('model_selection_method', 'gini_oot')
        self.max_train_oot_gap = kwargs.get('max_train_oot_gap', None)
        self.model_stability_weight = kwargs.get('model_stability_weight', 0.0)
        self.min_gini_threshold = kwargs.get('min_gini_threshold', 0.5)

        # Feature selection settings
        self.use_boruta = kwargs.get('use_boruta', True)
        self.forward_1se = kwargs.get('forward_1se', True)
        self.max_features = kwargs.get('max_features', 20)
        self.min_features = kwargs.get('min_features', 3)
        self.use_noise_sentinel = kwargs.get('use_noise_sentinel', True)

        # Imputation settings
        self.imputation_strategy = kwargs.get('imputation_strategy', 'median')

        # RAW pipeline settings
        self.raw_outlier_method = kwargs.get('raw_outlier_method', 'none')
        self.raw_outlier_threshold = kwargs.get('raw_outlier_threshold', 3.0)
        self.raw_scaler_type = kwargs.get('raw_scaler_type', 'standard')

        # Dual pipeline settings
        self.enable_dual_pipeline = kwargs.get('enable_dual_pipeline', True)

        # Output settings
        self.output_folder = kwargs.get('output_folder', 'output')
        self.output_excel_path = kwargs.get('output_excel_path', None)
        self.write_csv = kwargs.get('write_csv', False)
        self.run_id = kwargs.get('run_id', None)

        # Data splitting settings
        self.use_test_split = kwargs.get('use_test_split', True)
        self.train_ratio = kwargs.get('train_ratio', 0.60)
        self.test_ratio = kwargs.get('test_ratio', 0.20)
        self.oot_ratio = kwargs.get('oot_ratio', 0.20)
        self.oot_months = kwargs.get('oot_months', None)
        self.min_oot_size = kwargs.get('min_oot_size', 50)

        # Initialize orchestrator config
        self.orchestrator = OrchestratorConfig()

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

    def to_dict(self):
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'orchestrator':
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)
