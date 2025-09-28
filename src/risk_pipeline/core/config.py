"""
Unified Configuration System for Risk Model Pipeline
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import warnings
import os
import math


@dataclass
class Config:
    """
    Comprehensive configuration for Risk Model Pipeline.
    All features controlled through this single configuration.
    """
    
    # ==================== DATA COLUMNS ====================
    target_column: str = 'target'
    id_column: Optional[str] = None
    time_column: Optional[str] = None
    weight_column: Optional[str] = None
    
    # ==================== DATA SPLITTING ====================
    # Test split configuration
    create_test_split: bool = True
    use_test_split: bool = True  # Backward-compatible alias for historical flag
    test_size: float = 0.2
    stratify_test: bool = True  # Preserve event rate in splits
    train_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    oot_ratio: Optional[float] = None
    equal_default_splits: bool = False
    
    # OOT (Out-of-Time) configuration
    oot_months: int = 3  # Last N months for OOT
    oot_size: float = 0.2  # If no time column, use random split
    
    # ==================== SCORING ====================
    enable_scoring: bool = False  # Default disabled
    score_model_name: str = 'best'  # Which model to use for scoring
    
    # ==================== DATA PREPROCESSING ====================
    # Numeric variables
    numeric_imputation_strategy: str = 'median'  # 'mean', 'median', 'mode', 'target'
    numeric_outlier_method: str = 'clip'  # 'clip', 'remove', 'none'
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    
    # Categorical variables
    categorical_imputation_strategy: str = 'missing'  # 'mode', 'missing'
    rare_category_threshold: float = 0.01
    max_categories: int = 20
    enable_tsfresh_features: bool = False
    tsfresh_feature_set: str = 'minimal'
    tsfresh_window: Optional[int] = None
    tsfresh_max_ids: Optional[int] = None
    tsfresh_auto_max_ids: Optional[int] = 3000
    tsfresh_n_jobs: int = 0
    tsfresh_cpu_fraction: Optional[float] = None
    tsfresh_use_multiprocessing: bool = True
    tsfresh_force_sequential_notebook: bool = False
    tsfresh_custom_fc_parameters: Optional[Dict[str, Any]] = None

    
    # ==================== WOE CONFIGURATION ====================
    # WOE calculation settings
    calculate_woe_all: bool = True  # Calculate WOE for all variables
    woe_optimization_metric: str = 'iv'  # 'iv' or 'gini'
    woe_max_bins: int = 10
    woe_min_bins: int = 2
    woe_min_bin_size: float = 0.05
    woe_max_bin_share: float = 1.0
    woe_micro_bins: int = 256
    woe_binning_strategy: str = 'iv_optimal'
    woe_monotonic_numeric: bool = True  # Enforce monotonicity for numeric
    woe_merge_insignificant: bool = True  # Merge insignificant bins for categorical
    woe_special_values: Optional[List] = None  # Special values to handle separately
    
    # ==================== UNIVARIATE ANALYSIS ====================
    calculate_univariate_gini: bool = True  # Calculate uni gini for all variables
    check_woe_degradation: bool = True  # Check if WOE reduces gini
    woe_degradation_threshold: float = 0.05  # Alert if gini drops more than this
    
    # ==================== FEATURE SELECTION ====================
    # Selection pipeline steps (order matters)
    selection_steps: List[str] = field(default_factory=lambda: [
        'univariate',    # Filter by univariate gini/IV
        'psi',          # Population Stability Index filter
        'vif',          # Variance Inflation Factor filter
        'correlation',   # Correlation clustering
        'iv',           # Information Value filter
        'boruta',       # Boruta selection
        'stepwise'      # Stepwise selection
    ])
    
    # Selection thresholds
    min_univariate_gini: float = 0.05
    max_psi: float = 0.25
    max_vif: float = 5.0
    max_correlation: float = 0.95
    min_iv: float = 0.02
    vif_sample_size: int = 5000
    univariate_gini_threshold: Optional[float] = None
    iv_threshold: Optional[float] = None
    psi_threshold: Optional[float] = None
    correlation_threshold: Optional[float] = None
    monthly_psi_threshold: Optional[float] = None
    oot_psi_threshold: Optional[float] = None
    vif_threshold: Optional[float] = None
    
    # Stepwise configuration
    stepwise_method: str = 'forward'  # 'forward', 'backward', 'stepwise', 'forward_1se'
    stepwise_max_features: int = 30
    stepwise_min_features: int = 5
    stepwise_cv_folds: int = 5
    
    # Boruta configuration
    boruta_estimator: str = 'lightgbm'  # 'lightgbm' or 'randomforest'
    boruta_max_iter: int = 100
    boruta_alpha: float = 0.05
    
    # Noise sentinel
    use_noise_sentinel: bool = True
    noise_threshold: float = 0.5  # Drop if noise ranks higher than this percentile
    
    # ==================== MODEL TRAINING ====================
    # Algorithms to use
    algorithms: List[str] = field(default_factory=lambda: [
        'logistic',
        'gam',  # Generalized Additive Model
        'catboost',
        'lightgbm',
        'xgboost',
        'randomforest',
        'extratrees'
    ])
    
    # Training configuration
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'
    early_stopping_rounds: int = 50
    
    # Hyperparameter optimization
    use_optuna: bool = True
    n_trials: int = 100
    optuna_timeout: int = 3600  # seconds
    try_mlp: bool = False
    hpo_method: str = 'optuna'
    hpo_trials: int = 50
    hpo_timeout_sec: Optional[int] = None
    max_train_oot_gap: Optional[float] = None
    model_selection_method: str = 'gini_oot'
    model_stability_weight: float = 0.0
    min_gini_threshold: float = 0.5
    
    # Dual pipeline (WOE + RAW)
    enable_dual: bool = True
    
    # ==================== CALIBRATION ====================
    # Stage 1 calibration
    calibration_method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    calibration_stage1_method: Optional[str] = None
    calibration_cv_folds: int = 3
    
    # Stage 2 calibration
    enable_stage2_calibration: bool = True
    stage2_target_rate: Optional[float] = None
    stage2_lower_bound: float = 0.8
    stage2_upper_bound: float = 1.2
    stage2_confidence_level: float = 0.95
    stage2_method: str = 'lower_mean'
    calibration_stage2_method: Optional[str] = None
    
    # ==================== RISK BANDS ====================
    optimize_risk_bands: bool = True
    n_risk_bands: int = 10
    risk_band_method: str = 'pd_constraints'  # 'pd_constraints', 'quantile', 'equal_width', 'optimal'
    risk_band_min_bins: int = 7
    risk_band_max_bins: int = 10
    risk_band_micro_bins: int = 1000
    risk_band_min_weight: float = 0.05
    risk_band_max_weight: float = 0.30
    risk_band_hhi_threshold: float = 0.15
    risk_band_binomial_pass_weight: float = 0.85
    risk_band_alpha: float = 0.05
    risk_band_pd_dr_tolerance: float = 1e-4
    risk_band_max_iterations: int = 100
    risk_band_max_phase_iterations: int = 50
    risk_band_early_stop_rounds: int = 10
    
    # Statistical tests for risk bands
    risk_band_tests: List[str] = field(default_factory=lambda: [
        'binomial',
        'hosmer_lemeshow',
        'herfindahl'
    ])
    
    # Business risk ratings (optional)
    business_risk_ratings: Optional[List[str]] = field(default_factory=lambda: [
        'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'
    ])
    
    # ==================== REPORTING ====================
    # SHAP analysis
    calculate_shap: bool = True
    shap_sample_size: int = 1000
    gam_sample_size: int = 20000
    gam_max_iter: int = 500
    
    # Variable dictionary
    include_variable_dictionary: bool = True
    
    # Report components
    report_components: List[str] = field(default_factory=lambda: [
        'model_comparison',
        'feature_importance',
        'woe_bins',
        'univariate_analysis',
        'psi_analysis',
        'calibration_curves',
        'risk_bands',
        'statistical_tests',
        'shap_analysis'
    ])
    
    # ==================== OUTPUT ====================
    output_folder: str = 'outputs'
    model_name_prefix: str = 'risk_model'
    save_models: bool = True
    save_reports: bool = True
    save_plots: bool = True
    
    # File formats
    report_format: str = 'excel'  # 'excel', 'html', 'pdf'
    plot_format: str = 'png'  # 'png', 'svg', 'pdf'
    
    # ==================== SYSTEM ====================
    random_state: int = 42
    n_jobs: int = -1
    cpu_fraction: float = 0.8
    verbose: bool = True
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Memory optimization
    use_memory_optimization: bool = True
    chunk_size: Optional[int] = None
    
    def _apply_ratio_defaults(self) -> None:
        """Normalize ratio-based split inputs into size parameters."""
        provided: Dict[str, float] = {}
        for attr in ('train_ratio', 'test_ratio', 'oot_ratio'):
            value = getattr(self, attr, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{attr} must be a numeric value between 0 and 1") from exc
            if numeric < 0 or numeric > 1:
                raise ValueError(f"{attr} must be between 0 and 1")
            provided[attr] = numeric

        if not provided:
            fallback_test = getattr(self, 'test_size', None)
            if fallback_test is not None:
                provided['test_ratio'] = float(fallback_test)
            fallback_oot = getattr(self, 'oot_size', None)
            if fallback_oot is not None:
                provided['oot_ratio'] = float(fallback_oot)
            remaining = max(0.0, 1.0 - provided.get('test_ratio', 0.0) - provided.get('oot_ratio', 0.0))
            if remaining > 0:
                provided['train_ratio'] = remaining

        if not provided:
            return

        total = sum(provided.values())
        if total > 1.0 + 1e-6:
            raise ValueError("train_ratio + test_ratio + oot_ratio cannot exceed 1.0")

        if 'test_ratio' in provided:
            self.test_size = provided['test_ratio']
            self.test_ratio = provided['test_ratio']
        else:
            fallback = getattr(self, 'test_size', 0.0)
            self.test_ratio = self.test_size = float(fallback or 0.0)

        if 'oot_ratio' in provided:
            self.oot_size = provided['oot_ratio']
            self.oot_ratio = provided['oot_ratio']
        else:
            fallback = getattr(self, 'oot_size', 0.0)
            self.oot_ratio = self.oot_size = float(fallback or 0.0)

        if 'train_ratio' in provided:
            self.train_ratio = provided['train_ratio']
        else:
            remaining = max(0.0, 1.0 - self.test_ratio - self.oot_ratio)
            self.train_ratio = remaining if remaining > 0 else None

        # Keep create_test_split and use_test_split in sync
        if not self.use_test_split and self.create_test_split:
            self.create_test_split = False
        if self.create_test_split and not self.use_test_split:
            self.use_test_split = True

    def validate(self) -> None:
        """Validate configuration parameters"""
        # Validate splits
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.oot_size <= 0 or self.oot_size >= 1:
            raise ValueError("oot_size must be between 0 and 1")
        
        if self.test_size + self.oot_size >= 0.8:
            warnings.warn("Test + OOT size is very large, leaving little for training")
        
        # Validate WOE parameters
        if self.woe_min_bins < 2:
            raise ValueError("woe_min_bins must be at least 2")
        
        if self.woe_max_bins < self.woe_min_bins:
            raise ValueError("woe_max_bins must be >= woe_min_bins")
        
        if self.woe_min_bin_size <= 0 or self.woe_min_bin_size > 0.5:
            raise ValueError("woe_min_bin_size must be between 0 and 0.5")
        
        # Validate selection parameters
        if self.max_correlation <= 0 or self.max_correlation > 1:
            raise ValueError("max_correlation must be between 0 and 1")
        
        if self.max_vif <= 1:
            raise ValueError("max_vif must be greater than 1")
        
        # Validate model parameters
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        
        # Validate risk bands
        if self.n_risk_bands < 2:
            raise ValueError("n_risk_bands must be at least 2")
        
        # Check algorithm availability
        valid_algorithms = [
            'logistic', 'gam', 'catboost', 'lightgbm', 
            'xgboost', 'randomforest', 'extratrees', 'woe_boost', 'woe_li', 'shao', 'xbooster'
        ]
        for algo in self.algorithms:
            if algo not in valid_algorithms:
                raise ValueError(f"Unknown algorithm: {algo}")
        
        # Check stepwise method
        valid_stepwise = ['forward', 'backward', 'stepwise', 'forward_1se']
        if self.stepwise_method not in valid_stepwise:
            raise ValueError(f"stepwise_method must be one of {valid_stepwise}")
        
        # Check calibration method
        valid_calibration = ['isotonic', 'sigmoid']
        if self.calibration_method not in valid_calibration:
            raise ValueError(f"calibration_method must be one of {valid_calibration}")

        if self.stage2_target_rate is not None:
            if not 0 <= float(self.stage2_target_rate) <= 1:
                raise ValueError("stage2_target_rate must be between 0 and 1")

        if self.stage2_lower_bound <= 0 or self.stage2_upper_bound <= 0:
            raise ValueError("stage2 lower/upper bounds must be positive")
        if self.stage2_lower_bound > self.stage2_upper_bound:
            raise ValueError("stage2_lower_bound cannot exceed stage2_upper_bound")
        if self.cpu_fraction <= 0 or self.cpu_fraction > 1:
            raise ValueError("cpu_fraction must be between 0 and 1")

        cpu_fraction = getattr(self, 'tsfresh_cpu_fraction', None)
        if cpu_fraction is not None:
            frac = float(cpu_fraction)
            if frac < 0 or frac > 1:
                raise ValueError("tsfresh_cpu_fraction must be between 0 and 1")


    def _resolve_parallel_jobs(self, value: Optional[int]) -> int:
        """Resolve requested parallel jobs into an absolute worker count."""

        cpu_total = max(1, os.cpu_count() or 1)
        fraction = float(getattr(self, 'cpu_fraction', 0.8) or 0.8)
        if value is None:
            return max(1, math.floor(cpu_total * fraction))
        if value < 0:
            return max(1, math.floor(cpu_total * fraction))
        return int(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def __post_init__(self):
        """Post-initialization validation and legacy compatibility"""
        self._apply_ratio_defaults()
        self.validate()
        self._set_backward_compatibility_aliases()
        self.n_jobs = self._resolve_parallel_jobs(getattr(self, 'n_jobs', None))
        if getattr(self, 'tsfresh_cpu_fraction', None) is None:
            self.tsfresh_cpu_fraction = self.cpu_fraction

    def _set_backward_compatibility_aliases(self) -> None:
        """Expose legacy attribute names expected by older modules"""
        # Column aliases
        self.id_col = getattr(self, 'id_column', None)
        self.time_col = getattr(self, 'time_column', None)
        self.target_col = getattr(self, 'target_column', None)

        self.use_test_split = getattr(self, 'use_test_split', getattr(self, 'create_test_split', True))
        self.create_test_split = self.use_test_split
        test_ratio_attr = getattr(self, 'test_ratio', None)
        if test_ratio_attr is None:
            test_ratio_attr = getattr(self, 'test_size', 0.2)
        self.test_ratio = float(test_ratio_attr or 0.0)
        self.test_size = self.test_ratio

        oot_ratio_attr = getattr(self, 'oot_ratio', None)
        if oot_ratio_attr is None:
            oot_ratio_attr = getattr(self, 'oot_size', 0.0)
        self.oot_ratio = float(oot_ratio_attr or 0.0)
        self.oot_size = self.oot_ratio

        if getattr(self, 'train_ratio', None) is None:
            remaining = max(0.0, 1.0 - self.test_ratio - self.oot_ratio)
            self.train_ratio = remaining if remaining > 0 else None
        else:
            self.train_ratio = float(self.train_ratio)
        if not hasattr(self, 'stratify_test_split'):
            self.stratify_test_split = getattr(self, 'stratify_test', True)

        if not hasattr(self, 'numeric_imputation'):
            self.numeric_imputation = getattr(self, 'numeric_imputation_strategy', 'median')
        if not hasattr(self, 'outlier_method'):
            self.outlier_method = getattr(self, 'numeric_outlier_method', 'clip')
        if not hasattr(self, 'min_category_freq'):
            self.min_category_freq = getattr(self, 'rare_category_threshold', 0.01)

        # WOE/binning aliases
        binning_default = getattr(self, 'woe_binning_strategy', getattr(self, 'woe_binning_method', 'optimized'))
        self.binning_method = getattr(self, 'binning_method', binning_default)
        self.woe_binning_method = self.binning_method
        self.woe_binning_strategy = self.binning_method
        self.max_bins = getattr(self, 'max_bins', getattr(self, 'woe_max_bins', 10))
        self.min_bin_size = getattr(self, 'min_bin_size', getattr(self, 'woe_min_bin_size', 0.05))
        self.monotonic_woe = getattr(self, 'monotonic_woe', getattr(self, 'woe_monotonic_numeric', True))
        self.max_abs_woe = getattr(self, 'max_abs_woe', getattr(self, 'woe_max_abs', None))

        # Selection aliases
        if not hasattr(self, 'selection_order'):
            self.selection_order = getattr(self, 'selection_steps', [
                'psi',
                'univariate',
                'iv',
                'correlation',
                'boruta',
                'stepwise'
            ])

        if getattr(self, 'psi_threshold', None) is not None:
            self.max_psi = float(self.psi_threshold)
        else:
            self.psi_threshold = getattr(self, 'max_psi', 0.25)

        if getattr(self, 'monthly_psi_threshold', None) is None:
            self.monthly_psi_threshold = max(0.05, self.psi_threshold / 2)
        if getattr(self, 'oot_psi_threshold', None) is None:
            self.oot_psi_threshold = self.psi_threshold

        if getattr(self, 'correlation_threshold', None) is not None:
            self.max_correlation = float(self.correlation_threshold)
        else:
            self.correlation_threshold = getattr(self, 'max_correlation', 0.95)

        if getattr(self, 'vif_threshold', None) is not None:
            self.max_vif = float(self.vif_threshold)
        else:
            self.vif_threshold = getattr(self, 'max_vif', 5.0)

        if getattr(self, 'iv_threshold', None) is not None:
            self.min_iv = float(self.iv_threshold)
        else:
            self.iv_threshold = getattr(self, 'min_iv', 0.02)

        if getattr(self, 'univariate_gini_threshold', None) is not None:
            self.min_univariate_gini = float(self.univariate_gini_threshold)
        else:
            self.univariate_gini_threshold = getattr(self, 'min_univariate_gini', getattr(self, 'gini_threshold', 0.05))

        self.gam_sample_size = int(getattr(self, 'gam_sample_size', 20000))
        self.gam_max_iter = int(getattr(self, 'gam_max_iter', 500))
        self.max_features = getattr(self, 'max_features', getattr(self, 'stepwise_max_features', 30))
        self.max_features_per_cluster = getattr(self, 'max_features_per_cluster', 1)

        if not hasattr(self, 'band_method'):
            self.band_method = getattr(self, 'risk_band_method', 'pd_constraints')
        if not hasattr(self, 'selection_method'):
            self.selection_method = getattr(self, 'stepwise_method', 'forward')

        # Noise sentinel defaults
        if not hasattr(self, 'enable_noise_sentinel'):
            self.enable_noise_sentinel = getattr(self, 'use_noise_sentinel', False)
        self.noise_threshold = getattr(self, 'noise_threshold', 0.5)

        if getattr(self, 'calibration_stage1_method', None):
            self.calibration_method = self.calibration_stage1_method
        else:
            self.calibration_stage1_method = self.calibration_method

        if getattr(self, 'calibration_stage2_method', None):
            self.stage2_method = self.calibration_stage2_method
        else:
            self.calibration_stage2_method = self.stage2_method

        # Hyperparameter optimization aliases
        if not hasattr(self, 'hpo_method'):
            self.hpo_method = 'optuna' if getattr(self, 'use_optuna', False) else 'random'
        self.hpo_trials = int(getattr(self, 'hpo_trials', getattr(self, 'n_trials', 100)))
        self.n_trials = self.hpo_trials
        optuna_timeout = getattr(self, 'optuna_timeout', None)
        if getattr(self, 'hpo_timeout_sec', None) is None:
            self.hpo_timeout_sec = optuna_timeout
        else:
            self.optuna_timeout = self.hpo_timeout_sec

        self.max_train_oot_gap = getattr(self, 'max_train_oot_gap', getattr(self, 'max_train_oot_delta', None))
        self.model_selection_method = getattr(self, 'model_selection_method', 'gini_oot')
        self.model_stability_weight = float(getattr(self, 'model_stability_weight', 0.0))
        self.min_gini_threshold = float(getattr(self, 'min_gini_threshold', 0.5))
        self.try_mlp = bool(getattr(self, 'try_mlp', False))

        algorithms_attr = getattr(self, 'algorithms', [])
        if isinstance(algorithms_attr, str):
            self.algorithms = [algorithms_attr]
        else:
            self.algorithms = list(algorithms_attr) if isinstance(algorithms_attr, (list, tuple, set)) else list(algorithms_attr or [])
        default_algorithms = ['logistic', 'gam', 'catboost', 'lightgbm', 'xgboost', 'randomforest', 'extratrees', 'woe_boost', 'woe_li', 'shao', 'xbooster']
        if not getattr(self, 'model_type', None):
            if self.algorithms and set(self.algorithms) != set(default_algorithms):
                self.model_type = list(self.algorithms)
            else:
                self.model_type = 'all'

        # Pipeline toggles
        self.enable_dual_pipeline = getattr(self, 'enable_dual_pipeline', getattr(self, 'enable_dual', False))
        if not hasattr(self, 'enable_woe'):
            self.enable_woe = getattr(self, 'use_woe', True)

        if not hasattr(self, 'enable_calibration'):
            self.enable_calibration = getattr(self, 'enable_stage2_calibration', False)
        if not hasattr(self, 'stage2_method'):
            self.stage2_method = 'lower_mean'

        # Calendar defaults
        self.snapshot_column = getattr(self, 'snapshot_column', getattr(self, 'snapshot_month_column', 'snapshot_month'))
        self.oot_months = getattr(self, 'oot_months', getattr(self, 'n_oot_months', 3))

        # TSFresh compatibility
        legacy_tsfresh_flags = [getattr(self, 'use_tsfresh', None), getattr(self, 'enable_tsfresh', None)]
        for flag in legacy_tsfresh_flags:
            if flag is not None:
                self.enable_tsfresh_features = bool(flag)

        legacy_fc = getattr(self, 'tsfresh_params', None) or getattr(self, 'tsfresh_settings', None) or getattr(self, 'tsfresh_fc_parameters', None)
        if legacy_fc and not getattr(self, 'tsfresh_custom_fc_parameters', None):
            self.tsfresh_custom_fc_parameters = legacy_fc

        legacy_jobs = getattr(self, 'tsfresh_jobs', None)
        if legacy_jobs is not None:
            try:
                self.tsfresh_n_jobs = int(legacy_jobs)
            except (TypeError, ValueError):
                pass

        if getattr(self, 'tsfresh_feature_set', None) is None:
            self.tsfresh_feature_set = 'minimal'
