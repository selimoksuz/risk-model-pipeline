"""
Unified Configuration System for Risk Model Pipeline
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import warnings


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
    test_size: float = 0.2
    stratify_test: bool = True  # Preserve event rate in splits
    
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
    tsfresh_n_jobs: int = 0
    tsfresh_custom_fc_parameters: Optional[Dict[str, Any]] = None

    
    # ==================== WOE CONFIGURATION ====================
    # WOE calculation settings
    calculate_woe_all: bool = True  # Calculate WOE for all variables
    woe_optimization_metric: str = 'iv'  # 'iv' or 'gini'
    woe_max_bins: int = 10
    woe_min_bins: int = 2
    woe_min_bin_size: float = 0.05
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
    
    # Dual pipeline (WOE + RAW)
    enable_dual: bool = True
    
    # ==================== CALIBRATION ====================
    # Stage 1 calibration
    calibration_method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    calibration_cv_folds: int = 3
    
    # Stage 2 calibration
    enable_stage2_calibration: bool = True
    stage2_lower_bound: float = 0.8
    stage2_upper_bound: float = 1.2
    
    # ==================== RISK BANDS ====================
    optimize_risk_bands: bool = True
    n_risk_bands: int = 10
    risk_band_method: str = 'quantile'  # 'quantile', 'equal_width', 'optimal'
    
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
    verbose: bool = True
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Memory optimization
    use_memory_optimization: bool = True
    chunk_size: Optional[int] = None
    
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
            'xgboost', 'randomforest', 'extratrees'
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
        self.validate()
        self._set_backward_compatibility_aliases()

    def _set_backward_compatibility_aliases(self) -> None:
        """Expose legacy attribute names expected by older modules"""
        # Column aliases
        self.id_col = getattr(self, 'id_column', None)
        self.time_col = getattr(self, 'time_column', None)
        self.target_col = getattr(self, 'target_column', None)

        # Pipeline toggles
        self.enable_dual_pipeline = getattr(self, 'enable_dual', False)
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