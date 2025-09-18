"""
Recommended configuration settings for different scenarios
"""

from risk_pipeline.core.config import Config

def get_conservative_config(**overrides):
    """
    Conservative configuration that keeps more features
    Good for initial exploration
    """
    config_dict = {
        # Basic settings
        'target_col': 'target',
        'id_col': 'app_id',
        'time_col': 'app_dt',
        'random_state': 42,

        # Very permissive thresholds
        'iv_min': 0.0001,  # Keep almost all features
        'iv_threshold': 0.0001,  # Very low IV threshold
        'iv_high_threshold': 10.0,  # Only remove extremely suspicious features
        'psi_threshold': 1.0,  # Allow high PSI (normal is 0.25)
        'rho_threshold': 0.95,  # Only remove very high correlations
        'correlation_threshold': 0.95,  # Alternative name
        'vif_threshold': 20.0,  # Allow some multicollinearity
        'rare_threshold': 0.005,  # 0.5% threshold for rare categories

        # WOE settings
        'enable_woe': True,
        'n_bins': 5,  # Fewer bins for stability
        'max_bins': 10,
        'min_bin_size': 0.05,
        'woe_monotonic': False,  # Don't force monotonic for more flexibility

        # Pipeline settings
        'enable_dual_pipeline': False,  # Single pipeline for simplicity
        'enable_noise_sentinel': False,  # No noise sentinel
        'enable_calibration': True,
        'enable_scoring': False,

        # Feature selection - minimal filtering
        'selection_order': ['psi', 'correlation', 'iv'],  # Skip VIF and complex methods
        'selection_method': 'forward',
        'max_features': 30,  # Allow many features
        'min_features': 3,

        # Model settings
        'model_type': 'all',  # Try all models
        'use_optuna': False,  # No optimization for speed
        'n_trials': 10,
        'cv_folds': 3,

        # Split settings
        'test_ratio': 0.2,
        'oot_months': 0,  # No OOT initially
        'equal_default_splits': False,  # Don't force equal splits

        # Output settings
        'output_folder': 'output',
        'save_plots': True,
        'save_model': True
    }

    # Apply overrides
    config_dict.update(overrides)

    return Config(**config_dict)


def get_standard_config(**overrides):
    """
    Standard configuration with typical thresholds
    Good for production models
    """
    config_dict = {
        # Basic settings
        'target_col': 'target',
        'id_col': 'app_id',
        'time_col': 'app_dt',
        'random_state': 42,

        # Standard thresholds
        'iv_min': 0.02,  # Standard IV threshold
        'iv_threshold': 0.02,
        'iv_high_threshold': 2.0,  # Remove very high IV (likely leakage)
        'psi_threshold': 0.25,  # Standard PSI threshold
        'rho_threshold': 0.85,  # Standard correlation threshold
        'correlation_threshold': 0.85,
        'vif_threshold': 10.0,  # Standard VIF threshold
        'rare_threshold': 0.01,  # 1% threshold for rare categories

        # WOE settings
        'enable_woe': True,
        'n_bins': 10,  # Standard number of bins
        'max_bins': 20,
        'min_bin_size': 0.05,
        'woe_monotonic': True,  # Force monotonic for interpretability

        # Pipeline settings
        'enable_dual_pipeline': True,  # Try both WOE and raw
        'enable_noise_sentinel': True,  # Check for overfitting
        'enable_calibration': True,
        'enable_scoring': False,

        # Feature selection - full pipeline
        'selection_order': ['psi', 'vif', 'correlation', 'iv', 'stepwise'],
        'selection_method': 'forward',
        'max_features': 15,  # Reasonable number of features
        'min_features': 5,

        # Model settings
        'model_type': 'all',  # Try all models
        'use_optuna': False,  # Can enable for better performance
        'n_trials': 50,
        'cv_folds': 5,

        # Split settings
        'test_ratio': 0.2,
        'oot_months': 3,  # 3 months OOT
        'equal_default_splits': True,  # Equal default rates

        # Output settings
        'output_folder': 'output',
        'save_plots': True,
        'save_model': True
    }

    # Apply overrides
    config_dict.update(overrides)

    return Config(**config_dict)


def get_strict_config(**overrides):
    """
    Strict configuration with tight thresholds
    Good for high-stakes models requiring stability
    """
    config_dict = {
        # Basic settings
        'target_col': 'target',
        'id_col': 'app_id',
        'time_col': 'app_dt',
        'random_state': 42,

        # Strict thresholds
        'iv_min': 0.1,  # High IV requirement
        'iv_threshold': 0.1,
        'iv_high_threshold': 1.0,  # Remove high IV features
        'psi_threshold': 0.1,  # Very strict PSI
        'rho_threshold': 0.7,  # Lower correlation threshold
        'correlation_threshold': 0.7,
        'vif_threshold': 5.0,  # Strict VIF threshold
        'rare_threshold': 0.02,  # 2% threshold for rare categories

        # WOE settings
        'enable_woe': True,
        'n_bins': 5,  # Fewer bins for stability
        'max_bins': 10,
        'min_bin_size': 0.1,  # Larger minimum bin size
        'woe_monotonic': True,  # Force monotonic

        # Pipeline settings
        'enable_dual_pipeline': False,  # Single pipeline
        'enable_noise_sentinel': True,  # Check for overfitting
        'enable_calibration': True,
        'enable_scoring': False,

        # Feature selection - all methods
        'selection_order': ['psi', 'vif', 'correlation', 'iv', 'boruta', 'stepwise'],
        'selection_method': 'stepwise',  # Full stepwise selection
        'max_features': 10,  # Limit features
        'min_features': 3,

        # Model settings
        'model_type': 'logistic',  # Simple, interpretable model
        'use_optuna': True,  # Optimize hyperparameters
        'n_trials': 100,
        'cv_folds': 10,

        # Split settings
        'test_ratio': 0.3,  # Larger test set
        'oot_months': 6,  # 6 months OOT
        'equal_default_splits': True,  # Equal default rates

        # Output settings
        'output_folder': 'output',
        'save_plots': True,
        'save_model': True
    }

    # Apply overrides
    config_dict.update(overrides)

    return Config(**config_dict)


# Usage examples:
if __name__ == "__main__":
    # Example 1: Conservative config for initial exploration
    config1 = get_conservative_config(
        target_col='default_flag',
        output_folder='exploration_output'
    )
    print("Conservative config created")

    # Example 2: Standard config for production
    config2 = get_standard_config(
        enable_dual_pipeline=False,  # Override to single pipeline
        max_features=20  # Allow more features
    )
    print("Standard config created")

    # Example 3: Strict config for regulated environment
    config3 = get_strict_config(
        model_type='logistic',  # Only logistic regression
        use_optuna=False  # No black-box optimization
    )
    print("Strict config created")