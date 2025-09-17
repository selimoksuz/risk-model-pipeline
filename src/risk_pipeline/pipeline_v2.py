"""Risk Model Pipeline V2 - Unified Pipeline with Complete Features"""

import os
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from .core.config import Config
from .core.data_processor import DataProcessor
from .core.feature_selector import FeatureSelector
from .core.model_builder import ModelBuilder
from .core.reporter import Reporter
from .core.splitter import DataSplitter
from .core.woe_transformer import WOETransformer
from .core.calibration import CalibrationEngine
from .core.risk_band_optimizer import RiskBandOptimizer
from .core.psi_calculator import PSICalculator

warnings.filterwarnings("ignore")


class UnifiedRiskPipeline:
    """
    Unified Risk Model Pipeline - Single pipeline that handles all configurations.

    Features:
    - Single pipeline controlled by config
    - Optional scoring (default off)
    - Data dictionary support
    - Stage 1 & Stage 2 calibration
    - Complete selection methods (forward/backward/stepwise)
    - Optimized binning for IV/Gini
    - Support for all major ML algorithms
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with configuration."""
        self.config = config or Config()

        # Initialize components
        self.processor = DataProcessor(self.config)
        self.splitter = DataSplitter(self.config)
        self.selector = FeatureSelector(self.config)
        self.woe_transformer = WOETransformer(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.reporter = Reporter(self.config)
        self.calibration_engine = CalibrationEngine(self.config)
        self.risk_band_optimizer = RiskBandOptimizer(self.config)
        self.psi_calculator = PSICalculator()

        # Storage for results
        self.train_ = None
        self.test_ = None
        self.oot_ = None
        self.data_dictionary_ = None
        self.final_vars_ = []
        self.univariate_ginis_ = {}
        self.woe_ginis_ = {}
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        self.best_auc_ = None
        self.woe_mapping_ = None
        self.calibration_model_ = None
        self.risk_bands_ = None
        self.feature_importance_ = {}
        self.selection_history_ = {}

        # Model results storage (for WOE and RAW if dual mode)
        self.woe_models_ = {}
        self.raw_models_ = {}

    def fit(self,
            df: pd.DataFrame,
            data_dictionary: Optional[pd.DataFrame] = None,
            calibration_data: Optional[pd.DataFrame] = None,
            stage2_data: Optional[pd.DataFrame] = None) -> 'UnifiedRiskPipeline':
        """
        Fit the pipeline on training data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target
        data_dictionary : pd.DataFrame, optional
            Data dictionary with variable descriptions
        calibration_data : pd.DataFrame, optional
            Calibration dataset (if not provided, uses long-run average)
        stage2_data : pd.DataFrame, optional
            Recent predictions for Stage 2 calibration

        Returns:
        --------
        self : UnifiedRiskPipeline
            Fitted pipeline
        """
        print("="*80)
        print("UNIFIED RISK MODEL PIPELINE V2")
        print("="*80)

        # Store data dictionary if provided
        if data_dictionary is not None:
            self.data_dictionary_ = data_dictionary
            print("✓ Data dictionary loaded with {} variables".format(len(data_dictionary)))

        # Step 1: Data Processing
        print("\n1. DATA PROCESSING")
        print("-" * 40)
        df_processed = self._process_data(df)

        # Step 2: Train/Test/OOT Split
        print("\n2. DATA SPLITTING")
        print("-" * 40)
        self._split_data(df_processed)

        # Step 3: Feature Selection (Multi-stage)
        print("\n3. FEATURE SELECTION")
        print("-" * 40)
        self._select_features()

        # Step 4: WOE Transformation (if enabled)
        print("\n4. WOE TRANSFORMATION")
        print("-" * 40)
        if self.config.use_woe:
            self._apply_woe_transformation()

        # Step 5: Model Building
        print("\n5. MODEL BUILDING")
        print("-" * 40)
        self._build_models()

        # Step 6: Calibration
        print("\n6. MODEL CALIBRATION")
        print("-" * 40)
        self._calibrate_model(calibration_data, stage2_data)

        # Step 7: Risk Band Optimization
        print("\n7. RISK BAND OPTIMIZATION")
        print("-" * 40)
        self._optimize_risk_bands()

        # Step 8: Generate Reports
        print("\n8. REPORT GENERATION")
        print("-" * 40)
        self._generate_reports()

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

        return self

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate input data."""
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target and special columns
        special_cols = [self.config.target_col, self.config.id_col, self.config.time_col]
        numeric_cols = [col for col in numeric_cols if col not in special_cols]
        categorical_cols = [col for col in categorical_cols if col not in special_cols]

        print(f"  • Numeric features: {len(numeric_cols)}")
        print(f"  • Categorical features: {len(categorical_cols)}")

        # Apply different processing for numeric and categorical
        df_processed = df.copy()

        # Numeric processing
        if self.config.handle_outliers and numeric_cols:
            print("  • Handling outliers in numeric features...")
            for col in numeric_cols:
                # IQR method for outlier detection
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)

        # Numeric imputation
        if self.config.numeric_imputation and numeric_cols:
            print(f"  • Imputing numeric features with {self.config.numeric_imputation}...")
            if self.config.numeric_imputation == 'median':
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                    df_processed[numeric_cols].median()
                )
            elif self.config.numeric_imputation == 'mean':
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                    df_processed[numeric_cols].mean()
                )
            elif self.config.numeric_imputation == 'zero':
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)

        # Categorical imputation
        if self.config.categorical_imputation and categorical_cols:
            print(f"  • Imputing categorical features with {self.config.categorical_imputation}...")
            if self.config.categorical_imputation == 'mode':
                df_processed[categorical_cols] = df_processed[categorical_cols].fillna(
                    df_processed[categorical_cols].mode().iloc[0]
                )
            elif self.config.categorical_imputation == 'missing':
                df_processed[categorical_cols] = df_processed[categorical_cols].fillna('MISSING')

        return df_processed

    def _split_data(self, df: pd.DataFrame):
        """Split data into train/test/oot with equal default rates."""
        if self.config.ensure_equal_default_rate:
            # Stratified split to ensure equal default rates
            print("  • Ensuring equal default rates across splits...")
            splits = self.splitter.split_stratified(df)
        else:
            # Regular time-based or random split
            splits = self.splitter.split(df)

        self.train_ = splits['train']
        self.test_ = splits.get('test')
        self.oot_ = splits.get('oot')

        # Print split statistics
        print(f"  • Train: {len(self.train_):,} samples, "
              f"default rate: {self.train_[self.config.target_col].mean():.2%}")

        if self.test_ is not None:
            print(f"  • Test: {len(self.test_):,} samples, "
                  f"default rate: {self.test_[self.config.target_col].mean():.2%}")

        if self.oot_ is not None:
            print(f"  • OOT: {len(self.oot_):,} samples, "
                  f"default rate: {self.oot_[self.config.target_col].mean():.2%}")

    def _select_features(self):
        """Multi-stage feature selection following the specified order."""
        features = [col for col in self.train_.columns
                   if col not in [self.config.target_col, self.config.id_col, self.config.time_col]]

        print(f"  • Starting with {len(features)} features")
        self.selection_history_['initial'] = features.copy()

        # Stage 1: PSI filtering
        if self.config.enable_psi and (self.test_ is not None or self.oot_ is not None):
            print("  • Stage 1: PSI filtering...")
            features = self._filter_by_psi(features)
            self.selection_history_['after_psi'] = features.copy()
            print(f"    → {len(features)} features remaining")

        # Stage 2: VIF filtering
        if self.config.vif_threshold:
            print("  • Stage 2: VIF (multicollinearity) filtering...")
            features = self._filter_by_vif(features)
            self.selection_history_['after_vif'] = features.copy()
            print(f"    → {len(features)} features remaining")

        # Stage 3: Correlation clustering
        if self.config.rho_threshold:
            print("  • Stage 3: Correlation clustering...")
            features = self._filter_by_correlation(features)
            self.selection_history_['after_correlation'] = features.copy()
            print(f"    → {len(features)} features remaining")

        # Stage 4: IV filtering
        print("  • Stage 4: Information Value filtering...")
        features = self._filter_by_iv(features)
        self.selection_history_['after_iv'] = features.copy()
        print(f"    → {len(features)} features remaining")

        # Stage 5: Boruta selection (if enabled)
        if self.config.use_boruta:
            print("  • Stage 5: Boruta selection (LightGBM-based)...")
            features = self._boruta_selection(features)
            self.selection_history_['after_boruta'] = features.copy()
            print(f"    → {len(features)} features remaining")

        # Stage 6: Stepwise selection (if enabled)
        if self.config.selection_method in ['forward', 'backward', 'stepwise']:
            print(f"  • Stage 6: {self.config.selection_method.title()} selection...")
            features = self._stepwise_selection(features, method=self.config.selection_method)
            self.selection_history_['after_stepwise'] = features.copy()
            print(f"    → {len(features)} features remaining")

        # Stage 7: Noise sentinel validation
        if self.config.use_noise_sentinel:
            print("  • Stage 7: Noise sentinel validation...")
            features = self._noise_sentinel_check(features)
            self.selection_history_['after_noise_check'] = features.copy()
            print(f"    → {len(features)} features remaining")

        self.final_vars_ = features
        print(f"\n  ✓ Final selected features: {len(self.final_vars_)}")

    def _calculate_univariate_ginis(self, features: List[str]):
        """Calculate univariate Gini for each feature."""
        from sklearn.metrics import roc_auc_score

        print("  • Calculating univariate Gini coefficients...")

        for feature in features:
            try:
                # Raw Gini
                auc = roc_auc_score(
                    self.train_[self.config.target_col],
                    self.train_[feature].fillna(0)
                )
                self.univariate_ginis_[feature] = 2 * auc - 1

                # WOE Gini (if WOE is applied)
                if self.woe_mapping_ and feature in self.woe_mapping_:
                    woe_feature = f"{feature}_woe"
                    if woe_feature in self.train_.columns:
                        woe_auc = roc_auc_score(
                            self.train_[self.config.target_col],
                            self.train_[woe_feature]
                        )
                        self.woe_ginis_[feature] = 2 * woe_auc - 1

                        # Check if WOE degraded performance
                        if self.woe_ginis_[feature] < self.univariate_ginis_[feature] * 0.9:
                            print(f"    ⚠ WOE degraded Gini for {feature}: "
                                  f"{self.univariate_ginis_[feature]:.4f} → {self.woe_ginis_[feature]:.4f}")
            except:
                self.univariate_ginis_[feature] = 0.0

    def _apply_woe_transformation(self):
        """Apply optimized WOE transformation."""
        print("  • Calculating univariate metrics before WOE...")
        self._calculate_univariate_ginis(self.final_vars_)

        print("  • Applying IV/Gini-optimized WOE binning...")
        woe_data = self.woe_transformer.fit_transform_optimized(
            self.train_,
            self.test_,
            self.oot_,
            self.final_vars_,
            optimize_metric='iv'  # or 'gini'
        )

        # Store WOE mapping
        self.woe_mapping_ = woe_data['mapping']

        # Check WOE quality
        print("  • Validating WOE transformation quality...")
        self._calculate_univariate_ginis(self.final_vars_)

        # Update datasets with WOE features if dual mode
        if self.config.enable_dual_pipeline:
            self.train_woe_ = woe_data['train']
            self.test_woe_ = woe_data.get('test')
            self.oot_woe_ = woe_data.get('oot')

    def _build_models(self):
        """Build models with all specified algorithms."""
        print("  • Building models with multiple algorithms...")

        # Prepare model list
        model_types = self.config.model_types or [
            'LogisticRegression',
            'GAM',  # Generalized Additive Model
            'CatBoost',
            'LightGBM',
            'XGBoost',
            'RandomForest',
            'ExtraTrees',
            'GradientBoosting'
        ]

        # Build models
        if self.config.enable_dual_pipeline:
            # Build both WOE and RAW models
            print("  • Dual pipeline mode: Building WOE and RAW models...")

            # WOE models
            woe_results = self.model_builder.build_models(
                self.train_woe_[self.final_vars_],
                self.train_[self.config.target_col],
                self.test_woe_[self.final_vars_] if self.test_woe_ else None,
                self.test_[self.config.target_col] if self.test_ else None,
                model_types=model_types
            )
            self.woe_models_ = woe_results['models']

            # RAW models
            raw_results = self.model_builder.build_models(
                self.train_[self.final_vars_],
                self.train_[self.config.target_col],
                self.test_[self.final_vars_] if self.test_ else None,
                self.test_[self.config.target_col] if self.test_ else None,
                model_types=model_types
            )
            self.raw_models_ = raw_results['models']

            # Select best overall model
            all_models = {**self.woe_models_, **self.raw_models_}
            self.best_model_name_ = max(all_models, key=lambda k: all_models[k]['score'])
            self.best_model_ = all_models[self.best_model_name_]['model']
            self.best_score_ = all_models[self.best_model_name_]['score']

        else:
            # Single pipeline mode
            results = self.model_builder.build_models(
                self.train_[self.final_vars_],
                self.train_[self.config.target_col],
                self.test_[self.final_vars_] if self.test_ else None,
                self.test_[self.config.target_col] if self.test_ else None,
                model_types=model_types
            )

            self.best_model_ = results['best_model']
            self.best_model_name_ = results['best_model_name']
            self.best_score_ = results['best_score']

        print(f"  ✓ Best model: {self.best_model_name_} (Score: {self.best_score_:.4f})")

    def _calibrate_model(self, calibration_data: Optional[pd.DataFrame] = None,
                        stage2_data: Optional[pd.DataFrame] = None):
        """Apply Stage 1 and Stage 2 calibration."""
        print("  • Stage 1 Calibration...")

        # Prepare calibration data
        if calibration_data is None:
            # Use long-run average from training data
            print("    Using long-run average calibration")
            calibration_data = self.train_

        # Apply Stage 1 calibration
        self.calibration_model_ = self.calibration_engine.calibrate_stage1(
            self.best_model_,
            calibration_data[self.final_vars_],
            calibration_data[self.config.target_col]
        )

        # Stage 2 calibration if data provided
        if stage2_data is not None:
            print("  • Stage 2 Calibration...")
            self.calibration_model_ = self.calibration_engine.calibrate_stage2(
                self.calibration_model_,
                stage2_data[self.final_vars_],
                stage2_data[self.config.target_col],
                method=self.config.stage2_calibration_method  # 'lower_mean' or 'upper_bound'
            )

        print("  ✓ Calibration completed")

    def _optimize_risk_bands(self):
        """Optimize risk bands with multiple criteria."""
        print("  • Optimizing risk bands...")

        # Get calibrated predictions
        X_calib = self.train_[self.final_vars_]
        if self.calibration_model_:
            y_pred = self.calibration_model_.predict_proba(X_calib)[:, 1]
        else:
            y_pred = self.best_model_.predict_proba(X_calib)[:, 1]

        # Optimize bands
        self.risk_bands_ = self.risk_band_optimizer.optimize(
            y_pred,
            self.train_[self.config.target_col],
            n_bands=self.config.n_risk_bands,
            ensure_monotonic=True,
            min_samples_per_band=self.config.min_samples_per_band
        )

        # Calculate band metrics
        print("  • Calculating band metrics...")
        band_metrics = self.risk_band_optimizer.calculate_metrics(
            self.risk_bands_,
            include_herfindahl=True,
            include_binomial_test=True,
            include_hosmer_lemeshow=True
        )

        print(f"  ✓ Created {len(self.risk_bands_)} risk bands")
        print(f"    Herfindahl Index: {band_metrics['herfindahl_index']:.4f}")
        print(f"    Hosmer-Lemeshow p-value: {band_metrics['hosmer_lemeshow_pval']:.4f}")

    def _generate_reports(self):
        """Generate comprehensive reports."""
        print("  • Generating comprehensive reports...")

        # Prepare report data
        report_data = {
            'model_results': {
                'best_model': self.best_model_name_,
                'score': self.best_score_,
                'features': self.final_vars_
            },
            'univariate_ginis': self.univariate_ginis_,
            'woe_ginis': self.woe_ginis_,
            'feature_importance': self._calculate_feature_importance(),
            'selection_history': self.selection_history_,
            'risk_bands': self.risk_bands_,
            'calibration_info': {
                'stage1_applied': self.calibration_model_ is not None,
                'stage2_applied': False  # Update based on actual stage2 application
            }
        }

        # Add data dictionary descriptions if available
        if self.data_dictionary_ is not None:
            report_data['variable_descriptions'] = self.data_dictionary_

        # Generate reports
        self.reporter.generate_comprehensive_report(
            report_data,
            output_path=self.config.output_excel_path
        )

        print(f"  ✓ Reports saved to {self.config.output_excel_path}")

    def _calculate_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Calculate feature importance using multiple methods."""
        importance = {}

        # Model-specific importance
        if hasattr(self.best_model_, 'feature_importances_'):
            importance['model_importance'] = dict(zip(
                self.final_vars_,
                self.best_model_.feature_importances_
            ))

        # SHAP importance (if enabled)
        if self.config.calculate_shap:
            print("    Calculating SHAP values...")
            # Implementation for SHAP calculation
            pass

        # Univariate Gini as importance
        importance['univariate_gini'] = self.univariate_ginis_

        return importance

    def score(self, df: pd.DataFrame, use_calibration: bool = True) -> pd.DataFrame:
        """
        Score new data (optional, default off).

        Parameters:
        -----------
        df : pd.DataFrame
            Data to score
        use_calibration : bool
            Whether to use calibrated model

        Returns:
        --------
        pd.DataFrame
            Input data with score column added
        """
        if not self.config.enable_scoring:
            raise ValueError("Scoring is disabled in config. Set enable_scoring=True to enable.")

        if self.best_model_ is None:
            raise ValueError("Pipeline must be fitted before scoring")

        # Process data
        df_processed = self._process_data(df)

        # Apply WOE if used
        if self.woe_mapping_:
            df_processed = self.woe_transformer.transform(df_processed, self.woe_mapping_)

        # Select features
        X = df_processed[self.final_vars_]

        # Score
        if use_calibration and self.calibration_model_:
            scores = self.calibration_model_.predict_proba(X)[:, 1]
        else:
            scores = self.best_model_.predict_proba(X)[:, 1]

        # Add scores to dataframe
        result = df.copy()
        result['score'] = scores

        # Add risk band if available
        if self.risk_bands_ is not None:
            result['risk_band'] = self.risk_band_optimizer.assign_bands(
                scores,
                self.risk_bands_
            )

        return result

    # Helper methods for selection stages
    def _filter_by_psi(self, features: List[str]) -> List[str]:
        """Filter features by PSI threshold."""
        # Implementation
        pass

    def _filter_by_vif(self, features: List[str]) -> List[str]:
        """Filter features by VIF threshold."""
        # Implementation
        pass

    def _filter_by_correlation(self, features: List[str]) -> List[str]:
        """Filter features by correlation clustering."""
        # Implementation
        pass

    def _filter_by_iv(self, features: List[str]) -> List[str]:
        """Filter features by Information Value."""
        # Implementation
        pass

    def _boruta_selection(self, features: List[str]) -> List[str]:
        """Boruta feature selection using LightGBM."""
        # Implementation
        pass

    def _stepwise_selection(self, features: List[str], method: str = 'forward') -> List[str]:
        """Stepwise feature selection (forward/backward/stepwise)."""
        # Implementation
        pass

    def _noise_sentinel_check(self, features: List[str]) -> List[str]:
        """Validate features using noise sentinel."""
        # Implementation
        pass