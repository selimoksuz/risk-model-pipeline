"""
Unified Risk Model Pipeline - Single pipeline with complete configuration control
Author: Risk Analytics Team
Date: 2024
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import joblib
from scipy import stats

# Internal imports
from .core.config import Config
from .core.data_processor import DataProcessor
from .core.woe_transformer import EnhancedWOETransformer
from .core.feature_selector_enhanced import AdvancedFeatureSelector
from .core.model_builder import ComprehensiveModelBuilder
from .core.calibration import TwoStageCalibrator
from .core.risk_band_optimizer import OptimalRiskBandAnalyzer
from .core.reporter import EnhancedReporter
from .core.splitter import SmartDataSplitter

warnings.filterwarnings('ignore')


class UnifiedRiskPipeline:
    """
    Single unified pipeline that handles all risk modeling tasks.
    Everything controlled through configuration - no need for multiple pipeline classes.
    """

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        """
        Initialize pipeline with configuration.

        Parameters:
        -----------
        config : Config or dict, optional
            Pipeline configuration. If None, uses default settings.
        """
        # Handle config
        if config is None:
            self.config = Config()
        elif isinstance(config, dict):
            self.config = Config(**config)
        else:
            self.config = config

        # Set default: scoring disabled
        if not hasattr(self.config, 'enable_scoring'):
            self.config.enable_scoring = False

        # Initialize components
        self._initialize_components()

        # Storage for results
        self.results_ = {}
        self.models_ = {}
        self.transformers_ = {}
        self.data_ = {}

    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.data_processor = DataProcessor(self.config)
        self.splitter = SmartDataSplitter(self.config)
        self.woe_transformer = EnhancedWOETransformer(self.config)
        self.feature_selector = AdvancedFeatureSelector(self.config)
        self.model_builder = ComprehensiveModelBuilder(self.config)
        self.calibrator = TwoStageCalibrator(self.config)
        self.risk_band_optimizer = OptimalRiskBandAnalyzer(self.config)
        self.reporter = EnhancedReporter(self.config)

    def fit(self,
            df: pd.DataFrame,
            data_dictionary: Optional[pd.DataFrame] = None,
            calibration_df: Optional[pd.DataFrame] = None,
            stage2_df: Optional[pd.DataFrame] = None,
            score_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Main pipeline execution method.

        Parameters:
        -----------
        df : pd.DataFrame
            Main dataset for training/testing/OOT
        data_dictionary : pd.DataFrame, optional
            Variable descriptions for reporting
        calibration_df : pd.DataFrame, optional
            Data for Stage 1 calibration (if None, uses full data)
        stage2_df : pd.DataFrame, optional
            Recent data for Stage 2 calibration
        score_df : pd.DataFrame, optional
            Data to score (if enable_scoring=True)

        Returns:
        --------
        dict : Dictionary containing all results
        """
        print("="*80)
        print("UNIFIED RISK PIPELINE EXECUTION")
        print("="*80)

        # Store data dictionary
        self.data_dictionary = data_dictionary

        try:
            # Step 1: Data Processing
            print("\n[Step 1/10] Data Processing...")
            processed_data = self._process_data(df)

            # Step 2: Data Splitting
            print("\n[Step 2/10] Data Splitting...")
            splits = self._split_data(processed_data)

            # Step 3: WOE Transformation & Univariate Analysis
            print("\n[Step 3/10] WOE Transformation & Univariate Analysis...")
            woe_results = self._apply_woe_transformation(splits)

            # Step 4: Feature Selection
            print("\n[Step 4/10] Feature Selection...")
            selection_results = self._select_features(splits, woe_results)

            # Step 5: Model Training
            print("\n[Step 5/10] Model Training...")
            model_results = self._train_models(splits, selection_results)

            # Step 6: Stage 1 Calibration
            if self.config.enable_calibration:
                print("\n[Step 6/10] Stage 1 Calibration...")
                stage1_results = self._apply_stage1_calibration(
                    model_results,
                    calibration_df if calibration_df is not None else df
                )
            else:
                print("\n[Step 6/10] Stage 1 Calibration... Skipped")
                stage1_results = {'calibrated_model': model_results.get('best_model')}
                model_results['calibrated_model'] = model_results.get('best_model')

            # Step 7: Stage 2 Calibration (if data provided)
            if stage2_df is not None:
                print("\n[Step 7/10] Stage 2 Calibration...")
                stage2_results = self._apply_stage2_calibration(
                    stage1_results, stage2_df
                )
            else:
                print("\n[Step 7/10] Stage 2 Calibration... Skipped (no data)")
                stage2_results = stage1_results

            # Step 8: Risk Band Optimization
            print("\n[Step 8/10] Risk Band Optimization...")
            risk_bands = self._optimize_risk_bands(stage2_results, splits)

            # Step 9: Scoring (if enabled and data provided)
            if self.config.enable_scoring and score_df is not None:
                print("\n[Step 9/10] Scoring...")
                scoring_results = self._score_data(score_df, stage2_results)
            else:
                print("\n[Step 9/10] Scoring... Skipped")
                scoring_results = None

            # Step 10: Generate Reports
            print("\n[Step 10/10] Generating Reports...")
            reports = self._generate_reports()

            # Compile final results
            self.results_ = {
                'processed_data': processed_data,
                'splits': splits,
                'woe_results': woe_results,
                'selection_results': selection_results,
                'model_results': model_results,
                'calibration_stage1': stage1_results,
                'calibration_stage2': stage2_results,
                'risk_bands': risk_bands,
                'scoring_results': scoring_results,
                'reports': reports,
                'config': self.config.__dict__,
                'selected_features': self.selected_features_,
                'best_model_name': model_results.get('best_model_name'),
                'scores': model_results.get('scores', {})
            }

            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*80)

            return self.results_

        except Exception as e:
            print(f"\nERROR: Pipeline failed at step: {str(e)}")
            raise

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with separate handling for numeric/categorical."""

        df_processed = df.copy()

        # Validate and freeze data
        df_processed = self.data_processor.validate_and_freeze(df_processed)

        # Identify variable types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target and special columns
        special_cols = [self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month']
        numeric_cols = [c for c in numeric_cols if c not in special_cols]
        categorical_cols = [c for c in categorical_cols if c not in special_cols]

        print(f"  Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

        # Impute missing values for numeric columns
        if numeric_cols:
            from sklearn.impute import SimpleImputer
            numeric_imputer = SimpleImputer(strategy=self.config.numeric_imputation)
            df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])

        # Handle categorical missing values
        if categorical_cols:
            for col in categorical_cols:
                df_processed[col] = df_processed[col].fillna('MISSING')

                # Group rare categories
                if hasattr(self.config, 'min_category_freq'):
                    freq = df_processed[col].value_counts(normalize=True)
                    rare = freq[freq < self.config.min_category_freq].index.tolist()
                    if rare:
                        df_processed.loc[df_processed[col].isin(rare), col] = 'RARE'

        # Outlier handling for numeric columns
        if numeric_cols and self.config.outlier_method == 'clip':
            for col in numeric_cols:
                p1 = df_processed[col].quantile(0.01)
                p99 = df_processed[col].quantile(0.99)
                df_processed[col] = df_processed[col].clip(p1, p99)

        # Add noise sentinel if enabled
        if self.config.enable_noise_sentinel:
            df_processed['noise_sentinel'] = np.random.normal(0, 1, len(df_processed))

        self.data_['processed'] = df_processed
        return df_processed

    def _split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data with equal default rates if configured."""

        if self.config.equal_default_splits:
            splits = self.splitter.split_equal_default_rate(df)
        else:
            splits = self.splitter.split(df)

        # Print split statistics
        for split_name, split_df in splits.items():
            if split_df is not None:
                default_rate = split_df[self.config.target_col].mean()
                print(f"  {split_name}: {len(split_df)} samples, default rate: {default_rate:.2%}")

        self.data_['splits'] = splits
        return splits

    def _apply_woe_transformation(self, splits: Dict) -> Dict:
        """Apply WOE transformation and calculate univariate Gini."""

        results = {}

        # Fit WOE on train data
        train_df = splits['train']
        feature_cols = [c for c in train_df.columns
                       if c not in [self.config.target_col, self.config.id_col, self.config.time_col]]

        # Calculate WOE for each variable
        woe_values = {}
        univariate_gini = {}

        for col in feature_cols:
            print(f"  Processing {col}...")

            # Calculate WOE
            woe_result = self.woe_transformer.fit_transform_single(
                train_df[col],
                train_df[self.config.target_col]
            )
            woe_values[col] = woe_result

            # Calculate univariate Gini (both raw and WOE)
            gini_raw = self._calculate_gini(
                train_df[col],
                train_df[self.config.target_col]
            )
            gini_woe = self._calculate_gini(
                woe_result['transformed'],
                train_df[self.config.target_col]
            )

            univariate_gini[col] = {
                'gini_raw': gini_raw,
                'gini_woe': gini_woe,
                'gini_drop': gini_raw - gini_woe
            }

            # Check WOE quality
            if gini_woe < gini_raw * 0.8:  # If WOE drops Gini by >20%
                print(f"    WARNING: WOE significantly reduces Gini for {col}")

        results['woe_values'] = woe_values
        results['univariate_gini'] = univariate_gini

        # Transform all splits
        if self.config.enable_woe:
            # Create a list of splits to transform (avoid modifying dict during iteration)
            splits_to_transform = list(splits.items())
            for split_name, split_df in splits_to_transform:
                if split_df is not None:
                    splits[f'{split_name}_woe'] = self.woe_transformer.transform(
                        split_df, woe_values
                    )

        self.transformers_['woe'] = self.woe_transformer
        return results

    def _select_features(self, splits: Dict, woe_results: Dict) -> Dict:
        """
        Apply feature selection in specified order:
        PSI -> VIF -> Correlation -> IV -> Boruta -> Stepwise
        """

        train_df = splits['train_woe'] if self.config.enable_woe else splits['train']
        feature_cols = [c for c in train_df.columns
                       if c not in [self.config.target_col, self.config.id_col, self.config.time_col]]

        selected_features = feature_cols.copy()
        selection_history = []

        # Selection order from config
        for method in self.config.selection_order:
            print(f"  Applying {method} selection...")

            if method == 'psi':
                selected = self.feature_selector.select_by_psi(
                    train_df[selected_features],
                    splits.get('test', train_df)[selected_features],
                    threshold=self.config.psi_threshold
                )
            elif method == 'vif':
                selected = self.feature_selector.select_by_vif(
                    train_df[selected_features],
                    threshold=self.config.vif_threshold
                )
            elif method == 'correlation':
                selected = self.feature_selector.select_by_correlation(
                    train_df[selected_features],
                    train_df[self.config.target_col],
                    threshold=self.config.correlation_threshold
                )
            elif method == 'iv':
                selected = self.feature_selector.select_by_iv(
                    woe_results['woe_values'],
                    selected_features,
                    threshold=self.config.iv_threshold
                )
            elif method == 'boruta':
                selected = self.feature_selector.select_by_boruta_lgbm(
                    train_df[selected_features],
                    train_df[self.config.target_col]
                )
            elif method == 'stepwise':
                if self.config.selection_method == 'forward':
                    # Pass test data for validation to avoid overfitting
                    test_df = splits.get('test_woe' if self.config.enable_woe else 'test', train_df)
                    selected = self.feature_selector.forward_selection(
                        train_df[selected_features],
                        train_df[self.config.target_col],
                        max_features=self.config.max_features,
                        X_val=test_df[selected_features],
                        y_val=test_df[self.config.target_col]
                    )
                elif self.config.selection_method == 'backward':
                    selected = self.feature_selector.backward_selection(
                        train_df[selected_features],
                        train_df[self.config.target_col]
                    )
                else:  # stepwise
                    selected = self.feature_selector.stepwise_selection(
                        train_df[selected_features],
                        train_df[self.config.target_col],
                        max_features=self.config.max_features
                    )

            selection_history.append({
                'method': method,
                'before': len(selected_features),
                'after': len(selected),
                'removed': set(selected_features) - set(selected)
            })

            selected_features = selected
            print(f"    Remaining features: {len(selected_features)}")

        return {
            'selected_features': selected_features,
            'selection_history': selection_history
        }

    def _train_models(self, splits: Dict, selection_results: Dict) -> Dict:
        """Train all configured models."""

        selected_features = selection_results['selected_features']

        # Prepare training data
        if self.config.enable_woe:
            X_train = splits['train_woe'][selected_features]
            y_train = splits['train'][self.config.target_col]
            X_test = splits.get('test_woe', pd.DataFrame())[selected_features] if 'test_woe' in splits else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None
        else:
            X_train = splits['train'][selected_features]
            y_train = splits['train'][self.config.target_col]
            X_test = splits.get('test', pd.DataFrame())[selected_features] if 'test' in splits else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None

        # Train models
        model_results = self.model_builder.train_all_models(
            X_train, y_train, X_test, y_test
        )

        # Check noise sentinel if enabled
        if self.config.enable_noise_sentinel and 'noise_sentinel' in selected_features:
            print("  WARNING: Noise sentinel was selected - feature selection may be overfitting!")

        self.models_ = model_results['models']
        self.selected_features_ = model_results.get('selected_features', [])
        return model_results

    def _apply_stage1_calibration(self, model_results: Dict, calibration_df: pd.DataFrame) -> Dict:
        """Apply Stage 1 calibration (long-run average)."""

        # Check if we have a model to calibrate
        if model_results.get('best_model') is None:
            print("  Skipping calibration: No model available")
            return {}

        # Prepare calibration data
        selected_features = model_results['selected_features']

        # Check if we have features
        if not selected_features:
            print("  Skipping calibration: No features selected")
            return {}

        # Process calibration data
        if self.config.enable_woe:
            # First process the data to add snapshot_month etc
            cal_processed = self._process_data(calibration_df)
            # WOE transformer handles processed data
            cal_woe = self.woe_transformer.transform(cal_processed)

            # Use same features as training - important for model compatibility
            # Remove non-feature columns
            feature_cols = [c for c in cal_woe.columns
                          if c not in [self.config.target_col, self.config.id_col, self.config.time_col]]

            # Filter to only selected features
            X_cal = cal_woe[selected_features] if all(f in cal_woe.columns for f in selected_features) else cal_woe[feature_cols]

            # Ensure all columns are numeric (WOE transformed)
            for col in X_cal.columns:
                if X_cal[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_cal[col]):
                    # Convert to numeric or fill with 0
                    X_cal[col] = pd.to_numeric(X_cal[col], errors='coerce').fillna(0)
        else:
            # Only for non-WOE case, process data
            cal_processed = self._process_data(calibration_df)
            X_cal = cal_processed[selected_features]

            # Ensure all columns are numeric
            for col in X_cal.columns:
                if X_cal[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_cal[col]):
                    X_cal[col] = pd.to_numeric(X_cal[col], errors='coerce').fillna(0)

        y_cal = calibration_df[self.config.target_col]

        # Calibrate best model
        best_model = model_results['best_model']
        calibrated_model = self.calibrator.calibrate_stage1(
            best_model, X_cal, y_cal,
            method=self.config.calibration_method
        )

        return {
            'calibrated_model': calibrated_model,
            'calibration_metrics': self.calibrator.evaluate_calibration(
                calibrated_model, X_cal, y_cal
            )
        }

    def _apply_stage2_calibration(self, stage1_results: Dict, stage2_df: pd.DataFrame) -> Dict:
        """Apply Stage 2 calibration (recent period adjustment)."""

        # Check if we have a calibrated model from stage 1
        if not stage1_results or 'calibrated_model' not in stage1_results:
            print("  Skipping Stage 2 calibration: No Stage 1 model available")
            return {}

        # Prepare Stage 2 data
        selected_features = self.selected_features_

        if not selected_features:
            print("  Skipping Stage 2 calibration: No features selected")
            return {}

        # Process Stage 2 data
        if self.config.enable_woe:
            # WOE transformer handles raw data directly
            stage2_woe = self.woe_transformer.transform(stage2_df)

            # Same fix as Stage 1 - ensure features are properly WOE transformed
            available_features = [f for f in selected_features if f in stage2_woe.columns]
            for feat in available_features:
                if stage2_woe[feat].dtype == 'object':
                    if feat in self.woe_transformer.woe_maps_:
                        woe_info = self.woe_transformer.woe_maps_[feat]
                        if woe_info['type'] == 'categorical':
                            stage2_woe[feat] = stage2_woe[feat].map(woe_info['woe_map']).fillna(0)

            X_stage2 = stage2_woe[available_features]
        else:
            # Only for non-WOE case, process data
            stage2_processed = self._process_data(stage2_df)
            X_stage2 = stage2_processed[selected_features]

        y_stage2 = stage2_df[self.config.target_col]

        # Apply Stage 2 calibration
        stage2_model = self.calibrator.calibrate_stage2(
            stage1_results['calibrated_model'],
            X_stage2, y_stage2,
            method=self.config.stage2_method
        )

        return {
            'calibrated_model': stage2_model,
            'stage1_metrics': stage1_results['calibration_metrics'],
            'stage2_metrics': self.calibrator.evaluate_calibration(
                stage2_model, X_stage2, y_stage2
            )
        }

    def _optimize_risk_bands(self, model_results: Dict, splits: Dict) -> Dict:
        """Optimize risk bands with multiple metrics."""

        # Check if we have a model
        if 'calibrated_model' not in model_results or model_results['calibrated_model'] is None:
            print("  Skipping risk bands: No model available")
            return {}

        # Get predictions
        model = model_results['calibrated_model']
        selected_features = self.selected_features_

        if not selected_features:
            print("  Skipping risk bands: No features selected")
            return {}

        if 'test' in splits:
            if self.config.enable_woe:
                X_test = splits['test_woe'][selected_features]
            else:
                X_test = splits['test'][selected_features]
            y_test = splits['test'][self.config.target_col]
        else:
            if self.config.enable_woe:
                X_test = splits['train_woe'][selected_features]
            else:
                X_test = splits['train'][selected_features]
            y_test = splits['train'][self.config.target_col]

        predictions = model.predict_proba(X_test)[:, 1]

        # Optimize bands
        risk_bands = self.risk_band_optimizer.optimize_bands(
            predictions, y_test,
            n_bands=self.config.n_risk_bands,
            method=self.config.band_method
        )

        # Calculate metrics
        metrics = self.risk_band_optimizer.calculate_band_metrics(
            risk_bands, predictions, y_test
        )

        return {
            'bands': risk_bands,
            'metrics': metrics
        }

    def _score_data(self, score_df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Score new data using calibrated model."""

        # Check if we have a model
        if 'calibrated_model' not in model_results or model_results['calibrated_model'] is None:
            print("  Skipping scoring: No model available")
            return score_df

        # Process scoring data
        score_processed = self._process_data(score_df)

        # Transform if WOE enabled
        selected_features = self.selected_features_

        if not selected_features:
            print("  Skipping scoring: No features selected")
            return score_df

        if self.config.enable_woe:
            X_score = self.woe_transformer.transform(score_processed)[selected_features]
        else:
            X_score = score_processed[selected_features]

        # Get predictions
        model = model_results['calibrated_model']
        scores = model.predict_proba(X_score)[:, 1]

        # Create output
        result_df = score_df.copy()
        result_df['risk_score'] = scores
        result_df['risk_band'] = self.risk_band_optimizer.assign_bands(
            scores, self.results_['risk_bands']['bands']
        )

        return result_df

    def _generate_reports(self) -> Dict:
        """Generate comprehensive reports."""

        reports = {}

        # Model performance report
        if self.models_:
            reports['model_performance'] = self.reporter.generate_model_report(
                self.models_, self.data_dictionary
            )

        # Feature importance report
        if self.models_ and 'woe_results' in self.results_:
            reports['feature_importance'] = self.reporter.generate_feature_report(
                self.models_, self.results_['woe_results'], self.data_dictionary
            )

        # Risk band report
        if 'risk_bands' in self.results_ and self.results_['risk_bands']:
            reports['risk_bands'] = self.reporter.generate_risk_band_report(
                self.results_['risk_bands']
            )

        # Calibration report
        if self.results_.get('calibration_stage1') or self.results_.get('calibration_stage2'):
            reports['calibration'] = self.reporter.generate_calibration_report(
                self.results_.get('calibration_stage1'),
                self.results_.get('calibration_stage2')
            )

        return reports

    def _calculate_gini(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(actuals, predictions)
            return 2 * auc - 1
        except:
            return 0.0

    def save_pipeline(self, path: str):
        """Save pipeline to disk."""
        joblib.dump(self, path)
        print(f"Pipeline saved to {path}")

    @staticmethod
    def load_pipeline(path: str):
        """Load pipeline from disk."""
        pipeline = joblib.load(path)
        print(f"Pipeline loaded from {path}")
        return pipeline
