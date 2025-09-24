"""
Unified Risk Model Pipeline - Single pipeline with complete configuration control
Author: Risk Analytics Team
Date: 2024
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import joblib
from scipy import stats
from .utils.scoring import calculate_psi, calculate_ks_statistic, calculate_gini as scoring_gini, create_scoring_report

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

        if df is None:
            raise ValueError("Model dataset 'df' is required.")
        if getattr(df, 'empty', False):
            raise ValueError("Model dataset 'df' cannot be empty.")

        df = df.copy()

        # Store data dictionary
        self.data_dictionary = data_dictionary

        try:
            # Step 1: Data Processing
            print("\n[Step 1/10] Data Processing...")
            processed_data = self._process_data(df)

            # Step 2: Data Splitting
            print("\n[Step 2/10] Data Splitting...")
            splits = self._split_data(processed_data)

            def _run_single_flow(use_woe: bool) -> Dict[str, Any]:
                """Run selection->modeling->calibration->bands for one mode (WOE or RAW)."""
                original_enable_woe = getattr(self.config, 'enable_woe', True)
                # Backup state
                state_backup = {
                    'models_': self.models_.copy(),
                    'transformers_': self.transformers_.copy(),
                    'data_': self.data_.copy(),
                    'results_': self.results_.copy(),
                    'selected_features_': getattr(self, 'selected_features_', []),
                }

                self.config.enable_woe = use_woe

                # Step 3: WOE Transformation & Univariate Analysis
                print("\n[Step 3/10] WOE Transformation & Univariate Analysis...")
                woe_res = self._apply_woe_transformation(splits)

                # Step 4: Feature Selection
                print("\n[Step 4/10] Feature Selection...")
                sel_res = self._select_features(splits, woe_res)

                # Step 5: Model Training
                print("\n[Step 5/10] Model Training...")
                mdl_res = self._train_models(splits, sel_res)

                # Step 6: Stage 1 Calibration
                if self.config.enable_calibration:
                    print("\n[Step 6/10] Stage 1 Calibration...")
                    stg1 = self._apply_stage1_calibration(
                        mdl_res,
                        calibration_df if calibration_df is not None else df
                    )
                else:
                    print("\n[Step 6/10] Stage 1 Calibration... Skipped")
                    stg1 = {'calibrated_model': mdl_res.get('best_model')}
                    mdl_res['calibrated_model'] = mdl_res.get('best_model')

                # Step 7: Stage 2 Calibration (if data provided)
                if stage2_df is not None:
                    print("\n[Step 7/10] Stage 2 Calibration...")
                    stg2 = self._apply_stage2_calibration(stg1, stage2_df)
                else:
                    print("\n[Step 7/10] Stage 2 Calibration... Skipped (no data)")
                    stg2 = stg1

                # Step 8: Risk Band Optimization
                print("\n[Step 8/10] Risk Band Optimization...")
                bands = self._optimize_risk_bands(stg2, splits)
                self.results_['risk_bands'] = bands

                # Step 9: Scoring (if enabled and data provided)
                if self.config.enable_scoring and score_df is not None:
                    print("\n[Step 9/10] Scoring...")
                    score_out = self._score_data(score_df, stg2, sel_res, woe_res, mdl_res, splits)
                else:
                    print("\n[Step 9/10] Scoring... Skipped")
                    score_out = {'dataframe': None, 'metrics': None, 'reports': {}}

                # Evaluate best score
                best_name = mdl_res.get('best_model_name')
                scores = mdl_res.get('scores', {})
                def _score_of(name: Optional[str]) -> float:
                    if not name or name not in scores:
                        return -1e9
                    s = scores[name]
                    return s.get('test_auc') or s.get('train_auc') or 0.0

                best_auc = _score_of(best_name)

                out = {
                    'use_woe': use_woe,
                    'woe_results': woe_res,
                    'selection_results': sel_res,
                    'model_results': mdl_res,
                    'stage1': stg1,
                    'stage2': stg2,
                    'risk_bands': bands,
                    'scoring_output': score_out,
                    'scoring_results': score_out.get('dataframe'),
                    'best_auc': best_auc,
                    'selected_features': getattr(self, 'selected_features_', []),
                }

                # Restore state for the next flow
                self.models_ = state_backup['models_']
                self.transformers_ = state_backup['transformers_']
                self.data_ = state_backup['data_']
                self.results_ = state_backup['results_']
                self.selected_features_ = state_backup['selected_features_']
                self.config.enable_woe = original_enable_woe
                return out

            if getattr(self.config, 'enable_dual_pipeline', False):
                print("\n[DUAL] Running RAW and WOE flows and selecting the best by AUC...")
                flow_raw = _run_single_flow(False)
                flow_woe = _run_single_flow(True)
                best_flow = flow_woe if flow_woe['best_auc'] >= flow_raw['best_auc'] else flow_raw
                chosen = 'WOE' if best_flow['use_woe'] else 'RAW'
                print(f"[DUAL] Selected {chosen} flow with AUC={best_flow['best_auc']:.4f}")

                # Step 10: Generate Reports for chosen flow
                print("\n[Step 10/10] Generating Reports...")
                # Set current state to chosen flow for reporter
                self.models_ = best_flow['model_results'].get('models', {})
                self.selected_features_ = best_flow.get('selected_features', [])
                self.results_ = {
                    'woe_results': best_flow['woe_results'],
                    'risk_bands': best_flow['risk_bands'],
                    'scoring_output': best_flow.get('scoring_output'),
                }
                reports = self._generate_reports()

                # Compile final results (chosen flow)
                self.results_ = {
                    'processed_data': processed_data,
                    'splits': splits,
                    'woe_results': best_flow['woe_results'],
                    'selection_results': best_flow['selection_results'],
                    'model_results': best_flow['model_results'],
                    'calibration_stage1': best_flow['stage1'],
                    'calibration_stage2': best_flow['stage2'],
                    'risk_bands': best_flow['risk_bands'],
                    'stage2_details': best_flow['stage2'].get('stage2_details') if isinstance(best_flow['stage2'], dict) else None,
                    'scoring_output': best_flow.get('scoring_output'),
                    'scoring_results': best_flow.get('scoring_output', {}).get('dataframe') if best_flow.get('scoring_output') else None,
                    'scoring_metrics': best_flow.get('scoring_output', {}).get('metrics') if best_flow.get('scoring_output') else None,
                    'scoring_reports': best_flow.get('scoring_output', {}).get('reports') if best_flow.get('scoring_output') else None,
                    'reports': reports,
                    'config': self.config.__dict__,
                    'selected_features': best_flow.get('selected_features', []),
                    'best_model_name': best_flow['model_results'].get('best_model_name'),
                    'scores': best_flow['model_results'].get('scores', {}),
                    'chosen_flow': chosen,
                    'chosen_auc': best_flow['best_auc'],
                }

                self._persist_model_artifacts()
            else:
                # Single-flow path (default behaviour)
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

                self.results_.update({
                    'processed_data': processed_data,
                    'splits': splits,
                    'woe_results': woe_results,
                    'selection_results': selection_results,
                    'model_results': model_results,
                    'calibration_stage1': stage1_results,
                    'calibration_stage2': stage2_results,
                    'stage2_details': stage2_results.get('stage2_details') if isinstance(stage2_results, dict) else None,
                })

                # Step 8: Risk Band Optimization
                print("\n[Step 8/10] Risk Band Optimization...")
                risk_bands = self._optimize_risk_bands(stage2_results, splits)
                self.results_['risk_bands'] = risk_bands
                scoring_output = {'dataframe': None, 'metrics': None, 'reports': {}}

                # Step 9: Scoring (if enabled and data provided)
                if self.config.enable_scoring and score_df is not None:
                    print("\n[Step 9/10] Scoring...")
                    scoring_output = self._score_data(score_df, stage2_results, selection_results, woe_results, model_results, splits)
                else:
                    print("\n[Step 9/10] Scoring... Skipped")

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
                    'stage2_details': stage2_results.get('stage2_details') if isinstance(stage2_results, dict) else None,
                    'scoring_output': scoring_output,
                    'scoring_results': scoring_output.get('dataframe'),
                    'scoring_metrics': scoring_output.get('metrics'),
                    'scoring_reports': scoring_output.get('reports'),
                    'reports': reports,
                    'config': self.config.__dict__,
                    'selected_features': self.selected_features_,
                    'best_model_name': model_results.get('best_model_name'),
                    'scores': model_results.get('scores', {})
                }

                self._persist_model_artifacts()

            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*80)

            return self.results_

        except Exception as e:
            print(f"\nERROR: Pipeline failed at step: {str(e)}")
            raise

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with separate handling for numeric/categorical."""

        df_processed = self.data_processor.validate_and_freeze(df.copy())

        tsfresh_features = self.data_processor.generate_tsfresh_features(df_processed)
        if not tsfresh_features.empty:
            tsfresh_features = tsfresh_features.copy()
            tsfresh_features.index.name = '__tsfresh_id__'
            df_processed = df_processed.copy()
            df_processed['__tsfresh_id__'] = df_processed[self.config.id_col].astype(str)
            df_processed = df_processed.merge(
                tsfresh_features,
                how='left',
                left_on='__tsfresh_id__',
                right_index=True,
            )
            df_processed.drop(columns='__tsfresh_id__', inplace=True)
            print(f"  Added {tsfresh_features.shape[1]} tsfresh Ã¶zelliÄŸi")

        # Identify variable types
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

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
            df_processed = df_processed.copy()
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

    def _inject_categorical_woe(self, splits: Dict, categorical_cols: List[str]) -> None:
        """Replace categorical columns with their WOE-transformed values."""

        if not categorical_cols:
            return

        for split_name, split_df in list(splits.items()):
            if split_name.endswith('_woe') or split_df is None:
                continue

            woe_df = splits.get(f'{split_name}_woe')
            if woe_df is None:
                continue

            updated = split_df.copy()
            for col in categorical_cols:
                if col in woe_df.columns:
                    updated[col] = woe_df[col]
            splits[split_name] = updated

    def _apply_woe_transformation(self, splits: Dict) -> Dict:
        """Apply WOE transformation and calculate univariate Gini."""

        results = {}

        # Fit WOE on train data
        train_df = splits['train']
        exclude_cols = {self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month'}
        feature_cols = [
            c
            for c in train_df.columns
            if c not in exclude_cols and not is_datetime64_any_dtype(train_df[c])
        ]

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

        # Transform all splits so WOE-ready versions are always available
        splits_to_transform = list(splits.items())
        for split_name, split_df in splits_to_transform:
            if split_name.endswith('_woe') or split_df is None:
                continue
            splits[f'{split_name}_woe'] = self.woe_transformer.transform(
                split_df, woe_values
            )

        self.transformers_['woe'] = self.woe_transformer
        return results

    def _select_features(self, splits: Dict, woe_results: Dict) -> Dict:
        """
        Apply feature selection in specified order and capture diagnostics.
        """

        raw_train = splits['train']
        exclude_cols = {self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month'}
        feature_cols = [
            col
            for col in raw_train.columns
            if col not in exclude_cols and not is_datetime64_any_dtype(raw_train[col])
        ]

        categorical_cols = [
            col
            for col in feature_cols
            if is_object_dtype(raw_train[col]) or is_categorical_dtype(raw_train[col])
        ]

        if not self.config.enable_woe:
            self._inject_categorical_woe(splits, categorical_cols)
            train_df = splits['train']
        else:
            train_df = splits['train_woe']

        train_woe_df = splits.get('train_woe', train_df)
        psi_monthly_frames = self._prepare_monthly_frames(splits)

        selected_features = feature_cols.copy()
        selection_history = []

        for method in self.config.selection_order:
            print(f"  Applying {method} selection...")
            if not selected_features:
                print(f"    Skipping {method}: no features remaining.")
                selection_history.append({
                    'method': method,
                    'before': 0,
                    'after': 0,
                    'removed': set(),
                    'details': {'info': 'no_features'}
                })
                continue

            step_details: Optional[Dict[str, Any]] = None

            if method == 'psi':
                train_for_psi = train_woe_df[selected_features]

                test_candidate = splits.get('test_woe')
                if test_candidate is None:
                    test_candidate = splits.get('test')
                test_for_psi = (
                    test_candidate[selected_features]
                    if test_candidate is not None and not test_candidate.empty
                    else None
                )

                oot_candidate = splits.get('oot_woe')
                if oot_candidate is None:
                    oot_candidate = splits.get('oot')
                oot_for_psi = (
                    oot_candidate[selected_features]
                    if oot_candidate is not None and not oot_candidate.empty
                    else None
                )

                monthly_subset = {
                    label: frame[selected_features]
                    for label, frame in psi_monthly_frames.items()
                }

                selected, step_details = self.feature_selector.select_by_psi(
                    train_for_psi,
                    test_for_psi,
                    threshold=self.config.psi_threshold,
                    oot_df=oot_for_psi,
                    monthly_frames=monthly_subset,
                    monthly_threshold=self.config.monthly_psi_threshold,
                    oot_threshold=self.config.oot_psi_threshold,
                )

            elif method == 'univariate':
                gini_map = woe_results.get('univariate_gini', {})
                selected = []
                uni_details: Dict[str, Any] = {}
                for col in selected_features:
                    info = gini_map.get(col, {}) or {}
                    gini_woe = info.get('gini_woe')
                    gini_raw = info.get('gini_raw')
                    gini_val = gini_woe if gini_woe is not None else gini_raw
                    detail = {
                        'gini_woe': gini_woe,
                        'gini_raw': gini_raw,
                        'threshold': self.config.min_univariate_gini,
                        'status': 'kept'
                    }
                    if gini_val is None:
                        gini_val = 0.0
                    if gini_val < self.config.min_univariate_gini:
                        detail['status'] = 'dropped'
                        detail['drop_reason'] = (
                            f"univariate gini {gini_val:.3f} < {self.config.min_univariate_gini:.3f}"
                        )
                        print(
                            f"    Removing {col}: univariate gini {gini_val:.3f} < {self.config.min_univariate_gini:.3f}"
                        )
                    else:
                        selected.append(col)
                    uni_details[col] = detail
                step_details = uni_details

            elif method == 'vif':
                selected = self.feature_selector.select_by_vif(
                    train_df[selected_features],
                    threshold=self.config.vif_threshold
                )
                step_details = {'threshold': self.config.vif_threshold}

            elif method == 'correlation':
                selected = self.feature_selector.select_by_correlation(
                    train_df[selected_features],
                    train_df[self.config.target_col],
                    threshold=self.config.correlation_threshold,
                    max_per_cluster=self.config.max_features_per_cluster
                )
                step_details = {
                    'threshold': self.config.correlation_threshold,
                    'max_per_cluster': self.config.max_features_per_cluster
                }

            elif method == 'iv':
                iv_details: Dict[str, Any] = {}
                selected = []
                woe_values = woe_results.get('woe_values', {})
                for col in selected_features:
                    iv_value = None
                    if col in woe_values:
                        iv_value = woe_values[col].get('iv')
                    detail = {
                        'iv': iv_value,
                        'threshold': self.config.iv_threshold,
                        'status': 'kept'
                    }
                    if iv_value is not None and iv_value < self.config.iv_threshold:
                        detail['status'] = 'dropped'
                        detail['drop_reason'] = (
                            f"IV {iv_value:.3f} < {self.config.iv_threshold:.3f}"
                        )
                        print(f"    Removing {col}: IV {iv_value:.3f} < {self.config.iv_threshold:.3f}")
                    else:
                        selected.append(col)
                    iv_details[col] = detail
                step_details = iv_details

            elif method == 'boruta':
                selected = self.feature_selector.select_by_boruta_lgbm(
                    train_df[selected_features],
                    train_df[self.config.target_col]
                )

            elif method == 'stepwise':
                if self.config.selection_method == 'forward':
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
                step_details = {'selection_method': self.config.selection_method}

            else:
                print(f"    WARNING: Unknown selection method '{method}' - skipping")
                selected = selected_features.copy()
                step_details = {'warning': 'unknown_method'}

            removed = set(selected_features) - set(selected)
            selection_history.append({
                'method': method,
                'before': len(selected_features),
                'after': len(selected),
                'removed': removed,
                'details': step_details
            })

            selected_features = selected
            print(f"    Remaining features: {len(selected_features)}")

        return {
            'selected_features': selected_features,
            'selection_history': selection_history
        }

    def _prepare_monthly_frames(self, splits: Dict) -> Dict[str, pd.DataFrame]:
        """Build monthly slices for PSI stability checks."""

        frames: Dict[str, pd.DataFrame] = {}
        time_col = self.config.time_col or 'snapshot_month'

        if 'train' not in splits:
            return frames

        train_df = splits['train']
        if not time_col or time_col not in train_df.columns:
            return frames

        woe_df = splits.get('train_woe', train_df)
        time_series = train_df[time_col]

        for label in time_series.dropna().unique():
            mask = time_series == label
            frames[str(label)] = woe_df.loc[mask].reset_index(drop=True)

        return frames

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
            X_cal = X_cal.copy()

            # Ensure all columns are numeric (WOE transformed)
            for col in X_cal.columns:
                if X_cal[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_cal[col]):
                    # Convert to numeric or fill with 0
                    X_cal.loc[:, col] = pd.to_numeric(X_cal[col], errors='coerce').fillna(0)
        else:
            # Only for non-WOE case, process data
            cal_processed = self._process_data(calibration_df)
            X_cal = cal_processed[selected_features].copy()

            # Ensure all columns are numeric
            for col in X_cal.columns:
                if X_cal[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_cal[col]):
                    X_cal.loc[:, col] = pd.to_numeric(X_cal[col], errors='coerce').fillna(0)

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
        stage2_processed = self._process_data(stage2_df.copy())

        if self.config.enable_woe:
            stage2_woe = self.woe_transformer.transform(stage2_processed)
            available_features = [f for f in selected_features if f in stage2_woe.columns]
            X_stage2 = stage2_woe[available_features].copy()
        else:
            stage2_woe = self.woe_transformer.transform(stage2_processed)
            categorical_cols = [
                col
                for col in selected_features
                if col in stage2_processed.columns
                and (is_object_dtype(stage2_processed[col]) or is_categorical_dtype(stage2_processed[col]))
            ]
            for col in categorical_cols:
                if col in stage2_woe.columns:
                    stage2_processed[col] = stage2_woe[col]
            X_stage2 = stage2_processed[selected_features].copy()

        for col in X_stage2.columns:
            if is_object_dtype(X_stage2[col]) or is_datetime64_any_dtype(X_stage2[col]):
                X_stage2.loc[:, col] = pd.to_numeric(X_stage2[col], errors='coerce').fillna(0)

        y_stage2 = stage2_processed[self.config.target_col]

        # Apply Stage 2 calibration
        stage2_model = self.calibrator.calibrate_stage2(
            stage1_results['calibrated_model'],
            X_stage2, y_stage2,
            method=self.config.stage2_method
        )

        stage2_metrics = self.calibrator.evaluate_calibration(
                stage2_model, X_stage2, y_stage2
            )
        stage2_details = getattr(self.calibrator, 'stage2_metadata_', {}) or {}

        response = {
            'calibrated_model': stage2_model,
            'stage1_metrics': stage1_results['calibration_metrics'],
            'stage2_metrics': stage2_metrics,
            'stage2_details': stage2_details
        }

        for key in ['target_rate', 'recent_rate', 'stage1_rate', 'adjustment_factor', 'lower_ci', 'upper_ci', 'confidence_level', 'achieved_rate']:
            if key in stage2_details:
                response[key] = stage2_details[key]

        return response


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
            'band_edges': self.risk_band_optimizer.bands_,
            'metrics': metrics
        }

    def _build_woe_mapping(self, woe_results: Optional[Dict]) -> Dict[str, Any]:
        """Convert internal WOE representation to exportable mapping."""

        mapping: Dict[str, Any] = {'variables': {}}
        if not woe_results:
            return mapping

        if isinstance(woe_results, dict) and 'woe_values' in woe_results:
            woe_values = woe_results['woe_values']
        else:
            woe_values = woe_results

        for feature, info in (woe_values or {}).items():
            if not isinstance(info, dict):
                continue
            feature_type = info.get('type', 'numeric')
            if feature_type == 'categorical':
                groups: List[Dict[str, Any]] = []
                for stat in info.get('stats', []) or []:
                    if not isinstance(stat, dict):
                        continue
                    members = stat.get('members')
                    if members is None:
                        members = [stat.get('label', stat.get('category', 'group'))]
                    elif isinstance(members, (str, bytes)):
                        members = [members]
                    else:
                        members = list(members)
                    groups.append({
                        'label': str(stat.get('label', stat.get('category', 'group'))),
                        'members': members,
                        'woe': float(stat.get('woe', 0) or 0.0),
                    })
                mapping['variables'][feature] = {'type': 'categorical', 'groups': groups}
            else:
                bins: List[Dict[str, Any]] = []
                stats = info.get('stats') or []
                if stats:
                    for stat in stats:
                        if not isinstance(stat, dict):
                            continue
                        left = stat.get('bin_left', stat.get('score_min'))
                        right = stat.get('bin_right', stat.get('score_max'))
                        left = None if left is None or (isinstance(left, float) and not np.isfinite(left)) else float(left)
                        right = None if right is None or (isinstance(right, float) and not np.isfinite(right)) else float(right)
                        bins.append({'left': left, 'right': right, 'woe': float(stat.get('woe', 0) or 0.0)})
                else:
                    woe_map = info.get('woe_map', {}) or {}
                    for value in woe_map.values():
                        bins.append({'left': None, 'right': None, 'woe': float(value or 0.0)})
                mapping['variables'][feature] = {'type': 'numeric', 'bins': bins}

        return mapping


    def _score_data(
            self,
            score_df: pd.DataFrame,
            stage_results: Dict,
            selection_results: Dict,
            woe_results: Dict,
            model_results: Dict,
            splits: Dict,
        ) -> Dict[str, Any]:
        """Score new data using the calibrated model and compute monitoring metrics."""

        if not stage_results or stage_results.get('calibrated_model') is None:
            print("  Skipping scoring: No calibrated model available")
            return {'dataframe': score_df.copy(), 'metrics': None, 'reports': {}}

        final_model = stage_results.get('calibrated_model')
        final_features = selection_results.get('selected_features') or getattr(self, 'selected_features_', [])

        if not final_features:
            print("  Skipping scoring: No features selected")
            return {'dataframe': score_df.copy(), 'metrics': None, 'reports': {}}

        score_processed = self._process_data(score_df)

        if self.config.enable_woe:
            score_matrix = self.woe_transformer.transform(score_processed)[final_features]
        else:
            score_matrix = score_processed[final_features]
        score_matrix = score_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

        try:
            proba = final_model.predict_proba(score_matrix)
            raw_scores = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        except AttributeError:
            predictions = final_model.predict(score_matrix)
            raw_scores = predictions.ravel() if hasattr(predictions, 'ravel') else np.asarray(predictions)

        scores = raw_scores

        result_df = score_df.copy()
        result_df['risk_score'] = scores
        band_edges = None
        if 'risk_bands' in self.results_ and self.results_['risk_bands']:
            band_edges = self.results_['risk_bands'].get('band_edges')
        if band_edges is None:
            band_edges = self.risk_band_optimizer.bands_
        result_df['risk_band'] = self.risk_band_optimizer.assign_bands(scores, band_edges)

        training_scores = None
        if isinstance(splits, dict) and 'train' in splits:
            if self.config.enable_woe and 'train_woe' in splits:
                train_matrix = splits['train_woe'][final_features]
            else:
                train_matrix = splits['train'][final_features]
            train_matrix = train_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
            try:
                train_proba = final_model.predict_proba(train_matrix)
                training_scores = train_proba[:, 1] if train_proba.ndim == 2 else train_proba.ravel()
            except AttributeError:
                train_pred = final_model.predict(train_matrix)
                training_scores = train_pred.ravel() if hasattr(train_pred, 'ravel') else np.asarray(train_pred)

        target_col = self.config.target_col
        has_target = (
            score_df[target_col].notna()
            if target_col in score_df.columns
            else pd.Series([False] * len(score_df), index=score_df.index)
        )

        metrics: Dict[str, Any] = {
            'scores': scores,
            'raw_scores': raw_scores,
            'has_target_mask': has_target.to_numpy(dtype=bool),
            'n_total': int(len(score_df)),
            'n_with_target': int(has_target.sum()),
            'n_without_target': int((~has_target).sum()),
            'psi_score': None,
            'calibration_applied': bool(stage_results.get('method') or stage_results is not model_results),
        }

        if training_scores is not None and training_scores.size > 0:
            try:
                metrics['psi_score'] = float(calculate_psi(training_scores, scores))
            except Exception as exc:
                print(f"  WARNING: PSI calculation failed: {exc}")
                metrics['psi_score'] = None

        if metrics['n_with_target'] > 0:
            target_mask = has_target.to_numpy(dtype=bool)
            y_true = score_df.loc[has_target, target_col].astype(float).to_numpy()
            y_scores = scores[target_mask]
            try:
                from sklearn.metrics import roc_auc_score

                auc = float(roc_auc_score(y_true, y_scores))
            except Exception:
                auc = float('nan')
            gini = float(scoring_gini(y_true, y_scores))
            ks = float(calculate_ks_statistic(y_true, y_scores))
            default_rate = float(y_true.mean())

            metrics['with_target'] = {
                'n_records': int(metrics['n_with_target']),
                'default_rate': default_rate,
                'auc': auc,
                'gini': gini,
                'ks': ks,
                'score_stats': self._describe_scores(y_scores),
            }
        if metrics['n_without_target'] > 0:
            mask = ~has_target.to_numpy(dtype=bool)
            metrics['without_target'] = {
                'n_records': int(metrics['n_without_target']),
                'score_stats': self._describe_scores(scores[mask]),
            }

        reports = create_scoring_report(metrics)

        return {
            'dataframe': result_df,
            'metrics': metrics,
            'reports': reports,
        }

    def _generate_reports(self) -> Dict:
        """Generate comprehensive reports."""

        reports = {}

        model_results = self.results_.get('model_results')
        woe_results = self.results_.get('woe_results')

        if getattr(self, 'data_dictionary', None) is not None:
            self.reporter.register_data_dictionary(self.data_dictionary)

        # Model performance report
        if model_results:
            reports['model_performance'] = self.reporter.generate_model_report(
                model_results, self.data_dictionary
            )

        # Feature importance & best model reports
        if model_results and woe_results:
            feature_df = self.reporter.generate_feature_report(
                model_results, woe_results, self.data_dictionary
            )
            reports['feature_importance'] = feature_df

            best_reports = self.reporter.generate_best_model_reports(
                model_results, woe_results, self.data_dictionary
            )
            reports.update(best_reports)

        # Risk band report
        if 'risk_bands' in self.results_ and self.results_['risk_bands']:
            reports['risk_bands'] = self.reporter.generate_risk_band_report(
                self.results_['risk_bands']
            )
            summary_table = self.reporter.reports_.get('risk_bands_summary_table')
            if summary_table is not None:
                reports['risk_bands_summary'] = summary_table

        # Calibration report
        if self.results_.get('calibration_stage1') or self.results_.get('calibration_stage2'):
            reports['calibration'] = self.reporter.generate_calibration_report(
                self.results_.get('calibration_stage1'),
                self.results_.get('calibration_stage2')
            )

        scoring_output = self.results_.get('scoring_output')
        if scoring_output:
            scoring_reports = self.reporter.generate_scoring_report(scoring_output)
            if scoring_reports:
                reports['scoring'] = scoring_reports

        if 'data_dictionary' in self.reporter.reports_:
            reports['data_dictionary'] = self.reporter.reports_['data_dictionary']

        if reports:
            output_dir = getattr(self.config, 'output_folder', None) or '.'
            os.makedirs(output_dir, exist_ok=True)
            run_id = getattr(self.config, 'run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            excel_path = getattr(self.config, 'output_excel_path', None) or os.path.join(output_dir, f"risk_report_{run_id}.xlsx")
            try:
                self.reporter.export_to_excel(excel_path)
                reports['excel_path'] = excel_path
            except Exception as exc:
                print(f"Failed to export report to Excel: {exc}")

        return reports

    def _persist_model_artifacts(self) -> None:
        """Persist final model artifacts when configuration allows."""

        if not getattr(self.config, 'save_model', True):
            return

        final_model = None
        stage2 = self.results_.get('calibration_stage2') or {}
        stage1 = self.results_.get('calibration_stage1') or {}
        model_results = self.results_.get('model_results') or {}
        final_model = stage2.get('calibrated_model') or stage1.get('calibrated_model') or model_results.get('best_model')

        if final_model is None:
            return

        output_dir = getattr(self.config, 'output_folder', 'output') or 'output'
        os.makedirs(output_dir, exist_ok=True)
        run_id = getattr(self.config, 'run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))

        final_model_path = os.path.join(output_dir, f"final_model_{run_id}.joblib")
        try:
            joblib.dump(final_model, final_model_path)
        except Exception as exc:
            print(f"  WARNING: Failed to persist final model: {exc}")

        best_model = model_results.get('best_model')
        if best_model is not None:
            raw_model_path = os.path.join(output_dir, f"best_model_raw_{run_id}.joblib")
            try:
                joblib.dump(best_model, raw_model_path)
            except Exception as exc:
                print(f"  WARNING: Failed to persist raw best model: {exc}")

        selected_features = (
            self.results_.get('selection_results', {}).get('selected_features')
            or self.results_.get('selected_features')
            or []
        )
        try:
            with open(os.path.join(output_dir, f"final_features_{run_id}.json"), 'w', encoding='utf-8') as handle:
                json.dump({'features': selected_features}, handle, default=self._json_default, indent=2)
        except Exception as exc:
            print(f"  WARNING: Failed to persist feature list: {exc}")

        woe_values = self.results_.get('woe_results', {})
        if woe_values:
            try:
                mapping = self._build_woe_mapping(woe_values)
                with open(os.path.join(output_dir, f"woe_mapping_{run_id}.json"), 'w', encoding='utf-8') as handle:
                    json.dump(mapping, handle, default=self._json_default, indent=2)
            except Exception as exc:
                print(f"  WARNING: Failed to persist WOE mapping: {exc}")

        stage2_details = stage2.get('stage2_details') or {}
        if stage2_details:
            try:
                with open(os.path.join(output_dir, f"stage2_details_{run_id}.json"), 'w', encoding='utf-8') as handle:
                    json.dump(stage2_details, handle, default=self._json_default, indent=2)
            except Exception as exc:
                print(f"  WARNING: Failed to persist stage 2 metadata: {exc}")

    @staticmethod
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray, list, tuple)):
            return [UnifiedRiskPipeline._json_default(v) for v in value]
        if hasattr(value, 'isoformat'):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _describe_scores(scores: np.ndarray) -> Dict[str, float]:
        if scores is None or len(scores) == 0:
            return {}
        scores = np.asarray(scores, dtype=float)
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q50': float(np.percentile(scores, 50)),
            'q75': float(np.percentile(scores, 75)),
            'max': float(np.max(scores)),
        }

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
