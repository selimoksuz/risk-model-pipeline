"""
Unified Risk Model Pipeline - Single pipeline with complete configuration control
Author: Risk Analytics Team
Date: 2024
"""

import os
import json
import warnings
import re
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
from .core.utils import predict_positive_proba

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
        self.feature_name_map: Dict[str, str] = {}
        self.noise_sentinel_name = 'noise_sentinel'

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
            risk_band_df: Optional[pd.DataFrame] = None,
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
        risk_band_df : pd.DataFrame, optional
            Dataset used exclusively for score band optimisation
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
            processed_data = self._process_data(df, create_map=True)

            # Step 2: Data Splitting
            print("\n[Step 2/10] Data Splitting...")
            splits = self._split_data(processed_data)

            risk_band_reference: Optional[pd.DataFrame] = None
            if risk_band_df is not None:
                print("\n[INFO] Aligning dedicated risk band dataset with learned preprocessing maps...")
                risk_band_reference = self._process_data(risk_band_df, create_map=False)
                self.data_['risk_band_reference'] = risk_band_reference
            else:
                self.data_['risk_band_reference'] = None

            self.data_['stage2_source'] = stage2_df.copy() if stage2_df is not None else None

            woe_cache: Dict[str, Any] = {}
            woe_transformer_cache: Optional[EnhancedWOETransformer] = None

            def _ensure_woe_results() -> Dict[str, Any]:
                nonlocal woe_cache, woe_transformer_cache
                if not woe_cache:
                    original_flag = getattr(self.config, 'enable_woe', True)
                    self.config.enable_woe = True
                    results_local = self._apply_woe_transformation(splits)
                    woe_transformer_cache = self.woe_transformer
                    woe_cache['results'] = results_local
                    self.config.enable_woe = original_flag
                if woe_transformer_cache is not None:
                    self.woe_transformer = woe_transformer_cache
                    self.transformers_['woe'] = woe_transformer_cache
                return woe_cache.get('results', {})

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
                woe_res = _ensure_woe_results()

                # Step 4: Feature Selection
                print("\n[Step 4/10] Feature Selection...")
                sel_res = self._select_features(splits, woe_res)

                # Step 5: Model Training
                print("\n[Step 5/10] Model Training...")
                mdl_res = self._train_models(splits, sel_res)
                mdl_res['mode'] = 'WOE' if use_woe else 'RAW'

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
                bands = self._optimize_risk_bands(stg2, splits, data_override=risk_band_reference, stage1_results=stg1)
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
                    'mode': 'WOE' if use_woe else 'RAW',
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
                    'selection_history': sel_res.get('selection_history'),
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                    'mode': 'WOE' if self.config.enable_woe else 'RAW',
                    'best_model_mode': 'WOE' if self.config.enable_woe else 'RAW',
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
                    'model_results': best_flow['model_results'],
                    'selection_results': best_flow['selection_results'],
                    'calibration_stage1': best_flow['stage1'],
                    'calibration_stage2': best_flow['stage2'],
                    'tsfresh_metadata': best_flow.get('tsfresh_metadata'),
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
                    'selection_history': best_flow['selection_results'].get('selection_history'),
                    'tsfresh_metadata': best_flow.get('tsfresh_metadata'),
                    'scoring_metrics': best_flow.get('scoring_output', {}).get('metrics') if best_flow.get('scoring_output') else None,
                    'scoring_reports': best_flow.get('scoring_output', {}).get('reports') if best_flow.get('scoring_output') else None,
                    'reports': reports,
                    'config': self.config.__dict__,
                    'selected_features': best_flow.get('selected_features', []),
                    'best_model_name': best_flow['model_results'].get('best_model_name'),
                    'scores': best_flow['model_results'].get('scores', {}),
                    'chosen_flow': chosen,
                    'chosen_flow_mode': chosen,
                    'chosen_flow_use_woe': best_flow.get('use_woe'),
                    'best_model_mode': best_flow['model_results'].get('mode'),
                    'chosen_auc': best_flow['best_auc'],
                }

                band_info = best_flow.get('risk_bands') or {}
                self.results_.update({
                    'calibration_stage1_curve': best_flow.get('stage1', {}).get('calibration_curve') if isinstance(best_flow.get('stage1'), dict) else None,
                    'calibration_stage2_curve': best_flow.get('stage2', {}).get('stage2_curve') if isinstance(best_flow.get('stage2'), dict) else None,
                    'feature_name_map': self.data_.get('feature_name_map', getattr(self, 'feature_name_map', {})),
                    'imputation_stats': getattr(self.data_processor, 'imputation_stats_', {}),
                    'risk_band_source': band_info.get('source'),
                    'risk_band_n_records': band_info.get('n_records'),
                    'risk_band_edges': band_info.get('band_edges'),
                })

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
                risk_bands = self._optimize_risk_bands(stage2_results, splits, data_override=risk_band_reference, stage1_results=stage1_results)
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
                self.results_.update({
                    'model_results': model_results,
                    'selection_results': selection_results,
                    'calibration_stage1': stage1_results,
                    'calibration_stage2': stage2_results,
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                })
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
                    'scores': model_results.get('scores', {}),
                    'selection_history': selection_results.get('selection_history'),
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                    'mode': 'WOE' if self.config.enable_woe else 'RAW',
                    'best_model_mode': 'WOE' if self.config.enable_woe else 'RAW',
                }

                band_info = risk_bands or {}
                self.results_.update({
                    'calibration_stage1_curve': stage1_results.get('calibration_curve') if isinstance(stage1_results, dict) else None,
                    'calibration_stage2_curve': stage2_results.get('stage2_curve') if isinstance(stage2_results, dict) else None,
                    'feature_name_map': self.data_.get('feature_name_map', getattr(self, 'feature_name_map', {})),
                    'imputation_stats': getattr(self.data_processor, 'imputation_stats_', {}),
                    'risk_band_source': band_info.get('source'),
                    'risk_band_n_records': band_info.get('n_records'),
                    'risk_band_edges': band_info.get('band_edges'),
                })

                self._persist_model_artifacts()

            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*80)

            return self.results_

        except Exception as e:
            print(f"\nERROR: Pipeline failed at step: {str(e)}")
            raise

    def _process_data(self, df: pd.DataFrame, create_map: bool = False) -> pd.DataFrame:
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
            print(f"  Added {tsfresh_features.shape[1]} tsfresh features")

        tsfresh_meta = getattr(self.data_processor, 'tsfresh_metadata_', None)
        if tsfresh_meta is not None:
            self.data_['tsfresh_metadata'] = tsfresh_meta.copy()

        # Identify variable types
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target and special columns
        special_cols = [self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month', self.noise_sentinel_name]
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


        df_processed = self._sanitize_feature_columns(df_processed, create_map=create_map)

        self.data_['processed'] = df_processed
        return df_processed

    def _sanitize_feature_columns(self, df: pd.DataFrame, create_map: bool) -> pd.DataFrame:
        """Sanitize feature names to avoid characters that break LightGBM/JSON."""
        exclude = {self.config.target_col, self.config.id_col, self.config.time_col, self.config.weight_column, 'snapshot_month', self.noise_sentinel_name}
        exclude = {col for col in exclude if col}
        mapping = {} if create_map or not getattr(self, 'feature_name_map', None) else dict(self.feature_name_map)
        used = set(mapping.values())

        def _sanitize(name: str) -> str:
            sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
            if not sanitized:
                sanitized = "feature"
            base = sanitized
            counter = 1
            while sanitized in used:
                counter += 1
                sanitized = f"{base}_{counter}"
            used.add(sanitized)
            return sanitized

        rename_map: Dict[str, str] = {}
        for col in df.columns:
            if col in exclude:
                mapping.setdefault(col, col)
                continue
            sanitized = mapping.get(col)
            if sanitized is None:
                sanitized = _sanitize(col)
                mapping[col] = sanitized
            if sanitized != col:
                rename_map[col] = sanitized

        if rename_map:
            df = df.rename(columns=rename_map)

        self.feature_name_map = mapping
        self.data_['feature_name_map'] = mapping.copy()

        meta = self.data_.get('tsfresh_metadata')
        if meta is not None and not meta.empty:
            meta = meta.copy()
            meta['feature'] = meta['feature'].map(lambda f: mapping.get(f, f))
            self.data_['tsfresh_metadata'] = meta

        return df

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
        exclude_cols = {self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month', self.noise_sentinel_name}
        feature_cols = [
            c
            for c in train_df.columns
            if c not in exclude_cols
            and c != getattr(self, 'noise_sentinel_name', 'noise_sentinel')
            and not is_datetime64_any_dtype(train_df[c])
        ]

        # Calculate WOE for each variable
        woe_values = {}
        univariate_gini = {}

        for col in feature_cols:

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

        print(f"  WOE hesaplandi: {len(feature_cols)} degisken")

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
            if col not in exclude_cols
            and col != getattr(self, 'noise_sentinel_name', 'noise_sentinel')
            and not is_datetime64_any_dtype(raw_train[col])
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
        noise_name = getattr(self, 'noise_sentinel_name', 'noise_sentinel')
        noise_sentinel_flag = noise_name in raw_train.columns
        if noise_name in selected_features:
            selected_features.remove(noise_name)

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
                threshold = float(getattr(self.config, 'min_univariate_gini', 0.0) or 0.0)
                for col in selected_features:
                    info = gini_map.get(col, {}) or {}
                    gini_woe = info.get('gini_woe')
                    gini_raw = info.get('gini_raw')
                    gini_woe_val = float(gini_woe) if gini_woe is not None else 0.0
                    gini_raw_val = float(gini_raw) if gini_raw is not None else 0.0
                    keep_woe = gini_woe is not None and gini_woe_val >= threshold
                    keep_raw = gini_raw is not None and gini_raw_val >= threshold
                    detail = {
                        'gini_woe': gini_woe,
                        'gini_raw': gini_raw,
                        'threshold': threshold,
                        'status': 'kept',
                        'meets_woe_threshold': keep_woe,
                        'meets_raw_threshold': keep_raw
                    }
                    if keep_woe or keep_raw:
                        selected.append(col)
                    else:
                        detail['status'] = 'dropped'
                        detail['drop_reason'] = (
                            f"univariate gini raw={gini_raw_val:.3f}, woe={gini_woe_val:.3f} < {threshold:.3f}"
                        )
                        print(
                            f"    Removing {col}: univariate gini raw={gini_raw_val:.3f}, woe={gini_woe_val:.3f} < {threshold:.3f}"
                        )
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

            print(f"    {method}: {len(removed)} degisken cikarildi, {len(selected)} kaldi")

            selected_features = selected
        if noise_sentinel_flag and getattr(self.config, 'use_noise_sentinel', False):
            print('  Noise sentinel feature removed from processing scope (overfit check only).')
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

        if self.config.enable_woe:
            X_train = splits['train_woe'][selected_features]
            y_train = splits['train'][self.config.target_col]
            X_test = splits.get('test_woe', pd.DataFrame())[selected_features] if 'test_woe' in splits else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None
            oot_source = splits.get('oot_woe')
        else:
            X_train = splits['train'][selected_features]
            y_train = splits['train'][self.config.target_col]
            X_test = splits.get('test', pd.DataFrame())[selected_features] if 'test' in splits else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None
            oot_source = splits.get('oot')

        X_oot = None
        if oot_source is not None and not oot_source.empty:
            missing_cols = [col for col in selected_features if col not in oot_source.columns]
            if missing_cols:
                print(f"  Warning: OOT data missing {len(missing_cols)} selected features; skipping OOT evaluation.")
            else:
                X_oot = oot_source[selected_features].copy()

        y_oot = None
        oot_raw = splits.get('oot')
        if oot_raw is not None and not oot_raw.empty:
            target_col = self.config.target_col
            if target_col in oot_raw.columns:
                y_oot = oot_raw[target_col]
            else:
                print("  Warning: OOT dataset missing target column; skipping OOT evaluation.")

        model_results = self.model_builder.train_all_models(
            X_train, y_train, X_test, y_test, X_oot, y_oot
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
            cal_processed = self._process_data(calibration_df, create_map=False)
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
            cal_processed = self._process_data(calibration_df, create_map=False)
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

        metrics = self.calibrator.evaluate_calibration(
            calibrated_model, X_cal, y_cal
        )

        X_cal_filled = X_cal.fillna(0) if hasattr(X_cal, 'fillna') else X_cal
        base_scores = predict_positive_proba(best_model, X_cal_filled)
        calibrated_scores = predict_positive_proba(calibrated_model, X_cal_filled)
        calibration_curve = None
        if base_scores.size and calibrated_scores.size:
            quantiles = np.linspace(0, 1, 101)
            calibration_curve = pd.DataFrame({
                'quantile': quantiles.tolist(),
                'pre_calibrated': np.quantile(base_scores, quantiles).tolist(),
                'post_calibrated': np.quantile(calibrated_scores, quantiles).tolist(),
            })

        stage1_details = {
            'method': self.config.calibration_method,
            'long_run_rate': float(metrics.get('long_run_rate', y_cal.mean())),
            'base_rate': float(metrics.get('base_rate', y_cal.mean())),
            'n_samples': int(len(X_cal)),
        }
        if calibration_curve is not None:
            stage1_details['curve_points'] = len(calibration_curve)

        return {
            'calibrated_model': calibrated_model,
            'calibration_metrics': metrics,
            'stage1_details': stage1_details,
            'calibration_curve': calibration_curve
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

        stage2_input = stage2_df.copy()
        target_col = self.config.target_col
        y_stage2_source: Optional[pd.Series] = None
        if target_col and target_col in stage2_input.columns:
            y_stage2_source = stage2_input[target_col].astype(float)
        elif target_col:
            stage2_input[target_col] = np.nan

        # Process Stage 2 data
        stage2_processed = self._process_data(stage2_input, create_map=False)
        stage2_processed = stage2_processed.reset_index(drop=True)
        if y_stage2_source is not None:
            y_stage2_source = y_stage2_source.reset_index(drop=True)

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
        X_stage2 = X_stage2.reset_index(drop=True)

        y_stage2: Optional[pd.Series] = None
        X_stage2_cal = X_stage2
        if y_stage2_source is not None:
            y_stage2_aligned = y_stage2_source.loc[X_stage2.index]
            valid_mask = y_stage2_aligned.notna()
            if valid_mask.any():
                if not valid_mask.all():
                    X_stage2_cal = X_stage2.loc[valid_mask].copy()
                    y_stage2 = y_stage2_aligned.loc[valid_mask].reset_index(drop=True)
                else:
                    y_stage2 = y_stage2_aligned.reset_index(drop=True)


        if target_col and target_col in stage2_processed.columns:
            target_values = stage2_processed[target_col]
            if target_values.notna().any():
                y_stage2 = target_values.astype(float)
                valid_mask = y_stage2.notna()
                if not valid_mask.all():
                    X_stage2_cal = X_stage2.loc[valid_mask].copy()
                    y_stage2 = y_stage2.loc[valid_mask]
                else:
                    X_stage2_cal = X_stage2.copy()
                y_stage2 = y_stage2.reset_index(drop=True)
                X_stage2_cal = X_stage2_cal.reset_index(drop=True)
            else:
                X_stage2_cal = X_stage2.copy()
        else:
            X_stage2_cal = X_stage2.copy()

        if y_stage2 is None:
            print("  Stage 2 calibration: no observed targets; using configured target rates.")

        # Apply Stage 2 calibration
        stage2_model = self.calibrator.calibrate_stage2(
            stage1_results['calibrated_model'],
            X_stage2_cal,
            y_stage2,
            method=getattr(self.config, 'stage2_method', 'lower_mean')
        )

        stage2_metrics = self.calibrator.evaluate_calibration(
            stage2_model, X_stage2_cal, y_stage2
        )
        stage2_details = getattr(self.calibrator, 'stage2_metadata_', {}) or {}
        if isinstance(X_stage2_cal, pd.DataFrame):
            stage2_details.setdefault('n_recent_samples', int(len(X_stage2_cal)))

        X_stage2_eval = X_stage2_cal if isinstance(X_stage2_cal, pd.DataFrame) else X_stage2
        stage2_curve = None
        if X_stage2_eval is not None and hasattr(X_stage2_eval, 'fillna'):
            X_eval_filled = X_stage2_eval.fillna(0)
            stage1_scores = predict_positive_proba(stage1_results['calibrated_model'], X_eval_filled)
            stage2_scores = predict_positive_proba(stage2_model, X_eval_filled)
            if stage1_scores.size and stage2_scores.size:
                quantiles = np.linspace(0, 1, 101)
                stage2_curve = pd.DataFrame({
                    'quantile': quantiles.tolist(),
                    'stage1_score': np.quantile(stage1_scores, quantiles).tolist(),
                    'stage2_score': np.quantile(stage2_scores, quantiles).tolist(),
                })
                stage2_details['curve_points'] = len(stage2_curve)

        response = {
            'calibrated_model': stage2_model,
            'stage1_metrics': stage1_results.get('calibration_metrics'),
            'stage2_metrics': stage2_metrics,
            'stage2_details': stage2_details,
            'stage2_curve': stage2_curve,
            'stage1_model': stage1_results.get('calibrated_model') if isinstance(stage1_results, dict) else None
        }

        for key in ['target_rate', 'recent_rate', 'stage1_rate', 'adjustment_factor', 'lower_ci', 'upper_ci', 'confidence_level', 'achieved_rate']:
            if key in stage2_details:
                response[key] = stage2_details[key]

        return response


    def _optimize_risk_bands(self, model_results: Dict, splits: Dict, *, data_override: Optional[pd.DataFrame] = None, stage1_results: Optional[Dict] = None) -> Dict:
        """Optimize risk bands with multiple metrics."""

        if 'calibrated_model' not in model_results or model_results['calibrated_model'] is None:
            print("  Skipping risk bands: No model available")
            return {}

        model = model_results['calibrated_model']
        selected_features = self.selected_features_
        target_col = self.config.target_col

        if not selected_features:
            print("  Skipping risk bands: No features selected")
            return {}

        X_eval: Optional[pd.DataFrame] = None
        y_eval: Optional[pd.Series] = None
        source = 'split'

        if data_override is not None and isinstance(data_override, pd.DataFrame) and not data_override.empty:
            override_df = data_override.copy()
            if target_col not in override_df.columns:
                print('  Risk band override dataset missing target column; falling back to split data.')
            else:
                if self.config.enable_woe:
                    transformed = self.woe_transformer.transform(override_df)
                else:
                    transformed = override_df
                missing = [col for col in selected_features if col not in transformed.columns]
                if missing:
                    print(f"  Risk band override dataset missing {len(missing)} selected features; falling back to split data.")
                else:
                    X_eval = transformed[selected_features].copy()
                    for col in X_eval.columns:
                        if is_object_dtype(X_eval[col]) or is_datetime64_any_dtype(X_eval[col]):
                            X_eval.loc[:, col] = pd.to_numeric(X_eval[col], errors='coerce').fillna(0)
                    y_eval = override_df[target_col].astype(float)
                    valid_mask = y_eval.notna()
                    if valid_mask.any():
                        if not valid_mask.all():
                            X_eval = X_eval.loc[valid_mask].copy()
                            y_eval = y_eval.loc[valid_mask]
                        else:
                            y_eval = y_eval.copy()
                        source = 'override'
                    else:
                        print('  Risk band override dataset has no valid targets; falling back to split data.')
                        X_eval = None
                        y_eval = None

        if X_eval is None or y_eval is None or X_eval.empty:
            if 'test' in splits:
                if self.config.enable_woe:
                    X_eval = splits['test_woe'][selected_features]
                else:
                    X_eval = splits['test'][selected_features]
                y_eval = splits['test'][target_col]
            else:
                if self.config.enable_woe:
                    X_eval = splits['train_woe'][selected_features]
                else:
                    X_eval = splits['train'][selected_features]
                y_eval = splits['train'][target_col]
            X_eval = X_eval.copy()
            y_eval = y_eval.astype(float)
            for col in X_eval.columns:
                if is_object_dtype(X_eval[col]) or is_datetime64_any_dtype(X_eval[col]):
                    X_eval.loc[:, col] = pd.to_numeric(X_eval[col], errors='coerce').fillna(0)
            source = 'split'

        min_sample = int(getattr(self.config, 'risk_band_min_sample_size', 0) or 0)
        if min_sample > 0 and X_eval is not None and len(X_eval) < min_sample:
            reference_df = self.data_.get('risk_band_reference')
            if (
                data_override is None
                and isinstance(reference_df, pd.DataFrame)
                and not reference_df.empty
                and target_col in reference_df.columns
                and len(reference_df) >= min_sample
            ):
                ref_df = reference_df.copy()
                if self.config.enable_woe:
                    ref_woe = self.woe_transformer.transform(ref_df)
                    available_features = [f for f in selected_features if f in ref_woe.columns]
                    missing = set(selected_features) - set(available_features)
                    if missing:
                        print(f"  Warning: Reference dataset missing {len(missing)} selected features; using available subset.")
                    X_eval = ref_woe[available_features].copy()
                else:
                    available_features = [f for f in selected_features if f in ref_df.columns]
                    missing = set(selected_features) - set(available_features)
                    if missing:
                        print(f"  Warning: Reference dataset missing {len(missing)} selected features; using available subset.")
                    X_eval = ref_df[available_features].copy()
                y_eval = ref_df[target_col].astype(float)
                source = 'override_reference'
            else:
                print(f"  Warning: Risk band optimisation sample contains {len(X_eval)} records (< {min_sample}); results may be unstable.")

        predictions = predict_positive_proba(model, X_eval)
        predictions = np.asarray(predictions, dtype=float).ravel()

        if np.unique(predictions).size <= 1:
            print('  Stage 2 predictions lack variation; falling back to Stage 1 for risk bands.')
            fallback_stage1 = stage1_results if isinstance(stage1_results, dict) else {}
            stage1_model = fallback_stage1.get('calibrated_model') if fallback_stage1 else None
            if stage1_model is None:
                stage1_model = model_results.get('stage1_model')
            if stage1_model is None:
                cached_stage1 = self.results_.get('calibration_stage1') or {}
                stage1_model = cached_stage1.get('calibrated_model')
            if stage1_model is None:
                base_results = self.results_.get('model_results', {})
                stage1_model = base_results.get('calibrated_model') or base_results.get('best_model')
            if stage1_model is not None:
                predictions = predict_positive_proba(stage1_model, X_eval)
                predictions = np.asarray(predictions, dtype=float).ravel()
                if np.unique(predictions).size <= 1:
                    print('  Risk band optimization skipped: predictions still lack variation.')
                    return {}
                model = stage1_model
            else:
                print('  Risk band optimization skipped: no suitable model for fallback.')
                return {}

        band_frame: pd.DataFrame
        risk_bands = self.risk_band_optimizer.optimize_bands(
            predictions, y_eval,
            n_bands=self.config.n_risk_bands,
            method=self.config.band_method
        )
        if isinstance(risk_bands, pd.DataFrame):
            band_frame = risk_bands.copy()
        elif isinstance(risk_bands, dict):
            band_frame = pd.DataFrame(risk_bands)
        else:
            band_frame = pd.DataFrame(risk_bands) if risk_bands is not None else pd.DataFrame()

        if not band_frame.empty:
            standardized_cols = {}
            for col in band_frame.columns:
                if isinstance(col, str):
                    key = col.strip().lower()
                    standardized_cols.setdefault(col, col)
                    if key == 'bad_rate' and col != 'bad_rate':
                        standardized_cols[col] = 'bad_rate'
                    elif key == 'observed_dr' and 'bad_rate' not in band_frame.columns:
                        standardized_cols[col] = 'bad_rate'
                else:
                    standardized_cols[col] = col
            band_frame = band_frame.rename(columns=standardized_cols)
            if 'bad_rate' not in band_frame.columns:
                if {'bad_count', 'count'}.issubset(band_frame.columns):
                    counts = band_frame['count'].to_numpy(dtype=float)
                    band_frame['bad_rate'] = np.divide(
                        band_frame['bad_count'],
                        counts,
                        out=np.zeros_like(counts, dtype=float),
                        where=counts > 0,
                    )
                elif 'observed_dr' in band_frame.columns:
                    band_frame['bad_rate'] = band_frame['observed_dr']
        else:
            band_frame = pd.DataFrame(columns=['band', 'bad_rate', 'count', 'pct_count'])

        metrics = self.risk_band_optimizer.calculate_band_metrics(
            band_frame, predictions, y_eval
        )

        return {
            'bands': band_frame,
            'band_stats': band_frame.copy(),
            'band_edges': self.risk_band_optimizer.bands_,
            'metrics': metrics,
            'source': source,
            'n_records': int(len(predictions))
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

        score_processed = self._process_data(score_df, create_map=False)

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
        if self.config.enable_woe:
            for col in final_features:
                result_df['{}_woe'.format(col)] = score_matrix[col].to_numpy()

        stage1_model = (self.results_.get('calibration_stage1') or {}).get('calibrated_model')
        stage1_scores = None
        if stage1_model is not None and stage1_model is not final_model:
            try:
                stage1_scores = predict_positive_proba(stage1_model, score_matrix)
                result_df['stage1_score'] = np.asarray(stage1_scores, dtype=float).ravel()
            except Exception:
                stage1_scores = None

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
        if stage1_scores is not None:
            metrics['stage1_scores_preview'] = self._describe_scores(np.asarray(stage1_scores, dtype=float).ravel())

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

        self.reporter.register_tsfresh_metadata(self.results_.get('tsfresh_metadata'))
        self.reporter.register_selection_history(self.results_.get('selection_results'))

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

            woe_tables = self.reporter.generate_woe_tables(
                woe_results, model_results.get('selected_features', [])
            )
            if woe_tables:
                reports['woe_tables'] = woe_tables
            woe_mapping = self.reporter.reports_.get('woe_mapping')
            if isinstance(woe_mapping, pd.DataFrame) and not woe_mapping.empty:
                reports['woe_mapping'] = woe_mapping

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

        def _dump_json(name: str, payload: Any) -> None:
            if payload is None:
                return
            try:
                with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as handle:
                    json.dump(payload, handle, default=self._json_default, indent=2)
            except Exception as exc:
                print(f"  WARNING: Failed to persist {name}: {exc}")

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

        tsfresh_meta = self.results_.get('tsfresh_metadata')
        if tsfresh_meta is None or (isinstance(tsfresh_meta, pd.DataFrame) and tsfresh_meta.empty):
            tsfresh_meta = self.data_.get('tsfresh_metadata')
        if isinstance(tsfresh_meta, pd.DataFrame) and not tsfresh_meta.empty:
            _dump_json(f"tsfresh_metadata_{run_id}.json", tsfresh_meta.to_dict(orient='records'))

        stage1_details = stage1.get('stage1_details') or {}
        if stage1_details:
            _dump_json(f"stage1_details_{run_id}.json", stage1_details)

        stage2_details = stage2.get('stage2_details') or {}
        if stage2_details:
            _dump_json(f"stage2_details_{run_id}.json", stage2_details)

        stage1_curve = stage1.get('calibration_curve')
        if isinstance(stage1_curve, pd.DataFrame):
            stage1_curve_payload = stage1_curve.to_dict(orient='records')
        else:
            stage1_curve_payload = stage1_curve
        if stage1_curve_payload:
            _dump_json(f"stage1_curve_{run_id}.json", stage1_curve_payload)

        stage2_curve = stage2.get('stage2_curve')
        if isinstance(stage2_curve, pd.DataFrame):
            stage2_curve_payload = stage2_curve.to_dict(orient='records')
        else:
            stage2_curve_payload = stage2_curve
        if stage2_curve_payload:
            _dump_json(f"stage2_curve_{run_id}.json", stage2_curve_payload)

        feature_map = self.data_.get('feature_name_map', getattr(self, 'feature_name_map', {}))
        if feature_map:
            _dump_json(f"feature_map_{run_id}.json", feature_map)

        imputation_stats = getattr(self.data_processor, 'imputation_stats_', {})
        if imputation_stats:
            _dump_json(f"imputation_stats_{run_id}.json", imputation_stats)

        band_info = self.results_.get('risk_bands') or {}
        if band_info:
            band_payload = {
                'bands': band_info.get('bands'),
                'band_edges': band_info.get('band_edges'),
                'metrics': band_info.get('metrics'),
                'source': band_info.get('source'),
                'n_records': band_info.get('n_records'),
            }
            _dump_json(f"risk_bands_{run_id}.json", band_payload)

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



