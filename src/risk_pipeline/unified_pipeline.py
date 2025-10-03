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
from .core.run_logger import RunLogger
from .core.env_check import apply_runtime_feature_gates
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
        self.selected_features_: List[str] = []
        self.noise_sentinel_name = 'noise_sentinel'
        self._noise_counter = 0
        self._current_raw_preprocessor: Optional[Dict[str, Any]] = None
        # Install run logger (capture prints) if enabled
        try:
            if getattr(self.config, 'enable_run_logging', True):
                folder = getattr(self.config, 'logs_folder', 'logs')
                filename = getattr(self.config, 'log_filename', 'last_run.log')
                self._run_logger = RunLogger(folder=folder, filename=filename)
                self._run_logger.install()
            else:
                self._run_logger = None
        except Exception:
            self._run_logger = None

        # Apply environment feature gates (python version / optional libs)
        try:
            env_diag = apply_runtime_feature_gates(self.config)
            print('Environment diagnostics:')
            print('  Python:', env_diag.get('python_version'))
            if env_diag.get('disabled_algorithms'):
                print('  Disabled algorithms:', ', '.join(env_diag['disabled_algorithms']))
            if env_diag.get('disabled_features'):
                print('  Disabled features:', ', '.join(env_diag['disabled_features']))
            for note in env_diag.get('notes', []):
                print('  Note:', note)
        except Exception:
            pass
        # Ensure data_dictionary attribute always exists for reporter usage
        self.data_dictionary: Optional[pd.DataFrame] = None

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
        if data_dictionary is not None:
            self.results_['data_dictionary'] = data_dictionary

        try:
            # Step 1: Data Processing
            print("\n[Step 1/10] Data Processing...")
            processed_data = self._process_data(df, create_map=True, include_noise=False)

            # Step 2: Data Splitting
            print("\n[Step 2/10] Data Splitting...")
            splits = self._split_data(processed_data)

            self.data_['calibration_longrun'] = calibration_df.copy() if calibration_df is not None else None

            risk_band_reference: Optional[pd.DataFrame] = None
            if risk_band_df is not None:
                print("\n[INFO] Aligning dedicated risk band dataset with learned preprocessing maps...")
                risk_band_reference = self._process_data(risk_band_df, create_map=False, include_noise=False)
                self.data_['risk_band_reference'] = risk_band_reference
                self.data_['risk_band_reference_source'] = risk_band_df.copy()
            else:
                self.data_['risk_band_reference'] = None
                self.data_['risk_band_reference_source'] = None

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
                mode_label = 'WOE' if use_woe else 'RAW'
                state_backup = {
                    'models_': self.models_.copy(),
                    'transformers_': self.transformers_.copy(),
                    'data_': self.data_.copy(),
                    'results_': self.results_.copy(),
                    'selected_features_': getattr(self, 'selected_features_', []),
                }

                self.config.enable_woe = use_woe

                print("\n[Step 3/10] WOE Transformation & Univariate Analysis...")
                woe_res = _ensure_woe_results()

                print("\n[Step 4/10] Feature Selection...")
                sel_res = self._select_features(splits, woe_res)

                print("\n[Step 5/10] Model Training...")
                mdl_res = self._train_models(splits, sel_res, mode_label=mode_label)
                mdl_res['mode'] = mode_label

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

                if stage2_df is not None:
                    print("\n[Step 7/10] Stage 2 Calibration...")
                    stg2 = self._apply_stage2_calibration(stg1, stage2_df)
                else:
                    print("\n[Step 7/10] Stage 2 Calibration... Skipped (no data)")
                    stg2 = stg1

                print("\n[Step 8/10] Risk Band Optimization...")
                bands = self._optimize_risk_bands(stg2, splits, data_override=risk_band_reference, stage1_results=stg1)

                if self.config.enable_scoring and score_df is not None:
                    print("\n[Step 9/10] Scoring...")
                    score_out = self._score_data(score_df, stg2, sel_res, woe_res, mdl_res, splits)
                else:
                    print("\n[Step 9/10] Scoring... Skipped")
                    score_out = {'dataframe': None, 'metrics': None, 'reports': {}, 'noise_sentinel': None}

                best_name = mdl_res.get('best_model_name')
                scores = mdl_res.get('scores', {})

                def _score_of(name: Optional[str]) -> float:
                    if not name or name not in scores:
                        return -1e9
                    s = scores[name]
                    return s.get('test_auc') or s.get('train_auc') or 0.0

                best_auc = _score_of(best_name)

                registry_records = self._build_model_registry(mode_label, mdl_res)
                out = {
                    'use_woe': use_woe,
                    'mode': mode_label,
                    'mode_label': mode_label,
                    'woe_results': woe_res,
                    'selection_results': sel_res,
                    'model_results': mdl_res,
                    'stage1': stg1,
                    'stage2': stg2,
                    'risk_bands': bands,
                    'scoring_output': score_out,
                    'scoring_results': score_out.get('dataframe'),
                    'best_auc': best_auc,
                    'selected_features': sel_res.get('selected_features', []),
                    'selection_history': sel_res.get('selection_history'),
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                    'best_model_mode': mode_label,
                    'noise_sentinel_diagnostics': score_out.get('noise_sentinel'),
                    'model_registry': registry_records,
                    'model_scores': mdl_res.get('scores', {}),
                }

                self.models_ = state_backup['models_']
                self.transformers_ = state_backup['transformers_']
                self.data_ = state_backup['data_']
                self.results_ = state_backup['results_']
                self.selected_features_ = state_backup['selected_features_']
                self.config.enable_woe = original_enable_woe
                return out
            if getattr(self.config, 'enable_dual', getattr(self.config, 'enable_dual_pipeline', False)):
                print("\n[DUAL] Running RAW and WOE flows and selecting the best by AUC...")
                flow_raw = _run_single_flow(False)
                flow_woe = _run_single_flow(True)
                flows_by_mode = {flow_raw['mode']: flow_raw, flow_woe['mode']: flow_woe}
                best_flow = flow_woe if flow_woe['best_auc'] >= flow_raw['best_auc'] else flow_raw
                chosen = 'WOE' if best_flow['use_woe'] else 'RAW'
                print(f"[DUAL] Selected {chosen} flow with AUC={best_flow['best_auc']:.4f}")

                flow_registry = {
                    label: {
                        'mode': flow['mode'],
                        'best_auc': flow['best_auc'],
                        'best_model_name': flow['model_results'].get('best_model_name'),
                        'selected_features': flow.get('selected_features', []),
                        'noise_sentinel': flow.get('noise_sentinel_diagnostics'),
                    }
                    for label, flow in flows_by_mode.items()
                }
                model_registry_records: List[Dict[str, Any]] = []
                model_object_registry: Dict[str, Any] = {}
                model_scores_registry: Dict[str, Any] = {}

                for label, flow in flows_by_mode.items():
                    model_registry_records.extend(flow.get('model_registry', []))
                    model_object_registry[label] = flow['model_results'].get('models', {})
                    model_scores_registry[label] = flow['model_results'].get('scores', {})

                print("\n[Step 10/10] Generating Reports...")
                # Expose both RAW and WOE models for availability list
                combined_models: Dict[str, Any] = {}
                for label, flow in flows_by_mode.items():
                    modmap = flow['model_results'].get('models', {}) or {}
                    combined_models.update(modmap)
                self.models_ = combined_models
                self.selected_features_ = best_flow.get('selected_features', [])
                # Include dual registries before first report generation so reporter can aggregate RAW+WOE
                self.results_ = {
                    'woe_results': best_flow['woe_results'],
                    'risk_bands': best_flow['risk_bands'],
                    'scoring_output': best_flow.get('scoring_output'),
                    'model_results': best_flow['model_results'],
                    'selection_results': best_flow['selection_results'],
                    'calibration_stage1': best_flow['stage1'],
                    'calibration_stage2': best_flow['stage2'],
                    'tsfresh_metadata': best_flow.get('tsfresh_metadata'),
                    'noise_sentinel_diagnostics': best_flow.get('noise_sentinel_diagnostics'),
                    'model_registry': model_registry_records,
                    'model_object_registry': model_object_registry,
                    'model_scores_registry': model_scores_registry,
                    'flow_registry': flow_registry,
                }
                reports = self._generate_reports()

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
                    'flow_registry': flow_registry,
                    'model_registry': model_registry_records,
                    'model_object_registry': model_object_registry,
                    'model_scores_registry': model_scores_registry,
                    'chosen_flow_selected_features': best_flow.get('selected_features', []),
                    'chosen_flow_scores': best_flow['model_results'].get('scores', {}),
                    'noise_sentinel_diagnostics': best_flow.get('noise_sentinel_diagnostics'),
                    'chosen_flow_noise_sentinel': best_flow.get('noise_sentinel_diagnostics'),
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
                print("\n[Step 3/10] WOE Transformation & Univariate Analysis...")
                woe_results = self._apply_woe_transformation(splits)

                print("\n[Step 4/10] Feature Selection...")
                selection_results = self._select_features(splits, woe_results)

                mode_label = 'WOE' if self.config.enable_woe else 'RAW'

                print("\n[Step 5/10] Model Training...")
                model_results = self._train_models(splits, selection_results, mode_label=mode_label)

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

                print("\n[Step 8/10] Risk Band Optimization...")
                risk_bands = self._optimize_risk_bands(stage2_results, splits, data_override=risk_band_reference, stage1_results=stage1_results)
                self.results_['risk_bands'] = risk_bands
                scoring_output = {'dataframe': None, 'metrics': None, 'reports': {}, 'noise_sentinel': None}

                if self.config.enable_scoring and score_df is not None:
                    print("\n[Step 9/10] Scoring...")
                    scoring_output = self._score_data(score_df, stage2_results, selection_results, woe_results, model_results, splits)
                else:
                    print("\n[Step 9/10] Scoring... Skipped")

                noise_diag = scoring_output.get('noise_sentinel') if isinstance(scoring_output, dict) else None

                print("\n[Step 10/10] Generating Reports...")
                self.results_.update({
                    'model_results': model_results,
                    'selection_results': selection_results,
                    'calibration_stage1': stage1_results,
                    'calibration_stage2': stage2_results,
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                    'noise_sentinel_diagnostics': noise_diag,
                })
                reports = self._generate_reports()

                scores_dict = model_results.get('scores', {}) or {}
                best_model_name = model_results.get('best_model_name')
                registry_records = self._build_model_registry(mode_label, model_results)
                best_auc = None
                if best_model_name and best_model_name in scores_dict:
                    best_entry = scores_dict[best_model_name] or {}
                    best_auc = best_entry.get('test_auc') or best_entry.get('train_auc') or best_entry.get('oot_auc') or 0.0

                flow_registry = {
                    mode_label: {
                        'mode': mode_label,
                        'best_auc': best_auc,
                        'best_model_name': best_model_name,
                        'selected_features': self.selected_features_,
                        'noise_sentinel': noise_diag,
                    }
                }
                model_object_registry = {mode_label: model_results.get('models', {})}
                model_scores_registry = {mode_label: scores_dict}

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
                    'scores': scores_dict,
                    'selection_history': selection_results.get('selection_history'),
                    'tsfresh_metadata': self.data_.get('tsfresh_metadata'),
                    'mode': mode_label,
                    'best_model_mode': mode_label,
                    'flow_registry': flow_registry,
                    'model_registry': registry_records,
                    'model_object_registry': model_object_registry,
                    'model_scores_registry': model_scores_registry,
                    'chosen_flow': mode_label,
                    'chosen_flow_mode': mode_label,
                    'chosen_flow_use_woe': self.config.enable_woe,
                    'chosen_auc': best_auc,
                    'chosen_flow_selected_features': self.selected_features_,
                    'chosen_flow_scores': scores_dict,
                    'noise_sentinel_diagnostics': noise_diag,
                    'chosen_flow_noise_sentinel': noise_diag,
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



    def _process_data(self, df: pd.DataFrame, create_map: bool = False, *, include_noise: Optional[bool] = None) -> pd.DataFrame:
        """Process raw data while preserving raw, numeric-prepped, and WOE layers."""

        df_processed = self.data_processor.validate_and_freeze(df.copy())

        tsfresh_features = self.data_processor.generate_tsfresh_features(df_processed)
        merge_mode = getattr(self.data_processor, 'tsfresh_merge_mode', 'id')
        if not tsfresh_features.empty:
            tsfresh_features = tsfresh_features.copy()
            if merge_mode == 'row':
                df_processed = df_processed.copy()
                df_processed['__tsfresh_row_id__'] = np.arange(len(df_processed)).astype(str)
                df_processed = df_processed.merge(
                    tsfresh_features,
                    how='left',
                    left_on='__tsfresh_row_id__',
                    right_index=True,
                )
                df_processed.drop(columns='__tsfresh_row_id__', inplace=True)
            else:
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


        coverage_df = getattr(self.data_processor, 'tsfresh_coverage_', None)
        if isinstance(coverage_df, pd.DataFrame):
            self.data_['tsfresh_coverage'] = coverage_df.copy()
        tsfresh_meta = getattr(self.data_processor, 'tsfresh_metadata_', None)
        if tsfresh_meta is not None:
            self.data_['tsfresh_metadata'] = tsfresh_meta.copy()

        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        try:
            numeric_cols = [c for c in numeric_cols if not pd.api.types.is_bool_dtype(df_processed[c])]
        except Exception:
            pass
        categorical_cols = df_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        noise_col = getattr(self, 'noise_sentinel_name', 'noise_sentinel')
        special_cols = [self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month', noise_col]
        numeric_cols = [c for c in numeric_cols if c not in special_cols]
        categorical_cols = [c for c in categorical_cols if c not in special_cols]

        print(f"  Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

        if noise_col in df_processed.columns:
            df_processed = df_processed.drop(columns=noise_col)

        df_processed = self._sanitize_feature_columns(df_processed, create_map=create_map)

        mapping = self.feature_name_map or {}
        numeric_sanitized = [mapping.get(col, col) for col in numeric_cols]
        categorical_sanitized = [mapping.get(col, col) for col in categorical_cols]
        self.data_['numeric_features'] = numeric_sanitized
        self.data_['categorical_features'] = categorical_sanitized

        include_noise = self.config.enable_noise_sentinel if include_noise is None else bool(include_noise)
        if include_noise:
            df_processed = df_processed.copy()
            df_processed[self.noise_sentinel_name] = np.random.normal(0, 1, len(df_processed))

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

        splits = self._build_raw_numeric_layers(splits)
        self.data_['splits'] = splits
        return splits



    def _build_raw_numeric_layers(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create an imputed/clipped version of numeric features for RAW models."""

        numeric_cols = [col for col in self.data_.get('numeric_features', []) if col]
        if not numeric_cols:
            return splits

        train_df = splits.get('train')
        if train_df is None or train_df.empty:
            return splits

        available_train = [col for col in numeric_cols if col in train_df.columns]
        if not available_train:
            return splits

        strategy = getattr(self.config, 'numeric_imputation_strategy', 'median') or 'median'
        allowed_strategies = {'mean', 'median', 'most_frequent', 'constant'}
        if strategy not in allowed_strategies:
            strategy = 'median'

        from sklearn.impute import SimpleImputer

        imputer_kwargs: Dict[str, Any] = {'strategy': strategy}
        if strategy == 'constant':
            imputer_kwargs['fill_value'] = getattr(self.config, 'numeric_imputation_fill_value', 0.0)
        imputer = SimpleImputer(**imputer_kwargs)
        imputer.fit(train_df[available_train])

        outlier_method = getattr(self.config, 'numeric_outlier_method', 'clip')
        lower_bounds = upper_bounds = None
        if outlier_method == 'clip':
            lower_q = float(getattr(self.config, 'outlier_lower_quantile', 0.01) or 0.01)
            upper_q = float(getattr(self.config, 'outlier_upper_quantile', 0.99) or 0.99)
            base_numeric = train_df[available_train]
            lower_bounds = base_numeric.quantile(lower_q)
            upper_bounds = base_numeric.quantile(upper_q)

        processed_layers: Dict[str, pd.DataFrame] = {}
        for split_name, split_df in list(splits.items()):
            if split_df is None or getattr(split_df, 'empty', False):
                continue
            if split_name.endswith('_raw_prepped'):
                continue
            if not all(col in split_df.columns for col in available_train):
                continue

            updated_df = split_df.copy()
            transformed = imputer.transform(updated_df[available_train])
            transformed_df = pd.DataFrame(transformed, columns=available_train, index=updated_df.index)

            if outlier_method == 'clip' and lower_bounds is not None and upper_bounds is not None:
                transformed_df = transformed_df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

            updated_df.loc[:, available_train] = transformed_df

            layer_key = f"{split_name}_raw_prepped"
            splits[layer_key] = updated_df
            processed_layers[layer_key] = updated_df

        if processed_layers:
            self.data_['raw_numeric_layers'] = processed_layers
            self.data_['raw_numeric_statistics'] = {
                'strategy': strategy,
                'lower_bounds': lower_bounds.copy() if lower_bounds is not None else None,
                'upper_bounds': upper_bounds.copy() if upper_bounds is not None else None,
            }
            # Summarize preprocessing for reporting
            try:
                lb = lower_bounds if lower_bounds is not None else pd.Series(dtype=float)
                ub = upper_bounds if upper_bounds is not None else pd.Series(dtype=float)
                summary = pd.DataFrame([
                    {
                        'numeric_columns_prepped': len(available_train),
                        'imputation_strategy': strategy,
                        'outlier_method': outlier_method,
                        'lower_quantile': float(getattr(self.config, 'outlier_lower_quantile', 0.01) or 0.01),
                        'upper_quantile': float(getattr(self.config, 'outlier_upper_quantile', 0.99) or 0.99),
                        'lower_bounds_non_null': int(lb.notna().sum()) if hasattr(lb, 'notna') else 0,
                        'upper_bounds_non_null': int(ub.notna().sum()) if hasattr(ub, 'notna') else 0,
                    }
                ])
                self.data_['raw_preprocessing_summary'] = summary
            except Exception:
                pass
            self._current_raw_preprocessor = {
                'columns': available_train,
                'imputer': imputer,
                'outlier_method': outlier_method,
                'lower_bounds': lower_bounds.copy() if lower_bounds is not None else None,
                'upper_bounds': upper_bounds.copy() if upper_bounds is not None else None,
            }
        return splits


    def _transform_raw_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored raw numeric preprocessing (impute + clip) to a dataframe."""

        preprocessor = getattr(self, '_current_raw_preprocessor', None)
        if not preprocessor:
            return df

        columns = [col for col in preprocessor.get('columns', []) if col in df.columns]
        if not columns:
            return df

        imputer = preprocessor.get('imputer')
        transformed_df = df.copy()
        transformed_values = imputer.transform(transformed_df[columns])
        numeric_df = pd.DataFrame(transformed_values, columns=columns, index=transformed_df.index)

        if preprocessor.get('outlier_method') == 'clip':
            lower = preprocessor.get('lower_bounds')
            upper = preprocessor.get('upper_bounds')
            if lower is not None and upper is not None:
                lower = lower.reindex(columns)
                upper = upper.reindex(columns)
                numeric_df = numeric_df.clip(lower=lower, upper=upper, axis=1)

        transformed_df.loc[:, columns] = numeric_df
        return transformed_df


    def _inject_categorical_woe(self, splits: Dict, categorical_cols: List[str]) -> None:
        """Replace categorical columns with their WOE-transformed values."""

        if not categorical_cols:
            return

        for split_name, split_df in list(splits.items()):
            if split_df is None or split_name.endswith('_woe'):
                continue

            target_woe = splits.get(f'{split_name}_woe')
            base_name = split_name
            if target_woe is None and split_name.endswith('_raw_prepped'):
                base_name = split_name[:-len('_raw_prepped')]
                target_woe = splits.get(f'{base_name}_woe')
            if target_woe is None:
                continue

            updated = split_df.copy()
            for col in categorical_cols:
                if col in target_woe.columns and col in updated.columns:
                    updated[col] = target_woe[col]
            splits[split_name] = updated

    def _apply_woe_transformation(self, splits: Dict) -> Dict:
        """Apply WOE transformation and calculate univariate Gini."""

        results = {}

        # Fit WOE on train data
        train_df = splits['train']
        exclude_cols = {c for c in [self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month'] if c}
        noise_name = getattr(self, 'noise_sentinel_name', 'noise_sentinel')
        exclude_cols.add(noise_name)
        exclude_cols.update(getattr(self.config, 'exclude_woe_features', []) or [])
        suffix_blocklist = [s.lower() for s in getattr(self.config, 'exclude_woe_suffixes', []) or []]

        feature_cols = [
            c
            for c in train_df.columns
            if c not in exclude_cols
            and not is_datetime64_any_dtype(train_df[c])
            and not any(c.lower().endswith(suffix) for suffix in suffix_blocklist)
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
        exclude_cols = {c for c in [self.config.target_col, self.config.id_col, self.config.time_col, 'snapshot_month'] if c}
        noise_name = getattr(self, 'noise_sentinel_name', 'noise_sentinel')
        exclude_cols.add(noise_name)
        exclude_cols.update(getattr(self.config, 'exclude_woe_features', []) or [])
        suffix_blocklist = [s.lower() for s in getattr(self.config, 'exclude_woe_suffixes', []) or []]

        feature_cols = [
            col
            for col in raw_train.columns
            if col not in exclude_cols
            and not is_datetime64_any_dtype(raw_train[col])
            and not any(col.lower().endswith(suffix) for suffix in suffix_blocklist)
        ]

        categorical_cols = [
            col
            for col in feature_cols
            if is_object_dtype(raw_train[col]) or is_categorical_dtype(raw_train[col])
        ]

        if not self.config.enable_woe:
            self._inject_categorical_woe(splits, categorical_cols)
            train_df = splits.get('train_raw_prepped', splits['train'])
        else:
            train_df = splits['train_woe']

        if self.config.enable_woe:
            reference_df = splits.get('train_woe', train_df)
        else:
            reference_df = train_df
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
                train_for_psi = reference_df[selected_features]

                if self.config.enable_woe:
                    test_candidate = splits.get('test_woe')
                    if test_candidate is None:
                        test_candidate = splits.get('test')
                    oot_candidate = splits.get('oot_woe')
                    if oot_candidate is None:
                        oot_candidate = splits.get('oot')
                else:
                    test_candidate = splits.get('test_raw_prepped')
                    if test_candidate is None:
                        test_candidate = splits.get('test')
                    oot_candidate = splits.get('oot_raw_prepped')
                    if oot_candidate is None:
                        oot_candidate = splits.get('oot')

                test_for_psi = (
                    test_candidate[selected_features]
                    if test_candidate is not None and not test_candidate.empty
                    else None
                )

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
                vif_summary_df = getattr(self.feature_selector, 'last_vif_summary_', None)
                if isinstance(vif_summary_df, pd.DataFrame):
                    preview = vif_summary_df.head(10).to_dict(orient='records')
                    step_details = {
                        'threshold': self.config.vif_threshold,
                        'removed': int((vif_summary_df.get('status') == 'dropped').sum()),
                        'summary_preview': preview,
                    }
                else:
                    step_details = {'threshold': self.config.vif_threshold}

            elif method == 'correlation':
                selected = self.feature_selector.select_by_correlation(
                    train_df[selected_features],
                    train_df[self.config.target_col],
                    threshold=self.config.correlation_threshold,
                    max_per_cluster=self.config.max_features_per_cluster
                )
                corr_clusters = getattr(self.feature_selector, 'last_correlation_clusters_', None)
                if isinstance(corr_clusters, pd.DataFrame):
                    step_details = {
                        'threshold': self.config.correlation_threshold,
                        'max_per_cluster': self.config.max_features_per_cluster,
                        'clusters_preview': corr_clusters.head(10).to_dict(orient='records'),
                        'cluster_count': int(len(corr_clusters))
                    }
                else:
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
                    if self.config.enable_woe:
                        test_df = splits.get('test_woe', train_df)
                    else:
                        test_df = splits.get('test_raw_prepped', splits.get('test', train_df))
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
            'selection_history': selection_history,
            'vif_summary': getattr(self.feature_selector, 'last_vif_summary_', None),
            'correlation_clusters': getattr(self.feature_selector, 'last_correlation_clusters_', None)
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

        if self.config.enable_woe:
            reference_df = splits.get('train_woe', train_df)
        else:
            reference_df = splits.get('train_raw_prepped', train_df)
        time_series = train_df[time_col]

        for label in time_series.dropna().unique():
            mask = time_series == label
            frames[str(label)] = reference_df.loc[mask].reset_index(drop=True)

        return frames

    def _train_models(self, splits: Dict, selection_results: Dict, mode_label: Optional[str] = None) -> Dict:
        """Train all configured models."""

        selected_features = selection_results['selected_features']
        mode_label = (mode_label or ('WOE' if self.config.enable_woe else 'RAW')).upper()

        if self.config.enable_woe:
            X_train = splits['train_woe'][selected_features]
            y_train = splits['train'][self.config.target_col]
            X_test = splits.get('test_woe', pd.DataFrame())[selected_features] if 'test_woe' in splits else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None
            oot_source = splits.get('oot_woe')
        else:
            train_source = splits.get('train_raw_prepped', splits['train'])
            X_train = train_source[selected_features]
            y_train = splits['train'][self.config.target_col]
            test_source = splits.get('test_raw_prepped')
            if test_source is None:
                test_source = splits.get('test', pd.DataFrame())
            X_test = test_source[selected_features] if isinstance(test_source, pd.DataFrame) and not test_source.empty else None
            y_test = splits.get('test', pd.DataFrame())[self.config.target_col] if 'test' in splits else None
            oot_source = splits.get('oot_raw_prepped')
            if oot_source is None:
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
            X_train, y_train, X_test, y_test, X_oot, y_oot,
            mode=mode_label,
            name_prefix=f"{mode_label}_"
        )

        if self.config.enable_noise_sentinel and 'noise_sentinel' in selected_features:
            print("  WARNING: Noise sentinel was selected - feature selection may be overfitting!")

        model_results['mode'] = mode_label
        # Accumulate available models across calls
        try:
            current_models = model_results.get('models', {}) or {}
            union = {}
            if isinstance(getattr(self, 'models_', None), dict):
                union.update(self.models_)
            union.update(current_models)
            self.models_ = union
        except Exception:
            self.models_ = model_results.get('models', {}) or {}
        # Expose per-mode registries early for notebooks/diagnostics
        try:
            reg = self.results_.setdefault('model_object_registry', {})
            reg[mode_label] = model_results.get('models', {}) or {}
            sc = self.results_.setdefault('model_scores_registry', {})
            sc[mode_label] = model_results.get('scores', {}) or {}
        except Exception:
            pass
        self.selected_features_ = model_results.get('selected_features', [])
        self._set_active_model(model_results, mode_label=mode_label)
        return model_results
    @staticmethod
    def _model_supports_probability(model: Any) -> bool:
        if model is None:
            return False
        for attr in ('predict_proba', 'decision_function'):
            candidate = getattr(model, attr, None)
            if callable(candidate):
                return True
        return False

    def _set_active_model(self, model_results: Dict, *, mode_label: Optional[str] = None) -> None:
        models = model_results.get('models') or {}
        models = {name: mdl for name, mdl in models.items() if mdl is not None}
        preferred = getattr(self.config, 'score_model_name', 'best')
        preferred = (preferred or 'best').strip()
        best_name = model_results.get('best_model_name')

        candidate_sequence: List[str] = []
        if preferred and preferred.lower() != 'best' and preferred in models:
            candidate_sequence.append(preferred)
        if best_name and best_name not in candidate_sequence:
            candidate_sequence.append(best_name)
        if models:
            logistic_candidates = [
                name for name in models
                if 'logistic' in name.lower() or 'lr' in name.lower()
            ]
            remaining = [name for name in models if name not in logistic_candidates]
            for name in logistic_candidates + remaining:
                if name not in candidate_sequence:
                    candidate_sequence.append(name)
        if not candidate_sequence and models:
            candidate_sequence = list(models.keys())

        active_name = candidate_sequence[0] if candidate_sequence else best_name
        active_model = models.get(active_name) or model_results.get('best_model')
        if active_model is None and models:
            active_name, active_model = next(iter(models.items()))

        if not self._model_supports_probability(active_model):
            for name in candidate_sequence:
                candidate = models.get(name)
                if self._model_supports_probability(candidate):
                    if active_name != name:
                        print(f"  Active model '{active_name}' lacks probability interface; switching to '{name}'.")
                    active_name = name
                    active_model = candidate
                    break

        if not self._model_supports_probability(active_model):
            fallback_model = model_results.get('best_model')
            if self._model_supports_probability(fallback_model):
                print("  Active model lacks probability interface; using best model for downstream steps.")
                active_model = fallback_model
                active_name = model_results.get('best_model_name', active_name)

        model_results['active_model_name'] = active_name
        model_results['active_model'] = active_model
        model_results['active_model_mode'] = mode_label

    def _apply_stage1_calibration(self, model_results: Dict, calibration_df: pd.DataFrame) -> Dict:
        """Apply Stage 1 calibration (long-run average)."""

        # Check if we have a model to calibrate
        base_model = model_results.get('active_model') or model_results.get('best_model')
        base_name = model_results.get('active_model_name') or model_results.get('best_model_name')
        if base_model is None:
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
            cal_processed = self._process_data(calibration_df, create_map=False, include_noise=False)
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
            cal_processed = self._process_data(calibration_df, create_map=False, include_noise=False)
            cal_prepped = self._transform_raw_numeric(cal_processed)
            categorical_feats = self.data_.get('categorical_features', [])
            if categorical_feats:
                try:
                    cal_woe = self.woe_transformer.transform(cal_processed)
                except Exception:
                    cal_woe = None
                if cal_woe is not None:
                    for col in categorical_feats:
                        if col in cal_prepped.columns and col in cal_woe.columns:
                            cal_prepped[col] = cal_woe[col]
            X_cal = cal_prepped[selected_features].copy()

            for col in X_cal.columns:
                if X_cal[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_cal[col]):
                    X_cal.loc[:, col] = pd.to_numeric(X_cal[col], errors='coerce').fillna(0)
        y_cal = calibration_df[self.config.target_col]

        # Calibrate best model
        calibrated_model = self.calibrator.calibrate_stage1(
            base_model, X_cal, y_cal,
            method=self.config.calibration_method
        )

        metrics = self.calibrator.evaluate_calibration(
            calibrated_model, X_cal, y_cal
        )

        X_cal_filled = X_cal.fillna(0) if hasattr(X_cal, 'fillna') else X_cal
        base_scores = predict_positive_proba(base_model, X_cal_filled)
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
            'model_name': base_name,
        }
        if calibration_curve is not None:
            stage1_details['curve_points'] = len(calibration_curve)

        metrics['model_name'] = base_name
        return {
            'calibrated_model': calibrated_model,
            'calibration_metrics': metrics,
            'stage1_details': stage1_details,
            'calibration_curve': calibration_curve,
            'selected_features': model_results.get('selected_features', []),
            'base_model_name': base_name,
        }

    def _apply_stage2_calibration(self, stage1_results: Dict, stage2_df: pd.DataFrame) -> Dict:
        """Apply Stage 2 calibration (recent period adjustment)."""

        # Check if we have a calibrated model from stage 1
        if not stage1_results or 'calibrated_model' not in stage1_results:
            print("  Skipping Stage 2 calibration: No Stage 1 model available")
            return {}

        # Prepare Stage 2 data
        selected_features = list(self.selected_features_) if getattr(self, 'selected_features_', None) else []
        if not selected_features and isinstance(stage1_results, dict):
            selected_features = list(stage1_results.get('selected_features', []))
        if not selected_features and isinstance(self.results_.get('model_results'), dict):
            selected_features = list(self.results_['model_results'].get('selected_features', []))

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
        stage2_processed = self._process_data(stage2_input, create_map=False, include_noise=False)
        stage2_processed = stage2_processed.reset_index(drop=True)
        if y_stage2_source is not None:
            y_stage2_source = y_stage2_source.reset_index(drop=True)

        if self.config.enable_woe:
            stage2_woe = self.woe_transformer.transform(stage2_processed)
            available_features = [f for f in selected_features if f in stage2_woe.columns]
            X_stage2 = stage2_woe[available_features].copy()
        else:
            stage2_prepped = self._transform_raw_numeric(stage2_processed)
            categorical_feats = self.data_.get('categorical_features', [])
            try:
                stage2_woe = self.woe_transformer.transform(stage2_processed)
            except Exception:
                stage2_woe = None
            if stage2_woe is not None:
                for col in categorical_feats:
                    if col in stage2_prepped.columns and col in stage2_woe.columns:
                        stage2_prepped[col] = stage2_woe[col]
            X_stage2 = stage2_prepped[selected_features].copy()

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

        primary_results = model_results if isinstance(model_results, dict) else {}
        stage1_results = stage1_results or {}
        if (not primary_results or primary_results.get('calibrated_model') is None) and stage1_results.get('calibrated_model') is not None:
            print('  Risk bands: using Stage-1 calibrated model (Stage-2 unavailable).')
            primary_results = stage1_results

        if not primary_results or primary_results.get('calibrated_model') is None:
            print('  Skipping risk bands: No model available')
            return {}

        model = primary_results['calibrated_model']
        selected_features = self.selected_features_
        target_col = self.config.target_col

        if not selected_features:
            print('  Skipping risk bands: No features selected')
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
                    transformed = self._transform_raw_numeric(override_df)
                    categorical_feats = self.data_.get('categorical_features', [])
                    try:
                        transformed_woe = self.woe_transformer.transform(override_df)
                    except Exception:
                        transformed_woe = None
                    if transformed_woe is not None:
                        for col in categorical_feats:
                            if col in transformed.columns and col in transformed_woe.columns:
                                transformed[col] = transformed_woe[col]
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
                    X_eval_source = splits['test_woe']
                else:
                    X_eval_source = splits.get('test_raw_prepped', splits['test'])
                y_eval = splits['test'][target_col]
            else:
                if self.config.enable_woe:
                    X_eval_source = splits['train_woe']
                else:
                    X_eval_source = splits.get('train_raw_prepped', splits['train'])
                y_eval = splits['train'][target_col]
            X_eval = X_eval_source[selected_features].copy()
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
                    ref_prepped = self._transform_raw_numeric(ref_df)
                    try:
                        ref_woe = self.woe_transformer.transform(ref_df)
                    except Exception:
                        ref_woe = None
                    if ref_woe is not None:
                        for col in self.data_.get('categorical_features', []):
                            if col in ref_prepped.columns and col in ref_woe.columns:
                                ref_prepped[col] = ref_woe[col]
                    available_features = [f for f in selected_features if f in ref_prepped.columns]
                    missing = set(selected_features) - set(available_features)
                    if missing:
                        print(f"  Warning: Reference dataset missing {len(missing)} selected features; using available subset.")
                    X_eval = ref_prepped[available_features].copy()
                y_eval = ref_df[target_col].astype(float)
                source = 'override_reference'
            else:
                print(f"  Warning: Risk band optimisation sample contains {len(X_eval)} records (< {min_sample}); results may be unstable.")
        def _probabilities_from(candidate):
            if candidate is None:
                return None
            # Align features to candidate's expectations if available
            def _prepare_X_for_model(model, X):
                if X is None or not hasattr(X, 'copy'):
                    return None
                names = getattr(model, 'feature_names_in_', None)
                Xc = X.copy()
                for col in Xc.columns:
                    if is_object_dtype(Xc[col]) or is_datetime64_any_dtype(Xc[col]):
                        Xc.loc[:, col] = pd.to_numeric(Xc[col], errors='coerce').fillna(0)
                if names is None:
                    return Xc
                # Ensure all expected columns exist, add missing as zeros
                expected = list(names)
                missing = [c for c in expected if c not in Xc.columns]
                for m in missing:
                    Xc.loc[:, m] = 0.0
                # Reorder to model's expected order
                Xc = Xc[expected]
                try:
                    present = len(expected) - len(missing)
                    print(f"  Risk bands: aligned features to model ({present}/{len(expected)} present; added {len(missing)}).")
                except Exception:
                    pass
                return Xc
            try:
                Xc = _prepare_X_for_model(candidate, X_eval)
                if Xc is None:
                    return None
                return np.asarray(predict_positive_proba(candidate, Xc), dtype=float).ravel()
            except ValueError:
                try:
                    Xc = _prepare_X_for_model(candidate, X_eval)
                    if Xc is None:
                        return None
                    proba = candidate.predict_proba(Xc)
                except AttributeError:
                    try:
                        Xc = _prepare_X_for_model(candidate, X_eval)
                        if Xc is None:
                            return None
                        pred = candidate.predict(Xc)
                    except Exception:
                        return None
                    arr = np.asarray(pred, dtype=float).ravel()
                    if arr.size == 0 or not np.all(np.isfinite(arr)):
                        return None
                    if arr.min() < 0 or arr.max() > 1:
                        return None
                    return arr
                except Exception:
                    return None
                else:
                    proba = np.asarray(proba)
                    if proba.ndim == 2:
                        if proba.shape[1] == 1:
                            return proba[:, 0]
                        return proba[:, -1]
                    return proba.ravel()

        candidate_models = [model]
        fallback_candidates = []
        if isinstance(primary_results, dict):
            fallback_candidates.extend([
                primary_results.get('calibrated_model'),
                primary_results.get('stage1_model'),
                primary_results.get('best_model'),
                primary_results.get('active_model'),
            ])
            models_dict = primary_results.get('models') or {}
            if isinstance(models_dict, dict):
                fallback_candidates.extend(models_dict.values())
        if isinstance(stage1_results, dict):
            fallback_candidates.extend([
                stage1_results.get('calibrated_model'),
                stage1_results.get('best_model'),
                stage1_results.get('active_model'),
            ])
        if isinstance(model_results, dict):
            fallback_candidates.extend([
                model_results.get('calibrated_model'),
                model_results.get('stage1_model'),
                model_results.get('best_model'),
                model_results.get('active_model'),
            ])
            models_dict = model_results.get('models') or {}
            if isinstance(models_dict, dict):
                fallback_candidates.extend(models_dict.values())
        cache_results = self.results_.get('model_results', {}) if isinstance(self.results_, dict) else {}
        if isinstance(cache_results, dict):
            fallback_candidates.extend([
                cache_results.get('calibrated_model'),
                cache_results.get('stage1_model'),
                cache_results.get('best_model'),
                cache_results.get('active_model'),
            ])
            models_dict = cache_results.get('models') or {}
            if isinstance(models_dict, dict):
                fallback_candidates.extend(models_dict.values())
        fallback_candidates = [cand for cand in fallback_candidates if cand is not None]
        for cand in fallback_candidates:
            if cand in candidate_models:
                continue
            candidate_models.append(cand)

        predictions = None
        for cand in candidate_models:
            candidate_scores = _probabilities_from(cand)
            if candidate_scores is not None:
                model = cand
                predictions = candidate_scores
                if cand is not primary_results.get('calibrated_model'):
                    print('  Risk bands: using fallback model for probability estimation.')
                break
        if predictions is None:
            print('  Risk band optimization skipped: no model with probability interface available.')
            return {}

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
                # Align features to the stage1 model if needed
                try:
                    Xc = _prepare_X_for_model(stage1_model, X_eval)
                except Exception:
                    Xc = X_eval
                try:
                    predictions = predict_positive_proba(stage1_model, Xc)
                    predictions = np.asarray(predictions, dtype=float).ravel()
                except Exception:
                    # As last resort, scan cached models for any probability-capable candidate
                    predictions = None
                    cached_models = []
                    for source in (primary_results, stage1_results, model_results, cache_results):
                        if isinstance(source, dict):
                            cached_models.extend([source.get('active_model'), source.get('best_model'), source.get('calibrated_model')])
                            mdl_map = source.get('models') or {}
                            if isinstance(mdl_map, dict):
                                cached_models.extend(mdl_map.values())
                    for cand in cached_models:
                        if cand is None:
                            continue
                        Xc2 = _prepare_X_for_model(cand, X_eval)
                        if Xc2 is None:
                            continue
                        try:
                            predictions = predict_positive_proba(cand, Xc2)
                            predictions = np.asarray(predictions, dtype=float).ravel()
                            model = cand
                            print('  Risk bands: using cached probability-capable model (stage1 fallback failed).')
                            break
                        except Exception:
                            continue
                if predictions is None or np.unique(predictions).size <= 1:
                    print('  Risk band optimization skipped: predictions still lack variation.')
                    return {}
                model = stage1_model if predictions is not None else model
            else:
                print('  Risk band optimization skipped: no suitable model for fallback.')
                return {}

        band_frame: pd.DataFrame
        risk_bands = self.risk_band_optimizer.optimize_bands(
            predictions, y_eval,
            n_bands=self.config.n_risk_bands,
            method=getattr(self.config, 'risk_band_method', getattr(self.config, 'band_method', 'quantile'))
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
            return {'dataframe': score_df.copy(), 'metrics': None, 'reports': {}, 'noise_sentinel': None}

        final_model = stage_results.get('calibrated_model')
        final_features = selection_results.get('selected_features') or getattr(self, 'selected_features_', [])

        if not final_features:
            print("  Skipping scoring: No features selected")
            return {'dataframe': score_df.copy(), 'metrics': None, 'reports': {}, 'noise_sentinel': None}

        score_df = score_df.copy()
        target_col = getattr(self.config, 'target_col', None)
        if target_col and target_col not in score_df.columns:
            score_df[target_col] = np.nan

        noise_col = getattr(self, 'noise_sentinel_name', 'noise_sentinel')
        score_processed = self._process_data(score_df, create_map=False, include_noise=False)

        noise_series: Optional[np.ndarray] = None
        if getattr(self.config, 'enable_noise_sentinel', False):
            noise_series = np.random.normal(0, 1, len(score_processed))
            score_processed = score_processed.copy()
            score_processed[noise_col] = noise_series

        if self.config.enable_woe:
            score_matrix = self.woe_transformer.transform(score_processed)[final_features]
        else:
            score_prepped = self._transform_raw_numeric(score_processed)
            categorical_feats = self.data_.get('categorical_features', [])
            try:
                score_woe = self.woe_transformer.transform(score_processed)
            except Exception:
                score_woe = None
            if score_woe is not None:
                for col in categorical_feats:
                    if col in score_prepped.columns and col in score_woe.columns:
                        score_prepped[col] = score_woe[col]
            score_matrix = score_prepped[final_features]
        score_matrix = score_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

        stage1_model = (self.results_.get('calibration_stage1') or {}).get('calibrated_model')

        def _probabilities_from(candidate):
            if candidate is None:
                return None
            try:
                return np.asarray(predict_positive_proba(candidate, score_matrix), dtype=float).ravel()
            except ValueError:
                try:
                    proba = candidate.predict_proba(score_matrix)
                except AttributeError:
                    try:
                        pred = candidate.predict(score_matrix)
                    except Exception:
                        return None
                    arr = np.asarray(pred, dtype=float).ravel()
                    if arr.size == 0 or not np.all(np.isfinite(arr)):
                        return None
                    if arr.min() < 0 or arr.max() > 1:
                        return None
                    return arr
                except Exception:
                    return None
                else:
                    proba = np.asarray(proba)
                    if proba.ndim == 2:
                        if proba.shape[1] == 1:
                            return proba[:, 0]
                        return proba[:, -1]
                    return proba.ravel()

        candidate_models = [final_model]
        fallback_candidates = []
        if isinstance(stage_results, dict):
            fallback_candidates.extend([
                stage_results.get('calibrated_model'),
                stage_results.get('stage1_model'),
                stage_results.get('best_model'),
            ])
        if isinstance(model_results, dict):
            fallback_candidates.extend([
                model_results.get('calibrated_model'),
                model_results.get('stage1_model'),
                model_results.get('best_model'),
            ])
            models_dict = model_results.get('models') or {}
            if isinstance(models_dict, dict):
                fallback_candidates.extend(models_dict.values())
        cache_results = self.results_.get('model_results', {}) if isinstance(self.results_, dict) else {}
        if isinstance(cache_results, dict):
            fallback_candidates.extend([
                cache_results.get('calibrated_model'),
                cache_results.get('stage1_model'),
                cache_results.get('best_model'),
            ])
            models_dict = cache_results.get('models') or {}
            if isinstance(models_dict, dict):
                fallback_candidates.extend(models_dict.values())
        if stage1_model is not None:
            fallback_candidates.append(stage1_model)
        fallback_candidates = [cand for cand in fallback_candidates if cand is not None]
        for cand in fallback_candidates:
            if cand in candidate_models:
                continue
            candidate_models.append(cand)

        raw_scores = None
        for cand in candidate_models:
            scores_candidate = _probabilities_from(cand)
            if scores_candidate is not None:
                final_model = cand
                raw_scores = scores_candidate
                break
        if raw_scores is None:
            print('  Skipping scoring: models lack probability interface.')
            return {'dataframe': score_df.copy(), 'metrics': None, 'reports': {}, 'noise_sentinel': None}

        scores = raw_scores

        result_df = score_df.copy()
        if self.config.enable_woe:
            for col in final_features:
                result_df[f"{col}_woe"] = score_matrix[col].to_numpy()
        if noise_series is not None:
            result_df[noise_col] = noise_series

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
            band_edges = getattr(self.risk_band_optimizer, 'bands_', None)
        if band_edges is not None:
            result_df['risk_band'] = self.risk_band_optimizer.assign_bands(scores, band_edges)
        else:
            result_df['risk_band'] = np.nan

        training_scores = None

        if isinstance(splits, dict) and 'train' in splits:
            if self.config.enable_woe and 'train_woe' in splits:
                train_source = splits['train_woe']
            else:
                train_source = splits.get('train_raw_prepped', splits['train'])
            train_matrix = train_source[final_features]
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

        noise_report = None
        if noise_series is not None:
            noise_report = self._compute_noise_sentinel_report(noise_series, scores)
            noise_report['selected_in_features'] = noise_col in final_features
            metrics['noise_sentinel_column'] = noise_col
            metrics['noise_sentinel'] = noise_report
        else:
            metrics['noise_sentinel_column'] = None
            metrics['noise_sentinel'] = None

        reports = create_scoring_report(metrics)

        return {
            'dataframe': result_df,
            'metrics': metrics,
            'reports': reports,
            'noise_sentinel': noise_report,
        }

    def _compute_noise_sentinel_report(self, noise_values: np.ndarray, score_values: np.ndarray) -> Dict[str, Any]:
        """Summarise noise sentinel behaviour against model scores."""

        report: Dict[str, Any] = {}
        noise_arr = np.asarray(noise_values, dtype=float)
        score_arr = np.asarray(score_values, dtype=float)
        if noise_arr.size == 0:
            return {'n': 0}

        report['n'] = int(noise_arr.size)
        report['mean'] = float(np.mean(noise_arr))
        report['std'] = float(np.std(noise_arr))
        report['min'] = float(np.min(noise_arr))
        report['max'] = float(np.max(noise_arr))
        report['quantiles'] = {
            'p10': float(np.percentile(noise_arr, 10)),
            'p50': float(np.percentile(noise_arr, 50)),
            'p90': float(np.percentile(noise_arr, 90)),
        }

        if score_arr.size > 1:
            try:
                pearson = float(np.corrcoef(noise_arr, score_arr)[0, 1])
            except Exception:
                pearson = float('nan')
            if np.isfinite(pearson):
                report['pearson_correlation'] = pearson
            try:
                spearman, pvalue = stats.spearmanr(noise_arr, score_arr)
            except Exception:
                spearman, pvalue = float('nan'), float('nan')
            if np.isfinite(spearman):
                report['spearman_correlation'] = float(spearman)
            if np.isfinite(pvalue):
                report['spearman_p_value'] = float(pvalue)

            alert_threshold = max(0.1, float(getattr(self.config, 'noise_threshold', 0.5)))
            alerts = []
            if np.isfinite(pearson) and abs(pearson) >= alert_threshold:
                alerts.append({'metric': 'pearson', 'value': pearson, 'threshold': alert_threshold})
            if np.isfinite(spearman) and abs(spearman) >= alert_threshold:
                alerts.append({'metric': 'spearman', 'value': float(spearman), 'threshold': alert_threshold})
            if alerts:
                report['alerts'] = alerts

            order = np.argsort(score_arr)
            bucket = max(1, order.size // 10)
            if bucket > 0 and order.size >= bucket:
                top_idx = order[-bucket:]
                bottom_idx = order[:bucket]
                report['noise_decile_mean'] = {
                    'top': float(np.mean(noise_arr[top_idx])),
                    'bottom': float(np.mean(noise_arr[bottom_idx])),
                }
                report['score_deciles'] = {
                    'p10': float(np.percentile(score_arr, 10)),
                    'p50': float(np.percentile(score_arr, 50)),
                    'p90': float(np.percentile(score_arr, 90)),
                }

        return report

    def _compute_performance_artifacts(self, model_results: Dict[str, Any], splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Derive confusion matrices, lift tables, and metric summaries for major splits."""

        from .utils.metrics import calculate_metrics
        from .monitoring import _build_lift_table

        artifacts: Dict[str, pd.DataFrame] = {}
        if not model_results or not splits:
            return artifacts

        best_model = model_results.get('best_model')
        if best_model is None:
            return artifacts

        selected_features = model_results.get('selected_features', [])
        if not selected_features:
            return artifacts

        mode = str(model_results.get('mode') or ('WOE' if self.config.enable_woe else 'RAW')).upper()
        dataset_specs = []

        def _append_dataset(label: str, feature_key: str, target_key: str) -> None:
            x_source = splits.get(feature_key)
            target_df = splits.get(target_key)
            if x_source is None or target_df is None:
                return
            if self.config.target_col not in target_df.columns:
                return
            available = [feat for feat in selected_features if feat in x_source.columns]
            if not available:
                return
            X_matrix = x_source[available].copy()
            y_series = target_df[self.config.target_col].astype(float)
            if y_series.empty:
                return
            dataset_specs.append((label, X_matrix, y_series))

        if mode == 'WOE':
            _append_dataset('train', 'train_woe', 'train')
            _append_dataset('test', 'test_woe', 'test')
            _append_dataset('oot', 'oot_woe', 'oot')
        else:
            _append_dataset('train', 'train_raw_prepped', 'train')
            _append_dataset('test', 'test_raw_prepped', 'test')
            _append_dataset('oot', 'oot_raw_prepped', 'oot')

        if not dataset_specs:
            return artifacts

        performance_rows: List[Dict[str, Any]] = []
        confusion_rows: List[Dict[str, Any]] = []
        lift_frames: List[pd.DataFrame] = []
        baseline_metrics_row: Optional[Dict[str, Any]] = None
        baseline_lift = None

        for label, X_matrix, y_series in dataset_specs:
            try:
                predictions = predict_positive_proba(best_model, X_matrix)
            except Exception:
                continue
            if len(predictions) != len(y_series):
                continue
            metrics = calculate_metrics(y_series.values, predictions)
            performance_rows.append({
                'dataset': label,
                'auc': float(metrics.get('auc', float('nan'))),
                'gini': float(metrics.get('gini', float('nan'))),
                'ks': float(metrics.get('ks_statistic', float('nan'))),
                'brier': float(metrics.get('brier_score', float('nan'))),
                'log_loss': float(metrics.get('log_loss', float('nan'))),
                'accuracy': float(metrics.get('accuracy', float('nan'))),
                'precision': float(metrics.get('precision', float('nan'))),
                'recall': float(metrics.get('recall', float('nan'))),
                'f1': float(metrics.get('f1', float('nan'))),
            })
            confusion_rows.append({
                'dataset': label,
                'true_negatives': int(metrics.get('true_negatives', 0)),
                'false_positives': int(metrics.get('false_positives', 0)),
                'false_negatives': int(metrics.get('false_negatives', 0)),
                'true_positives': int(metrics.get('true_positives', 0)),
            })
            try:
                lift_df = _build_lift_table(y_series.values, predictions)
            except Exception:
                lift_df = pd.DataFrame()
            if isinstance(lift_df, pd.DataFrame) and not lift_df.empty:
                lift_df = lift_df.copy()
                lift_df.insert(0, 'dataset', label)
                lift_frames.append(lift_df)
                if label == 'train':
                    baseline_lift = lift_df
            if label == 'train':
                baseline_metrics_row = {
                    'dataset': label,
                    **{key: float(metrics.get(key, float('nan'))) for key in (
                        'auc', 'gini', 'ks_statistic', 'brier_score', 'log_loss', 'accuracy', 'precision', 'recall', 'f1'
                    )}
                }

        if performance_rows:
            artifacts['performance_report'] = pd.DataFrame(performance_rows)
        if confusion_rows:
            artifacts['confusion_matrix'] = pd.DataFrame(confusion_rows)
        if lift_frames:
            artifacts['lift_table'] = pd.concat(lift_frames, ignore_index=True)
        if baseline_metrics_row is not None:
            artifacts['baseline_metrics'] = pd.DataFrame([baseline_metrics_row])
        if baseline_lift is not None:
            artifacts['baseline_lift_table'] = baseline_lift

        return artifacts

    def _generate_reports(self) -> Dict:
        """Generate comprehensive reports."""

        reports = {}

        model_results = self.results_.get('model_results')
        woe_results = self.results_.get('woe_results')

        dictionary_candidate = getattr(self, 'data_dictionary', None)
        if dictionary_candidate is None:
            dictionary_candidate = self.results_.get('data_dictionary') or self.data_.get('dictionary')
        if dictionary_candidate is not None:
            self.data_dictionary = dictionary_candidate
            self.results_['data_dictionary'] = dictionary_candidate
            self.reporter.register_data_dictionary(dictionary_candidate)
        else:
            self.reporter.register_data_dictionary(None)

        self.reporter.register_tsfresh_metadata(self.results_.get('tsfresh_metadata'))
        self.reporter.register_selection_history(self.results_.get('selection_results'))

        # Model performance report
        if model_results:
            dual_scores = self.results_.get('model_scores_registry')
            dual_models = self.results_.get('model_object_registry')
            if isinstance(dual_scores, dict) and dual_scores:
                combined_scores = dict(model_results.get('scores') or {})
                for score_map in dual_scores.values():
                    if isinstance(score_map, dict):
                        combined_scores.update(score_map)
                if combined_scores:
                    model_results = dict(model_results)
                    model_results['scores'] = combined_scores
            if isinstance(dual_models, dict) and dual_models:
                aggregated_models = dict(model_results.get('models') or {})
                for mode_map in dual_models.values():
                    if isinstance(mode_map, dict):
                        aggregated_models.update(mode_map)
                if aggregated_models:
                    model_results = dict(model_results)
                    model_results['models'] = aggregated_models
                    # Ensure pipeline exposes all available models even in reporting-only runs
                    try:
                        self.models_ = aggregated_models
                    except Exception:
                        pass

            # Re-compute best model globally across RAW+WOE using OOT/Test/Train AUC preference
            try:
                scores = model_results.get('scores') or {}
                def _metric_of(name: str) -> float:
                    s = scores.get(name) or {}
                    if s.get('oot_auc') is not None:
                        return float(s.get('oot_auc'))
                    if s.get('test_auc') is not None:
                        return float(s.get('test_auc'))
                    if s.get('train_auc') is not None:
                        return float(s.get('train_auc'))
                    return -1e9
                if scores:
                    best_name = max(scores.keys(), key=_metric_of)
                    model_results['best_model_name'] = best_name
                    best_obj = (model_results.get('models') or {}).get(best_name)
                    if best_obj is not None:
                        model_results['best_model'] = best_obj
                        model_results['active_model_name'] = best_name
                        model_results['active_model'] = best_obj
                # Reflect back to cached results for notebook convenience
                self.results_['model_results'] = model_results
            except Exception:
                pass
            reports['model_performance'] = self.reporter.generate_model_report(
                model_results, self.data_dictionary
            )
            # Bubble up models_summary and best_model dataframes to top-level reports for notebook convenience
            ms = self.reporter.reports_.get('models_summary')
            if isinstance(ms, pd.DataFrame) and not ms.empty:
                reports['models_summary'] = ms
            bm = self.reporter.reports_.get('best_model')
            if isinstance(bm, pd.DataFrame) and not bm.empty:
                reports['best_model'] = bm
            perf_artifacts = self._compute_performance_artifacts(model_results, self.results_.get('splits', {}))
            for key, value in perf_artifacts.items():
                if isinstance(value, pd.DataFrame):
                    reports[key] = value
                    self.reporter.reports_[key] = value

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

            feature_importance = model_results.get('feature_importance', {})
            best_model_name = model_results.get('best_model_name')
            shap_df = None
            if isinstance(feature_importance, dict) and best_model_name in feature_importance:
                shap_candidate = feature_importance.get(best_model_name)
                if isinstance(shap_candidate, pd.DataFrame):
                    shap_df = shap_candidate.copy()
            if isinstance(shap_df, pd.DataFrame) and not shap_df.empty:
                reports['shap_importance'] = shap_df
                self.reporter.reports_['shap_importance'] = shap_df

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
            band_metrics = self.reporter.reports_.get('risk_bands_metrics')
            if isinstance(band_metrics, pd.DataFrame) and not band_metrics.empty:
                reports['band_metrics'] = band_metrics

        # Calibration tables (decile/band level observed vs predicted with CI & binomial p)
        try:
            from .core.calibration_analyzer import CalibrationAnalyzer
            stage2 = self.results_.get('calibration_stage2') or {}
            stage1 = self.results_.get('calibration_stage1') or {}
            model_obj = None
            for candidate in (
                stage2.get('calibrated_model') if isinstance(stage2, dict) else None,
                stage1.get('calibrated_model') if isinstance(stage1, dict) else None,
                (self.results_.get('model_results') or {}).get('active_model') if isinstance(self.results_.get('model_results'), dict) else None,
                (self.results_.get('model_results') or {}).get('best_model') if isinstance(self.results_.get('model_results'), dict) else None,
            ):
                if candidate is not None:
                    model_obj = candidate
                    break
            splits = self.results_.get('splits') or {}
            target_col = getattr(self.config, 'target_col', 'target')
            X_eval_df = None
            y_eval_series = None
            # Prefer test split if available
            if 'test' in splits and isinstance(splits['test'], pd.DataFrame) and not splits['test'].empty:
                base = splits['test']
                if self.config.enable_woe and 'test_woe' in splits:
                    X_eval_df = splits['test_woe']
                else:
                    X_eval_df = splits.get('test_raw_prepped', base)
                y_eval_series = base[target_col] if target_col in base.columns else None
            elif 'train' in splits and isinstance(splits['train'], pd.DataFrame):
                base = splits['train']
                if self.config.enable_woe and 'train_woe' in splits:
                    X_eval_df = splits['train_woe']
                else:
                    X_eval_df = splits.get('train_raw_prepped', base)
                y_eval_series = base[target_col] if target_col in base.columns else None
            if model_obj is not None and X_eval_df is not None and y_eval_series is not None:
                features = list(self.selected_features_) if getattr(self, 'selected_features_', None) else X_eval_df.columns.tolist()
                features = [f for f in features if f in X_eval_df.columns]
                X_eval = X_eval_df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
                y_eval = y_eval_series.astype(float).to_numpy()
                try:
                    from .core.utils import predict_positive_proba
                    y_pred = predict_positive_proba(model_obj, X_eval)
                    analyzer = CalibrationAnalyzer()
                    cal_res = analyzer.analyze_calibration(y_eval, np.asarray(y_pred, dtype=float).ravel(), use_deciles=True)
                    segments = cal_res.get('segments')
                    if isinstance(segments, pd.DataFrame):
                        self.reporter.reports_['calibration_tables'] = segments
                        reports['calibration_tables'] = segments
                except Exception:
                    pass
        except Exception:
            pass

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

        noise_diag = self.results_.get('noise_sentinel_diagnostics')
        if noise_diag:
            if isinstance(noise_diag, dict):
                reports['noise_sentinel_check'] = pd.DataFrame([noise_diag])
            elif isinstance(noise_diag, pd.DataFrame):
                reports['noise_sentinel_check'] = noise_diag
            self.reporter.reports_['noise_sentinel_check'] = reports['noise_sentinel_check']

        if 'data_dictionary' in self.reporter.reports_:
            reports['data_dictionary'] = self.reporter.reports_['data_dictionary']

        if 'pipeline_overview' not in self.reporter.reports_:
            overview_rows = [
                {'item': 'target_column', 'value': getattr(self.config, 'target_col', None)},
                {'item': 'id_column', 'value': getattr(self.config, 'id_col', None)},
                {'item': 'time_column', 'value': getattr(self.config, 'time_col', None)},
                {'item': 'selection_steps', 'value': ', '.join(getattr(self.config, 'selection_steps', []))},
                {'item': 'algorithms', 'value': ', '.join(getattr(self.config, 'algorithms', []))},
                {'item': 'enable_woe', 'value': getattr(self.config, 'enable_woe', True)},
                {'item': 'enable_scoring', 'value': getattr(self.config, 'enable_scoring', False)},
            ]
            self.reporter.reports_['pipeline_overview'] = pd.DataFrame(overview_rows)
        # Data layers overview (train/test/oot and prepared variants)
        try:
            splits = self.results_.get('splits', {}) or {}
            layer_rows = []
            for key in ['train', 'train_woe', 'train_raw_prepped', 'test', 'test_woe', 'test_raw_prepped', 'oot', 'oot_woe', 'oot_raw_prepped']:
                df_layer = splits.get(key)
                n_rows = int(len(df_layer)) if isinstance(df_layer, pd.DataFrame) else 0
                n_cols = int(df_layer.shape[1]) if isinstance(df_layer, pd.DataFrame) else 0
                layer_rows.append({'layer': key, 'rows': n_rows, 'cols': n_cols})
            self.reporter.reports_['data_layers_overview'] = pd.DataFrame(layer_rows)
        except Exception:
            pass
        # Bubble up RAW preprocessing summary from pipeline state for reporting convenience
        try:
            raw_prep = self.data_.get('raw_preprocessing_summary')
            if isinstance(raw_prep, pd.DataFrame) and not raw_prep.empty:
                self.reporter.reports_['raw_preprocessing_summary'] = raw_prep
        except Exception:
            pass
        # Convenience: include these in top-level reports dict
        dlov = self.reporter.reports_.get('data_layers_overview')
        if isinstance(dlov, pd.DataFrame) and not dlov.empty:
            reports['data_layers_overview'] = dlov
        rps = self.reporter.reports_.get('raw_preprocessing_summary')
        if isinstance(rps, pd.DataFrame) and not rps.empty:
            reports['raw_preprocessing_summary'] = rps
        if 'operations_notes' not in self.reporter.reports_:
            ops_rows = [
                {'item': 'psi_threshold', 'value': getattr(self.config, 'psi_threshold', None)},
                {'item': 'iv_threshold', 'value': getattr(self.config, 'iv_threshold', None)},
                {'item': 'risk_band_method', 'value': getattr(self.config, 'risk_band_method', None)},
                {'item': 'risk_band_hhi_threshold', 'value': getattr(self.config, 'risk_band_hhi_threshold', None)},
            ]
            self.reporter.reports_['operations_notes'] = pd.DataFrame(ops_rows)
        if 'git_notes' not in self.reporter.reports_:
            git_rows = [
                {'item': 'branch', 'value': 'development'},
                {'item': 'latest_run', 'value': self.config.__dict__.get('run_id')},
            ]
            self.reporter.reports_['git_notes'] = pd.DataFrame(git_rows)

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

    def _build_model_registry(self, mode_label: str, model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Summarise trained models for reporting tables."""
        if not isinstance(model_results, dict):
            return []
        scores = model_results.get('scores', {}) or {}
        selected_features = model_results.get('selected_features') or []
        if not isinstance(selected_features, list):
            try:
                selected_features = list(selected_features)
            except TypeError:
                selected_features = []
        selected_features = selected_features.copy() if isinstance(selected_features, list) else []

        records: List[Dict[str, Any]] = []
        for model_name, metrics in scores.items():
            if not isinstance(metrics, dict):
                continue
            train_auc = metrics.get('train_auc')
            test_auc = metrics.get('test_auc')
            oot_auc_raw = metrics.get('oot_auc')
            if oot_auc_raw is not None:
                oot_auc = oot_auc_raw
                auc_source = 'oot'
            elif test_auc is not None:
                oot_auc = test_auc
                auc_source = 'test'
            elif train_auc is not None:
                oot_auc = train_auc
                auc_source = 'train'
            else:
                oot_auc = None
                auc_source = 'none'

            train_gini = metrics.get('train_gini')
            test_gini = metrics.get('test_gini')
            oot_gini_raw = metrics.get('oot_gini')
            if oot_gini_raw is not None:
                oot_gini = oot_gini_raw
                gini_source = 'oot'
            elif test_gini is not None:
                oot_gini = test_gini
                gini_source = 'test'
            elif train_gini is not None:
                oot_gini = train_gini
                gini_source = 'train'
            else:
                oot_gini = None
                gini_source = 'none'

            records.append({
                'mode': mode_label,
                'model_name': model_name,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'oot_auc': oot_auc,
                'oot_auc_source': auc_source,
                'oot_auc_raw': oot_auc_raw,
                'train_gini': train_gini,
                'test_gini': test_gini,
                'oot_gini': oot_gini,
                'oot_gini_source': gini_source,
                'oot_gini_raw': oot_gini_raw,
                'train_oot_gap': metrics.get('train_oot_gap'),
                'n_features': len(selected_features),
                'selected_features': selected_features.copy(),
            })
        return records

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



    def run_process(self, df: pd.DataFrame, *, create_map: bool = True, include_noise: bool = False, force: bool = False) -> pd.DataFrame:
        """Execute the processing step and cache the result."""

        if not force and 'processed_data' in self.results_:
            return self.results_['processed_data']
        processed = self._process_data(df, create_map=create_map, include_noise=include_noise)
        self.results_['processed_data'] = processed
        return processed

    def run_split(self, df: Optional[pd.DataFrame] = None, *, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Execute splitting and cache the resulting partitions."""

        if not force and 'splits' in self.results_:
            return self.results_['splits']
        if df is None:
            df = self.results_.get('processed_data')
            if df is None:
                raise ValueError('No processed data available. Call run_process first.')
        splits = self._split_data(df)
        self.results_['splits'] = splits
        return splits

    def run_woe(self, splits: Optional[Dict[str, pd.DataFrame]] = None, *, force: bool = False) -> Dict[str, Any]:
        """Execute WOE transformation and cache the derived values."""

        if not force and 'woe_results' in self.results_:
            return self.results_['woe_results']
        splits = splits or self.results_.get('splits')
        if splits is None:
            raise ValueError('No data splits available. Call run_split first.')
        woe = self._apply_woe_transformation(splits)
        self.results_['woe_results'] = woe
        return woe

    def run_selection(self, mode: str = 'WOE', *, splits: Optional[Dict[str, pd.DataFrame]] = None,
                      woe_results: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, Any]:
        """Run feature selection for the requested mode and cache the output."""

        mode_label = str(mode or 'RAW').upper()
        cache_key = f'selection_results_{mode_label}'
        if not force and cache_key in self.results_:
            return self.results_[cache_key]
        splits = splits or self.results_.get('splits')
        if splits is None:
            raise ValueError('No data splits available. Call run_split first.')
        woe_results = woe_results or self.results_.get('woe_results', {})
        original_flag = getattr(self.config, 'enable_woe', True)
        self.config.enable_woe = (mode_label == 'WOE')
        try:
            selection = self._select_features(splits, woe_results)
        finally:
            self.config.enable_woe = original_flag
        self.results_[cache_key] = selection
        self.results_['selection_results'] = selection
        return selection

    def run_modeling(self, mode: str = 'WOE', *, splits: Optional[Dict[str, pd.DataFrame]] = None,
                     selection_results: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, Any]:
        """Train models for the requested mode and cache the results."""

        mode_label = str(mode or 'RAW').upper()
        cache_key = f'model_results_{mode_label}'
        if not force and cache_key in self.results_:
            cached = self.results_[cache_key]
            if isinstance(cached, dict):
                self.selected_features_ = cached.get('selected_features', self.selected_features_)
            return cached
        splits = splits or self.results_.get('splits')
        if splits is None:
            raise ValueError('No data splits available. Call run_split first.')
        selection_results = selection_results or self.results_.get(f'selection_results_{mode_label}')
        if selection_results is None:
            selection_results = self.run_selection(mode=mode_label, splits=splits, woe_results=self.results_.get('woe_results', {}), force=force)
        original_flag = getattr(self.config, 'enable_woe', True)
        self.config.enable_woe = (mode_label == 'WOE')
        try:
            model_results = self._train_models(splits, selection_results, mode_label=mode_label)
        finally:
            self.config.enable_woe = original_flag
        model_results['mode'] = mode_label
        self.results_[cache_key] = model_results
        self.results_['model_results'] = model_results
        return model_results

    def run_stage1_calibration(self, model_results: Optional[Dict[str, Any]] = None,
                               calibration_df: Optional[pd.DataFrame] = None, *, force: bool = False) -> Dict[str, Any]:
        """Run stage-1 calibration and cache the output."""

        if not force and 'calibration_stage1' in self.results_:
            return self.results_['calibration_stage1']
        model_results = model_results or self.results_.get('model_results')
        if model_results is None:
            raise ValueError('Model results required for calibration. Call run_modeling first.')
        if calibration_df is None:
            calibration_df = self.data_.get('calibration_longrun')
        if calibration_df is None:
            calibration_df = self.results_.get('processed_data')
        if calibration_df is None:
            raise ValueError('Calibration dataframe unavailable. Provide calibration_df or run run_process first.')
        stage1 = self._apply_stage1_calibration(model_results, calibration_df)
        self.results_['calibration_stage1'] = stage1
        return stage1

    def run_stage2_calibration(self, stage1_results: Optional[Dict[str, Any]] = None,
                               recent_df: Optional[pd.DataFrame] = None, *, force: bool = False) -> Dict[str, Any]:
        """Run stage-2 calibration on recent data and cache the output."""

        if not force and 'calibration_stage2' in self.results_:
            return self.results_['calibration_stage2']
        stage1_results = stage1_results or self.results_.get('calibration_stage1')
        if stage1_results is None:
            raise ValueError('Stage-1 calibration results missing. Call run_stage1_calibration first.')
        recent_df = recent_df if recent_df is not None else self.data_.get('stage2_source')
        if recent_df is None:
            raise ValueError('Recent dataframe required for stage-2 calibration.')
        stage2 = self._apply_stage2_calibration(stage1_results, recent_df)
        self.results_['calibration_stage2'] = stage2
        return stage2

    def run_risk_bands(self, stage2_results: Optional[Dict[str, Any]] = None,
                       splits: Optional[Dict[str, pd.DataFrame]] = None, *,
                       stage1_results: Optional[Dict[str, Any]] = None,
                       data_override: Optional[pd.DataFrame] = None,
                       force: bool = False) -> Dict[str, Any]:
        """Optimise risk bands using cached artifacts."""

        if not force and 'risk_bands' in self.results_:
            return self.results_['risk_bands']
        stage2_results = stage2_results or self.results_.get('calibration_stage2') or {}
        stage1_results = stage1_results or self.results_.get('calibration_stage1') or {}
        splits = splits or self.results_.get('splits')
        if splits is None:
            raise ValueError('Data splits missing. Call run_split first.')
        if data_override is None:
            data_override = self.data_.get('risk_band_reference') or self.data_.get('risk_band_reference_source')
        bands = self._optimize_risk_bands(stage2_results, splits, data_override=data_override, stage1_results=stage1_results)
        self.results_['risk_bands'] = bands
        return bands

    def run_scoring(self, score_df: pd.DataFrame, *, stage2_results: Optional[Dict[str, Any]] = None,
                     selection_results: Optional[Dict[str, Any]] = None, woe_results: Optional[Dict[str, Any]] = None,
                     model_results: Optional[Dict[str, Any]] = None, splits: Optional[Dict[str, pd.DataFrame]] = None,
                     force: bool = False) -> Dict[str, Any]:
        """Score new data using existing artifacts and cache the outcome."""

        if not force and self.results_.get('scoring_output'):
            return self.results_['scoring_output']
        stage2_results = stage2_results or self.results_.get('calibration_stage2')
        model_results = model_results or self.results_.get('model_results')
        selection_results = selection_results or self.results_.get('selection_results')
        woe_results = woe_results or self.results_.get('woe_results')
        splits = splits or self.results_.get('splits')
        scoring = self._score_data(score_df, stage2_results, selection_results, woe_results, model_results, splits or {})
        self.results_['scoring_output'] = scoring
        if isinstance(scoring, dict):
            self.results_['noise_sentinel_diagnostics'] = scoring.get('noise_sentinel')
        return scoring

    def run_reporting(self, *, force: bool = False) -> Dict[str, Any]:
        """Generate reports using cached artifacts."""

        if not force and 'reports' in self.results_:
            return self.results_['reports']
        reports = self._generate_reports()
        self.results_['reports'] = reports
        return reports
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





