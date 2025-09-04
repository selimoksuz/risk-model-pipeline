"""Modular Risk Model Pipeline - Main Orchestrator"""

import os
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

from typing import Optional
from dataclasses import dataclass

# Import core modules
from .core import (
    BasePipeline,
    DataProcessor,
    FeatureEngineer,
    ModelTrainer,
    ReportGenerator,
    Timer,
    utils
)

# Import existing config (keep compatibility)
from .pipeline16 import Config

class RiskModelPipeline(BasePipeline):
    """Main pipeline orchestrator using modular components"""
    
    def __init__(self, config: Config):
        """Initialize pipeline with modular components"""
        super().__init__(config)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        self.report_generator = ReportGenerator(config)
        
        # Pipeline state
        self.df_ = None
        self.final_vars_ = []
        self.best_model_name_ = None
        
        # Results storage
        self.models_ = {}
        self.models_summary_ = None
        self.woe_map = {}
        
        # Dual pipeline support
        if config.enable_dual_pipeline:
            self.woe_final_vars_ = []
            self.raw_final_vars_ = []
            self.woe_models_ = {}
            self.raw_models_ = {}
            self.woe_models_summary_ = None
            self.raw_models_summary_ = None
    
    def run(self, df: pd.DataFrame):
        """Main pipeline execution"""
        
        # Step 1: Data loading & preparation
        with Timer("1) Veri yukleme & hazirlik", self._log):
            self.df_ = df
            self._log(f"   - Veri boyutu: {df.shape[0]:,} satir x {df.shape[1]} sutun")
            self._log(f"   - Target orani: {df[self.cfg.target_col].mean():.2%}")
            
            # Set random seed for reproducibility
            if self.cfg.random_state is not None:
                np.random.seed(self.cfg.random_state)
                random.seed(self.cfg.random_state)
                self._log(f"   - Random seed: {self.cfg.random_state}")
        
        # Step 2: Data validation & freezing
        if self.cfg.orchestrator.enable_validate:
            with Timer("2) Giris dogrulama & sabitleme", self._log):
                self._activate("validate")
                self.data_processor.validate_and_freeze(self.df_)
                self.data_processor.downcast_inplace(self.df_)
        
        # Step 3: Variable classification
        if self.cfg.orchestrator.enable_classify:
            with Timer("3) Degisken siniflamasi", self._log):
                self._activate("classify")
                var_catalog = self.data_processor.classify_variables(self.df_)
                self.data_processor.var_catalog_ = var_catalog
                
                num_count = (var_catalog.variable_group == 'numeric').sum()
                cat_count = (var_catalog.variable_group == 'categorical').sum()
                self._log(f"   - numeric={num_count}, categorical={cat_count}")
        
        # Step 4: Missing & rare value policy
        with Timer("4) Eksik & Nadir deger politikasi", self._log):
            self._activate("missing_policy")
            # Policy is embedded in WOE transformation
        
        # Step 5: Time-based splitting
        if self.cfg.orchestrator.enable_split:
            with Timer("5) Zaman bolmesi (Train/Test/OOT)", self._log):
                self._activate("split")
                train_idx, test_idx, oot_idx = self.data_processor.split_time(self.df_)
                
                self._log(f"   - Train={len(train_idx)}, "
                         f"Test={len(test_idx) if test_idx is not None else 0}, "
                         f"OOT={len(oot_idx)}")
        
        # Get train/test/OOT data
        X_tr, y_tr, X_te, y_te, X_oot, y_oot = self.data_processor.get_train_test_oot_data(
            self.df_, train_idx, test_idx, oot_idx
        )
        
        # Step 6: WOE binning (train only)
        if self.cfg.orchestrator.enable_woe:
            with Timer("6) WOE binleme (yalniz Train; adaptif)", self._log):
                self._activate("woe")
                
                train_df = self.df_.iloc[train_idx]
                policy = {"missing_strategy": "as_category"}
                
                self.woe_map = self.feature_engineer.fit_woe_mapping(
                    train_df, 
                    self.data_processor.var_catalog_,
                    policy
                )
                
                self._log(f"   - WOE hazir: {len(self.woe_map)} degisken")
                self._log("   - Not: WOE haritasi SADECE TRAIN'de ogrenildi")
        
        # Step 7: PSI screening
        if self.cfg.orchestrator.enable_psi:
            with Timer("7) PSI (vektorize)", self._log):
                self._activate("psi")
                
                # Calculate PSI between train and OOT
                psi_results = self.feature_engineer.calculate_psi(
                    self.df_.iloc[train_idx],
                    self.df_.iloc[oot_idx],
                    list(self.woe_map.keys())
                )
                
                keep_vars = psi_results[psi_results['status'] == 'KEEP']['variable'].tolist()
                drop_vars = psi_results[psi_results['status'] == 'DROP']['variable'].tolist()
                
                self._log(f"   * PSI özet: KEEP={len(keep_vars)} | DROP={len(drop_vars)} | WARN=0")
                self._log(f"   - PSI sonrasi kalan: {len(keep_vars)}")
                
                self.feature_engineer.psi_summary_ = psi_results
        else:
            keep_vars = list(self.woe_map.keys())
        
        # Identify high IV variables
        high_iv = []
        iv_high_threshold = getattr(self.cfg, 'iv_high_threshold', 0.5)
        for var in keep_vars:
            if var in self.woe_map and self.woe_map[var].iv > iv_high_threshold:
                high_iv.append(var)
        
        if high_iv:
            self._log(f"   - High IV flags: {','.join(high_iv)}")
            self.feature_engineer.high_iv_flags_ = high_iv
        
        # Step 8: WOE transformation
        if self.cfg.orchestrator.enable_transform:
            with Timer("8) WOE transform (Train/Test/OOT)", self._log):
                self._activate("transform")
                
                X_tr_woe = self.feature_engineer.apply_woe_transform(X_tr, keep_vars)
                X_te_woe = self.feature_engineer.apply_woe_transform(X_te, keep_vars) if X_te is not None else None
                X_oot_woe = self.feature_engineer.apply_woe_transform(X_oot, keep_vars)
                
                self._log(f"   - X_train={X_tr_woe.shape}, "
                         f"X_test={X_te_woe.shape if X_te_woe is not None else None}, "
                         f"X_oot={X_oot_woe.shape}")
        
        # Step 9: Correlation & clustering
        if getattr(self.cfg.orchestrator, 'enable_corr', True):
            with Timer("9) Korelasyon & cluster", self._log):
                self._activate("correlation")
                
                cluster_vars = self.feature_engineer.correlation_clustering(
                    X_tr_woe, 
                    threshold=self.cfg.rho_threshold
                )
                
                self._log(f"   - cluster temsilcisi={len(cluster_vars)}")
        else:
            cluster_vars = list(X_tr_woe.columns)
        
        # Step 10: Feature selection
        if getattr(self.cfg.orchestrator, 'enable_selection', True):
            with Timer("10) Feature selection (Forward+1SE)", self._log):
                self._activate("selection")
                
                # Boruta selection
                boruta_vars = cluster_vars
                if getattr(self.cfg, 'use_boruta', True):
                    try:
                        boruta_vars = self.feature_engineer.boruta_selection(
                            X_tr_woe[cluster_vars], 
                            y_tr
                        )
                        self._log(f"   - Boruta: {len(boruta_vars)}/{len(cluster_vars)} kaldi")
                    except Exception:
                        self._log("   - Boruta kullanilamadi")
                
                # Forward selection with 1SE rule
                if self.cfg.forward_1se:
                    selected_vars = self.feature_engineer.forward_selection(
                        X_tr_woe[boruta_vars],
                        y_tr,
                        max_features=self.cfg.max_features,
                        cv_folds=self.cfg.cv_folds
                    )
                    
                    # Ensure minimum features
                    if len(selected_vars) < self.cfg.min_features:
                        selected_vars = boruta_vars[:self.cfg.min_features]
                    
                    self._log(f"   - Forward+1SE secti: {len(selected_vars)}")
                else:
                    selected_vars = boruta_vars[:self.cfg.max_features]
                
                self._log(f"   - baseline degisken={len(selected_vars)}")
                pre_final = selected_vars
        else:
            pre_final = cluster_vars
        
        # Step 11: Final correlation filter
        if self.cfg.orchestrator.enable_final_corr:
            with Timer("11) Nihai korelasyon filtresi", self._log):
                self._activate("final_correlation")
                
                final_vars = self.feature_engineer.correlation_clustering(
                    X_tr_woe[pre_final],
                    threshold=self.cfg.rho_threshold
                )
                
                self._log(f"   - corr sonrasi={len(final_vars)}")
        else:
            final_vars = pre_final
        
        # Step 12: Noise sentinel check
        if self.cfg.orchestrator.enable_noise and self.cfg.use_noise_sentinel:
            with Timer("12) Gurultu (noise) sentineli", self._log):
                self._activate("noise_sentinel")
                
                # Add noise features for checking
                np.random.seed(self.cfg.random_state)
                X_noise = X_tr_woe[final_vars].copy()
                X_noise["__noise_g"] = np.random.normal(size=len(X_noise))
                X_noise["__noise_p"] = np.random.permutation(X_tr_woe[final_vars[0]].values) if final_vars else np.random.normal(size=len(X_noise))
                
                # Re-run feature selection with noise
                noise_vars = self.feature_engineer.forward_selection(
                    X_noise,
                    y_tr,
                    max_features=len(final_vars) + 2,
                    cv_folds=self.cfg.cv_folds
                )
                
                # Remove noise variables
                final_vars = [v for v in noise_vars if not v.startswith("__noise")]
                
                self._log(f"   - final degisken={len(final_vars)}")
        
        self.final_vars_ = final_vars
        
        # Step 13: Model training & evaluation
        if self.cfg.orchestrator.enable_model and final_vars:
            with Timer("13) Modelleme & degerlendirme (WOE)", self._log):
                self._activate("modeling")
                
                self.models_summary_ = self.model_trainer.train_and_evaluate_models(
                    X_tr_woe, y_tr,
                    X_te_woe, y_te,
                    X_oot_woe, y_oot,
                    final_vars,
                    prefix="WOE_" if self.cfg.enable_dual_pipeline else ""
                )
                
                self.models_ = self.model_trainer.models_.copy()
        
        # Dual pipeline: RAW variables
        if self.cfg.enable_dual_pipeline:
            self._log("\n" + "="*80)
            self._log("DUAL PIPELINE: RAW VARIABLES (Ham Degiskenler)")
            self._log("="*80)
            
            # Step 8b: Raw transformation
            with Timer("8b) Raw transform (Train/Test/OOT)", self._log):
                # Get numeric variables only for raw pipeline
                numeric_vars = self.data_processor.var_catalog_[
                    self.data_processor.var_catalog_['variable_group'] == 'numeric'
                ]['variable'].tolist()
                
                # Remove target and special columns
                numeric_vars = [v for v in numeric_vars 
                               if v not in [self.cfg.target_col, self.cfg.id_col, self.cfg.time_col]]
                
                # Apply raw transformations
                X_tr_raw, imputer, scaler = self.data_processor.apply_raw_transformations(
                    X_tr[numeric_vars], fit=True
                )
                
                X_te_raw = None
                if X_te is not None:
                    X_te_raw, _, _ = self.data_processor.apply_raw_transformations(
                        X_te[numeric_vars], fit=False, imputer=imputer, scaler=scaler
                    )
                
                X_oot_raw, _, _ = self.data_processor.apply_raw_transformations(
                    X_oot[numeric_vars], fit=False, imputer=imputer, scaler=scaler
                )
                
                self._log(f"   - X_train_raw={X_tr_raw.shape}, "
                         f"X_test_raw={X_te_raw.shape if X_te_raw is not None else None}, "
                         f"X_oot_raw={X_oot_raw.shape}")
            
            # Step 10b: Feature selection for raw
            with Timer("10b) Feature selection RAW (Forward+1SE)", self._log):
                # Boruta for raw
                raw_vars = list(X_tr_raw.columns)
                
                if self.cfg.use_boruta:
                    try:
                        raw_vars = self.feature_engineer.boruta_selection(
                            X_tr_raw, 
                            y_tr
                        )
                        self._log(f"   - Boruta: {len(raw_vars)}/{len(X_tr_raw.columns)} kaldi")
                    except Exception:
                        self._log("   - Boruta kullanilamadi")
                
                # Forward selection for raw
                if self.cfg.forward_1se:
                    raw_selected = self.feature_engineer.forward_selection(
                        X_tr_raw[raw_vars],
                        y_tr,
                        max_features=self.cfg.max_features,
                        cv_folds=self.cfg.cv_folds
                    )
                    
                    # Ensure minimum features
                    if len(raw_selected) < self.cfg.min_features:
                        raw_selected = raw_vars[:self.cfg.min_features]
                    
                    self._log(f"   - Forward+1SE secti: {len(raw_selected)}")
                else:
                    raw_selected = raw_vars[:self.cfg.max_features]
                
                self._log(f"   - raw baseline degisken={len(raw_selected)}")
            
            # Step 11b: Final correlation filter for raw
            with Timer("11b) Nihai korelasyon filtresi RAW", self._log):
                raw_final = self.feature_engineer.correlation_clustering(
                    X_tr_raw[raw_selected],
                    threshold=self.cfg.rho_threshold
                )
                
                self._log(f"   - raw corr sonrasi={len(raw_final)}")
            
            # Step 12b: Noise sentinel for raw
            if self.cfg.use_noise_sentinel:
                with Timer("12b) Gurultu sentineli RAW", self._log):
                    # Add noise features
                    np.random.seed(self.cfg.random_state)
                    X_noise = X_tr_raw[raw_final].copy()
                    X_noise["__noise_g"] = np.random.normal(size=len(X_noise))
                    X_noise["__noise_p"] = np.random.permutation(X_tr_raw[raw_final[0]].values) if raw_final else np.random.normal(size=len(X_noise))
                    
                    # Re-run selection
                    noise_vars = self.feature_engineer.forward_selection(
                        X_noise,
                        y_tr,
                        max_features=len(raw_final) + 2,
                        cv_folds=self.cfg.cv_folds
                    )
                    
                    # Remove noise
                    raw_final = [v for v in noise_vars if not v.startswith("__noise")]
                    
                    self._log(f"   - raw final degisken={len(raw_final)}")
            
            self.raw_final_vars_ = raw_final
            
            # Step 13b: Model training for raw
            if raw_final:
                with Timer("13b) Modelleme & degerlendirme (RAW)", self._log):
                    self.raw_models_summary_ = self.model_trainer.train_and_evaluate_models(
                        X_tr_raw, y_tr,
                        X_te_raw, y_te,
                        X_oot_raw, y_oot,
                        raw_final,
                        prefix="RAW_"
                    )
                    
                    self.raw_models_ = self.model_trainer.models_.copy()
            
            # Combine dual pipeline results
            self._combine_dual_pipeline_results()
        
        # Step 14: Best model selection
        if self.cfg.orchestrator.enable_best_select:
            with Timer("14) En iyi model secimi", self._log):
                self._activate("best_select")
                
                self.best_model_name_ = self.model_trainer.select_best_model(
                    self.models_summary_
                )
                
                self._log(f"   - best={self.best_model_name_}")
        
        # Step 15: Report generation
        if self.cfg.orchestrator.enable_report:
            with Timer("15) Rapor tablolari", self._log):
                self._activate("report")
                
                # Get best model
                best_model = self.models_.get(self.best_model_name_) if self.best_model_name_ else None
                
                # Generate reports
                sheets = self.report_generator.generate_full_report(
                    self.models_summary_,
                    self.best_model_name_,
                    best_model,
                    self.final_vars_,
                    self.woe_map,
                    self.feature_engineer.psi_summary_,
                    X_tr_woe if 'X_tr_woe' in locals() else None,
                    y_tr
                )
                
                self.report_sheets_ = sheets
            
            # Step 15b: Export
            with Timer("15b) Export (Excel/Parquet)", self._log):
                # Export to Excel
                if self.cfg.output_excel_path:
                    excel_path = os.path.join(
                        self.cfg.output_folder,
                        self.cfg.output_excel_path
                    )
                else:
                    excel_path = os.path.join(
                        self.cfg.output_folder,
                        f"model_report_{self.cfg.run_id}.xlsx"
                    )
                
                self.report_generator.export_to_excel(excel_path, sheets)
                
                # Export artifacts
                self.report_generator.export_artifacts(
                    self.cfg.output_folder,
                    self.cfg.run_id,
                    self.woe_map,
                    self.final_vars_,
                    best_model
                )
        
        self._log(f"[{utils.now_str()}] >> RUN tamam - run_id={self.cfg.run_id}{utils.sys_metrics()}")
        
        # Close log file
        self.close()
        
        return self
    
    def _combine_dual_pipeline_results(self):
        """Combine results from dual pipeline"""
        if not self.cfg.enable_dual_pipeline:
            return
        
        # Store WOE results
        self.woe_final_vars_ = self.final_vars_
        self.woe_models_ = self.models_.copy()
        self.woe_models_summary_ = self.models_summary_.copy() if self.models_summary_ is not None else None
        
        # Combine model summaries
        if self.woe_models_summary_ is not None and self.raw_models_summary_ is not None:
            # Add pipeline column
            woe_summary = self.woe_models_summary_.copy()
            woe_summary['pipeline'] = 'WOE'
            
            raw_summary = self.raw_models_summary_.copy()
            raw_summary['pipeline'] = 'RAW'
            
            # Combine
            self.models_summary_ = pd.concat([woe_summary, raw_summary], ignore_index=True)
            
            # Combine models dict
            self.models_ = {}
            self.models_.update(self.woe_models_)
            self.models_.update(self.raw_models_)
            
            # Log summary
            self._log("\n" + "="*80)
            self._log("DUAL PIPELINE SUMMARY")
            self._log("="*80)
            self._log(f"WOE Pipeline: {len(self.woe_final_vars_)} variables, {len(self.woe_models_)} models")
            self._log(f"RAW Pipeline: {len(self.raw_final_vars_)} variables, {len(self.raw_models_)} models")
            
            # Best models from each pipeline
            if self.woe_models_summary_ is not None and not self.woe_models_summary_.empty:
                best_woe = self.woe_models_summary_.nlargest(1, 'Gini_OOT').iloc[0]
                self._log(f"Best WOE Model: {best_woe['model_name']} - Gini OOT: {best_woe['Gini_OOT']:.4f}")
            
            if self.raw_models_summary_ is not None and not self.raw_models_summary_.empty:
                best_raw = self.raw_models_summary_.nlargest(1, 'Gini_OOT').iloc[0]
                self._log(f"Best RAW Model: {best_raw['model_name']} - Gini OOT: {best_raw['Gini_OOT']:.4f}")
    
    def export_reports(self):
        """Export reports (compatibility method)"""
        if hasattr(self, 'report_sheets_'):
            excel_path = os.path.join(
                self.cfg.output_folder,
                self.cfg.output_excel_path or f"model_report_{self.cfg.run_id}.xlsx"
            )
            self.report_generator.export_to_excel(excel_path, self.report_sheets_)

# Keep for compatibility
import os