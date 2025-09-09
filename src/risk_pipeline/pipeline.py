"""Fixed Modular Risk Model Pipeline - Main Orchestrator"""
"""Main pipeline orchestrator using modular components"""
"""Initialize pipeline with modular components"""

        """Main pipeline execution"""
        """Combine results from dual pipeline"""
        """Export reports (compatibility method)"""
    """
    Dual Pipeline that automatically runs both WOE and RAW pipelines
    and selects the best performing model.
    """
        """Initialize dual pipeline with config ensuring dual mode is enabled"""
        """
        Fit the dual pipeline on training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series or np.array
            Training target
        X_valid : pd.DataFrame, optional
            Validation features
        y_valid : pd.Series or np.array, optional
            Validation target
        X_oot : pd.DataFrame, optional
            Out-of-time features
        y_oot : pd.Series or np.array, optional
            Out-of-time target

        Returns:
        --------
        self : DualPipeline
            Fitted pipeline
        """
        """Predict using the best model from dual pipeline"""
        """Predict probabilities using the best model from dual pipeline"""
        """Predict using best WOE model"""
        """Predict using best RAW model"""
        """Apply WOE transformation to features"""
        """Apply RAW transformation to features"""
        """Get summary of dual pipeline results"""

                from risk_pipeline.reporting.shap_utils import compute_shap_values, summarize_shap
from .core import (
from .core.config import Config
import os
import pandas as pd
import random
import warnings



warnings.filterwarnings("ignore")


# Import core modules
    BasePipeline,
    DataProcessor,
    FeatureEngineer,
    ModelTrainer,
    ReportGenerator,
    Timer,
    utils
)

# Import existing config (keep compatibility)


class RiskModelPipeline(BasePipeline):

    def __init__(self, config: Config):
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
        if getattr(config, 'enable_dual_pipeline', False):
            self.woe_final_vars_ = []
            self.raw_final_vars_ = []
            self.woe_models_ = {}
            self.raw_models_ = {}
            self.woe_models_summary_ = None
            self.raw_models_summary_ = None

    def run(self, df: pd.DataFrame):

        # Get config attributes with defaults
        target_col = getattr(self.cfg, 'target_col', 'target')
        id_col = getattr(self.cfg, 'id_col', 'app_id')
        time_col = getattr(self.cfg, 'time_col', 'app_dt')
        random_state = getattr(self.cfg, 'random_state', 42)

        # Step 1: Data loading & preparation
        with Timer("1) Data loading & preparation", self._log):
            self.df_ = df
            self._log(f"   - Data size: {df.shape[0]:, } rows x {df.shape[1]} columns")
            self._log(f"   - Target ratio: {df[target_col].mean():.2%}")

            # Set random seed for reproducibility
            if random_state is not None:
                np.random.seed(random_state)
                random.seed(random_state)
                self._log(f"   - Random seed: {random_state}")

        # Step 2: Data validation & freezing
        if getattr(self.cfg.orchestrator, 'enable_validate', True):
            with Timer("2) Input validation & freezing", self._log):
                self._activate("validate")
                self.data_processor.validate_and_freeze(self.df_)
                self.data_processor.downcast_inplace(self.df_)

        # Step 3: Variable classification
        if getattr(self.cfg.orchestrator, 'enable_classify', True):
            with Timer("3) Variable classification", self._log):
                self._activate("classify")
                var_catalog = self.data_processor.classify_variables(self.df_)
                self.data_processor.var_catalog_ = var_catalog

                num_count = (var_catalog.variable_group == 'numeric').sum()
                cat_count = (var_catalog.variable_group == 'categorical').sum()
                self._log(f"   - numeric={num_count}, categorical={cat_count}")

        # Step 4: Missing & rare value policy
        with Timer("4) Missing & Rare value policy", self._log):
            self._activate("missing_policy")
            # Policy is embedded in WOE transformation

        # Step 5: Time-based splitting
        if getattr(self.cfg.orchestrator, 'enable_split', True):
            with Timer("5) Time splitting (Train/Test/OOT)", self._log):
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
        if getattr(self.cfg.orchestrator, 'enable_woe', True):
            with Timer("6) WOE binning(Train only
    adaptive)", self._log):
                self._activate("woe")

                train_df=self.df_.iloc[train_idx]
                policy={"missing_strategy": "as_category"}

                self.woe_map=self.feature_engineer.fit_woe_mapping(
                    train_df,
                    self.data_processor.var_catalog_,
                    policy
                )

                self._log(f"   - WOE ready: {len(self.woe_map)} variables")
                self._log("   - Note: WOE mapping learned ONLY on TRAIN")

        # Step 7: PSI screening
        if getattr(self.cfg.orchestrator, 'enable_psi', True):
            with Timer("7) PSI (vectorized)", self._log):
                self._activate("psi")

                # Calculate PSI between train and OOT
                psi_results = self.feature_engineer.calculate_psi(
                    self.df_.iloc[train_idx],
                    self.df_.iloc[oot_idx],
                    list(self.woe_map.keys())
                )

                keep_vars = psi_results[psi_results['status'] == 'KEEP']['variable'].tolist()
                drop_vars = psi_results[psi_results['status'] == 'DROP']['variable'].tolist()

                self._log(f"   * PSI summary: KEEP={len(keep_vars)} | DROP={len(drop_vars)} | WARN = 0")
                self._log(f"   - Remaining after PSI: {len(keep_vars)}")

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
            self._log(f"   - High IV flags: {', '.join(high_iv)}")
            self.feature_engineer.high_iv_flags_ = high_iv

        # Step 8: WOE transformation
        if getattr(self.cfg.orchestrator, 'enable_transform', True):
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
            with Timer("9) Correlation & clustering", self._log):
                self._activate("correlation")

                rho_threshold = getattr(self.cfg, 'rho_threshold', 0.95)
                cluster_vars = self.feature_engineer.correlation_clustering(
                    X_tr_woe,
                    threshold = rho_threshold
                )

                self._log(f"   - cluster representatives={len(cluster_vars)}")
        else:
            cluster_vars = list(X_tr_woe.columns)

        # Step 10: Feature selection
        if getattr(self.cfg.orchestrator, 'enable_selection', True):
            with Timer("10) Feature selection (Forward + 1SE)", self._log):
                self._activate("selection")

                # Boruta selection
                boruta_vars = cluster_vars
                use_boruta = getattr(self.cfg, 'use_boruta', True)
                if use_boruta:
                    try:
                        boruta_vars = self.feature_engineer.boruta_selection(
                          X_tr_woe[cluster_vars],
                          y_tr
                        )
                        self._log(f"   - Boruta: {len(boruta_vars)}/{len(cluster_vars)} remained")
                    except Exception:
                        self._log("   - Boruta could not be used")

                # Forward selection with 1SE rule
                forward_1se = getattr(self.cfg, 'forward_1se', True)
                if forward_1se:
                    max_features = getattr(self.cfg, 'max_features', 20)
                    cv_folds = getattr(self.cfg, 'cv_folds', 5)

                    selected_vars = self.feature_engineer.forward_selection(
                        X_tr_woe[boruta_vars],
                        y_tr,
                        max_features = max_features,
                        cv_folds = cv_folds
                    )

                    # Ensure minimum features
                    min_features = getattr(self.cfg, 'min_features', 3)
                    if len(selected_vars) < min_features:
                        selected_vars = boruta_vars[:min_features]

                    self._log(f"   - Forward + 1SE selected: {len(selected_vars)}")
                else:
                    max_features = getattr(self.cfg, 'max_features', 20)
                    selected_vars = boruta_vars[:max_features]

                self._log(f"   - baseline variables={len(selected_vars)}")
                pre_final = selected_vars
        else:
            pre_final = cluster_vars

        # Step 11: Final correlation filter
        if getattr(self.cfg.orchestrator, 'enable_final_corr', True):
            with Timer("11) Final correlation filter", self._log):
                self._activate("final_correlation")

                rho_threshold = getattr(self.cfg, 'rho_threshold', 0.95)
                final_vars = self.feature_engineer.correlation_clustering(
                    X_tr_woe[pre_final],
                    threshold = rho_threshold
                )

                self._log(f"   - after correlation={len(final_vars)}")
        else:
            final_vars = pre_final

        # Step 12: Noise sentinel check
        enable_noise = getattr(self.cfg.orchestrator, 'enable_noise', True)
        use_noise_sentinel = getattr(self.cfg, 'use_noise_sentinel', True)

        if enable_noise and use_noise_sentinel:
            with Timer("12) Noise sentinel", self._log):
                self._activate("noise_sentinel")

                # Add noise features for checking
                np.random.seed(random_state)
                X_noise = X_tr_woe[final_vars].copy()
                X_noise["__noise_g"] = np.random.normal(size = len(X_noise))
                X_noise["__noise_p"] = np.random.permutation(X_tr_woe[final_vars[0]].values) if final_vars else np.random.normal(size = len(X_noise))

                # Re-run feature selection with noise
                cv_folds = getattr(self.cfg, 'cv_folds', 5)
                noise_vars = self.feature_engineer.forward_selection(
                    X_noise,
                    y_tr,
                    max_features = len(final_vars) + 2,
                    cv_folds = cv_folds
                )

                # Remove noise variables
                final_vars = [v for v in noise_vars if not v.startswith("__noise")]

                self._log(f"   - final variables={len(final_vars)}")

        self.final_vars_ = final_vars

        # Step 13: Model training & evaluation
        enable_model = getattr(self.cfg.orchestrator, 'enable_model', True)
        if enable_model and final_vars:
            with Timer("13) Modeling & evaluation (WOE)", self._log):
                self._activate("modeling")

                enable_dual = getattr(self.cfg, 'enable_dual_pipeline', False)
                self.models_summary_ = self.model_trainer.train_and_evaluate_models(
                    X_tr_woe, y_tr,
                    X_te_woe, y_te,
                    X_oot_woe, y_oot,
                    final_vars,
                    prefix="WOE_" if enable_dual else ""
                )

                self.models_ = self.model_trainer.models_.copy()

        # Dual pipeline: RAW variables
        enable_dual = getattr(self.cfg, 'enable_dual_pipeline', False)
        if enable_dual:
            self._log("\n" + "="*80)
            self._log("DUAL PIPELINE: RAW VARIABLES")
            self._log("="*80)

            # Step 8b: Raw transformation
            with Timer("8b) Raw transform (Train/Test/OOT)", self._log):
                # Get numeric variables only for raw pipeline
                numeric_vars = self.data_processor.var_catalog_[
                    self.data_processor.var_catalog_['variable_group'] == 'numeric'
                ]['variable'].tolist()

                # Remove target and special columns
                numeric_vars = [v for v in numeric_vars
                             if v not in [target_col, id_col, time_col]]

                # Apply raw transformations
                X_tr_raw, imputer, scaler = self.data_processor.apply_raw_transformations(
                    X_tr[numeric_vars], fit = True
                )

                # Save imputer and scaler for later use
                self.data_processor.imputer_ = imputer
                self.data_processor.scaler_ = scaler

                X_te_raw = None
                if X_te is not None:
                    X_te_raw, _, _ = self.data_processor.apply_raw_transformations(
                        X_te[numeric_vars], fit = False, imputer = imputer, scaler = scaler
                    )

                X_oot_raw, _, _ = self.data_processor.apply_raw_transformations(
                    X_oot[numeric_vars], fit = False, imputer = imputer, scaler = scaler
                )

                self._log(f"   - X_train_raw={X_tr_raw.shape}, "
                         f"X_test_raw={X_te_raw.shape if X_te_raw is not None else None}, "
                         f"X_oot_raw={X_oot_raw.shape}")

            # Step 10b: Feature selection for raw
            with Timer("10b) Feature selection RAW (Forward + 1SE)", self._log):
                # Boruta for raw
                raw_vars = list(X_tr_raw.columns)

                if use_boruta:
                    try:
                        raw_vars = self.feature_engineer.boruta_selection(
                          X_tr_raw,
                          y_tr
                        )
                        self._log(f"   - Boruta: {len(raw_vars)}/{len(X_tr_raw.columns)} remained")
                    except Exception:
                        self._log("   - Boruta could not be used")

                # Forward selection for raw
                if forward_1se:
                    max_features = getattr(self.cfg, 'max_features', 20)
                    cv_folds = getattr(self.cfg, 'cv_folds', 5)

                    raw_selected = self.feature_engineer.forward_selection(
                        X_tr_raw[raw_vars],
                        y_tr,
                        max_features = max_features,
                        cv_folds = cv_folds
                    )

                    # Ensure minimum features
                    min_features = getattr(self.cfg, 'min_features', 3)
                    if len(raw_selected) < min_features:
                        raw_selected = raw_vars[:min_features]

                    self._log(f"   - Forward + 1SE selected: {len(raw_selected)}")
                else:
                    max_features = getattr(self.cfg, 'max_features', 20)
                    raw_selected = raw_vars[:max_features]

                self._log(f"   - raw baseline variables={len(raw_selected)}")

            # Step 11b: Final correlation filter for raw
            with Timer("11b) Final correlation filter RAW", self._log):
                rho_threshold = getattr(self.cfg, 'rho_threshold', 0.95)
                raw_final = self.feature_engineer.correlation_clustering(
                    X_tr_raw[raw_selected],
                    threshold = rho_threshold
                )

                self._log(f"   - raw after correlation={len(raw_final)}")

            # Step 12b: Noise sentinel for raw
            if use_noise_sentinel:
                with Timer("12b) Noise sentinel RAW", self._log):
                    # Add noise features
                    np.random.seed(random_state)
                    X_noise = X_tr_raw[raw_final].copy()
                    X_noise["__noise_g"] = np.random.normal(size = len(X_noise))
                    X_noise["__noise_p"] = np.random.permutation(X_tr_raw[raw_final[0]].values) if raw_final else np.random.normal(size = len(X_noise))

                    # Re-run selection
                    cv_folds = getattr(self.cfg, 'cv_folds', 5)
                    noise_vars = self.feature_engineer.forward_selection(
                        X_noise,
                        y_tr,
                        max_features = len(raw_final) + 2,
                        cv_folds = cv_folds
                    )

                    # Remove noise
                    raw_final = [v for v in noise_vars if not v.startswith("__noise")]

                    self._log(f"   - raw final variables={len(raw_final)}")

            self.raw_final_vars_ = raw_final

            # Step 13b: Model training for raw
            if raw_final:
                with Timer("13b) Modeling & evaluation (RAW)", self._log):
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
        enable_best_select = getattr(self.cfg.orchestrator, 'enable_best_select', True)
        if enable_best_select:
            with Timer("14) Best model selection", self._log):
                self._activate("best_select")

                self.best_model_name_ = self.model_trainer.select_best_model(
                    self.models_summary_
                )

                # Update final_vars based on which pipeline was selected
                if self.best_model_name_ and self.best_model_name_.startswith("RAW_"):
                    # If RAW model was selected, use raw_final_vars
                    if hasattr(self, 'raw_final_vars_'):
                        self.final_vars_ = self.raw_final_vars_
                elif self.best_model_name_ and self.best_model_name_.startswith("WOE_"):
                    # If WOE model was selected, use woe_final_vars
                    if hasattr(self, 'woe_final_vars_'):
                        self.final_vars_ = self.woe_final_vars_

                self._log(f"   - best={self.best_model_name_}")

        # Step 15: Report generation
        # Step 14b: SHAP analysis for best model
        shap_values = None
        shap_summary = None
        if self.best_model_name_ and self.final_vars_:
            try:
                with Timer("14b) SHAP analysis", self._log):
                    best_model = self.models_.get(self.best_model_name_)

                    # Get appropriate data for SHAP
                    if self.best_model_name_.startswith("WOE_"):
                        X_for_shap = X_tr_woe if 'X_tr_woe' in locals() else None
                    else:
                        X_for_shap = X_tr_raw if 'X_tr_raw' in locals() else None

                    if X_for_shap is not None and best_model is not None:
                        # Filter to only columns that exist in X_for_shap
                        shap_cols = [col for col in self.final_vars_ if col in X_for_shap.columns]

                        if shap_cols:
                          shap_values = compute_shap_values(
                              best_model,
                              X_for_shap[shap_cols],
                              shap_sample = min(1000, len(X_for_shap))
                          )
                          if shap_values:
                              shap_summary = summarize_shap(shap_values, shap_cols)
                          self._log(f"   - SHAP values computed for {len(self.final_vars_)} features")
                          self.shap_values_ = shap_values
                          self.shap_summary_ = shap_summary
            except Exception as e:
                self._log(f"   - SHAP analysis skipped: {str(e)}")

        enable_report = getattr(self.cfg.orchestrator, 'enable_report', True)
        if enable_report:
            with Timer("15) Report tables", self._log):
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
                    y_tr,
                    shap_values = shap_values,
                    shap_summary = shap_summary
                )

                self.report_sheets_ = sheets

            # Step 15b: Export
            with Timer("15b) Export (Excel/Parquet)", self._log):
                # Export to Excel
                output_folder = getattr(self.cfg, 'output_folder', 'output')
                output_excel = getattr(self.cfg, 'output_excel_path', None)
                run_id = getattr(self.cfg, 'run_id', utils.now_str())

                if output_excel:
                    excel_path = os.path.join(output_folder, output_excel)
                else:
                    excel_path = os.path.join(output_folder, f"model_report_{run_id}.xlsx")

                self.report_generator.export_to_excel(excel_path, sheets)

                # Export artifacts
                self.report_generator.export_artifacts(
                    output_folder,
                    run_id,
                    self.woe_map,
                    self.final_vars_,
                    best_model
                )

        self._log(f"[{utils.now_str()}] >> RUN complete - run_id={getattr(self.cfg,
            'run_id'
            'N/A')}{utils.sys_metrics()}")

        # Close log file
        self.close()

        return self

    def _combine_dual_pipeline_results(self):
        enable_dual = getattr(self.cfg, 'enable_dual_pipeline', False)
        if not enable_dual:
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
            self.models_summary_ = pd.concat([woe_summary, raw_summary], ignore_index = True)

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
        if hasattr(self, 'report_sheets_'):
            output_folder = getattr(self.cfg, 'output_folder', 'output')
            output_excel = getattr(self.cfg, 'output_excel_path', None)
            run_id = getattr(self.cfg, 'run_id', utils.now_str())

            if output_excel:
                excel_path = os.path.join(output_folder, output_excel)
            else:
                excel_path = os.path.join(output_folder, f"model_report_{run_id}.xlsx")

            self.report_generator.export_to_excel(excel_path, self.report_sheets_)


class DualPipeline(RiskModelPipeline):

    def __init__(self, config = None):
        if config is None:
            config = Config()

        # Ensure dual pipeline is enabled
        config.enable_dual_pipeline = True

        super().__init__(config)

        # Store results from both pipelines
        self.woe_pipeline_results = None
        self.raw_pipeline_results = None
        self.best_pipeline = None  # 'WOE' or 'RAW'

    def fit(self, X_train, y_train, X_valid = None, y_valid = None, X_oot = None, y_oot = None):
        # Prepare dataframe for pipeline

        df_train = X_train.copy()
        df_train[self.cfg.target_col] = y_train

        if X_valid is not None and y_valid is not None:
            df_valid = X_valid.copy()
            df_valid[self.cfg.target_col] = y_valid
            df_train = pd.concat([df_train, df_valid], ignore_index = True)

        if X_oot is not None and y_oot is not None:
            df_oot = X_oot.copy()
            df_oot[self.cfg.target_col] = y_oot

            # Add time column to distinguish OOT
            df_train[self.cfg.time_col] = pd.date_range('2022-01-01', periods = len(df_train), freq='D')
            df_oot[self.cfg.time_col] = pd.date_range('2023-01-01', periods = len(df_oot), freq='D')

            df = pd.concat([df_train, df_oot], ignore_index = True)
        else:
            df = df_train
            # Add dummy time column
            df[self.cfg.time_col] = pd.date_range('2022-01-01', periods = len(df), freq='D')

        # Run the pipeline
        self.run(df)

        # Determine best pipeline
        if self.models_summary_ is not None and not self.models_summary_.empty:
            best_model_row = self.models_summary_[
                self.models_summary_['model_name'] == self.best_model_name_
            ].iloc[0] if self.best_model_name_ else self.models_summary_.nlargest(1, 'Gini_OOT').iloc[0]

            self.best_pipeline = 'WOE' if 'WOE_' in best_model_row['model_name'] else 'RAW'

        return self

    def predict(self, X):
        if self.best_model_name_ and self.best_model_name_ in self.models_:
            model = self.models_[self.best_model_name_]

            # Determine which pipeline to use
            if 'WOE_' in self.best_model_name_:
                # Apply WOE transformation
                X_transformed = self.transform_woe(X)
                features = self.woe_final_vars_
            else:
                # Apply RAW transformation
                X_transformed = self.transform_raw(X)
                features = self.raw_final_vars_

            # Filter to final features
            X_final = X_transformed[features] if features else X_transformed

            # Make predictions
            if hasattr(model, 'predict'):
                return model.predict(X_final)
            else:
                raise AttributeError(f"Model {self.best_model_name_} does not have predict method")
        else:
            raise ValueError("No model has been trained yet. Call fit() first.")

    def predict_proba(self, X):
        if self.best_model_name_ and self.best_model_name_ in self.models_:
            model = self.models_[self.best_model_name_]

            # Determine which pipeline to use
            if 'WOE_' in self.best_model_name_:
                # Apply WOE transformation
                X_transformed = self.transform_woe(X)
                features = self.woe_final_vars_
            else:
                # Apply RAW transformation
                X_transformed = self.transform_raw(X)
                features = self.raw_final_vars_

            # Filter to final features
            X_final = X_transformed[features] if features else X_transformed

            # Make predictions
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_final)
            else:
                # For models without predict_proba, return predictions as probabilities
                preds = self.predict(X)
                return np.column_stack([1 - preds, preds])
        else:
            raise ValueError("No model has been trained yet. Call fit() first.")

    def predict_woe(self, X):
        woe_models = {k: v for k, v in self.models_.items() if 'WOE_' in k}
        if not woe_models:
            raise ValueError("No WOE models available")

        # Get best WOE model
        best_woe_name = max(woe_models.keys(),
                         key = lambda x: self.models_summary_[
                             self.models_summary_['model_name'] == x
                         ]['Gini_OOT'].values[0])

        model = woe_models[best_woe_name]
        X_transformed = self.transform_woe(X)
        features = self.woe_final_vars_
        X_final = X_transformed[features] if features else X_transformed

        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_final)[:, 1]
        else:
            return model.predict(X_final)

    def predict_raw(self, X):
        raw_models = {k: v for k, v in self.models_.items() if 'RAW_' in k}
        if not raw_models:
            raise ValueError("No RAW models available")

        # Get best RAW model
        best_raw_name = max(raw_models.keys(),
                         key = lambda x: self.models_summary_[
                             self.models_summary_['model_name'] == x
                         ]['Gini_OOT'].values[0])

        model = raw_models[best_raw_name]
        X_transformed = self.transform_raw(X)
        features = self.raw_final_vars_
        X_final = X_transformed[features] if features else X_transformed

        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_final)[:, 1]
        else:
            return model.predict(X_final)

    def transform_woe(self, X):
        if not hasattr(self, 'woe_map') or not self.woe_map:
            raise ValueError("WOE mapping not available. Run fit() first.")

        return self.feature_engineer.apply_woe_transform(X, list(self.woe_map.keys()))

    def transform_raw(self, X):
        if not hasattr(self.data_processor, 'imputer_') or not hasattr(self.data_processor, 'scaler_'):
            raise ValueError("RAW transformation not available. Run fit() first.")

        # Get numeric variables
        numeric_vars = self.data_processor.var_catalog_[
            self.data_processor.var_catalog_['variable_group'] == 'numeric'
        ]['variable'].tolist()

        # Remove target and special columns
        numeric_vars = [v for v in numeric_vars
                       if v not in [self.cfg.target_col, self.cfg.id_col, self.cfg.time_col]
                       and v in X.columns]

        X_transformed, _, _ = self.data_processor.apply_raw_transformations(
            X[numeric_vars],
            fit = False,
            imputer = self.data_processor.imputer_,
            scaler = self.data_processor.scaler_
        )

        return X_transformed

    def get_summary(self):
        summary = {
            'best_model': self.best_model_name_,
            'best_pipeline': self.best_pipeline,
            'best_gini': None,
            'n_features_woe': len(self.woe_final_vars_) if hasattr(self, 'woe_final_vars_') else 0,
            'n_features_raw': len(self.raw_final_vars_) if hasattr(self, 'raw_final_vars_') else 0,
            'n_models_total': len(self.models_) if hasattr(self, 'models_') else 0,
            'n_models_woe': len([m for m in self.models_.keys() if 'WOE_' in m]) if hasattr(self, 'models_') else 0,
            'n_models_raw': len([m for m in self.models_.keys() if 'RAW_' in m]) if hasattr(self, 'models_') else 0
        }

        if self.models_summary_ is not None and self.best_model_name_:
            best_row = self.models_summary_[
                self.models_summary_['model_name'] == self.best_model_name_
            ]
            if not best_row.empty:
                summary['best_gini'] = best_row.iloc[0]['Gini_OOT']

        return summary
