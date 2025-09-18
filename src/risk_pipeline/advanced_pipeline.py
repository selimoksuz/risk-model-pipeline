"""Advanced Risk Model Pipeline with Complete Features"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

from .core.config import Config
from .core.data_processor import DataProcessor
from .core.feature_selector import FeatureSelector
from .core.model_builder import ComprehensiveModelBuilder as ModelBuilder
from .core.reporter import EnhancedReporter as Reporter
from .core.splitter import SmartDataSplitter as DataSplitter
from .core.woe_transformer import EnhancedWOETransformer as WOETransformer
from .core.psi_calculator import PSICalculator
from .core.calibration_analyzer import CalibrationAnalyzer
from .core.risk_band_optimizer import OptimalRiskBandAnalyzer as RiskBandOptimizer

warnings.filterwarnings("ignore")


class AdvancedRiskPipeline:
    """Advanced risk model pipeline with all features"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize advanced pipeline"""
        self.config = config or Config()
        
        # Core components
        self.processor = DataProcessor(self.config)
        self.splitter = DataSplitter(self.config)
        self.selector = FeatureSelector(self.config)
        self.woe_transformer = WOETransformer(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.reporter = Reporter(self.config)
        
        # Advanced components
        self.psi_calculator = PSICalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.risk_band_optimizer = RiskBandOptimizer()
        
        # Storage
        self.train_ = None
        self.test_ = None
        self.oot_ = None
        self.calibration_ = None
        
        # Variable types
        self.numeric_vars_ = []
        self.categorical_vars_ = []
        self.all_vars_ = []
        
        # Selected features
        self.final_vars_ = []
        self.final_vars_raw_ = []
        self.final_vars_woe_ = []
        
        # Models
        self.best_model_woe_ = None
        self.best_model_raw_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        
        # Transformations
        self.woe_mapping_ = None
        self.imputer_ = None
        self.scaler_ = None
        
        # Results
        self.model_results_ = {}
        self.psi_results_ = {}
        self.calibration_results_ = {}
        self.risk_bands_ = None
        self.final_report_ = None
        
    def run(self, df: pd.DataFrame, 
            test_df: Optional[pd.DataFrame] = None,
            oot_df: Optional[pd.DataFrame] = None,
            calibration_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run complete advanced pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        test_df : pd.DataFrame, optional
            Test data (if None, will be split from df)
        oot_df : pd.DataFrame, optional
            Out-of-time validation data
        calibration_df : pd.DataFrame, optional
            Calibration data
        """
        
        print("\n" + "="*80)
        print("ADVANCED RISK MODEL PIPELINE")
        print("="*80)
        
        # 1. Data Validation and Processing
        print("\n1. DATA PROCESSING")
        print("-"*40)
        df_processed = self._process_data(df)
        
        # 2. Variable Classification
        print("\n2. VARIABLE CLASSIFICATION")
        print("-"*40)
        self._classify_variables(df_processed)
        
        # 3. Data Splitting
        print("\n3. DATA SPLITTING")
        print("-"*40)
        self._split_data(df_processed, test_df, oot_df, calibration_df)
        
        # 4. Imputation and Outlier Handling
        print("\n4. IMPUTATION & OUTLIER HANDLING")
        print("-"*40)
        self._handle_missing_and_outliers()
        
        # 5. Feature Selection with Multiple Methods
        print("\n5. FEATURE SELECTION")
        print("-"*40)
        self._select_features()
        
        # 6. WOE Transformation for All Variables
        print("\n6. WOE TRANSFORMATION")
        print("-"*40)
        woe_data = self._apply_woe_transformation()
        
        # 7. Build Models (WOE and RAW)
        print("\n7. MODEL BUILDING")
        print("-"*40)
        self._build_all_models(woe_data)
        
        # 8. PSI Analysis (WOE, Quantile, Score)
        print("\n8. PSI ANALYSIS")
        print("-"*40)
        self._calculate_all_psi()
        
        # 9. Calibration Analysis
        print("\n9. CALIBRATION ANALYSIS")
        print("-"*40)
        if self.calibration_ is not None or self.oot_ is not None:
            self._perform_calibration()
        
        # 10. Risk Band Optimization
        print("\n10. RISK BAND OPTIMIZATION")
        print("-"*40)
        self._optimize_risk_bands()
        
        # 11. Generate Comprehensive Reports
        print("\n11. REPORT GENERATION")
        print("-"*40)
        self._generate_comprehensive_reports()
        
        print("\n" + "="*80)
        print("Pipeline finished.")
        print("="*80)
        
        return self._get_final_results()
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate input data"""
        df_processed = self.processor.validate_and_freeze(df)
        print(f"  - Data shape: {df_processed.shape}")
        print(f"  - Target distribution: {df_processed[self.config.target_col].value_counts().to_dict()}")
        return df_processed
    
    def _classify_variables(self, df: pd.DataFrame):
        """Classify variables as numeric or categorical"""
        exclude_cols = [self.config.id_col, self.config.time_col, 
                       self.config.target_col, 'snapshot_month']
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's truly numeric or categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    self.categorical_vars_.append(col)
                else:
                    self.numeric_vars_.append(col)
            else:
                self.categorical_vars_.append(col)
        
        self.all_vars_ = self.numeric_vars_ + self.categorical_vars_
        print(f"  - Numeric variables: {len(self.numeric_vars_)}")
        print(f"  - Categorical variables: {len(self.categorical_vars_)}")
        print(f"  - Total variables: {len(self.all_vars_)}")
    
    def _split_data(self, df: pd.DataFrame, 
                    test_df: Optional[pd.DataFrame] = None,
                    oot_df: Optional[pd.DataFrame] = None,
                    calibration_df: Optional[pd.DataFrame] = None):
        """Split data into train/test/oot/calibration"""
        
        if test_df is None:
            # Use splitter to create train/test split
            splits = self.splitter.split(df)
            self.train_ = splits["train"]
            self.test_ = splits.get("test")
            self.oot_ = splits.get("oot")
        else:
            # Use provided splits
            self.train_ = df
            self.test_ = test_df
            self.oot_ = oot_df
        
        self.calibration_ = calibration_df
        
        print(f"  - Train: {len(self.train_)} samples")
        if self.test_ is not None:
            print(f"  - Test: {len(self.test_)} samples")
        if self.oot_ is not None:
            print(f"  - OOT: {len(self.oot_)} samples")
        if self.calibration_ is not None:
            print(f"  - Calibration: {len(self.calibration_)} samples")
    
    def _handle_missing_and_outliers(self):
        """Handle missing values and outliers"""
        
        # Imputation for numeric variables
        if self.numeric_vars_:
            print("  Imputing numeric variables...")
            imputation_strategy = self.config.imputation_strategy
            self.imputer_ = SimpleImputer(strategy=imputation_strategy)
            
            # Fit on train
            self.imputer_.fit(self.train_[self.numeric_vars_])
            
            # Transform all datasets
            self.train_[self.numeric_vars_] = self.imputer_.transform(
                self.train_[self.numeric_vars_]
            )
            
            if self.test_ is not None:
                self.test_[self.numeric_vars_] = self.imputer_.transform(
                    self.test_[self.numeric_vars_]
                )
            
            if self.oot_ is not None:
                self.oot_[self.numeric_vars_] = self.imputer_.transform(
                    self.oot_[self.numeric_vars_]
                )
            
            print(f"    - Strategy: {imputation_strategy}")
            print(f"    - Variables imputed: {len(self.numeric_vars_)}")
        
        # Outlier handling for RAW pipeline
        if self.config.raw_outlier_method != "none" and self.numeric_vars_:
            print("  Handling outliers...")
            method = self.config.raw_outlier_method
            threshold = self.config.raw_outlier_threshold
            
            if method == "clip":
                # Clip outliers using IQR
                for col in self.numeric_vars_:
                    q1 = self.train_[col].quantile(0.25)
                    q3 = self.train_[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    
                    self.train_[col] = self.train_[col].clip(lower, upper)
                    if self.test_ is not None:
                        self.test_[col] = self.test_[col].clip(lower, upper)
                    if self.oot_ is not None:
                        self.oot_[col] = self.oot_[col].clip(lower, upper)
            
            elif method == "remove":
                # Remove outliers from training only
                mask = np.ones(len(self.train_), dtype=bool)
                for col in self.numeric_vars_:
                    z_scores = np.abs((self.train_[col] - self.train_[col].mean()) / self.train_[col].std())
                    mask &= z_scores < threshold
                self.train_ = self.train_[mask]
            
            print(f"    - Method: {method}")
            print(f"    - Threshold: {threshold}")
    
    def _select_features(self):
        """Comprehensive feature selection"""
        
        print("  Running multi-stage feature selection...")
        
        # Get all features for selection
        feature_cols = self.all_vars_.copy()
        
        # Stage 1: IV filtering
        print("    1. Information Value filtering...")
        from .core.feature_engineer import FeatureEngineer
        engineer = FeatureEngineer(self.config)
        
        iv_scores = {}
        for col in feature_cols:
            try:
                # Handle categorical variables differently
                if col in self.categorical_vars_:
                    # For categorical, calculate IV directly
                    X = self.train_[[col]].copy()
                    X[col] = X[col].astype(str).fillna('missing')
                    y = self.train_[self.config.target_col]
                    
                    # Calculate IV for each category
                    crosstab = pd.crosstab(X[col], y, normalize='columns')
                    if crosstab.shape[1] == 2:
                        good_dist = crosstab[0]
                        bad_dist = crosstab[1]
                        woe = np.log((bad_dist + 0.0001) / (good_dist + 0.0001))
                        iv = ((bad_dist - good_dist) * woe).sum()
                        iv_scores[col] = abs(iv)
                    else:
                        iv_scores[col] = 0
                else:
                    # For numeric, use binning
                    iv = engineer.calculate_iv(
                        self.train_[col].fillna(0), 
                        self.train_[self.config.target_col]
                    )
                    iv_scores[col] = iv
            except Exception as e:
                print(f"        IV calculation failed for {col}: {e}")
                iv_scores[col] = 0
        
        # Filter by IV threshold
        iv_threshold = self.config.iv_threshold
        feature_cols = [col for col, iv in iv_scores.items() if iv >= iv_threshold]
        
        # If no features pass, keep top N by IV
        if len(feature_cols) == 0 and len(iv_scores) > 0:
            sorted_features = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
            feature_cols = [f[0] for f in sorted_features[:min(20, len(sorted_features))]]
            print(f"      No features passed IV threshold, keeping top {len(feature_cols)}")
        
        print(f"      After IV filter: {len(feature_cols)} features")
        
        # Stage 2: PSI filtering (if test data available)
        if self.test_ is not None and self.config.enable_psi:
            print("    2. PSI filtering...")
            psi_threshold = self.config.psi_threshold
            features_to_keep = []
            
            for col in feature_cols:
                try:
                    if col in self.numeric_vars_:
                        # Quantile PSI for numeric
                        psi_value, _ = self.psi_calculator.calculate_psi(
                            self.train_[col].values,
                            self.test_[col].values,
                            n_bins=10
                        )
                    else:
                        # Category PSI for categorical
                        psi_value = self.psi_calculator._calculate_categorical_psi(
                            self.train_[col],
                            self.test_[col]
                        )
                    
                    if psi_value < psi_threshold:
                        features_to_keep.append(col)
                except:
                    features_to_keep.append(col)
            
            feature_cols = features_to_keep
            print(f"      After PSI filter: {len(feature_cols)} features")
        
        # Stage 3: Correlation filtering
        print("    3. Correlation filtering...")
        corr_threshold = self.config.rho_threshold
        corr_matrix = self.train_[feature_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > corr_threshold)]
        feature_cols = [col for col in feature_cols if col not in to_drop]
        print(f"      After correlation filter: {len(feature_cols)} features")
        
        # Stage 4: Boruta selection
        if self.config.use_boruta and len(feature_cols) > 10:
            print("    4. Boruta selection...")
            try:
                X = self.train_[feature_cols].fillna(0)
                y = self.train_[self.config.target_col]
                selected = engineer.boruta_selection(X, y)
                if selected:
                    feature_cols = selected
                print(f"      After Boruta: {len(feature_cols)} features")
            except Exception as e:
                print(f"      Boruta failed: {e}")
        
        # Stage 5: Forward/Stepwise selection
        if self.config.forward_selection and len(feature_cols) > 5:
            print("    5. Forward selection...")
            try:
                X = self.train_[feature_cols].fillna(0)
                y = self.train_[self.config.target_col]
                selected = engineer.forward_selection(
                    X, y, 
                    max_features=self.config.max_features
                )
                if selected:
                    feature_cols = selected
                print(f"      After forward selection: {len(feature_cols)} features")
            except Exception as e:
                print(f"      Forward selection failed: {e}")
        
        # Stage 6: VIF check
        if self.config.vif_threshold and len(feature_cols) > 2:
            print("    6. VIF check...")
            try:
                vif_data = engineer.calculate_vif(
                    self.train_[feature_cols].fillna(0),
                    self.config.vif_threshold
                )
                if not vif_data.empty:
                    high_vif = vif_data[vif_data['VIF'] > self.config.vif_threshold]['feature'].tolist()
                    feature_cols = [col for col in feature_cols if col not in high_vif]
                print(f"      After VIF filter: {len(feature_cols)} features")
            except Exception as e:
                print(f"      VIF check failed: {e}")
        
        # Store selected features
        self.final_vars_ = feature_cols
        self.final_vars_raw_ = [col for col in feature_cols if col in self.numeric_vars_]
        self.final_vars_woe_ = feature_cols  # All will get WOE
        
        print(f"\n  Final selected features: {len(self.final_vars_)}")
        print(f"    - Numeric (for RAW): {len(self.final_vars_raw_)}")
        print(f"    - All (for WOE): {len(self.final_vars_woe_)}")
    
    def _apply_woe_transformation(self) -> Dict[str, pd.DataFrame]:
        """Apply WOE transformation to ALL selected variables"""
        
        print("  Fitting WOE transformation...")
        
        # Fit WOE on all selected features
        woe_result = self.woe_transformer.fit_transform(
            self.train_,
            self.test_,
            self.oot_,
            self.final_vars_woe_
        )
        
        self.woe_mapping_ = woe_result["mapping"]
        
        print(f"    - Variables transformed: {len(self.final_vars_woe_)}")
        print(f"    - WOE bins created: {sum(len(v['bins']) for v in self.woe_mapping_.values())}")
        
        return woe_result
    
    def _build_all_models(self, woe_data: Dict):
        """Build both WOE and RAW models, test all model types"""
        
        print("  Building WOE models...")
        
        # Build WOE models
        woe_results = self.model_builder.build_models(
            woe_data["train"],
            woe_data.get("test"),
            woe_data.get("oot")
        )
        
        self.best_model_woe_ = woe_results["best_model"]
        self.model_results_["woe"] = woe_results
        
        print(f"    Best WOE model: {woe_results['best_model_name']} (AUC: {woe_results['best_score']:.4f})")
        
        # Build RAW models if enabled
        if self.config.enable_dual_pipeline and self.final_vars_raw_:
            print("\n  Building RAW models...")
            
            # Scale features for RAW models
            if self.config.raw_scaler_type == "standard":
                self.scaler_ = StandardScaler()
            else:
                self.scaler_ = RobustScaler()
            
            # Prepare RAW data
            X_train_raw = self.train_[self.final_vars_raw_].fillna(0)
            X_train_scaled = pd.DataFrame(
                self.scaler_.fit_transform(X_train_raw),
                columns=self.final_vars_raw_,
                index=X_train_raw.index
            )
            
            raw_train = X_train_scaled.copy()
            raw_train[self.config.target_col] = self.train_[self.config.target_col]
            
            raw_test = None
            if self.test_ is not None:
                X_test_raw = self.test_[self.final_vars_raw_].fillna(0)
                X_test_scaled = pd.DataFrame(
                    self.scaler_.transform(X_test_raw),
                    columns=self.final_vars_raw_,
                    index=X_test_raw.index
                )
                raw_test = X_test_scaled.copy()
                raw_test[self.config.target_col] = self.test_[self.config.target_col]
            
            raw_oot = None
            if self.oot_ is not None:
                X_oot_raw = self.oot_[self.final_vars_raw_].fillna(0)
                X_oot_scaled = pd.DataFrame(
                    self.scaler_.transform(X_oot_raw),
                    columns=self.final_vars_raw_,
                    index=X_oot_raw.index
                )
                raw_oot = X_oot_scaled.copy()
                raw_oot[self.config.target_col] = self.oot_[self.config.target_col]
            
            # Build RAW models
            raw_results = self.model_builder.build_models(
                raw_train,
                raw_test,
                raw_oot
            )
            
            self.best_model_raw_ = raw_results["best_model"]
            self.model_results_["raw"] = raw_results
            
            print(f"    Best RAW model: {raw_results['best_model_name']} (AUC: {raw_results['best_score']:.4f})")
            
            # Select overall best model
            if raw_results["best_score"] > woe_results["best_score"]:
                self.best_model_ = self.best_model_raw_
                self.best_model_name_ = f"RAW_{raw_results['best_model_name']}"
                self.best_score_ = raw_results["best_score"]
                print(f"\n  Overall best: RAW pipeline")
            else:
                self.best_model_ = self.best_model_woe_
                self.best_model_name_ = f"WOE_{woe_results['best_model_name']}"
                self.best_score_ = woe_results["best_score"]
                print(f"\n  Overall best: WOE pipeline")
        else:
            self.best_model_ = self.best_model_woe_
            self.best_model_name_ = woe_results['best_model_name']
            self.best_score_ = woe_results['best_score']
        
        # Print all model results
        print("\n  All model results:")
        for pipeline_type, results in self.model_results_.items():
            print(f"\n    {pipeline_type.upper()} Pipeline:")
            for model_name, score in results.get("all_scores", {}).items():
                print(f"      - {model_name}: {score:.4f}")
    
    def _calculate_all_psi(self):
        """Calculate comprehensive PSI (WOE, Quantile, Score)"""
        
        if self.test_ is None and self.oot_ is None:
            print("  No test/OOT data for PSI calculation")
            return
        
        comparison_data = self.oot_ if self.oot_ is not None else self.test_
        comparison_name = "OOT" if self.oot_ is not None else "Test"
        
        # 1. WOE-based PSI for WOE variables
        if self.woe_mapping_:
            print(f"  Calculating WOE-based PSI (Train vs {comparison_name})...")
            
            woe_psi_results = {}
            for var in self.final_vars_woe_:
                if var in self.woe_mapping_:
                    try:
                        # Calculate using WOE bins
                        train_woe = self.train_[[var]]
                        comp_woe = comparison_data[[var]]
                        
                        psi_result = self.psi_calculator.calculate_woe_psi(
                            train_woe, comp_woe, {var: self.woe_mapping_[var]}
                        )
                        woe_psi_results[var] = psi_result.get(var, {}).get('psi', 0)
                    except:
                        woe_psi_results[var] = 0
            
            self.psi_results_["woe_psi"] = woe_psi_results
            high_psi_vars = [v for v, p in woe_psi_results.items() if p > 0.25]
            print(f"    - Variables with high PSI (>0.25): {len(high_psi_vars)}")
        
        # 2. Quantile-based PSI for RAW variables
        if self.final_vars_raw_:
            print(f"  Calculating Quantile-based PSI (Train vs {comparison_name})...")
            
            quantile_psi_results = {}
            for var in self.final_vars_raw_:
                try:
                    psi_value, _ = self.psi_calculator.calculate_psi(
                        self.train_[var].values,
                        comparison_data[var].values,
                        n_bins=10
                    )
                    quantile_psi_results[var] = psi_value
                except:
                    quantile_psi_results[var] = 0
            
            self.psi_results_["quantile_psi"] = quantile_psi_results
        
        # 3. Score PSI
        print(f"  Calculating Score PSI (Train vs {comparison_name})...")
        
        # Get predictions
        if "woe" in self.best_model_name_.lower():
            X_train = self.train_[self.final_vars_woe_].fillna(0)
            X_comp = comparison_data[self.final_vars_woe_].fillna(0)
        else:
            X_train = self.train_[self.final_vars_raw_].fillna(0)
            X_comp = comparison_data[self.final_vars_raw_].fillna(0)
            if self.scaler_:
                X_train = self.scaler_.transform(X_train)
                X_comp = self.scaler_.transform(X_comp)
        
        train_scores = self.best_model_.predict_proba(X_train)[:, 1]
        comp_scores = self.best_model_.predict_proba(X_comp)[:, 1]
        
        score_psi, score_df = self.psi_calculator.calculate_score_psi(
            train_scores, comp_scores, n_bins=10
        )
        
        self.psi_results_["score_psi"] = {
            "value": score_psi,
            "interpretation": self.psi_calculator._interpret_psi(score_psi),
            "details": score_df
        }
        
        print(f"    - Score PSI: {score_psi:.4f} ({self.psi_calculator._interpret_psi(score_psi)})")
    
    def _perform_calibration(self):
        """Perform calibration analysis with risk bands"""
        
        calib_data = self.calibration_ if self.calibration_ is not None else self.oot_
        calib_name = "Calibration" if self.calibration_ is not None else "OOT"
        
        print(f"  Performing calibration on {calib_name} data...")
        
        # Get predictions
        if "woe" in self.best_model_name_.lower():
            X_calib = calib_data[self.final_vars_woe_].fillna(0)
        else:
            X_calib = calib_data[self.final_vars_raw_].fillna(0)
            if self.scaler_:
                X_calib = self.scaler_.transform(X_calib)
        
        y_true = calib_data[self.config.target_col].values
        y_pred = self.best_model_.predict_proba(X_calib)[:, 1]
        
        # Analyze calibration with risk bands
        calib_results = self.calibration_analyzer.analyze_calibration(
            y_true, y_pred,
            use_deciles=False  # Use risk bands
        )
        
        self.calibration_results_ = calib_results
        
        # Print summary
        print(f"    - Hosmer-Lemeshow test p-value: {calib_results.get('hosmer_lemeshow_p', 0):.4f}")
        print(f"    - Brier score: {calib_results.get('brier_score', 0):.4f}")
        print(f"    - Calibration slope: {calib_results.get('calibration_slope', 0):.4f}")
        
        # Check binomial tests
        failed_bands = 0
        for band in calib_results.get('risk_bands', []):
            if band.get('binomial_p_value', 1) < 0.05:
                failed_bands += 1
        
        print(f"    - Risk bands failing binomial test: {failed_bands}/{len(calib_results.get('risk_bands', []))}")
        
        if calib_results.get('needs_calibration'):
            print("    [WARNING] Model needs calibration!")
    
    def _optimize_risk_bands(self):
        """Optimize risk bands based on IV/Gini"""
        
        print("  Optimizing risk bands...")
        
        # Get scores
        if "woe" in self.best_model_name_.lower():
            X_train = self.train_[self.final_vars_woe_].fillna(0)
        else:
            X_train = self.train_[self.final_vars_raw_].fillna(0)
            if self.scaler_:
                X_train = self.scaler_.transform(X_train)
        
        train_scores = self.best_model_.predict_proba(X_train)[:, 1]
        y_train = self.train_[self.config.target_col].values
        
        # Optimize bands
        optimization_metric = "gini"  # or "iv", "ks"
        n_bands = 10
        
        risk_bands = self.risk_band_optimizer.create_optimized_bands(
            train_scores, y_train,
            optimization_metric=optimization_metric,
            n_bands=n_bands,
            ensure_monotonic=True
        )
        
        self.risk_bands_ = risk_bands
        
        # Print band summary
        print(f"    - Optimization metric: {optimization_metric}")
        print(f"    - Number of bands: {len(risk_bands)}")
        print(f"    - Total {optimization_metric.upper()}: {risk_bands[-1].get('cumulative_' + optimization_metric, 0):.4f}")
        
        # Show top and bottom bands
        if risk_bands:
            print(f"\n    Risk Band Summary:")
            print(f"      Band 1: Score [{risk_bands[0]['min_score']:.4f}, {risk_bands[0]['max_score']:.4f}], "
                  f"Bad Rate: {risk_bands[0]['bad_rate']:.4f}")
            print(f"      Band {len(risk_bands)}: Score [{risk_bands[-1]['min_score']:.4f}, {risk_bands[-1]['max_score']:.4f}], "
                  f"Bad Rate: {risk_bands[-1]['bad_rate']:.4f}")
    
    def _generate_comprehensive_reports(self):
        """Generate all required reports"""
        
        print("  Generating comprehensive reports...")
        
        # Prepare report data
        report_data = {
            "model_info": {
                "best_model": self.best_model_name_,
                "best_score": self.best_score_,
                "n_features": len(self.final_vars_),
                "features": self.final_vars_
            },
            "data_splits": {
                "train_size": len(self.train_),
                "test_size": len(self.test_) if self.test_ is not None else 0,
                "oot_size": len(self.oot_) if self.oot_ is not None else 0,
                "calibration_size": len(self.calibration_) if self.calibration_ is not None else 0
            },
            "model_results": self.model_results_,
            "psi_results": self.psi_results_,
            "calibration_results": self.calibration_results_,
            "risk_bands": self.risk_bands_,
            "woe_mapping": self.woe_mapping_
        }
        
        # Generate Excel report
        self.reporter.generate_reports(
            train=self.train_,
            test=self.test_,
            oot=self.oot_,
            model=self.best_model_,
            features=self.final_vars_,
            woe_mapping=self.woe_mapping_,
            model_name=self.best_model_name_,
            scores={"best": self.best_score_}
        )
        
        self.final_report_ = report_data
        
        print("    - Reports saved to output folder")
        print("    - Excel report generated")
        
        # Add SHAP analysis if available
        try:
            import shap
            print("    - Generating SHAP analysis...")
            
            # Get sample data for SHAP
            if "woe" in self.best_model_name_.lower():
                X_sample = self.train_[self.final_vars_woe_].fillna(0).sample(min(100, len(self.train_)))
            else:
                X_sample = self.train_[self.final_vars_raw_].fillna(0).sample(min(100, len(self.train_)))
                if self.scaler_:
                    X_sample = pd.DataFrame(self.scaler_.transform(X_sample), columns=X_sample.columns)
            
            # Create SHAP explainer
            if "tree" in str(type(self.best_model_)).lower() or "forest" in str(type(self.best_model_)).lower():
                explainer = shap.TreeExplainer(self.best_model_)
            else:
                explainer = shap.LinearExplainer(self.best_model_, X_sample)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Store SHAP importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            self.final_report_["shap_importance"] = shap_importance
            print("    - SHAP analysis completed")
            
        except ImportError:
            print("    - SHAP not available, skipping SHAP analysis")
        except Exception as e:
            print(f"    - SHAP analysis failed: {e}")
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Get final pipeline results"""
        
        return {
            "best_model": self.best_model_,
            "best_model_name": self.best_model_name_,
            "best_score": self.best_score_,
            "selected_features": self.final_vars_,
            "numeric_features": self.numeric_vars_,
            "categorical_features": self.categorical_vars_,
            "woe_mapping": self.woe_mapping_,
            "model_results": self.model_results_,
            "psi_results": self.psi_results_,
            "calibration_results": self.calibration_results_,
            "risk_bands": self.risk_bands_,
            "report": self.final_report_
        }
    
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score new data with the best model"""
        
        if self.best_model_ is None:
            raise ValueError("Pipeline must be run before scoring")
        
        # Process data
        df_processed = self.processor.validate_and_freeze(df)
        
        # Impute if needed
        if self.imputer_ and self.numeric_vars_:
            num_cols = [c for c in self.numeric_vars_ if c in df_processed.columns]
            if num_cols:
                df_processed[num_cols] = self.imputer_.transform(df_processed[num_cols])
        
        # Prepare features based on model type
        if "woe" in self.best_model_name_.lower():
            # Apply WOE transformation
            df_woe = self.woe_transformer.transform(df_processed, self.woe_mapping_)
            X = df_woe[self.final_vars_woe_].fillna(0)
        else:
            # Use RAW features
            X = df_processed[self.final_vars_raw_].fillna(0)
            if self.scaler_:
                X = self.scaler_.transform(X)
        
        # Make predictions
        scores = self.best_model_.predict_proba(X)[:, 1]
        
        # Apply calibration if available
        if self.calibration_results_ and self.calibration_results_.get('calibration_function'):
            calibration_func = self.calibration_results_['calibration_function']
            scores = calibration_func(scores)
        
        # Create result DataFrame
        result = df.copy()
        result['score'] = scores
        
        # Add risk bands
        if self.risk_bands_:
            result['risk_band'] = pd.cut(
                scores,
                bins=[b['min_score'] for b in self.risk_bands_] + [self.risk_bands_[-1]['max_score']],
                labels=[f"Band_{i+1}" for i in range(len(self.risk_bands_))],
                include_lowest=True
            )
        
        return result