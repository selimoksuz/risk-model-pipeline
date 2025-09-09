"""Complete Risk Model Pipeline with All Features"""

import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

from .core.config import Config
from .core.data_processor import DataProcessor
from .core.feature_selector import FeatureSelector
from .core.model_builder import ModelBuilder
from .core.reporter import Reporter
from .core.splitter import DataSplitter
from .core.woe_transformer import WOETransformer
from .core.psi_calculator import PSICalculator
from .core.calibration_analyzer import CalibrationAnalyzer
from .core.risk_band_optimizer import RiskBandOptimizer
from .core.feature_engineer import FeatureEngineer

warnings.filterwarnings("ignore")


class CompletePipeline:
    """Complete risk model pipeline with all required features"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize complete pipeline"""
        self.config = config or Config()
        
        # Core components
        self.processor = DataProcessor(self.config)
        self.splitter = DataSplitter(self.config)
        self.selector = FeatureSelector(self.config)
        self.engineer = FeatureEngineer(self.config)
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
        
        # Variables
        self.numeric_vars_ = []
        self.categorical_vars_ = []
        self.final_vars_ = []
        
        # Models and results
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        self.all_models_ = {}
        
        # Transformations
        self.woe_mapping_ = None
        self.imputer_ = None
        self.scaler_ = None
        
        # Results
        self.psi_results_ = {}
        self.calibration_results_ = {}
        self.risk_bands_ = None
        self.model_comparison_ = None
        
    def run(self, df: pd.DataFrame, 
            test_df: Optional[pd.DataFrame] = None,
            oot_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run complete pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data (or full data if no test/oot provided)
        test_df : pd.DataFrame, optional
            Test data (if provided separately)
        oot_df : pd.DataFrame, optional
            Out-of-time validation data
        """
        
        print("\n" + "="*80)
        print("RISK MODEL PIPELINE")
        print("="*80)
        
        # 1. Data Processing and Validation
        print("\n1. DATA PROCESSING")
        print("-"*40)
        df_processed = self.processor.validate_and_freeze(df)
        
        # Process test/oot if provided
        if test_df is not None:
            test_df = self.processor.validate_and_freeze(test_df)
        if oot_df is not None:
            oot_df = self.processor.validate_and_freeze(oot_df)
        
        # 2. Variable Classification
        print("\n2. VARIABLE CLASSIFICATION")
        print("-"*40)
        var_catalog = self.processor.classify_variables(df_processed)
        self.numeric_vars_ = var_catalog[var_catalog['variable_group'] == 'numeric']['variable'].tolist()
        self.categorical_vars_ = var_catalog[var_catalog['variable_group'] == 'categorical']['variable'].tolist()
        
        print(f"  Numeric variables: {len(self.numeric_vars_)}")
        print(f"  Categorical variables: {len(self.categorical_vars_)}")
        
        # 3. Data Splitting (if not provided separately)
        print("\n3. DATA SPLITTING")
        print("-"*40)
        if test_df is None and oot_df is None:
            # Use splitter
            splits = self.splitter.split(df_processed)
            self.train_ = splits['train']
            self.test_ = splits.get('test')
            self.oot_ = splits.get('oot')
        else:
            # Use provided splits
            self.train_ = df_processed
            self.test_ = test_df
            self.oot_ = oot_df
        
        print(f"  Train: {len(self.train_)} samples")
        if self.test_ is not None:
            print(f"  Test: {len(self.test_)} samples")
        if self.oot_ is not None:
            print(f"  OOT: {len(self.oot_)} samples")
        
        # 4. Imputation and Outlier Handling
        print("\n4. IMPUTATION & OUTLIER HANDLING")
        print("-"*40)
        self._handle_missing_and_outliers()
        
        # 5. Feature Engineering and Selection
        print("\n5. FEATURE SELECTION")
        print("-"*40)
        selected = self.selector.select_features(self.train_, self.test_, self.oot_)
        self.final_vars_ = selected['final_features']
        print(f"  Selected features: {len(self.final_vars_)}")
        
        # 6. WOE Transformation for ALL variables (not just selected)
        print("\n6. WOE TRANSFORMATION")
        print("-"*40)
        all_features = self.numeric_vars_ + self.categorical_vars_
        print(f"  Applying WOE to {len(all_features)} variables")
        
        woe_data = self.woe_transformer.fit_transform(
            self.train_, 
            self.test_, 
            self.oot_, 
            all_features  # Apply to ALL features
        )
        self.woe_mapping_ = woe_data['mapping']
        
        # Select WOE features for modeling
        woe_train = woe_data['train'][self.final_vars_ + [self.config.target_col]]
        woe_test = woe_data.get('test')
        if woe_test is not None:
            woe_test = woe_test[self.final_vars_ + [self.config.target_col]]
        woe_oot = woe_data.get('oot')
        if woe_oot is not None:
            woe_oot = woe_oot[self.final_vars_ + [self.config.target_col]]
        
        # 7. Build ALL Model Types
        print("\n7. MODEL BUILDING")
        print("-"*40)
        print("  Testing all model types with HPO...")
        
        # Build WOE models
        woe_results = self.model_builder.build_models(
            woe_train, woe_test, woe_oot
        )
        
        self.best_model_ = woe_results['best_model']
        self.best_model_name_ = woe_results['best_model_name']
        self.best_score_ = woe_results['best_score']
        self.all_models_ = woe_results.get('all_models', {})
        
        # Print all model results
        print("\n  Model Performance:")
        for model_name, score in woe_results.get('all_scores', {}).items():
            print(f"    {model_name}: {score:.4f}")
        print(f"\n  Best Model: {self.best_model_name_} (AUC: {self.best_score_:.4f})")
        
        # 8. PSI Analysis (WOE, Quantile, Score)
        print("\n8. PSI ANALYSIS")
        print("-"*40)
        self._calculate_comprehensive_psi()
        
        # 9. Calibration Analysis
        print("\n9. CALIBRATION ANALYSIS")
        print("-"*40)
        if self.oot_ is not None:
            self._perform_calibration()
        
        # 10. Risk Band Optimization
        print("\n10. RISK BAND OPTIMIZATION")
        print("-"*40)
        self._optimize_risk_bands()
        
        # 11. Model Comparison and Reports
        print("\n11. COMPREHENSIVE REPORTS")
        print("-"*40)
        self._generate_reports()
        
        # 12. SHAP Analysis (if available)
        print("\n12. FEATURE IMPORTANCE ANALYSIS")
        print("-"*40)
        self._perform_shap_analysis()
        
        print("\n" + "="*80)
        print("Pipeline finished.")
        print("="*80)
        
        return self._get_results()
    
    def _handle_missing_and_outliers(self):
        """Handle missing values and outliers"""
        
        # Imputation for numeric variables
        if self.numeric_vars_:
            print("  Imputing numeric variables...")
            self.imputer_ = SimpleImputer(strategy=self.config.imputation_strategy)
            
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
        
        # Handle categorical missing values
        if self.categorical_vars_:
            print("  Handling categorical missing values...")
            for col in self.categorical_vars_:
                self.train_[col] = self.train_[col].fillna('Missing')
                if self.test_ is not None:
                    self.test_[col] = self.test_[col].fillna('Missing')
                if self.oot_ is not None:
                    self.oot_[col] = self.oot_[col].fillna('Missing')
        
        # Outlier handling
        if self.config.raw_outlier_method != "none" and self.numeric_vars_:
            print("  Handling outliers...")
            method = self.config.raw_outlier_method
            threshold = self.config.raw_outlier_threshold
            
            if method == "clip":
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
    
    def _calculate_comprehensive_psi(self):
        """Calculate WOE PSI, Quantile PSI, and Score PSI"""
        
        if self.test_ is None and self.oot_ is None:
            print("  No test/OOT data for PSI calculation")
            return
        
        comparison_data = self.oot_ if self.oot_ is not None else self.test_
        comparison_name = "OOT" if self.oot_ is not None else "Test"
        
        # 1. WOE-based PSI
        if self.woe_mapping_:
            print(f"  WOE-based PSI (Train vs {comparison_name}):")
            woe_psi = self.psi_calculator.calculate_woe_psi(
                self.train_[list(self.woe_mapping_.keys())],
                comparison_data[list(self.woe_mapping_.keys())],
                self.woe_mapping_
            )
            self.psi_results_['woe_psi'] = woe_psi
            
            # Print summary
            high_psi = [k for k, v in woe_psi.items() if v.get('psi', 0) > 0.25]
            print(f"    Variables with high PSI (>0.25): {len(high_psi)}")
        
        # 2. Quantile PSI for numeric variables
        if self.numeric_vars_:
            print(f"  Quantile PSI for numeric variables:")
            quantile_psi = {}
            for var in self.numeric_vars_:
                if var in self.train_.columns and var in comparison_data.columns:
                    psi_value, _ = self.psi_calculator.calculate_psi(
                        self.train_[var].values,
                        comparison_data[var].values,
                        n_bins=10
                    )
                    quantile_psi[var] = psi_value
            
            self.psi_results_['quantile_psi'] = quantile_psi
            high_psi = [k for k, v in quantile_psi.items() if v > 0.25]
            print(f"    Variables with high PSI (>0.25): {len(high_psi)}")
        
        # 3. Score PSI
        print(f"  Score PSI (Train vs {comparison_name}):")
        
        # Get predictions
        X_train = self.train_[self.final_vars_].fillna(0)
        X_comp = comparison_data[self.final_vars_].fillna(0)
        
        train_scores = self.best_model_.predict_proba(X_train)[:, 1]
        comp_scores = self.best_model_.predict_proba(X_comp)[:, 1]
        
        score_psi, score_df = self.psi_calculator.calculate_score_psi(
            train_scores, comp_scores, n_bins=10
        )
        
        self.psi_results_['score_psi'] = {
            'value': score_psi,
            'interpretation': self.psi_calculator._interpret_psi(score_psi),
            'details': score_df
        }
        
        print(f"    PSI: {score_psi:.4f} ({self.psi_calculator._interpret_psi(score_psi)})")
    
    def _perform_calibration(self):
        """Perform calibration analysis with binomial test and Brier score"""
        
        # Use OOT for calibration
        X_oot = self.oot_[self.final_vars_].fillna(0)
        y_true = self.oot_[self.config.target_col].values
        y_pred = self.best_model_.predict_proba(X_oot)[:, 1]
        
        # Analyze calibration
        calib_results = self.calibration_analyzer.analyze_calibration(
            y_true, y_pred, use_deciles=True
        )
        
        self.calibration_results_ = calib_results
        
        # Print summary
        print(f"  Hosmer-Lemeshow p-value: {calib_results.get('hosmer_lemeshow_p', 0):.4f}")
        print(f"  Brier score: {calib_results.get('brier_score', 0):.4f}")
        print(f"  Calibration slope: {calib_results.get('calibration_slope', 0):.4f}")
        
        # Check binomial tests
        if 'risk_bands' in calib_results:
            failed_bands = sum(1 for band in calib_results['risk_bands'] 
                             if band.get('binomial_p_value', 1) < 0.05)
            print(f"  Bands failing binomial test: {failed_bands}/{len(calib_results['risk_bands'])}")
        
        if calib_results.get('needs_calibration'):
            print("  [WARNING] Model needs calibration!")
    
    def _optimize_risk_bands(self):
        """Optimize risk bands based on Gini/IV"""
        
        X_train = self.train_[self.final_vars_].fillna(0)
        train_scores = self.best_model_.predict_proba(X_train)[:, 1]
        y_train = self.train_[self.config.target_col].values
        
        # Optimize bands
        risk_bands = self.risk_band_optimizer.create_optimized_bands(
            train_scores, y_train,
            optimization_metric='gini',
            n_bands=10,
            ensure_monotonic=True
        )
        
        self.risk_bands_ = risk_bands
        
        if risk_bands:
            print(f"  Created {len(risk_bands)} risk bands")
            print(f"  Band 1 bad rate: {risk_bands[0]['bad_rate']:.4f}")
            print(f"  Band {len(risk_bands)} bad rate: {risk_bands[-1]['bad_rate']:.4f}")
    
    def _generate_reports(self):
        """Generate comprehensive reports"""
        
        # Generate standard reports
        self.reporter.generate_reports(
            train=self.train_,
            test=self.test_,
            oot=self.oot_,
            model=self.best_model_,
            features=self.final_vars_,
            woe_mapping=self.woe_mapping_,
            model_name=self.best_model_name_,
            scores={'best': self.best_score_}
        )
        
        # Create model comparison
        if self.all_models_:
            comparison = []
            for name, model_info in self.all_models_.items():
                comparison.append({
                    'Model': name,
                    'Train_AUC': model_info.get('train_score', 0),
                    'Test_AUC': model_info.get('test_score', 0),
                    'OOT_AUC': model_info.get('oot_score', 0) if self.oot_ is not None else None
                })
            self.model_comparison_ = pd.DataFrame(comparison)
            
            # Save comparison
            self.model_comparison_.to_csv(
                f"{self.config.output_folder}/model_comparison.csv", 
                index=False
            )
        
        print("  Reports generated successfully")
    
    def _perform_shap_analysis(self):
        """Perform SHAP analysis for feature importance"""
        
        try:
            import shap
            
            # Get sample data
            X_sample = self.train_[self.final_vars_].fillna(0).sample(
                min(100, len(self.train_)), random_state=42
            )
            
            # Create explainer based on model type
            model_type = str(type(self.best_model_)).lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
                explainer = shap.TreeExplainer(self.best_model_)
            else:
                explainer = shap.LinearExplainer(self.best_model_, X_sample)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            feature_importance = pd.DataFrame({
                'feature': self.final_vars_,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            # Save SHAP importance
            feature_importance.to_csv(
                f"{self.config.output_folder}/shap_importance.csv",
                index=False
            )
            
            print("  SHAP analysis completed")
            print(f"  Top 3 features: {', '.join(feature_importance.head(3)['feature'].tolist())}")
            
        except ImportError:
            print("  SHAP not available, using model's feature importance")
            
            # Try to get feature importance from model
            if hasattr(self.best_model_, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.final_vars_,
                    'importance': self.best_model_.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance.to_csv(
                    f"{self.config.output_folder}/feature_importance.csv",
                    index=False
                )
                print(f"  Top 3 features: {', '.join(importance.head(3)['feature'].tolist())}")
            
        except Exception as e:
            print(f"  Feature importance analysis failed: {e}")
    
    def _get_results(self) -> Dict[str, Any]:
        """Get pipeline results"""
        
        return {
            'best_model': self.best_model_,
            'best_model_name': self.best_model_name_,
            'best_score': self.best_score_,
            'selected_features': self.final_vars_,
            'numeric_features': self.numeric_vars_,
            'categorical_features': self.categorical_vars_,
            'woe_mapping': self.woe_mapping_,
            'psi_results': self.psi_results_,
            'calibration_results': self.calibration_results_,
            'risk_bands': self.risk_bands_,
            'model_comparison': self.model_comparison_
        }
    
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score new data"""
        
        if self.best_model_ is None:
            raise ValueError("Pipeline must be run before scoring")
        
        # Process data
        df_processed = self.processor.validate_and_freeze(df)
        
        # Apply imputation
        if self.imputer_ and self.numeric_vars_:
            num_cols = [c for c in self.numeric_vars_ if c in df_processed.columns]
            if num_cols:
                df_processed[num_cols] = self.imputer_.transform(df_processed[num_cols])
        
        # Handle categorical missing
        if self.categorical_vars_:
            for col in self.categorical_vars_:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna('Missing')
        
        # Apply WOE transformation
        df_woe = self.woe_transformer.transform(df_processed, self.woe_mapping_)
        
        # Get features
        X = df_woe[self.final_vars_].fillna(0)
        
        # Make predictions
        scores = self.best_model_.predict_proba(X)[:, 1]
        
        # Apply calibration if available
        if self.calibration_results_ and self.calibration_results_.get('calibration_function'):
            calibration_func = self.calibration_results_['calibration_function']
            scores = calibration_func(scores)
        
        # Create result
        result = df.copy()
        result['score'] = scores
        
        # Add risk bands
        if self.risk_bands_:
            bins = [b['min_score'] for b in self.risk_bands_] + [self.risk_bands_[-1]['max_score']]
            labels = [f"Band_{i+1}" for i in range(len(self.risk_bands_))]
            result['risk_band'] = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
        
        return result