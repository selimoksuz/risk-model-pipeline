"""Risk Model Pipeline - Main Orchestrator"""

import os
import random
import warnings
from typing import Any, Dict, Optional, Union

import pandas as pd

from .core.config import Config
from .core.data_processor import DataProcessor
from .core.feature_selector import FeatureSelector
from .core.model_builder import ModelBuilder
from .core.reporter import Reporter
from .core.splitter import DataSplitter
from .core.woe_transformer import WOETransformer

warnings.filterwarnings("ignore")


class RiskModelPipeline:
    """Main risk model pipeline orchestrator."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with configuration."""
        self.config = config or Config()
        self.processor = DataProcessor(self.config)
        self.splitter = DataSplitter(self.config)
        self.selector = FeatureSelector(self.config)
        self.woe_transformer = WOETransformer(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.reporter = Reporter(self.config)

        # Results storage
        self.train_ = None
        self.test_ = None
        self.oot_ = None
        self.final_vars_ = []
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        self.best_auc_ = None
        self.woe_mapping_ = None

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete pipeline."""
        print("Starting Risk Model Pipeline...")

        # Process data
        print("1. Processing data...")
        df_processed = self.processor.validate_and_freeze(df)

        # Split data
        print("2. Splitting data...")
        splits = self.splitter.split(df_processed)
        self.train_ = splits["train"]
        self.test_ = splits["test"]
        self.oot_ = splits.get("oot")

        # Select features
        print("3. Selecting features...")
        selected_features = self.selector.select_features(self.train_, self.test_, self.oot_)
        self.final_vars_ = selected_features["final_features"]

        # WOE transformation
        print("4. Applying WOE transformation...")
        woe_data = self.woe_transformer.fit_transform(self.train_, self.test_, self.oot_, self.final_vars_)
        self.woe_mapping_ = woe_data["mapping"]

        # Build models
        print("5. Building models...")
        model_results = self.model_builder.build_models(woe_data["train"], woe_data["test"], woe_data.get("oot"))

        self.best_model_ = model_results["best_model"]
        self.best_model_name_ = model_results["best_model_name"]
        self.best_score_ = model_results["best_score"]
        self.best_auc_ = model_results.get("best_auc", self.best_score_)

        # Generate reports
        print("6. Generating reports...")
        self.reporter.generate_reports(
            train=self.train_,
            test=self.test_,
            oot=self.oot_,
            model=self.best_model_,
            features=self.final_vars_,
            woe_mapping=self.woe_mapping_,
            model_name=self.best_model_name_,
            scores={self.best_model_name_: self.best_score_},
        )

        print("Pipeline finished.")

        return {
            "best_model": self.best_model_,
            "best_model_name": self.best_model_name_,
            "best_score": self.best_score_,
            "features": self.final_vars_,
            "woe_mapping": self.woe_mapping_,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if self.best_model_ is None:
            raise ValueError("Pipeline must be run before making predictions")

        # Process data
        df_processed = self.processor.process(df)

        # Apply WOE transformation
        if self.woe_mapping_:
            df_woe = self.woe_transformer.transform(df_processed, self.woe_mapping_)
            X = df_woe[self.final_vars_]
        else:
            X = df_processed[self.final_vars_]

        # Make predictions
        predictions = self.best_model_.predict_proba(X)[:, 1]

        result = df.copy()
        result["score"] = predictions

        return result

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score new data (alias for predict)."""
        return self.predict(df)
    
    def extract_results(self) -> Dict[str, Any]:
        """Extract comprehensive results from the fitted pipeline."""
        if self.best_model_ is None:
            raise ValueError("Pipeline must be run before extracting results")
        
        results = {
            'best_model': self.best_model_name_,
            'best_model_object': self.best_model_,
            'best_auc': self.best_auc_,
            'best_gini': (self.best_auc_ * 2 - 1) if self.best_auc_ else None,
            'selected_features': self.final_vars_,
            'n_selected_features': len(self.final_vars_),
            'woe_mapping': self.woe_mapping_,
            'train_size': len(self.train_) if self.train_ is not None else 0,
            'test_size': len(self.test_) if self.test_ is not None else 0,
            'oot_size': len(self.oot_) if self.oot_ is not None else 0,
            'config': {
                'target_col': self.config.target_col,
                'enable_dual_pipeline': self.config.enable_dual_pipeline,
                'n_trials': self.config.n_trials,
                'use_boruta': self.config.use_boruta,
                'forward_selection': self.config.forward_selection,
                'use_noise_sentinel': self.config.use_noise_sentinel
            }
        }
        
        return results
    
    def extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from the fitted pipeline."""
        if self.best_model_ is None:
            raise ValueError("Pipeline must be run before extracting metrics")
        
        metrics = {}
        
        # Training metrics
        if self.train_ is not None and self.final_vars_:
            X_train = self.train_[self.final_vars_].fillna(0)
            y_train = self.train_[self.config.target_col]
            train_pred = self.best_model_.predict_proba(X_train)[:, 1]
            
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            from scipy.stats import ks_2samp
            
            metrics['train'] = {
                'auc': roc_auc_score(y_train, train_pred),
                'gini': roc_auc_score(y_train, train_pred) * 2 - 1,
                'ks': ks_2samp(train_pred[y_train == 0], train_pred[y_train == 1]).statistic,
                'accuracy': accuracy_score(y_train, (train_pred > 0.5).astype(int)),
                'precision': precision_score(y_train, (train_pred > 0.5).astype(int)),
                'recall': recall_score(y_train, (train_pred > 0.5).astype(int)),
                'f1': f1_score(y_train, (train_pred > 0.5).astype(int))
            }
        
        # Test metrics
        if self.test_ is not None and self.final_vars_:
            X_test = self.test_[self.final_vars_].fillna(0)
            y_test = self.test_[self.config.target_col]
            test_pred = self.best_model_.predict_proba(X_test)[:, 1]
            
            metrics['test'] = {
                'auc': roc_auc_score(y_test, test_pred),
                'gini': roc_auc_score(y_test, test_pred) * 2 - 1,
                'ks': ks_2samp(test_pred[y_test == 0], test_pred[y_test == 1]).statistic,
                'accuracy': accuracy_score(y_test, (test_pred > 0.5).astype(int)),
                'precision': precision_score(y_test, (test_pred > 0.5).astype(int)),
                'recall': recall_score(y_test, (test_pred > 0.5).astype(int)),
                'f1': f1_score(y_test, (test_pred > 0.5).astype(int))
            }
        
        # OOT metrics if available
        if self.oot_ is not None and self.final_vars_:
            X_oot = self.oot_[self.final_vars_].fillna(0)
            y_oot = self.oot_[self.config.target_col]
            oot_pred = self.best_model_.predict_proba(X_oot)[:, 1]
            
            metrics['oot'] = {
                'auc': roc_auc_score(y_oot, oot_pred),
                'gini': roc_auc_score(y_oot, oot_pred) * 2 - 1,
                'ks': ks_2samp(oot_pred[y_oot == 0], oot_pred[y_oot == 1]).statistic,
                'accuracy': accuracy_score(y_oot, (oot_pred > 0.5).astype(int)),
                'precision': precision_score(y_oot, (oot_pred > 0.5).astype(int)),
                'recall': recall_score(y_oot, (oot_pred > 0.5).astype(int)),
                'f1': f1_score(y_oot, (oot_pred > 0.5).astype(int))
            }
        
        # Model comparison if results were stored
        metrics['model_name'] = self.best_model_name_
        metrics['best_auc'] = self.best_auc_
        metrics['best_gini'] = (self.best_auc_ * 2 - 1) if self.best_auc_ else None
        
        return metrics


class DualRiskModelPipeline(RiskModelPipeline):
    """
    Dual Pipeline that automatically runs both WOE and RAW pipelines
    and selects the best performing model.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize dual pipeline with config ensuring dual mode is enabled."""
        super().__init__(config)
        self.config.enable_dual_pipeline = True

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the dual pipeline on training data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target

        Returns:
        --------
        dict
            Dictionary containing results from both pipelines
        """
        print("Starting Dual Risk Model Pipeline (WOE + RAW)...")

        # Run base pipeline which handles dual mode internally
        results = super().run(df)

        # The ModelBuilder already handles dual pipeline logic
        # when enable_dual_pipeline is True in config

        return results
