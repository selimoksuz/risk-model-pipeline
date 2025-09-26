"""
Report Generation Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ReportGenerator:
    """Generate comprehensive model reports"""
    
    def __init__(self, config):
        self.config = config
        
    def generate_report(
        self,
        report_data: Dict[str, Any],
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame],
        oot_data: pd.DataFrame,
        output_dir: Path
    ) -> Path:
        """
        Generate comprehensive report
        
        Parameters:
        -----------
        report_data : Dict
            All model results and metrics
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Test data
        oot_data : pd.DataFrame
            Out-of-time data
        output_dir : Path
            Output directory
            
        Returns:
        --------
        Path
            Path to generated report
        """
        
        # Create Excel writer
        report_path = output_dir / f"{self.config.model_name_prefix}_report.xlsx"
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            
            # 1. Model Comparison
            if 'model_comparison' in self.config.report_components:
                model_comparison = self._create_model_comparison(report_data['metrics'])
                model_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # 2. Feature Importance
            if 'feature_importance' in self.config.report_components:
                feature_importance = self._create_feature_importance(
                    report_data['selected_features'],
                    report_data['univariate_stats']
                )
                feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            
            # 3. WOE Bins
            if 'woe_bins' in self.config.report_components and report_data.get('woe_bins'):
                self._write_woe_bins(writer, report_data['woe_bins'])
            
            # 4. Univariate Analysis
            if 'univariate_analysis' in self.config.report_components:
                univariate_df = self._create_univariate_summary(report_data['univariate_stats'])
                univariate_df.to_excel(writer, sheet_name='Univariate Analysis', index=False)
            
            # 5. Risk Bands
            if 'risk_bands' in self.config.report_components and report_data.get('risk_bands'):
                risk_bands_df = report_data['risk_bands'].get('band_stats', pd.DataFrame())
                if not risk_bands_df.empty:
                    risk_bands_df.to_excel(writer, sheet_name='Risk Bands', index=False)
            
            # 6. Statistical Tests
            if 'statistical_tests' in self.config.report_components and report_data.get('risk_bands'):
                tests_df = self._create_statistical_tests_summary(report_data['risk_bands'])
                tests_df.to_excel(writer, sheet_name='Statistical Tests', index=False)
        
            # Variable dictionary
            if (
                self.config.include_variable_dictionary
                and report_data.get('variable_dictionary') is not None
            ):
                self._write_variable_dictionary(
                    writer,
                    report_data['variable_dictionary'],
                    report_data.get('selected_features', {}),
                    report_data.get('best_model')
                )

        print(f"Report saved to: {report_path}")
        return report_path
    
    def _create_model_comparison(self, metrics: Dict) -> pd.DataFrame:
        """Create model comparison table"""
        
        comparison_data = []
        
        for model_name, model_metrics in metrics.items():
            row = {'Model': model_name}
            
            # Add metrics for each dataset
            for dataset_name, dataset_metrics in model_metrics.items():
                row[f'{dataset_name}_auc'] = dataset_metrics.get('auc', 0)
                row[f'{dataset_name}_gini'] = dataset_metrics.get('gini', 0)
                row[f'{dataset_name}_ks'] = dataset_metrics.get('ks_statistic', 0)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by OOT AUC if available
        if 'oot_auc' in df.columns:
            df = df.sort_values('oot_auc', ascending=False)
        
        return df
    
    def _create_feature_importance(
        self,
        selected_features: Dict,
        univariate_stats: Dict
    ) -> pd.DataFrame:
        """Create feature importance summary"""
        
        importance_data = []
        
        # Get all unique features
        all_features = set()
        for features in selected_features.values():
            all_features.update(features)
        
        for feature in all_features:
            row = {'Feature': feature}
            
            # Add univariate stats
            if feature in univariate_stats:
                stats = univariate_stats[feature]
                row['IV'] = stats.get('iv', 0)
                row['Raw_Gini'] = stats.get('raw_gini', 0)
                row['WOE_Gini'] = stats.get('woe_gini', 0)
                row['WOE_Degradation'] = stats.get('woe_degradation', False)
            
            # Check which models use this feature
            models_using = []
            for model_name, features in selected_features.items():
                if feature in features:
                    models_using.append(model_name)
            
            row['Used_In_Models'] = ', '.join(models_using)
            row['N_Models'] = len(models_using)
            
            importance_data.append(row)
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('IV', ascending=False)
        
        return df
    
    def _write_woe_bins(self, writer, woe_transformers: Dict):
        """Write WOE bins to Excel"""
        
        for i, (feature, transformer) in enumerate(woe_transformers.items()):
            if i >= 20:  # Limit number of sheets
                break
            
            woe_df = transformer['woe_df']
            sheet_name = f'WOE_{feature[:20]}'  # Limit sheet name length
            
            # Clean sheet name
            for char in ['/', '\\', '?', '*', '[', ']', ':']:
                sheet_name = sheet_name.replace(char, '_')
            
            try:
                woe_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except:
                pass
    
    def _create_univariate_summary(self, univariate_stats: Dict) -> pd.DataFrame:
        """Create univariate analysis summary"""
        
        summary_data = []
        
        for feature, stats in univariate_stats.items():
            row = {
                'Feature': feature,
                'IV': stats.get('iv', 0),
                'Raw_Gini': stats.get('raw_gini', 0),
                'WOE_Gini': stats.get('woe_gini', 0),
                'Gini_Difference': stats.get('woe_gini', 0) - stats.get('raw_gini', 0),
                'WOE_Degradation': stats.get('woe_degradation', False)
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('IV', ascending=False)
        
        return df
    
    def _create_statistical_tests_summary(self, risk_bands: Dict) -> pd.DataFrame:
        """Create statistical tests summary"""
        
        test_results = risk_bands.get('test_results', {})
        
        summary_data = []
        
        # Binomial test
        if 'binomial' in test_results:
            binomial = test_results['binomial']
            summary_data.append({
                'Test': 'Binomial Test',
                'Result': 'Calibrated' if binomial.get('overall_calibrated', False) else 'Not Calibrated',
                'N_Significant_Bands': binomial.get('n_significant', 0),
                'Details': f"{binomial.get('n_significant', 0)} bands significant at 0.05 level"
            })
        
        # Hosmer-Lemeshow test
        if 'hosmer_lemeshow' in test_results:
            hl = test_results['hosmer_lemeshow']
            summary_data.append({
                'Test': 'Hosmer-Lemeshow',
                'Result': 'Calibrated' if hl.get('calibrated', False) else 'Not Calibrated',
                'Statistic': hl.get('statistic', 0),
                'P_Value': hl.get('p_value', 0)
            })
        
        # Herfindahl Index
        if 'herfindahl' in test_results:
            hhi = test_results['herfindahl']
            summary_data.append({
                'Test': 'Herfindahl Index',
                'Result': hhi.get('concentration', 'unknown'),
                'HHI': hhi.get('hhi', 0),
                'Normalized_HHI': hhi.get('normalized_hhi', 0)
            })
        
        # Monotonicity
        if 'monotonicity' in risk_bands:
            mono = risk_bands['monotonicity']
            summary_data.append({
                'Test': 'Monotonicity',
                'Result': 'Monotonic' if mono.get('is_monotonic', False) else 'Not Monotonic',
                'Spearman_Correlation': mono.get('spearman_correlation', 0),
                'P_Value': mono.get('correlation_p_value', 0)
            })
        
        return pd.DataFrame(summary_data)

    def _write_variable_dictionary(
        self,
        writer,
        variable_dictionary,
        selected_features: Dict[str, List[str]],
        best_model: Optional[str]
    ) -> None:
        """Append variable dictionary entries used by the best model"""

        if variable_dictionary is None:
            return

        if isinstance(variable_dictionary, pd.DataFrame):
            dict_df = variable_dictionary.copy()
        else:
            dict_df = pd.DataFrame(variable_dictionary)

        if dict_df.empty:
            return

        candidate_columns = ['Feature', 'feature', 'variable', 'Variable', 'name', 'Name']
        feature_col = None
        for col in candidate_columns:
            if col in dict_df.columns:
                feature_col = col
                break
        if feature_col is None:
            feature_col = dict_df.columns[0]

        best_features: List[str] = []
        if best_model and best_model in selected_features:
            best_features = selected_features.get(best_model, []) or []
        elif selected_features:
            first_key = next(iter(selected_features))
            best_features = selected_features.get(first_key, []) or []

        dict_df['Feature'] = dict_df[feature_col].astype(str)
        dict_df['Used_In_Best_Model'] = dict_df['Feature'].isin(best_features)
        if best_features:
            dict_df = dict_df[dict_df['Used_In_Best_Model']]

        if dict_df.empty:
            return

        dict_df.to_excel(writer, sheet_name='Variable Dictionary', index=False)
