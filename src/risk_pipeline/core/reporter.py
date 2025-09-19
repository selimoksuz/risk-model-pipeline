"""
Enhanced Reporter with comprehensive reporting capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class EnhancedReporter:
    """
    Comprehensive reporting module with WOE tables, univariate Gini,
    SHAP importance, and data dictionary integration.
    """

    def __init__(self, config):
        self.config = config
        self.reports_ = {}

    def generate_model_report(self, models: Dict, data_dictionary: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive model performance report."""

        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_scores': {},
            'best_model': None,
            'feature_count': 0
        }

        # Model performance summary
        if 'scores' in models:
            report['model_scores'] = models['scores']

        # Best model identification
        if 'best_model_name' in models:
            report['best_model'] = models['best_model_name']
            if 'scores' in models and models['best_model_name'] in models['scores']:
                report['best_auc'] = models['scores'][models['best_model_name']].get('test_auc', 0)

        # Feature count
        if 'selected_features' in models:
            report['feature_count'] = len(models['selected_features'])

        self.reports_['model_performance'] = report
        return report

    def generate_feature_report(self, models: Dict, woe_results: Dict,
                               data_dictionary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate feature importance report with WOE, univariate Gini, and descriptions.
        """

        features_data = []

        # Get selected features
        selected_features = models.get('selected_features', [])

        for feature in selected_features:
            feature_info = {
                'feature': feature,
                'description': self._get_feature_description(feature, data_dictionary),
                'category': self._get_feature_category(feature, data_dictionary)
            }

            # Add WOE information
            if feature in woe_results.get('woe_values', {}):
                woe_info = woe_results['woe_values'][feature]
                feature_info['iv'] = woe_info.get('iv', 0)
                feature_info['woe_min'] = min(woe_info.get('woe_map', {}).values()) if woe_info.get('woe_map') else 0
                feature_info['woe_max'] = max(woe_info.get('woe_map', {}).values()) if woe_info.get('woe_map') else 0
                feature_info['n_bins'] = len(woe_info.get('bins', [])) if woe_info.get('type') == 'numeric' else len(woe_info.get('categories', []))

            # Add univariate Gini
            if feature in woe_results.get('univariate_gini', {}):
                gini_info = woe_results['univariate_gini'][feature]
                feature_info['gini_raw'] = gini_info.get('gini_raw', 0)
                feature_info['gini_woe'] = gini_info.get('gini_woe', 0)
                feature_info['gini_drop'] = gini_info.get('gini_drop', 0)

            # Add model importance
            if 'feature_importance' in models:
                for model_name, importance_df in models['feature_importance'].items():
                    if not importance_df.empty and feature in importance_df['feature'].values:
                        imp_value = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                        feature_info[f'importance_{model_name}'] = imp_value

            features_data.append(feature_info)

        # Create DataFrame and sort by IV
        features_df = pd.DataFrame(features_data)
        if 'iv' in features_df.columns:
            features_df = features_df.sort_values('iv', ascending=False)

        self.reports_['features'] = features_df
        return features_df

    def generate_woe_tables(self, woe_results: Dict, selected_features: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate detailed WOE tables for each feature."""

        woe_tables = {}

        for feature in selected_features:
            if feature in woe_results.get('woe_values', {}):
                woe_info = woe_results['woe_values'][feature]

                if 'stats' in woe_info:
                    # Convert stats to DataFrame
                    woe_table = pd.DataFrame(woe_info['stats'])

                    # Add additional columns
                    woe_table['feature'] = feature
                    woe_table['type'] = woe_info.get('type', 'unknown')

                    # Sort by WOE
                    woe_table = woe_table.sort_values('woe')

                    woe_tables[feature] = woe_table

        self.reports_['woe_tables'] = woe_tables
        return woe_tables

    def generate_risk_band_report(self, risk_bands: Dict) -> pd.DataFrame:
        """Generate risk band analysis report."""

        if 'bands' not in risk_bands:
            return pd.DataFrame()

        bands_df = risk_bands['bands']
        metrics = risk_bands.get('metrics', {})

        # Create summary
        summary = {
            'Risk Bands Summary': [
                f"Number of bands: {len(bands_df)}",
                f"Herfindahl Index: {metrics.get('herfindahl_index', 0):.4f}",
                f"Entropy: {metrics.get('entropy', 0):.4f}",
                f"Gini Coefficient: {metrics.get('gini_coefficient', 0):.4f}",
                f"KS Statistic: {metrics.get('ks_stat', 0):.4f}",
                f"Hosmer-Lemeshow p-value: {metrics.get('hosmer_lemeshow_p', 0):.4f}",
                f"Top 20% Concentration: {metrics.get('cr_top20', 0):.2%}",
                f"Top 50% Concentration: {metrics.get('cr_top50', 0):.2%}"
            ]
        }

        # Add binomial test results
        if 'binomial_tests' in metrics:
            n_significant = sum(1 for band_result in metrics['binomial_tests'].values()
                              if band_result.get('significant', False))
            summary['Risk Bands Summary'].append(
                f"Significant bands (binomial test): {n_significant}/{len(metrics['binomial_tests'])}"
            )

        self.reports_['risk_bands'] = bands_df
        self.reports_['risk_bands_summary'] = summary

        return bands_df

    def generate_calibration_report(self, stage1_results: Optional[Dict],
                                   stage2_results: Optional[Dict]) -> Dict:
        """Generate calibration analysis report."""

        report = {}

        if stage1_results:
            report['stage1'] = {
                'method': self.config.calibration_method,
                'metrics': stage1_results.get('calibration_metrics', {})
            }

        if stage2_results:
            report['stage2'] = {
                'method': self.config.stage2_method if hasattr(self.config, 'stage2_method') else 'N/A',
                'metrics': stage2_results.get('stage2_metrics', {})
            }

        self.reports_['calibration'] = report
        return report

    def generate_final_summary(self) -> str:
        """Generate final text summary of all results."""

        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("RISK MODEL PIPELINE - FINAL SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append("")

        # Model Performance
        if 'model_performance' in self.reports_:
            mp = self.reports_['model_performance']
            summary_lines.append("MODEL PERFORMANCE:")
            summary_lines.append(f"  Best Model: {mp.get('best_model', 'N/A')}")
            summary_lines.append(f"  Best AUC: {mp.get('best_auc', 0):.4f}")
            summary_lines.append(f"  Features Used: {mp.get('feature_count', 0)}")
            summary_lines.append("")

        # Top Features
        if 'features' in self.reports_:
            features_df = self.reports_['features']
            summary_lines.append("TOP 5 FEATURES BY IV:")
            for i, row in features_df.head(5).iterrows():
                summary_lines.append(
                    f"  {i+1}. {row['feature']}: IV={row.get('iv', 0):.4f}, "
                    f"Gini={row.get('gini_woe', 0):.4f}"
                )
            summary_lines.append("")

        # Risk Bands
        if 'risk_bands_summary' in self.reports_:
            summary_lines.append("RISK BANDS ANALYSIS:")
            for line in self.reports_['risk_bands_summary']['Risk Bands Summary']:
                summary_lines.append(f"  {line}")
            summary_lines.append("")

        # Calibration
        if 'calibration' in self.reports_:
            cal = self.reports_['calibration']
            summary_lines.append("CALIBRATION:")
            if 'stage1' in cal:
                stage1_metrics = cal['stage1'].get('metrics', {})
                summary_lines.append(f"  Stage 1 ECE: {stage1_metrics.get('ece', 0):.4f}")
            if 'stage2' in cal:
                stage2_metrics = cal['stage2'].get('metrics', {})
                summary_lines.append(f"  Stage 2 ECE: {stage2_metrics.get('ece', 0):.4f}")
            summary_lines.append("")

        summary_lines.append("="*80)

        return "\n".join(summary_lines)

    def export_to_excel(self, filepath: str):
        """Export all reports to Excel file."""

        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Features report
            if 'features' in self.reports_:
                self.reports_['features'].to_excel(
                    writer, sheet_name='Features', index=False
                )

            # WOE tables
            if 'woe_tables' in self.reports_:
                for i, (feature, woe_table) in enumerate(self.reports_['woe_tables'].items()):
                    sheet_name = f'WOE_{i+1}' if i < 10 else f'WOE_{feature[:20]}'
                    woe_table.to_excel(writer, sheet_name=sheet_name, index=False)

            # Risk bands
            if 'risk_bands' in self.reports_:
                self.reports_['risk_bands'].to_excel(
                    writer, sheet_name='Risk_Bands', index=False
                )

            # Model scores
            if 'model_performance' in self.reports_:
                scores_df = pd.DataFrame(self.reports_['model_performance']['model_scores']).T
                scores_df.to_excel(writer, sheet_name='Model_Scores')

        print(f"Reports exported to {filepath}")

    def _get_feature_description(self, feature: str, data_dictionary: Optional[pd.DataFrame]) -> str:
        """Get feature description from data dictionary."""

        if data_dictionary is None:
            return ""

        if 'variable' in data_dictionary.columns and 'description' in data_dictionary.columns:
            match = data_dictionary[data_dictionary['variable'] == feature]
            if not match.empty:
                return match.iloc[0]['description']

        return ""

    def _get_feature_category(self, feature: str, data_dictionary: Optional[pd.DataFrame]) -> str:
        """Get feature category from data dictionary."""

        if data_dictionary is None:
            return ""

        if 'variable' in data_dictionary.columns and 'category' in data_dictionary.columns:
            match = data_dictionary[data_dictionary['variable'] == feature]
            if not match.empty:
                return match.iloc[0]['category']

        return ""

    def create_model_comparison_plot(self, models: Dict):
        """Create model comparison visualization."""

        if not models or 'scores' not in models:
            return None

        scores_data = []
        for model_name, model_info in models.items():
            if 'scores' in model_info:
                scores_data.append({
                    'Model': model_name,
                    'Train AUC': model_info['scores'].get('train_auc', 0),
                    'Test AUC': model_info['scores'].get('test_auc', 0)
                })

        if not scores_data:
            return None

        scores_df = pd.DataFrame(scores_data)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(scores_df))
        width = 0.35

        train_bars = ax.bar(x - width/2, scores_df['Train AUC'], width, label='Train AUC')
        test_bars = ax.bar(x + width/2, scores_df['Test AUC'], width, label='Test AUC')

        ax.set_xlabel('Model')
        ax.set_ylabel('AUC Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scores_df['Model'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bars in [train_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        plt.tight_layout()
        return fig
