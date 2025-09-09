"""Reporting module with segment-based PSI and calibration analysis"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

from .psi_calculator import PSICalculator
from .calibration_analyzer import CalibrationAnalyzer


class Reporter:
    """Handles report generation including segment analysis and calibration"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_folder)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.psi_calculator = PSICalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        
    def generate_reports(self, train, test=None, oot=None, model=None, 
                        features=None, woe_mapping=None, model_name=None, scores=None):
        """Generate comprehensive reports with segment analysis"""
        
        print("\nGenerating reports...")
        reports = {}
        
        # Model performance summary
        if scores:
            reports['model_summary'] = self._create_model_summary(scores)
        
        # Feature importance
        if model and features:
            reports['feature_importance'] = self._get_feature_importance(model, features)
        
        # WOE bins report
        if woe_mapping:
            reports['woe_bins'] = self._create_woe_report(woe_mapping)
        
        # PSI Analysis
        if train is not None and (test is not None or oot is not None):
            print("  Calculating PSI...")
            
            # WOE-based PSI for WOE variables
            if woe_mapping and features:
                X_train_woe = train[features] if features else train
                if test is not None:
                    X_test_woe = test[features] if features else test
                    reports['woe_psi_test'] = self.psi_calculator.calculate_woe_psi(
                        X_train_woe, X_test_woe, woe_mapping
                    )
                if oot is not None:
                    X_oot_woe = oot[features] if features else oot
                    reports['woe_psi_oot'] = self.psi_calculator.calculate_woe_psi(
                        X_train_woe, X_oot_woe, woe_mapping
                    )
            
            # Score PSI (model predictions)
            if model:
                X_train = train[features] if features else train
                train_scores = model.predict_proba(X_train)[:, 1]
                
                if test is not None:
                    X_test = test[features] if features else test
                    test_scores = model.predict_proba(X_test)[:, 1]
                    psi_value, psi_df = self.psi_calculator.calculate_score_psi(
                        train_scores, test_scores, n_bins=10
                    )
                    reports['score_psi_test'] = {
                        'psi_value': psi_value,
                        'segments': psi_df,
                        'interpretation': self.psi_calculator._interpret_psi(psi_value)
                    }
                
                if oot is not None:
                    X_oot = oot[features] if features else oot
                    oot_scores = model.predict_proba(X_oot)[:, 1]
                    psi_value, psi_df = self.psi_calculator.calculate_score_psi(
                        train_scores, oot_scores, n_bins=10
                    )
                    reports['score_psi_oot'] = {
                        'psi_value': psi_value,
                        'segments': psi_df,
                        'interpretation': self.psi_calculator._interpret_psi(psi_value)
                    }
        
        # Calibration analysis with risk bands
        if model and oot is not None:
            print("  Performing calibration analysis...")
            X_oot = oot[features] if features else oot
            y_true = oot[self.config.target_col]
            y_pred = model.predict_proba(X_oot)[:, 1]
            
            # Analyze with risk bands
            reports['calibration_risk_bands'] = self.calibration_analyzer.analyze_calibration(
                y_true, y_pred, use_deciles=False
            )
            
            # Also analyze with deciles for comparison
            reports['calibration_deciles'] = self.calibration_analyzer.analyze_calibration(
                y_true, y_pred, use_deciles=True
            )
        
        # Save reports
        self._save_reports(reports)
        
        return reports
    
    def _calculate_score_psi_segments_OLD(self, model, train, test=None, oot=None, features=None):
        """Calculate PSI for model scores with segment details"""
        
        results = {}
        
        # Get predictions
        X_train = train[features] if features else train
        train_scores = model.predict_proba(X_train)[:, 1]
        
        if test is not None:
            X_test = test[features] if features else test
            test_scores = model.predict_proba(X_test)[:, 1]
            
            # Calculate segment PSI
            psi_value, segment_df = self._compute_segment_psi(
                train_scores, test_scores, n_bins=10
            )
            
            results['test'] = {
                'psi_value': psi_value,
                'segments': segment_df,
                'interpretation': self._interpret_psi(psi_value)
            }
        
        if oot is not None:
            X_oot = oot[features] if features else oot
            oot_scores = model.predict_proba(X_oot)[:, 1]
            
            # Calculate segment PSI
            psi_value, segment_df = self._compute_segment_psi(
                train_scores, oot_scores, n_bins=10
            )
            
            results['oot'] = {
                'psi_value': psi_value,
                'segments': segment_df,
                'interpretation': self._interpret_psi(psi_value)
            }
        
        return results
    
    def _compute_segment_psi(self, baseline_scores, comparison_scores, n_bins=10):
        """Compute PSI with segment breakdown"""
        
        # Create bins from baseline
        _, bin_edges = pd.qcut(baseline_scores, q=n_bins, retbins=True, duplicates='drop')
        bin_edges[0] = 0  # Ensure coverage
        bin_edges[-1] = 1
        
        # Bin both distributions
        baseline_binned = pd.cut(baseline_scores, bins=bin_edges, include_lowest=True)
        comparison_binned = pd.cut(comparison_scores, bins=bin_edges, include_lowest=True)
        
        # Calculate distributions
        baseline_dist = baseline_binned.value_counts() / len(baseline_scores)
        comparison_dist = comparison_binned.value_counts() / len(comparison_scores)
        
        # Align indices
        all_bins = sorted(set(baseline_dist.index) | set(comparison_dist.index))
        
        segments = []
        total_psi = 0
        
        for i, bin_range in enumerate(all_bins):
            base_pct = baseline_dist.get(bin_range, 0.0001)
            comp_pct = comparison_dist.get(bin_range, 0.0001)
            
            # PSI calculation
            psi_contrib = (comp_pct - base_pct) * np.log(comp_pct / base_pct)
            total_psi += psi_contrib
            
            # Get actual values in this segment
            base_mask = baseline_binned == bin_range
            comp_mask = comparison_binned == bin_range
            
            segments.append({
                'segment': i + 1,
                'range': str(bin_range),
                'baseline_pct': base_pct * 100,
                'comparison_pct': comp_pct * 100,
                'baseline_count': base_mask.sum(),
                'comparison_count': comp_mask.sum(),
                'difference_pct': (comp_pct - base_pct) * 100,
                'psi_contribution': psi_contrib,
                'baseline_avg_score': baseline_scores[base_mask].mean() if base_mask.any() else 0,
                'comparison_avg_score': comparison_scores[comp_mask].mean() if comp_mask.any() else 0
            })
        
        segment_df = pd.DataFrame(segments)
        segment_df['cumulative_psi'] = segment_df['psi_contribution'].cumsum()
        
        return total_psi, segment_df
    
    def _calibration_analysis(self, model, data, features):
        """Perform calibration analysis with segments"""
        
        # Get predictions
        X = data[features] if features else data
        y_true = data[self.config.target_col]
        y_pred = model.predict_proba(X)[:, 1]
        
        # Create calibration segments
        n_bins = 10
        segments = []
        
        # Sort by predicted probability
        sorted_idx = np.argsort(y_pred)
        y_true_sorted = y_true.iloc[sorted_idx].values
        y_pred_sorted = y_pred[sorted_idx]
        
        bin_size = len(y_pred) // n_bins
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
            
            bin_y_true = y_true_sorted[start_idx:end_idx]
            bin_y_pred = y_pred_sorted[start_idx:end_idx]
            
            # Calculate statistics
            mean_predicted = bin_y_pred.mean()
            mean_actual = bin_y_true.mean()
            count = len(bin_y_true)
            events = bin_y_true.sum()
            
            # Binomial test for calibration
            from scipy import stats
            expected_events = mean_predicted * count
            
            # Binomial test
            if count > 0:
                p_value = stats.binom_test(events, count, mean_predicted, alternative='two-sided')
            else:
                p_value = 1.0
            
            # Confidence interval for actual rate
            if count > 0:
                conf_int = stats.binom.interval(0.95, count, mean_predicted) / count
            else:
                conf_int = (0, 0)
            
            segments.append({
                'decile': i + 1,
                'score_range': f"{bin_y_pred.min():.3f} - {bin_y_pred.max():.3f}",
                'count': count,
                'events': events,
                'mean_predicted_pd': mean_predicted * 100,
                'actual_default_rate': mean_actual * 100,
                'difference': (mean_actual - mean_predicted) * 100,
                'expected_events': expected_events,
                'ci_lower': conf_int[0] * 100,
                'ci_upper': conf_int[1] * 100,
                'p_value': p_value,
                'calibrated': 'Yes' if p_value > 0.05 else 'No'
            })
        
        calibration_df = pd.DataFrame(segments)
        
        # Calculate overall metrics
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
        brier_score = brier_score_loss(y_true, y_pred)
        
        # Expected Calibration Error (ECE)
        ece = 0
        for i in range(len(fraction_pos)):
            bin_size = len(y_pred) // len(fraction_pos)
            ece += (bin_size / len(y_pred)) * abs(fraction_pos[i] - mean_pred[i])
        
        # Hosmer-Lemeshow test
        from scipy.stats import chi2
        hl_statistic = 0
        for _, row in calibration_df.iterrows():
            expected = row['count'] * row['mean_predicted_pd'] / 100
            observed = row['events']
            if expected > 0:
                hl_statistic += (observed - expected) ** 2 / expected
        
        hl_p_value = 1 - chi2.cdf(hl_statistic, df=n_bins - 2)
        
        return {
            'segments': calibration_df,
            'brier_score': brier_score,
            'ece': ece,
            'hosmer_lemeshow': {
                'statistic': hl_statistic,
                'p_value': hl_p_value,
                'calibrated': 'Yes' if hl_p_value > 0.05 else 'No'
            },
            'calibration_curve': {
                'fraction_positive': fraction_pos.tolist(),
                'mean_predicted': mean_pred.tolist()
            }
        }
    
    def _interpret_psi(self, psi_value):
        """Interpret PSI value"""
        if psi_value < 0.1:
            return "Insignificant change (PSI < 0.1)"
        elif psi_value < 0.25:
            return "Small change (0.1 ≤ PSI < 0.25)"
        else:
            return "Significant change (PSI ≥ 0.25) - Investigation needed"
    
    def _create_model_summary(self, scores):
        """Create model performance summary"""
        summary = []
        for model_name, model_scores in scores.items():
            row = {'model': model_name}
            row.update(model_scores)
            summary.append(row)
        return pd.DataFrame(summary)
    
    def _get_feature_importance(self, model, features):
        """Get feature importance from model"""
        importance_df = pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'coefficient': model.coef_[0]
            }).sort_values('coefficient', ascending=False, key=lambda x: abs(x))
        
        return importance_df
    
    def _create_woe_report(self, woe_mapping):
        """Create WOE bins report"""
        woe_reports = {}
        
        for feature, mapping in woe_mapping.items():
            if hasattr(mapping, 'numeric_bins'):
                # Numeric variable
                bins_data = []
                for bin_info in mapping.numeric_bins:
                    bins_data.append({
                        'bin': f"[{bin_info.left:.2f}, {bin_info.right:.2f}]",
                        'woe': bin_info.woe,
                        'event_count': bin_info.event_count,
                        'nonevent_count': bin_info.nonevent_count,
                        'event_rate': bin_info.event_rate,
                        'iv_contribution': bin_info.iv_contrib
                    })
                woe_reports[feature] = pd.DataFrame(bins_data)
            
            elif hasattr(mapping, 'categorical_groups'):
                # Categorical variable
                groups_data = []
                for group in mapping.categorical_groups:
                    groups_data.append({
                        'group': group.label,
                        'members': ', '.join(map(str, group.members[:5])) + ('...' if len(group.members) > 5 else ''),
                        'woe': group.woe,
                        'event_count': group.event_count,
                        'nonevent_count': group.nonevent_count,
                        'event_rate': group.event_rate,
                        'iv_contribution': group.iv_contrib
                    })
                woe_reports[feature] = pd.DataFrame(groups_data)
        
        return woe_reports
    
    def _save_reports(self, reports):
        """Save reports to files"""
        
        # Save to Excel
        excel_path = self.output_dir / 'model_report.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for name, report in reports.items():
                if isinstance(report, pd.DataFrame):
                    report.to_excel(writer, sheet_name=name[:31], index=False)
                elif isinstance(report, dict):
                    # Handle nested dictionaries
                    for sub_name, sub_report in report.items():
                        if isinstance(sub_report, pd.DataFrame):
                            sheet_name = f"{name}_{sub_name}"[:31]
                            sub_report.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"  Reports saved to: {excel_path}")
        
        # Save PSI summary
        if 'score_psi' in reports:
            psi_summary = []
            for dataset, psi_data in reports['score_psi'].items():
                psi_summary.append({
                    'dataset': dataset,
                    'psi_value': psi_data['psi_value'],
                    'interpretation': psi_data['interpretation']
                })
            
            if psi_summary:
                psi_df = pd.DataFrame(psi_summary)
                psi_path = self.output_dir / 'psi_summary.csv'
                psi_df.to_csv(psi_path, index=False)
                print(f"  PSI summary saved to: {psi_path}")
        
        # Save calibration summary
        if 'calibration' in reports:
            cal_summary = {
                'brier_score': reports['calibration']['brier_score'],
                'ece': reports['calibration']['ece'],
                'hosmer_lemeshow_pvalue': reports['calibration']['hosmer_lemeshow']['p_value'],
                'calibrated': reports['calibration']['hosmer_lemeshow']['calibrated']
            }
            
            cal_path = self.output_dir / 'calibration_summary.json'
            import json
            with open(cal_path, 'w') as f:
                json.dump(cal_summary, f, indent=2)
            print(f"  Calibration summary saved to: {cal_path}")