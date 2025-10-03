"""Enhanced Reporter with comprehensive reporting capabilities."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


class EnhancedReporter:
    """Generates structured outputs for model performance and diagnostics."""

    def __init__(self, config: Any):
        self.config = config
        self.reports_: Dict[str, Any] = {}
        self.data_dictionary: Optional[pd.DataFrame] = None
        self.tsfresh_metadata: Optional[pd.DataFrame] = None
        self.selection_history: Optional[pd.DataFrame] = None

    def register_data_dictionary(self, dictionary: Optional[pd.DataFrame]) -> None:
        '''Store a normalized data dictionary for later lookups.'''
        normalized = self._normalize_data_dictionary(dictionary)
        if normalized is None:
            self.data_dictionary = None
            return



        self.data_dictionary = normalized
        self.reports_['data_dictionary'] = normalized


    def register_tsfresh_metadata(self, metadata: Optional[pd.DataFrame]) -> None:
        """Keep a copy of tsfresh-derived feature metadata for reporting."""

        if metadata is None:
            self.tsfresh_metadata = None
            self.reports_.pop('tsfresh_metadata', None)
            return

        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        meta = metadata.copy()
        if meta.empty:
            self.tsfresh_metadata = meta
            self.reports_['tsfresh_metadata'] = meta
            return

        rename_map: Dict[str, str] = {}
        for col in meta.columns:
            key = str(col).strip().lower()
            if key in {'base_variable', 'raw_feature', 'raw_variable', 'source', 'source_var'}:
                rename_map[col] = 'source_variable'
            elif key in {'stat', 'stats', 'aggregation'}:
                rename_map[col] = 'statistic'
            elif key in {'generator', 'method'}:
                rename_map[col] = 'generator'
        meta = meta.rename(columns=rename_map)
        if 'feature' in meta.columns:
            meta['feature'] = meta['feature'].astype(str)
        self.tsfresh_metadata = meta
        self.reports_['tsfresh_metadata'] = meta

    def register_selection_history(self, selection_results: Optional[Dict[str, Any]]) -> None:
        """Persist feature selection progression as a flat table."""

        history = None
        if isinstance(selection_results, dict):
            history = selection_results.get('selection_history')
            vif_summary = selection_results.get('vif_summary')
            if isinstance(vif_summary, pd.DataFrame) and not vif_summary.empty:
                self.reports_['vif_summary'] = vif_summary.copy()
            corr_clusters = selection_results.get('correlation_clusters')
            if isinstance(corr_clusters, pd.DataFrame) and not corr_clusters.empty:
                self.reports_['correlation_clusters'] = corr_clusters.copy()
        if not history:
            self.selection_history = None
            self.reports_.pop('selection_history', None)
            return

        rows: List[Dict[str, Any]] = []
        for step in history:
            if not isinstance(step, dict):
                continue
            removed = step.get('removed')
            if isinstance(removed, set):
                removed_list = sorted(str(item) for item in removed)
            elif isinstance(removed, (list, tuple)):
                removed_list = [str(item) for item in removed]
            elif removed is None:
                removed_list = []
            else:
                removed_list = [str(removed)]
            rows.append({
                'method': step.get('method'),
                'before': step.get('before'),
                'after': step.get('after'),
                'removed_count': len(removed_list),
                'removed_features': ', '.join(removed_list),
                'details': step.get('details'),
            })
        df = pd.DataFrame(rows)
        self.selection_history = df
        self.reports_['selection_history'] = df


    @staticmethod
    def _binomial_results_to_df(results: Any) -> pd.DataFrame:
        """Normalise binomial test outputs into a DataFrame."""

        if results is None:
            return pd.DataFrame()

        if isinstance(results, pd.DataFrame):
            df = results.copy()
        elif isinstance(results, (list, tuple)) and len(results) == 0:
            return pd.DataFrame()
        elif isinstance(results, dict):
            rows: List[Dict[str, Any]] = []
            for band, payload in results.items():
                row: Dict[str, Any] = {'band': band}
                if isinstance(payload, dict):
                    row.update(payload)
                else:
                    row['value'] = payload
                rows.append(row)
            df = pd.DataFrame(rows)
        elif isinstance(results, (list, tuple)):
            df = pd.DataFrame(list(results))
        else:
            df = pd.DataFrame()

        if df.empty:
            return df

        if 'band' not in df.columns:
            df.insert(0, 'band', range(1, len(df) + 1))
        else:
            df['band'] = pd.to_numeric(df['band'], errors='ignore')
        if 'significant' in df.columns:
            df['significant'] = df['significant'].astype(bool)
        return df


    def generate_model_report(
        self,
        models: Dict[str, Any],
        data_dictionary: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Build detailed model performance artefacts."""

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_scores": models.get("scores", {}),
            "best_model": models.get("best_model_name"),
            "feature_count": len(models.get("selected_features", [])),
        }

        model_scores = report["model_scores"] if isinstance(report["model_scores"], dict) else {}
        if model_scores:
            models_summary_df = pd.DataFrame(model_scores).T
            models_summary_df.index.name = 'model_name'
            models_summary_df = models_summary_df.reset_index()
            self.reports_['models_summary'] = models_summary_df
        else:
            models_summary_df = pd.DataFrame()
            self.reports_.pop('models_summary', None)

        best_model = report["best_model"]
        best_model_df = pd.DataFrame()
        if best_model:
            if not models_summary_df.empty:
                best_model_df = models_summary_df.loc[models_summary_df['model_name'] == best_model].copy()
            if best_model_df.empty:
                best_metrics = model_scores.get(best_model, {}) if isinstance(model_scores, dict) else {}
                best_model_df = pd.DataFrame([{**{'model_name': best_model}, **best_metrics}])
            self.reports_['best_model'] = best_model_df.reset_index(drop=True)
            if 'test_auc' in best_model_df.columns and not best_model_df.empty:
                try:
                    report["best_auc"] = float(best_model_df['test_auc'].iloc[0])
                except Exception:
                    report["best_auc"] = best_model_df['test_auc'].iloc[0]
        else:
            self.reports_.pop('best_model', None)

        selected_features = models.get("selected_features", []) or []
        feature_importance = models.get("feature_importance", {}) if isinstance(models.get("feature_importance"), dict) else {}
        best_importance_df = feature_importance.get(best_model) if best_model else None
        best_model_vars_df = pd.DataFrame()
        if isinstance(best_importance_df, pd.DataFrame) and not best_importance_df.empty:
            importance_df = best_importance_df.copy()
            if 'importance' in importance_df.columns:
                importance_df = importance_df.sort_values('importance', ascending=False)
            self.reports_['shap_importance'] = importance_df.reset_index(drop=True)
            if 'importance' in importance_df.columns and 'feature' in importance_df.columns:
                shap_summary = (
                    importance_df[['feature', 'importance']]
                    .groupby('feature', as_index=False)['importance']
                    .sum()
                    .sort_values('importance', ascending=False)
                )
                self.reports_['shap_summary'] = shap_summary.reset_index(drop=True)
            best_model_vars_df = importance_df[['feature']].copy()
            if 'importance' in importance_df.columns:
                best_model_vars_df['importance'] = importance_df['importance']
        elif selected_features:
            best_model_vars_df = pd.DataFrame({'feature': list(selected_features)})
        if not best_model_vars_df.empty:
            self.reports_['best_model_vars_df'] = best_model_vars_df.reset_index(drop=True)
        else:
            self.reports_.pop('best_model_vars_df', None)
            if selected_features:
                self.reports_['best_model_vars_df'] = pd.DataFrame({'feature': list(selected_features)})

        self.reports_["model_performance"] = report
        return report



    def generate_feature_report(
        self,
        models: Dict[str, Any],
        woe_results: Dict[str, Any],
        data_dictionary: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Render feature level metadata including dictionary context and WOE stats."""

        dictionary = self._normalize_data_dictionary(data_dictionary)
        selected_features = models.get("selected_features", [])
        woe_values = woe_results.get("woe_values", {}) if isinstance(woe_results, dict) else {}
        gini_values = woe_results.get("univariate_gini", {}) if isinstance(woe_results, dict) else {}

        tsfresh_lookup: Dict[str, Dict[str, Any]] = {}
        if isinstance(self.tsfresh_metadata, pd.DataFrame) and not self.tsfresh_metadata.empty:
            for _, row in self.tsfresh_metadata.iterrows():
                feature_key = str(row.get('feature', '')).strip()
                if feature_key:
                    tsfresh_lookup.setdefault(feature_key, {}).update(row.to_dict())

        records: List[Dict[str, Any]] = []

        for feature in selected_features:
            raw_feature = self._extract_raw_feature(feature)
            woe_key = self._resolve_woe_key(raw_feature, feature, woe_values)
            woe_info = woe_values.get(woe_key, {})
            stats = woe_info.get("stats") or []

            description = self._get_feature_description(raw_feature, dictionary)
            category = self._get_feature_category(raw_feature, dictionary)

            woe_candidates: List[float] = []
            if isinstance(woe_info.get("woe_map"), dict):
                woe_candidates.extend(woe_info["woe_map"].values())
            woe_candidates.extend(
                [row.get("woe") for row in stats if isinstance(row, dict) and "woe" in row]
            )

            record: Dict[str, Any] = {
                "feature": feature,
                "raw_feature": raw_feature,
                "description": description,
                "category": category,
                "iv": woe_info.get("iv", np.nan),
                "woe_min": min(woe_candidates) if woe_candidates else np.nan,
                "woe_max": max(woe_candidates) if woe_candidates else np.nan,
                "n_bins": self._infer_bin_count(woe_info),
                "type": woe_info.get("type"),
            }

            gini_info = gini_values.get(woe_key, {})
            if gini_info:
                record.update(
                    {
                        "gini_raw": gini_info.get("gini_raw", np.nan),
                        "gini_woe": gini_info.get("gini_woe", np.nan),
                        "gini_drop": gini_info.get("gini_drop", np.nan),
                    }
                )

            feature_importance = models.get("feature_importance", {})
            for model_name, importance_df in feature_importance.items():
                if not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
                    continue
                mask = importance_df["feature"].astype(str) == str(feature)
                if mask.any():
                    record[f"importance_{model_name}"] = importance_df.loc[mask, "importance"].iloc[0]

            ts_meta = tsfresh_lookup.get(str(feature)) or tsfresh_lookup.get(raw_feature)
            if ts_meta:
                record.update(
                    {
                        "is_tsfresh": True,
                        "tsfresh_source": ts_meta.get('source_variable'),
                        "tsfresh_statistic": ts_meta.get('statistic'),
                        "tsfresh_generator": ts_meta.get('generator'),
                        "tsfresh_parameters": ts_meta.get('parameters', ''),
                    }
                )
            else:
                record['is_tsfresh'] = False

            records.append(record)

        features_df = pd.DataFrame(records)
        if not features_df.empty:
            preferred_order = [
                "feature",
                "raw_feature",
                "description",
                "category",
                "iv",
                "gini_raw",
                "gini_woe",
                "gini_drop",
                "is_tsfresh",
                "tsfresh_source",
                "tsfresh_statistic",
                "tsfresh_generator",
                "tsfresh_parameters",
            ]
            ordered_cols = [c for c in preferred_order if c in features_df.columns]
            remaining_cols = [c for c in features_df.columns if c not in ordered_cols]
            features_df = features_df[ordered_cols + remaining_cols]
            if "iv" in features_df.columns:
                features_df = features_df.sort_values("iv", ascending=False).reset_index(drop=True)
        self.reports_["features"] = features_df
        return features_df

    def generate_best_model_reports(
        self,
        models: Dict[str, Any],
        woe_results: Dict[str, Any],
        data_dictionary: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Produce bin level outputs for the selected champion model."""

        dictionary = self._normalize_data_dictionary(data_dictionary)
        woe_values = woe_results.get("woe_values", {})
        selected_features = models.get("selected_features", [])

        bin_rows: List[Dict[str, Any]] = []
        final_rows: List[Dict[str, Any]] = []

        for feature in selected_features:
            raw_feature = self._extract_raw_feature(feature)
            woe_key = self._resolve_woe_key(raw_feature, feature, woe_values)
            woe_info = woe_values.get(woe_key, {})
            stats = woe_info.get("stats") or []

            description = self._get_feature_description(raw_feature, dictionary)
            category = self._get_feature_category(raw_feature, dictionary)

            for stat in stats:
                if not isinstance(stat, dict):
                    continue
                row = {
                    "feature": feature,
                    "raw_feature": raw_feature,
                    "description": description,
                    "category": category,
                }
                row.update(stat)
                bin_rows.append(row)

            woe_candidates: List[float] = []
            if isinstance(woe_info.get("woe_map"), dict):
                woe_candidates.extend(woe_info["woe_map"].values())
            woe_candidates.extend(
                [row.get("woe") for row in stats if isinstance(row, dict) and "woe" in row]
            )

            final_rows.append(
                {
                    "feature": feature,
                    "raw_feature": raw_feature,
                    "description": description,
                    "category": category,
                    "iv": woe_info.get("iv", np.nan),
                    "type": woe_info.get("type"),
                    "n_bins": self._infer_bin_count(woe_info),
                    "woe_min": min(woe_candidates) if woe_candidates else np.nan,
                    "woe_max": max(woe_candidates) if woe_candidates else np.nan,
                }
            )

        bins_df = pd.DataFrame(bin_rows)
        if not bins_df.empty:
            sort_cols = [c for c in ["raw_feature", "bin_index", "bin"] if c in bins_df.columns]
            if sort_cols:
                bins_df = bins_df.sort_values(sort_cols).reset_index(drop=True)

        final_vars_df = pd.DataFrame(final_rows)

        self.reports_["best_model_bins"] = bins_df
        self.reports_["final_variables"] = final_vars_df

        summary = {
            "best_model": models.get("best_model_name"),
            "best_model_bins": bins_df,
            "final_variables": final_vars_df,
        }
        return summary

    def generate_woe_tables(
        self,
        woe_results: Dict[str, Any],
        selected_features: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Return WOE detail tables for requested features."""

        tables: Dict[str, pd.DataFrame] = {}
        combined_frames: List[pd.DataFrame] = []
        woe_values = woe_results.get("woe_values", {}) or {}

        for feature in selected_features:
            raw_feature = self._extract_raw_feature(feature)
            woe_key = self._resolve_woe_key(raw_feature, feature, woe_values)
            woe_info = woe_values.get(woe_key)
            if not woe_info or "stats" not in woe_info:
                continue

            table = pd.DataFrame(woe_info["stats"])
            table["feature"] = feature
            table["raw_feature"] = raw_feature
            table["type"] = woe_info.get("type", "unknown")
            if "woe" in table.columns:
                table = table.sort_values("woe").reset_index(drop=True)
            tables[feature] = table
            combined_frames.append(table.copy())

        self.reports_["woe_tables"] = tables
        if combined_frames:
            self.reports_["woe_mapping"] = pd.concat(combined_frames, ignore_index=True)
        return tables



    def generate_risk_band_report(
        self,
        risk_bands: Dict[str, Any]
    ) -> pd.DataFrame:
        """Summarise risk band allocation and diagnostics."""

        risk_dict = risk_bands if isinstance(risk_bands, dict) else {}
        inner_bands = risk_dict.get('bands') if isinstance(risk_dict.get('bands'), dict) else None

        band_stats = risk_dict.get('band_stats')
        if (not isinstance(band_stats, pd.DataFrame) or band_stats.empty) and isinstance(inner_bands, dict):
            band_stats = inner_bands.get('band_stats')

        if isinstance(band_stats, pd.DataFrame) and not band_stats.empty:
            bands_df = band_stats.copy()
        else:
            bands_obj = risk_dict.get('bands')
            if isinstance(bands_obj, pd.DataFrame):
                bands_df = bands_obj.copy()
            elif isinstance(bands_obj, (list, tuple)):
                bands_df = pd.DataFrame(bands_obj)
            else:
                bands_df = pd.DataFrame()
            if isinstance(inner_bands, dict):
                stats = inner_bands.get('band_stats')
                if bands_df.empty and isinstance(stats, pd.DataFrame):
                    bands_df = stats.copy()

        if not bands_df.empty and 'band' not in bands_df.columns:
            bands_df = bands_df.reset_index().rename(columns={'index': 'band'})

        metrics = risk_dict.get('metrics', {})
        test_results = risk_dict.get('test_results', {}) if isinstance(risk_dict.get('test_results'), dict) else {}
        if not test_results and isinstance(inner_bands, dict):
            inner_tests = inner_bands.get('test_results')
            if isinstance(inner_tests, dict):
                test_results = inner_tests

        binomial_source = metrics.get('binomial_tests')
        if (binomial_source is None or (isinstance(binomial_source, pd.DataFrame) and binomial_source.empty)) and isinstance(test_results, dict):
            binomial_source = test_results.get('binomial')
        binomial_df = self._binomial_results_to_df(binomial_source)

        if not bands_df.empty and not binomial_df.empty and 'band' in binomial_df.columns:
            bands_df = bands_df.merge(binomial_df, on='band', how='left', suffixes=('', '_binom'))
            # Remove duplicated columns created by merge if any
            for col in list(bands_df.columns):
                if col.endswith('_binom') and col[:-6] in bands_df.columns:
                    bands_df.drop(columns=col, inplace=True)

        bands_df = bands_df.sort_values('band').reset_index(drop=True) if 'band' in bands_df.columns else bands_df

        # Enhance with scaled risk scores (300-850) and risk levels
        if isinstance(bands_df, pd.DataFrame) and not bands_df.empty and 'band' in bands_df.columns:
            n_bands = len(bands_df)
            try:
                scaled_scores = np.linspace(850, 300, n_bands).astype(int)
                bands_df['scaled_score'] = scaled_scores
            except Exception:
                pass

            def risk_level(score: int) -> str:
                if score >= 750:
                    return 'Very Low Risk'
                if score >= 650:
                    return 'Low Risk'
                if score >= 550:
                    return 'Medium Risk'
                if score >= 450:
                    return 'High Risk'
                return 'Very High Risk'

            if 'scaled_score' in bands_df.columns:
                bands_df['risk_level'] = bands_df['scaled_score'].apply(lambda x: risk_level(int(x)))

            # Build risk score mapping table for export
            if {'min_score', 'max_score'}.issubset(bands_df.columns):
                score_range = bands_df.apply(
                    lambda row: f"[{float(row['min_score']):.4f}, {float(row['max_score']):.4f})", axis=1
                )
                # Normalise naming for counts and event rates
                count_col = 'count' if 'count' in bands_df.columns else ('n_samples' if 'n_samples' in bands_df.columns else None)
                rate_col = 'bad_rate' if 'bad_rate' in bands_df.columns else ('event_rate' if 'event_rate' in bands_df.columns else None)
                mapping_cols = {
                    'band': bands_df['band'],
                    'score_range': score_range,
                    'scaled_score': bands_df.get('scaled_score'),
                    'risk_level': bands_df.get('risk_level'),
                }
                if count_col:
                    mapping_cols['n_samples'] = bands_df[count_col]
                if rate_col:
                    mapping_cols['event_rate'] = bands_df[rate_col]
                risk_mapping = pd.DataFrame(mapping_cols)
                self.reports_['risk_score_mapping'] = risk_mapping

        self.reports_['risk_bands'] = bands_df

        if not binomial_df.empty:
            self.reports_['risk_bands_tests'] = binomial_df

        if isinstance(metrics, dict) and metrics:
            metrics_df = pd.DataFrame([metrics])
            self.reports_['risk_bands_metrics'] = metrics_df
        else:
            metrics_df = pd.DataFrame()

        summary_lines: List[str] = []
        if not bands_df.empty:
            summary_lines.append(f"Number of bands: {len(bands_df)}")
            if 'bad_rate' in bands_df.columns:
                summary_lines.append(
                    f"Bad rate range: {bands_df['bad_rate'].min():.2%} - {bands_df['bad_rate'].max():.2%}"
                )
            if 'bad_capture' in bands_df.columns and bands_df['bad_capture'].notna().any():
                summary_lines.append(
                    f"Bad capture (top band): {bands_df['bad_capture'].iloc[-1]:.2%}"
                )

        if isinstance(metrics, dict):
            hhi = metrics.get('herfindahl_index')
            if hhi is not None:
                summary_lines.append(f"Herfindahl Index: {hhi:.4f}")
            entropy = metrics.get('entropy')
            if entropy is not None:
                summary_lines.append(f"Entropy: {entropy:.4f}")
            gini_coeff = metrics.get('gini_coefficient')
            if gini_coeff is not None:
                summary_lines.append(f"Gini Coefficient: {gini_coeff:.4f}")
            hl_p = metrics.get('hosmer_lemeshow_p')
            if hl_p is not None:
                summary_lines.append(f"Hosmer-Lemeshow p-value: {hl_p:.4f}")
            ks = metrics.get('ks_stat')
            if ks is not None:
                summary_lines.append(f"KS Statistic: {ks:.4f}")

        if not binomial_df.empty:
            if 'binomial_result' in binomial_df.columns:
                rejects = int((binomial_df['binomial_result'] == 'Reject').sum())
                summary_lines.append(
                    f"Binomial rejects: {rejects}/{len(binomial_df)}"
                )
            elif 'binomial_pass' in binomial_df.columns:
                rejects = int((~binomial_df['binomial_pass'].astype(bool)).sum())
                summary_lines.append(
                    f"Binomial rejects: {rejects}/{len(binomial_df)}"
                )

        if summary_lines:
            self.reports_['risk_bands_summary'] = {"Risk Bands Summary": summary_lines}
            self.reports_['risk_bands_summary_table'] = pd.DataFrame({'summary': summary_lines})
        else:
            self.reports_.pop('risk_bands_summary', None)
            self.reports_.pop('risk_bands_summary_table', None)

        return bands_df

    def generate_calibration_report(
        self,
        stage1_results: Optional[Dict[str, Any]],
        stage2_results: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect calibration metrics from stage 1 and stage 2 routines."""

        report: Dict[str, Any] = {}

        metrics_rows: List[Dict[str, Any]] = []
        if stage1_results:
            stage1_metrics = stage1_results.get("calibration_metrics", {})
            report["stage1"] = {
                "method": getattr(self.config, "calibration_method", "unknown"),
                "metrics": stage1_metrics,
            }
            if stage1_metrics:
                metrics_rows.append({"stage": "stage1", **stage1_metrics})
            details = stage1_results.get("stage1_details") or {}
            if details:
                report["stage1"]["details"] = details
                self.reports_["stage1_details"] = details

        if stage2_results:
            stage2_metrics = stage2_results.get("stage2_metrics", {})
            report["stage2"] = {
                "method": getattr(self.config, "stage2_method", "unknown"),
                "metrics": stage2_metrics,
            }
            if stage2_metrics:
                metrics_rows.append({"stage": "stage2", **stage2_metrics})
                hl_stats = stage2_metrics.get('hosmer_lemeshow')
                if isinstance(hl_stats, dict):
                    self.reports_['hosmer_lemeshow'] = pd.DataFrame([hl_stats])
            details = stage2_results.get("stage2_details") or {}
            if details:
                report["stage2"]["details"] = details
                self.reports_["stage2_details"] = details

        if metrics_rows:
            self.reports_["calibration_metrics"] = pd.DataFrame(metrics_rows)

        self.reports_["calibration"] = report
        return report

    def generate_scoring_report(self, scoring_output: Dict[str, Any]) -> Dict[str, Any]:
        '''Summarise scoring metrics and persist scored dataset references.'''
        metrics = scoring_output.get('metrics') or {}
        summary_row = {
            'n_total': metrics.get('n_total', 0),
            'n_with_target': metrics.get('n_with_target', 0),
            'n_without_target': metrics.get('n_without_target', 0),
            'psi_score': metrics.get('psi_score'),
            'calibration_applied': bool(metrics.get('calibration_applied')),
        }
        if 'with_target' in metrics:
            summary_row['auc_with_target'] = metrics['with_target'].get('auc')
            summary_row['default_rate'] = metrics['with_target'].get('default_rate')
            summary_row['gini'] = metrics['with_target'].get('gini')
            summary_row['ks'] = metrics['with_target'].get('ks')
        summary_df = pd.DataFrame([summary_row])
        reports = scoring_output.get('reports') or {}
        scored_df = scoring_output.get('dataframe')
        self.reports_['scoring_summary'] = summary_df
        self.reports_['scoring_reports'] = reports
        self.reports_['monitor_report'] = summary_df
        if isinstance(scored_df, pd.DataFrame):
            self.reports_['scored_data'] = scored_df
        return {'summary': summary_df, 'reports': reports, 'metrics': metrics, 'dataframe': scored_df}


    def generate_final_summary(self) -> str:
        """Assemble a human readable wrap-up of stored reports."""

        lines = ["=" * 80, "RISK MODEL PIPELINE - FINAL SUMMARY", "=" * 80, ""]

        model_perf = self.reports_.get("model_performance")
        if model_perf:
            lines.append("MODEL PERFORMANCE:")
            lines.append(f"  Best Model: {model_perf.get('best_model', 'N/A')}")
            lines.append(f"  Best AUC: {model_perf.get('best_auc', 0):.4f}")
            lines.append(f"  Features Used: {model_perf.get('feature_count', 0)}")
            lines.append("")

        features_df = self.reports_.get("features")
        if isinstance(features_df, pd.DataFrame) and not features_df.empty:
            lines.append("TOP FEATURES:")
            for _, row in features_df.head(5).iterrows():
                lines.append(
                    f"  {row.get('feature')} (IV={row.get('iv', np.nan):.3f})"
                )
            lines.append("")

        risk_summary = self.reports_.get("risk_bands_summary")
        if risk_summary:
            lines.append("RISK BANDS ANALYSIS:")
            for entry in risk_summary.get("Risk Bands Summary", []):
                lines.append(f"  {entry}")
            lines.append("")

        calibration = self.reports_.get("calibration")
        if calibration:
            lines.append("CALIBRATION:")
            if "stage1" in calibration:
                stage1_metrics = calibration["stage1"].get("metrics", {})
                lines.append(f"  Stage 1 ECE: {stage1_metrics.get('ece', 0):.4f}")
            if "stage2" in calibration:
                stage2_metrics = calibration["stage2"].get("metrics", {})
                lines.append(f"  Stage 2 ECE: {stage2_metrics.get('ece', 0):.4f}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)



    def export_to_excel(self, filepath: str) -> None:
        """Persist collected reports to an Excel workbook following the required structure."""
    
        used_names: set[str] = set()
    
        def safe_sheet_name(name: str) -> str:
            cleaned = re.sub(r"[^0-9A-Za-z _-]", "_", str(name)).strip()
            cleaned = cleaned or "Sheet"
            cleaned = cleaned[:31]
            base = cleaned
            counter = 1
            while cleaned in used_names:
                suffix = f"_{counter}"
                cleaned = (base[: 31 - len(suffix)] + suffix).strip() or f"Sheet_{counter}"
                counter += 1
            used_names.add(cleaned)
            return cleaned
    
        def ensure_df(candidate: Any, message: str) -> pd.DataFrame:
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                return candidate
            return pd.DataFrame([{'message': message}])
    
        model_perf = self.reports_.get('model_performance', {}) if isinstance(self.reports_.get('model_performance'), dict) else {}
        model_scores = model_perf.get('model_scores') if isinstance(model_perf, dict) else None
        if isinstance(model_scores, dict) and model_scores:
            models_summary_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={'index': 'model_name'})
        else:
            models_summary_df = None
    
        best_model_df = None
        if isinstance(model_perf, dict) and model_perf:
            best_model_df = pd.DataFrame([{key: model_perf.get(key) for key in ['timestamp', 'best_model', 'feature_count']}])
    
        final_vars = self.reports_.get('final_variables')
        features_df = self.reports_.get('features')
        shap_df = self.reports_.get('shap_importance')
    
        top20_iv = None
        if isinstance(final_vars, pd.DataFrame) and 'iv' in final_vars.columns:
            top20_iv = final_vars[['feature', 'iv']].dropna().sort_values('iv', ascending=False).head(20)
    
        top50_univariate = None
        if isinstance(features_df, pd.DataFrame):
            score_cols = [col for col in ['gini_raw', 'gini_woe', 'gini_drop'] if col in features_df.columns]
            if score_cols:
                top50_univariate = features_df[['feature', *score_cols]].copy().sort_values(score_cols[0], ascending=False).head(50)
    
        woe_degradation = None
        if isinstance(features_df, pd.DataFrame) and {'feature', 'gini_raw', 'gini_woe'} <= set(features_df.columns):
            temp = features_df[['feature', 'gini_raw', 'gini_woe']].copy()
            temp['gini_drop'] = temp['gini_raw'] - temp['gini_woe']
            woe_degradation = temp
    
        psi_summary = None
        if isinstance(self.selection_history, pd.DataFrame):
            psi_rows = self.selection_history[self.selection_history['method'] == 'psi']
            if not psi_rows.empty:
                psi_summary = psi_rows[['method', 'before', 'after', 'removed_features']]
    
        psi_dropped = None
        if isinstance(psi_summary, pd.DataFrame):
            psi_dropped = psi_summary[['removed_features']].rename(columns={'removed_features': 'psi_removed'})
    
        run_meta_rows = [
            {'parameter': 'target_column', 'value': getattr(self.config, 'target_column', None)},
            {'parameter': 'id_column', 'value': getattr(self.config, 'id_column', None)},
            {'parameter': 'time_column', 'value': getattr(self.config, 'time_column', None)},
            {'parameter': 'enable_tsfresh_features', 'value': getattr(self.config, 'enable_tsfresh_features', False)},
            {'parameter': 'selection_steps', 'value': ', '.join(getattr(self.config, 'selection_steps', []))},
            {'parameter': 'algorithms', 'value': ', '.join(getattr(self.config, 'algorithms', []))},
            {'parameter': 'random_state', 'value': getattr(self.config, 'random_state', None)},
        ]
        run_meta_df = pd.DataFrame(run_meta_rows)
    
        risk_mapping_sql = None
        risk_bands = self.reports_.get('risk_bands')
        if isinstance(risk_bands, pd.DataFrame) and {'band', 'min_score', 'max_score'} <= set(risk_bands.columns):
            risk_mapping_sql = risk_bands[['band', 'min_score', 'max_score']].copy()
    
        scoring_summary = None
        scoring_reports = self.reports_.get('scoring_reports')
        if isinstance(scoring_reports, dict):
            summary = scoring_reports.get('summary')
            if isinstance(summary, pd.DataFrame):
                scoring_summary = summary
            elif isinstance(summary, dict):
                scoring_summary = pd.DataFrame([summary])
    
        scored_data = None
        scoring_output = self.reports_.get('scored_data')
        if isinstance(scoring_output, pd.DataFrame):
            limit = getattr(self.config, 'report_scored_rows', 5000)
            scored_data = scoring_output.head(limit)
    
        sheet_map = {
            'models_summary': ensure_df(models_summary_df, 'Model performance summary not available'),
            'best_model': ensure_df(best_model_df, 'Best model details not available'),
            'best_model_vars_df': ensure_df(final_vars, 'Final variables not available'),
            'confusion_matrix': ensure_df(self.reports_.get('confusion_matrix'), 'Confusion matrix not available'),
            'performance_report': ensure_df(self.reports_.get('performance_report'), 'Performance metrics not available'),
            'lift_table': ensure_df(self.reports_.get('lift_table'), 'Lift table not available'),
            'baseline_metrics': ensure_df(self.reports_.get('baseline_metrics'), 'Baseline metrics unavailable'),
            'baseline_lift_table': ensure_df(self.reports_.get('baseline_lift_table'), 'Baseline lift table unavailable'),
            'shap_importance': ensure_df(shap_df, 'SHAP importance not available'),
            'final_vars': ensure_df(final_vars, 'Final variable list not available'),
            'top20_iv': ensure_df(top20_iv, 'IV ranking not available'),
            'top50_univariate': ensure_df(top50_univariate, 'Univariate ranking not available'),
            'selection_history': ensure_df(self.reports_.get('selection_history'), 'Selection history unavailable'),
            'correlation_clusters': ensure_df(self.reports_.get('correlation_clusters'), 'Correlation clusters unavailable'),
            'vif_summary': ensure_df(self.reports_.get('vif_summary'), 'VIF summary unavailable'),
            'noise_sentinel_check': ensure_df(self.reports_.get('noise_sentinel_check'), 'Noise sentinel diagnostics unavailable'),
            'variable_dictionary': ensure_df(self.reports_.get('data_dictionary'), 'Data dictionary unavailable'),
            'shap_summary': ensure_df(shap_df, 'SHAP summary not available'),
            'woe_mapping': ensure_df(self.reports_.get('woe_mapping'), 'WOE mapping unavailable'),
            'woe_bins': ensure_df(self.reports_.get('best_model_bins'), 'WOE bins unavailable'),
            'best_model_details': ensure_df(self.reports_.get('best_model_bins'), 'Best model binning unavailable'),
            'woe_degradation': ensure_df(woe_degradation, 'WOE vs raw gini comparison unavailable'),
            'psi_summary': ensure_df(psi_summary, 'PSI summary unavailable'),
            'psi_dropped_features': ensure_df(psi_dropped, 'PSI dropped features unavailable'),
            'WOE_PSI': ensure_df(self.reports_.get('woe_psi'), 'WOE PSI report unavailable'),
            'Score_PSI': ensure_df(self.reports_.get('score_psi'), 'Score PSI report unavailable'),
            'Quantile_PSI': ensure_df(self.reports_.get('quantile_psi'), 'Quantile PSI report unavailable'),
            'run_meta': run_meta_df,
            'monitor_report': ensure_df(self.reports_.get('monitor_report'), 'Monitoring report unavailable'),
            'calibration_metrics': ensure_df(self.reports_.get('calibration_metrics'), 'Calibration metrics unavailable'),
            'calibration_tables': ensure_df(self.reports_.get('calibration_tables'), 'Calibration tables unavailable'),
            'stage1': ensure_df(self.reports_.get('stage1_details'), 'Stage 1 details unavailable'),
            'stage2': ensure_df(self.reports_.get('stage2_details'), 'Stage 2 details unavailable'),
            'hosmer_lemeshow': ensure_df(self.reports_.get('hosmer_lemeshow'), 'Hosmer-Lemeshow results unavailable'),
            'risk_bands': ensure_df(self.reports_.get('risk_bands'), 'Risk bands unavailable'),
            'band_tests': ensure_df(self.reports_.get('risk_bands_tests'), 'Band test results unavailable'),
            'band_metrics': ensure_df(self.reports_.get('risk_bands_metrics'), 'Risk band metrics unavailable'),
            'risk_band_summary': ensure_df(self.reports_.get('risk_bands_summary_table'), 'Risk band summary unavailable'),
            'risk_score_mapping': ensure_df(risk_mapping_sql, 'Risk score mapping unavailable'),
            'scoring_summary': ensure_df(scoring_summary, 'Scoring summary unavailable'),
            'scored_data': ensure_df(scored_data, 'Scored data unavailable'),
            'micro_bins': ensure_df(self.reports_.get('micro_bins'), 'Micro bin results unavailable'),
            'evaluate_bins': ensure_df(self.reports_.get('evaluate_bins'), 'Bin evaluation unavailable'),
            'calculate_penalty': ensure_df(self.reports_.get('calculate_penalty'), 'Penalty breakdown unavailable'),
            'multi_start_optimization': ensure_df(self.reports_.get('multi_start_optimization'), 'Multi-start optimisation log unavailable'),
            'report_results': ensure_df(self.reports_.get('report_results'), 'Optimisation results unavailable'),
            'pipeline_overview': ensure_df(self.reports_.get('pipeline_overview'), 'Pipeline overview unavailable'),
            'operations_notes': ensure_df(self.reports_.get('operations_notes'), 'Operations notes unavailable'),
            'git_notes': ensure_df(self.reports_.get('git_notes'), 'Git structure notes unavailable'),
        }
    
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            def write_df(name: str, df: pd.DataFrame, *, index: bool = False) -> None:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return
                sheet = safe_sheet_name(name)
                df.to_excel(writer, sheet_name=sheet, index=index)
    
            for sheet_name, df in sheet_map.items():
                write_df(sheet_name, df)
    
        def create_model_comparison_plot(self, models: Dict[str, Any]):
            """Visualise train/test AUC for compared models."""
    
            scores_section = models.get("scores")
            if not scores_section:
                return None
    
            data = []
            for model_name, metrics in scores_section.items():
                data.append(
                    {
                        "Model": model_name,
                        "Train AUC": metrics.get("train_auc", 0.0),
                        "Test AUC": metrics.get("test_auc", 0.0),
                    }
                )
    
            if not data:
                return None
    
            scores_df = pd.DataFrame(data)
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(scores_df))
            width = 0.35
    
            train_bars = ax.bar(x - width / 2, scores_df["Train AUC"], width, label="Train AUC")
            test_bars = ax.bar(x + width / 2, scores_df["Test AUC"], width, label="Test AUC")
    
            ax.set_xlabel("Model")
            ax.set_ylabel("AUC Score")
            ax.set_title("Model Performance Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(scores_df["Model"], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
            for bars in (train_bars, test_bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )
    
            plt.tight_layout()
            return fig
    
        @staticmethod
        def _infer_bin_count(woe_info: Dict[str, Any]) -> int:
            if not woe_info:
                return 0
            stats = woe_info.get("stats")
            if isinstance(stats, list) and stats:
                return len(stats)
            if woe_info.get("type") == "numeric" and woe_info.get("bins"):
                return len(woe_info.get("bins", []))
            categories = woe_info.get("categories")
            if isinstance(categories, list):
                return len(categories)
            return 0
    
        @staticmethod
        def _extract_raw_feature(feature: str) -> str:
            if not feature:
                return feature
            candidates = ["_woe", "_bin", "_scaled", "_bucketed"]
            for suffix in candidates:
                if feature.endswith(suffix):
                    return feature[: -len(suffix)]
            return feature
    
    
def _resolve_woe_key(
    self,
    raw_feature: str,
    feature: str,
    woe_values: Dict[str, Any],
) -> Optional[str]:
    if not woe_values:
        return None

    candidates = [raw_feature, raw_feature.lower(), raw_feature.upper(), feature]
    for candidate in candidates:
        if candidate in woe_values:
            return candidate
    return feature if feature in woe_values else raw_feature

@staticmethod
def _normalize_data_dictionary(
    data_dictionary: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if data_dictionary is None or data_dictionary.empty:
        return None

    dictionary = data_dictionary.copy()
    rename_map: Dict[str, str] = {}
    for column in dictionary.columns:
        key = column.strip().lower()
        if key in {"variable", "feature", "field", "column", "raw_feature", "var_name"}:
            rename_map[column] = "variable"
        elif key in {"description", "desc", "desc_text", "feature_description"}:
            rename_map[column] = "description"
        elif key in {"category", "segment", "group"}:
            rename_map[column] = "category"
    dictionary = dictionary.rename(columns=rename_map)

    if "variable" not in dictionary.columns:
        return None

    dictionary["variable"] = dictionary["variable"].astype(str)
    dictionary["variable_key"] = dictionary["variable"].str.strip().str.lower()
    dictionary = dictionary.drop_duplicates(subset="variable_key")

    for column in ("description", "category"):
        if column in dictionary.columns:
            dictionary[column] = dictionary[column].fillna("")

    return dictionary

def _lookup_dictionary_value(
    self,
    raw_feature: str,
    dictionary: Optional[pd.DataFrame],
    column: str,
) -> str:
    if dictionary is None or column not in dictionary.columns:
        return ""

    key = (raw_feature or "").strip().lower()
    if not key:
        return ""

    match = dictionary[dictionary["variable_key"] == key]
    if match.empty:
        match = dictionary[dictionary["variable"] == raw_feature]
    if match.empty:
        return ""

    value = match.iloc[0][column]
    if pd.isna(value):
        return ""
    return str(value)

def _get_feature_description(
    self,
    raw_feature: str,
    dictionary: Optional[pd.DataFrame],
) -> str:
    return self._lookup_dictionary_value(raw_feature, dictionary, "description")

def _get_feature_category(
    self,
    raw_feature: str,
    dictionary: Optional[pd.DataFrame],
) -> str:
    return self._lookup_dictionary_value(raw_feature, dictionary, "category")
    def _lookup_dictionary_value(
        self,
        raw_feature: str,
        dictionary: Optional[pd.DataFrame],
        column: str,
    ) -> str:
        if dictionary is None or column not in dictionary.columns:
            return ""

        key = (raw_feature or "").strip().lower()
        if not key:
            return ""

        match = dictionary[dictionary["variable_key"] == key]
        if match.empty:
            match = dictionary[dictionary["variable"] == raw_feature]
        if match.empty:
            return ""

        value = match.iloc[0][column]
        if pd.isna(value):
            return ""
        return str(value)

    def _get_feature_description(
        self,
        raw_feature: str,
        dictionary: Optional[pd.DataFrame],
    ) -> str:
        return self._lookup_dictionary_value(raw_feature, dictionary, "description")

    def _get_feature_category(
        self,
        raw_feature: str,
        dictionary: Optional[pd.DataFrame],
    ) -> str:
        return self._lookup_dictionary_value(raw_feature, dictionary, "category")





def _enhanced_reporter_resolve_woe_key(self, raw_feature: str, feature: str, woe_values: Dict[str, Any]) -> Optional[str]:
    if not woe_values:
        return None
    candidates = [raw_feature, raw_feature.lower(), raw_feature.upper(), feature]
    for candidate in candidates:
        if candidate in woe_values:
            return candidate
    return feature if feature in woe_values else raw_feature


def _enhanced_reporter_normalize_dictionary(data_dictionary: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if data_dictionary is None or data_dictionary.empty:
        return None
    dictionary = data_dictionary.copy()
    rename_map: Dict[str, str] = {}
    for column in dictionary.columns:
        key = column.strip().lower()
        if key in {"variable", "feature", "field", "column", "raw_feature", "var_name"}:
            rename_map[column] = "variable"
        elif key in {"description", "desc", "desc_text", "feature_description"}:
            rename_map[column] = "description"
        elif key in {"category", "segment", "group"}:
            rename_map[column] = "category"
    dictionary = dictionary.rename(columns=rename_map)
    if "variable" not in dictionary.columns:
        return None
    dictionary["variable"] = dictionary["variable"].astype(str)
    dictionary["variable_key"] = dictionary["variable"].str.strip().str.lower()
    dictionary = dictionary.drop_duplicates(subset="variable_key")
    for column in ("description", "category"):
        if column in dictionary.columns:
            dictionary[column] = dictionary[column].fillna("")
    return dictionary


def _enhanced_reporter_lookup_dictionary_value(self, raw_feature: str, dictionary: Optional[pd.DataFrame], column: str) -> str:
    if dictionary is None or column not in dictionary.columns:
        return ""
    key = (raw_feature or "").strip().lower()
    if not key:
        return ""
    match = dictionary[dictionary["variable_key"] == key]
    if match.empty:
        match = dictionary[dictionary["variable"] == raw_feature]
    if match.empty:
        return ""
    value = match.iloc[0][column]
    if pd.isna(value):
        return ""
    return str(value)


def _enhanced_reporter_get_feature_description(self, raw_feature: str, dictionary: Optional[pd.DataFrame]) -> str:
    return _enhanced_reporter_lookup_dictionary_value(self, raw_feature, dictionary, "description")


def _enhanced_reporter_get_feature_category(self, raw_feature: str, dictionary: Optional[pd.DataFrame]) -> str:
    return _enhanced_reporter_lookup_dictionary_value(self, raw_feature, dictionary, "category")

EnhancedReporter._resolve_woe_key = _enhanced_reporter_resolve_woe_key
EnhancedReporter._normalize_data_dictionary = staticmethod(_enhanced_reporter_normalize_dictionary)
EnhancedReporter._lookup_dictionary_value = _enhanced_reporter_lookup_dictionary_value
EnhancedReporter._get_feature_description = _enhanced_reporter_get_feature_description
EnhancedReporter._get_feature_category = _enhanced_reporter_get_feature_category



def _enhanced_reporter_extract_raw_feature(feature: str) -> str:
    if not feature:
        return feature
    for suffix in ["_woe", "_bin", "_scaled", "_bucketed"]:
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature

EnhancedReporter._extract_raw_feature = staticmethod(_enhanced_reporter_extract_raw_feature)
EnhancedReporter._get_feature_description = _enhanced_reporter_get_feature_description
EnhancedReporter._get_feature_category = _enhanced_reporter_get_feature_category
EnhancedReporter._lookup_dictionary_value = _enhanced_reporter_lookup_dictionary_value



def _enhanced_reporter_infer_bin_count(woe_info: Dict[str, Any]) -> int:
    if not woe_info:
        return 0
    stats = woe_info.get("stats")
    if isinstance(stats, list) and stats:
        return len(stats)
    if woe_info.get("type") == "numeric" and woe_info.get("bins"):
        return len(woe_info.get("bins", []))
    categories = woe_info.get("categories")
    if isinstance(categories, list):
        return len(categories)
    return 0

EnhancedReporter._infer_bin_count = staticmethod(_enhanced_reporter_infer_bin_count)
