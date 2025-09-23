"""Enhanced Reporter with comprehensive reporting capabilities."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EnhancedReporter:
    """Generates structured outputs for model performance and diagnostics."""

    def __init__(self, config: Any):
        self.config = config
        self.reports_: Dict[str, Any] = {}
        self.data_dictionary: Optional[pd.DataFrame] = None

    def register_data_dictionary(self, dictionary: Optional[pd.DataFrame]) -> None:
        '''Store a normalized data dictionary for later lookups.'''
        normalized = self._normalize_data_dictionary(dictionary)
        if normalized is None:
            self.data_dictionary = None
            return
        self.data_dictionary = normalized
        self.reports_['data_dictionary'] = normalized

    def generate_model_report(
        self,
        models: Dict[str, Any],
        data_dictionary: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Build a compact model performance summary."""

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_scores": models.get("scores", {}),
            "best_model": models.get("best_model_name"),
            "feature_count": len(models.get("selected_features", [])),
        }

        best_model = report["best_model"]
        if best_model and best_model in report["model_scores"]:
            report["best_auc"] = report["model_scores"][best_model].get("test_auc", 0.0)

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
        woe_values = woe_results.get("woe_values", {})
        gini_values = woe_results.get("univariate_gini", {})

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
            ]
            ordered_cols = [c for c in preferred_order if c in features_df.columns]
            remaining_cols = [c for c in features_df.columns if c not in ordered_cols]
            features_df = features_df[ordered_cols + remaining_cols]
            if "iv" in features_df.columns:
                features_df = features_df.sort_values("iv", ascending=False).reset_index(drop=True)

        self.reports_["features"] = features_df
        if dictionary is not None:
            self.reports_["data_dictionary"] = dictionary
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

        for feature in selected_features:
            woe_info = woe_results.get("woe_values", {}).get(feature)
            if not woe_info or "stats" not in woe_info:
                continue

            table = pd.DataFrame(woe_info["stats"])
            table["feature"] = feature
            table["type"] = woe_info.get("type", "unknown")
            if "woe" in table.columns:
                table = table.sort_values("woe").reset_index(drop=True)
            tables[feature] = table

        self.reports_["woe_tables"] = tables
        return tables

    def generate_risk_band_report(self, risk_bands: Dict[str, Any]) -> pd.DataFrame:
        """Summarise risk band allocation and diagnostics."""

        if "bands" not in risk_bands:
            return pd.DataFrame()

        bands_df = risk_bands["bands"]
        metrics = risk_bands.get("metrics", {})

        summary_lines = [
            f"Number of bands: {len(bands_df)}",
            f"Herfindahl Index: {metrics.get('herfindahl_index', 0):.4f}",
            f"Entropy: {metrics.get('entropy', 0):.4f}",
            f"Gini Coefficient: {metrics.get('gini_coefficient', 0):.4f}",
            f"KS Statistic: {metrics.get('ks_stat', 0):.4f}",
            f"Hosmer-Lemeshow p-value: {metrics.get('hosmer_lemeshow_p', 0):.4f}",
            f"Top 20% Concentration: {metrics.get('cr_top20', 0):.2%}",
            f"Top 50% Concentration: {metrics.get('cr_top50', 0):.2%}",
        ]

        if "binomial_tests" in metrics:
            total = len(metrics["binomial_tests"])
            significant = sum(
                1 for result in metrics["binomial_tests"].values() if result.get("significant", False)
            )
            summary_lines.append(f"Significant bands (binomial test): {significant}/{total}")

        self.reports_["risk_bands"] = bands_df
        self.reports_["risk_bands_summary"] = {"Risk Bands Summary": summary_lines}
        return bands_df

    def generate_calibration_report(
        self,
        stage1_results: Optional[Dict[str, Any]],
        stage2_results: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect calibration metrics from stage 1 and stage 2 routines."""

        report: Dict[str, Any] = {}

        if stage1_results:
            report["stage1"] = {
                "method": getattr(self.config, "calibration_method", "unknown"),
                "metrics": stage1_results.get("calibration_metrics", {}),
            }

        if stage2_results:
            report["stage2"] = {
                "method": getattr(self.config, "stage2_method", "unknown"),
                "metrics": stage2_results.get("stage2_metrics", {}),
            }

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
        """Persist collected reports to an Excel workbook."""

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            features = self.reports_.get("features")
            if isinstance(features, pd.DataFrame) and not features.empty:
                features.to_excel(writer, sheet_name="Features", index=False)

            woe_tables = self.reports_.get("woe_tables", {})
            for index, (feature, table) in enumerate(woe_tables.items()):
                sheet_name = f"WOE_{index + 1}" if index < 10 else f"WOE_{feature[:20]}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)

            risk_bands = self.reports_.get("risk_bands")
            if isinstance(risk_bands, pd.DataFrame) and not risk_bands.empty:
                risk_bands.to_excel(writer, sheet_name="Risk_Bands", index=False)

            model_perf = self.reports_.get("model_performance", {})
            if model_perf.get("model_scores"):
                pd.DataFrame(model_perf["model_scores"]).T.to_excel(
                    writer, sheet_name="Model_Scores"
                )

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

