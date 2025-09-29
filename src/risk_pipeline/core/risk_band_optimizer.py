"""
Optimal Risk Band Analyzer with comprehensive metrics
Includes: Herfindahl Index, Hosmer-Lemeshow, Binomial Tests
"""

import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings('ignore')


class OptimalRiskBandAnalyzer:
    """
    Risk band optimization with comprehensive statistical tests.

    Features:
    - Band optimization for PSI stability
    - Event rate monotonicity
    - Herfindahl-Hirschman Index for concentration
    - Hosmer-Lemeshow test for calibration
    - Binomial tests for each band
    - Gini coefficient for distribution
    - Entropy measures
    """

    def __init__(self, config):
        self.config = config
        self.bands_ = None
        self.band_stats_ = None
        self.band_summary_ = None
        self.metrics_ = {}

    def optimize_bands(self, predictions: np.ndarray, actuals: np.ndarray,
                      n_bands: int = 10, method: str = 'quantile') -> pd.DataFrame:
        """
        Optimize risk bands for stability and discrimination.

        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions (probabilities)
        actuals : np.ndarray
            Actual outcomes (0/1)
        n_bands : int
            Number of risk bands
        method : str
            Banding method ('quantile', 'equal_width', 'optimal', 'pd_constraints')

        Returns:
        --------
        pd.DataFrame : Band statistics
        """

        print(f"    Optimizing {n_bands} risk bands (method={method})...")

        band_stats: Optional[pd.DataFrame] = None
        summary_meta: Optional[Dict[str, Any]] = None
        method_key = (method or 'quantile').lower()

        if method_key == 'quantile':
            bands = self._create_quantile_bands(predictions, n_bands)
        elif method_key == 'equal_width':
            bands = self._create_equal_width_bands(predictions, n_bands)
        elif method_key == 'optimal':
            bands = self._optimize_bands_iterative(predictions, actuals, n_bands)
        elif method_key in {'pd_constraints', 'score_constraints', 'constrained'}:
            bands, constrained_stats, summary_meta = self._optimize_bands_with_constraints(
                predictions, actuals, n_bands
            )
            if bands is None or constrained_stats is None:
                print('      Warning: constrained risk band optimization failed; falling back to quantile bands')
                bands = self._create_quantile_bands(predictions, n_bands)
                summary_meta = None
            else:
                band_stats = constrained_stats
        else:
            bands = self._create_quantile_bands(predictions, n_bands)

        if band_stats is None:
            band_assignments = self._assign_to_bands(predictions, bands)
            band_stats = self._calculate_band_statistics(
                predictions, actuals, band_assignments
            )
        else:
            if 'cum_count' not in band_stats.columns:
                band_assignments = self._assign_to_bands(predictions, bands)
                base_stats = self._calculate_band_statistics(
                    predictions, actuals, band_assignments
                )
                extra_cols = [col for col in band_stats.columns if col not in base_stats.columns]
                band_stats = base_stats.merge(
                    band_stats[['band'] + extra_cols],
                    on='band',
                    how='left'
                )

        if method_key in {'pd_constraints', 'score_constraints', 'constrained'} and summary_meta is None:
            penalty, augmented_stats, summary = self._score_band_configuration(band_stats)
            band_stats = augmented_stats
            summary['total_penalty'] = penalty
            summary['n_bins'] = int(len(augmented_stats))
            summary['method'] = method_key
            summary['table'] = augmented_stats
            summary_meta = summary

        is_monotonic = self._check_monotonicity(band_stats)
        if not is_monotonic:
            print('      Warning: Bands are not monotonic in bad rate')

        self.bands_ = bands
        self.band_stats_ = band_stats
        self.band_summary_ = summary_meta

        return band_stats

    def calculate_band_metrics(self, band_stats: pd.DataFrame,
                              predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Calculate comprehensive metrics for risk bands.

        Returns dict with:
        - herfindahl_index: Concentration measure
        - entropy: Distribution entropy
        - gini_coefficient: Inequality measure
        - hosmer_lemeshow_stat: Calibration test statistic
        - hosmer_lemeshow_p: Calibration test p-value
        - binomial_tests: Dict of binomial test results per band
        - ks_stat: Kolmogorov-Smirnov statistic
        """

        df = band_stats.copy() if isinstance(band_stats, pd.DataFrame) else pd.DataFrame()
        if df.empty:
            return {}

        if 'pct_count' not in df.columns and {'count'}.issubset(df.columns):
            total = float(df['count'].sum())
            if total > 0:
                df['pct_count'] = df['count'] / total
        if 'bad_rate' not in df.columns and {'bad_count', 'count'}.issubset(df.columns):
            counts = df['count'].to_numpy(dtype=float)
            df['bad_rate'] = np.divide(
                df['bad_count'],
                counts,
                out=np.zeros_like(counts, dtype=float),
                where=counts > 0,
            )
        if not {'pct_count', 'bad_rate'}.issubset(df.columns):
            return {}

        metrics = {}

        # Herfindahl-Hirschman Index
        metrics['herfindahl_index'] = self._calculate_herfindahl_index(df)
        print(f"      Herfindahl Index: {metrics['herfindahl_index']:.4f}")

        # Entropy
        metrics['entropy'] = self._calculate_entropy(band_stats)
        print(f"      Entropy: {metrics['entropy']:.4f}")

        # Gini Coefficient
        metrics['gini_coefficient'] = self._calculate_gini_coefficient(band_stats)
        print(f"      Gini Coefficient: {metrics['gini_coefficient']:.4f}")

        # Hosmer-Lemeshow Test
        hl_stat, hl_p = self._hosmer_lemeshow_test(predictions, actuals)
        metrics['hosmer_lemeshow_stat'] = hl_stat
        metrics['hosmer_lemeshow_p'] = hl_p
        print(f"      Hosmer-Lemeshow p-value: {hl_p:.4f}")

        # Binomial Tests
        metrics['binomial_tests'] = self._perform_binomial_tests(band_stats)

        # KS Statistic
        metrics['ks_stat'] = self._calculate_ks_stat(band_stats)
        print(f"      KS Statistic: {metrics['ks_stat']:.4f}")

        # Concentration Ratio
        metrics['cr_top20'] = self._calculate_concentration_ratio(band_stats, 0.2)
        metrics['cr_top50'] = self._calculate_concentration_ratio(band_stats, 0.5)

        if isinstance(getattr(self, 'band_summary_', None), dict):
            summary_meta = self.band_summary_
            metrics['ci_overlaps'] = summary_meta.get('ci_overlaps')
            metrics['binomial_pass_weight'] = summary_meta.get('binomial_pass_weight')
            metrics['binomial_pass_rate'] = summary_meta.get('binomial_pass_rate')
            metrics['risk_band_penalty'] = summary_meta.get('total_penalty')
            metrics['weight_violations'] = summary_meta.get('weight_violations')
            metrics['hhi_total'] = summary_meta.get('hhi_total')
            metrics['monotonic_pd'] = summary_meta.get('monotonic_pd')
            metrics['monotonic_dr'] = summary_meta.get('monotonic_dr')

        self.metrics_ = metrics
        return metrics

    def _create_quantile_bands(self, predictions: np.ndarray, n_bands: int) -> np.ndarray:
        """Create quantile-based bands."""

        percentiles = np.linspace(0, 100, n_bands + 1)
        bands = np.percentile(predictions, percentiles)

        # Ensure unique bands
        bands = np.unique(bands)

        if len(bands) < n_bands + 1:
            print(f"      Warning: Could only create {len(bands)-1} unique bands")

        return bands

    def _create_equal_width_bands(self, predictions: np.ndarray, n_bands: int) -> np.ndarray:
        """Create equal-width bands."""

        min_score = predictions.min()
        max_score = predictions.max()
        bands = np.linspace(min_score, max_score, n_bands + 1)

        return bands

    def _optimize_bands_iterative(self, predictions: np.ndarray, actuals: np.ndarray,
                                  n_bands: int) -> np.ndarray:
        """
        Optimize bands to maximize discrimination while ensuring stability.
        """

        # Start with quantile bands
        bands = self._create_quantile_bands(predictions, n_bands)

        # Iterate to improve
        for _ in range(10):
            # Calculate current performance
            band_assignments = self._assign_to_bands(predictions, bands)
            band_stats = self._calculate_band_statistics(predictions, actuals, band_assignments)

            # Check if improvement needed
            if self._check_monotonicity(band_stats):
                break

            # Adjust bands
            bands = self._adjust_bands_for_monotonicity(
                predictions, actuals, bands
            )

        return bands

    def _adjust_bands_for_monotonicity(self, predictions: np.ndarray, actuals: np.ndarray,
                                       bands: np.ndarray) -> np.ndarray:
        """Adjust bands to improve monotonicity."""

        # Calculate bad rates for each band
        bad_rates = []
        for i in range(len(bands) - 1):
            if i == 0:
                mask = predictions <= bands[i + 1]
            elif i == len(bands) - 2:
                mask = predictions > bands[i]
            else:
                mask = (predictions > bands[i]) & (predictions <= bands[i + 1])

            bad_rate = actuals[mask].mean() if mask.sum() > 0 else 0
            bad_rates.append(bad_rate)

        # Merge bands with non-monotonic bad rates
        new_bands = [bands[0]]
        for i in range(1, len(bands) - 1):
            if i < len(bad_rates) and i > 0:
                if bad_rates[i] > bad_rates[i-1]:  # Assuming higher score = lower risk
                    continue
            new_bands.append(bands[i])
        new_bands.append(bands[-1])

        return np.array(new_bands)

    def _optimize_bands_with_constraints(self, predictions: np.ndarray, actuals: np.ndarray,
                                         target_bins: int) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Construct bands using constrained optimization inspired by PD binning reference."""

        predictions = np.asarray(predictions, dtype=float)
        actuals = np.asarray(actuals, dtype=float)

        if predictions.size == 0 or np.allclose(np.nanstd(predictions), 0.0):
            return None, None, None

        cfg = self.config
        min_bins = int(getattr(cfg, 'risk_band_min_bins', max(2, min(target_bins, 7))))
        max_bins = int(getattr(cfg, 'risk_band_max_bins', max(target_bins, 10)))
        if target_bins:
            max_bins = max(min_bins, min(max_bins, target_bins))
            min_bins = min(min_bins, max_bins)

        n_records = predictions.size
        if n_records < min_bins:
            return None, None, None

        order = np.argsort(predictions)
        pred_sorted = predictions[order]
        actual_sorted = np.nan_to_num(actuals[order], nan=0.0)

        micro_target = int(getattr(cfg, 'risk_band_micro_bins', max_bins * 20))
        micro_bins = max(max_bins, min(micro_target, n_records))
        boundaries = [0]
        for i in range(1, micro_bins):
            idx = int(round(i * n_records / micro_bins))
            if idx > boundaries[-1]:
                boundaries.append(idx)
        if boundaries[-1] != n_records:
            boundaries.append(n_records)
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != n_records:
            boundaries.append(n_records)

        num_segments = len(boundaries) - 1
        if num_segments < min_bins:
            return None, None, None

        max_bins = min(max_bins, num_segments)
        min_bins = min(min_bins, max_bins)

        prefix_bad = np.cumsum(actual_sorted)
        prefix_pred = np.cumsum(pred_sorted)

        min_weight = float(getattr(cfg, 'risk_band_min_weight', 0.05))
        max_weight = float(getattr(cfg, 'risk_band_max_weight', 0.30))
        alpha = float(getattr(cfg, 'risk_band_alpha', 0.05))
        z_value = stats.norm.ppf(1 - alpha / 2) if 0 < alpha < 1 else stats.norm.ppf(0.975)

        segment_cache: List[List[Optional[Dict[str, Any]]]] = [[None] * (num_segments + 1) for _ in range(num_segments)]
        for i in range(num_segments):
            for j in range(i + 1, num_segments + 1):
                start = boundaries[i]
                end = boundaries[j]
                count = end - start
                if count <= 0:
                    continue
                bads = prefix_bad[end - 1] - (prefix_bad[start - 1] if start > 0 else 0)
                pred_sum = prefix_pred[end - 1] - (prefix_pred[start - 1] if start > 0 else 0)
                mean_pd = float(pred_sum / count)
                observed_dr = float(bads / count)
                pd_var = max(mean_pd * (1 - mean_pd), 1e-6)
                pd_se = math.sqrt(pd_var / count)
                ci_low = max(mean_pd - z_value * pd_se, 0.0)
                ci_high = min(mean_pd + z_value * pd_se, 1.0)
                binomial_pass = ci_low <= observed_dr <= ci_high
                weight = count / n_records
                dr_pd_diff = abs(observed_dr - mean_pd)

                penalty = 0.0
                if weight < min_weight:
                    penalty += (min_weight - weight) * 3000.0
                if weight > max_weight:
                    penalty += (weight - max_weight) * 3000.0
                penalty += dr_pd_diff * 200.0
                if not binomial_pass:
                    penalty += abs(dr_pd_diff) * 2000.0 + 500.0

                segment_cache[i][j] = {
                    'start': start,
                    'end': end,
                    'count': count,
                    'bads': float(bads),
                    'weight': weight,
                    'mean_pd': mean_pd,
                    'observed_dr': observed_dr,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'dr_pd_diff': dr_pd_diff,
                    'binomial_pass': binomial_pass,
                    'penalty': penalty,
                    'min_score': float(pred_sorted[start]),
                    'max_score': float(pred_sorted[end - 1]),
                }

        dp = [[math.inf] * (num_segments + 1) for _ in range(max_bins + 1)]
        back: List[List[Optional[int]]] = [[None] * (num_segments + 1) for _ in range(max_bins + 1)]
        dp[0][0] = 0.0

        for k in range(1, max_bins + 1):
            for j in range(1, num_segments + 1):
                for i in range(k - 1, j):
                    segment = segment_cache[i][j]
                    if segment is None:
                        continue
                    previous = dp[k - 1][i]
                    if not math.isfinite(previous):
                        continue
                    cost = previous + segment['penalty']
                    if cost < dp[k][j]:
                        dp[k][j] = cost
                        back[k][j] = i

        best_solution = None
        best_stats: Optional[pd.DataFrame] = None
        best_summary: Optional[Dict[str, Any]] = None
        best_penalty = math.inf

        for bins_count in range(min_bins, max_bins + 1):
            if not math.isfinite(dp[bins_count][num_segments]):
                continue
            chain = []
            curr = num_segments
            steps = bins_count
            valid = True
            while steps > 0:
                prev = back[steps][curr]
                if prev is None:
                    valid = False
                    break
                chain.append((prev, curr))
                curr = prev
                steps -= 1
            if not valid or curr != 0:
                continue
            chain.reverse()

            edges = [float(pred_sorted[0])]
            for seg_index, (start_idx, end_idx) in enumerate(chain[:-1]):
                boundary = segment_cache[start_idx][end_idx]['end']
                if boundary >= n_records:
                    continue
                left_val = float(pred_sorted[boundary - 1])
                right_val = float(pred_sorted[boundary]) if boundary < n_records else left_val
                if right_val == left_val:
                    boundary_value = left_val + 1e-6 * (seg_index + 1)
                else:
                    boundary_value = (left_val + right_val) / 2.0
                if boundary_value <= edges[-1]:
                    boundary_value = edges[-1] + 1e-6
                edges.append(boundary_value)
            edges.append(float(pred_sorted[-1]) + 1e-6)
            edges_array = np.array(edges, dtype=float)

            assignments = self._assign_to_bands(predictions, edges_array)
            band_stats = self._calculate_band_statistics(predictions, actuals, assignments)
            penalty, augmented, summary = self._score_band_configuration(band_stats)
            if not math.isfinite(penalty):
                continue

            if penalty < best_penalty:
                best_penalty = penalty
                best_solution = edges_array
                best_stats = augmented
                summary['total_penalty'] = penalty
                summary['n_bins'] = int(len(augmented))
                summary['method'] = 'pd_constraints'
                summary['table'] = augmented
                best_summary = summary

        if best_solution is None or best_stats is None or best_summary is None:
            return None, None, None

        return best_solution, best_stats, best_summary


    def _annotate_band_statistics(self, band_stats: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        '''Annotate per-band statistics with binomial and summary metrics.'''

        if band_stats is None or band_stats.empty:
            return pd.DataFrame(), {}

        df = band_stats.copy().reset_index(drop=True)
        if 'band' in df.columns:
            df = df.sort_values('band').reset_index(drop=True)

        if 'pct_count' not in df.columns and {'count'}.issubset(df.columns):
            total = float(df['count'].sum())
            if total > 0:
                df['pct_count'] = df['count'] / total

        if 'bad_rate' not in df.columns and {'bad_count', 'count'}.issubset(df.columns):
            df['bad_rate'] = np.divide(
                df['bad_count'],
                df['count'],
                out=np.zeros_like(df['bad_count'], dtype=float),
                where=df['count'].to_numpy(dtype=float) > 0
            )

        required_cols = {'count', 'pct_count', 'bad_rate', 'avg_score'}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame(), {}

        total = float(df['count'].sum())
        if total <= 0:
            return pd.DataFrame(), {}

        cfg = self.config
        alpha = float(getattr(cfg, 'risk_band_alpha', 0.05) or 0.05)
        if not 0 < alpha < 1:
            alpha = 0.05
        confidence = 1 - alpha

        min_weight = float(getattr(cfg, 'risk_band_min_weight', 0.05))
        max_weight = float(getattr(cfg, 'risk_band_max_weight', 0.30))

        df['pct_count'] = df['count'] / total
        df['weight'] = df['pct_count']
        df['avg_score'] = df['avg_score'].astype(float).clip(0.0, 1.0)
        df['bad_rate'] = df['bad_rate'].astype(float).clip(0.0, 1.0)
        df['expected_defaults'] = df['avg_score'] * df['count']
        df['dr_pd_diff'] = df['bad_rate'] - df['avg_score']
        df['abs_dr_pd_diff'] = df['dr_pd_diff'].abs()
        df['hhi_contrib'] = df['pct_count'] ** 2
        df['bin_range'] = df.apply(lambda row: f"[{row['min_score']:.4f}, {row['max_score']:.4f}]", axis=1)
        df['band_label'] = df['band'].apply(lambda band: f"Band {int(band)}")

        ci_lowers: List[float] = []
        ci_uppers: List[float] = []
        p_values: List[float] = []
        binomial_pass: List[bool] = []

        for _, row in df.iterrows():
            n_obs = int(row['count'])
            n_bad = int(row['bad_count'])
            if n_obs > 0:
                predicted_pd = float(np.clip(row['avg_score'], 1e-9, 1 - 1e-9))
                try:
                    binom_res = stats.binomtest(n_bad, n_obs, predicted_pd, alternative='two-sided')
                except TypeError:
                    binom_res = stats.binomtest(n_bad, n_obs, predicted_pd)
                p_val = float(binom_res.pvalue)
                try:
                    ci = binom_res.proportion_ci(confidence_level=confidence, method='wilson')
                except TypeError:
                    ci = stats.binomtest(n_bad, n_obs).proportion_ci(confidence_level=confidence, method='wilson')
                ci_low = float(ci.low)
                ci_high = float(ci.high)
                in_ci = ci_low <= predicted_pd <= ci_high
            else:
                p_val = np.nan
                ci_low = np.nan
                ci_high = np.nan
                in_ci = True
            ci_lowers.append(ci_low)
            ci_uppers.append(ci_high)
            p_values.append(p_val)
            binomial_pass.append(in_ci)

        df['ci_lower'] = ci_lowers
        df['ci_upper'] = ci_uppers
        df['ci_lower_observed'] = df['ci_lower']
        df['ci_upper_observed'] = df['ci_upper']
        df['binomial_p_value'] = p_values
        df['binomial_pass'] = binomial_pass
        df['binomial_result'] = df['binomial_pass'].map(lambda flag: 'Pass' if flag else 'Reject')
        df['predicted_within_ci'] = df['binomial_pass']
        df['mean_pd_gt_dr'] = (df['avg_score'] >= df['bad_rate']).astype(int)
        df['weight_pct'] = df['pct_count']
        df['mean_pd'] = df['avg_score']
        df['observed_dr'] = df['bad_rate']
        df['bad_minus_expected'] = df['bad_count'] - df['expected_defaults']
        df['binomial_distance'] = df['abs_dr_pd_diff']
        df['ci_width'] = df['ci_upper'] - df['ci_lower']

        ci_overlaps = int(np.sum(df['ci_upper'].values[:-1] > df['ci_lower'].values[1:] + 1e-12)) if len(df) > 1 else 0
        binomial_failures = int((~df['binomial_pass']).sum())
        binomial_pass_weight = float(df.loc[df['binomial_pass'], 'pct_count'].sum())
        binomial_pass_rate = float(df['binomial_pass'].mean()) if len(df) else 0.0
        avg_diff = float(df['dr_pd_diff'].mean()) if len(df) else 0.0
        avg_abs_diff = float(df['abs_dr_pd_diff'].mean()) if len(df) else 0.0
        mean_pd_failures = int(((~df['binomial_pass']) & (df['dr_pd_diff'] < 0)).sum())
        hhi_total = float(df['hhi_contrib'].sum())
        weight_violations = int(((df['pct_count'] < min_weight - 1e-9) | (df['pct_count'] > max_weight + 1e-9)).sum())
        monotonic_pd = bool(np.all(np.diff(df['avg_score']) >= -1e-6))
        monotonic_dr = bool(np.all(np.diff(df['bad_rate']) >= -1e-6))

        summary = {
            'ci_overlaps': ci_overlaps,
            'binomial_failures': binomial_failures,
            'binomial_pass_weight': binomial_pass_weight,
            'binomial_pass_rate': binomial_pass_rate,
            'dr_pd_diff_mean': avg_diff,
            'dr_pd_diff_mean_abs': avg_abs_diff,
            'mean_pd_gt_dr_failures': mean_pd_failures,
            'hhi_total': hhi_total,
            'weight_violations': weight_violations,
            'monotonic_pd': monotonic_pd,
            'monotonic_dr': monotonic_dr,
            'n_bins': int(len(df)),
        }

        return df, summary



    def _score_band_configuration(self, band_stats: pd.DataFrame) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
        '''Evaluate band statistics against business constraints and compute penalty.'''

        annotated_df, summary = self._annotate_band_statistics(band_stats)
        if annotated_df is None or annotated_df.empty:
            return math.inf, annotated_df, summary or {}

        cfg = self.config
        min_weight = float(getattr(cfg, 'risk_band_min_weight', 0.05))
        max_weight = float(getattr(cfg, 'risk_band_max_weight', 0.30))
        hhi_threshold = float(getattr(cfg, 'risk_band_hhi_threshold', 0.15))
        required_pass_weight = float(getattr(cfg, 'risk_band_binomial_pass_weight', 0.85))

        ci_overlaps = summary.get('ci_overlaps', 0)
        binomial_failures = summary.get('binomial_failures', 0)
        binomial_pass_weight = summary.get('binomial_pass_weight', 0.0)
        avg_abs_diff = summary.get('dr_pd_diff_mean_abs', 0.0)
        mean_pd_failures = summary.get('mean_pd_gt_dr_failures', 0)
        hhi_total = summary.get('hhi_total', 0.0)
        weight_violations = summary.get('weight_violations', 0)
        monotonic_pd = summary.get('monotonic_pd', True)
        monotonic_dr = summary.get('monotonic_dr', True)

        overlaps_penalty = ci_overlaps * 10000.0
        binomial_penalty = binomial_failures * 200.0
        binomial_weight_penalty = 0.0
        if binomial_pass_weight < required_pass_weight:
            binomial_weight_penalty = (required_pass_weight - binomial_pass_weight) * 500.0 * 1000.0
        dr_pd_penalty = avg_abs_diff * 80.0 * 100.0
        mean_pd_penalty = mean_pd_failures * 50.0
        hhi_penalty = max(0.0, hhi_total - hhi_threshold) * 100.0 * 20.0
        weight_penalty = weight_violations * 50.0 * 100.0
        monotonic_penalty = 0.0 if (monotonic_pd and monotonic_dr) else 1_000_000.0

        total_penalty = overlaps_penalty + binomial_penalty + binomial_weight_penalty + dr_pd_penalty + mean_pd_penalty + hhi_penalty + weight_penalty + monotonic_penalty

        summary = dict(summary)
        summary.update({
            'total_penalty': total_penalty,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'hhi_threshold': hhi_threshold,
            'required_pass_weight': required_pass_weight,
            'table': annotated_df,
        })

        return total_penalty, annotated_df, summary

    def _assign_to_bands(self, predictions: np.ndarray, bands: np.ndarray) -> np.ndarray:
        """Assign predictions to bands."""

        assignments = np.zeros(len(predictions), dtype=int)

        for i in range(len(bands) - 1):
            if i == 0:
                mask = predictions <= bands[i + 1]
            elif i == len(bands) - 2:
                mask = predictions > bands[i]
            else:
                mask = (predictions > bands[i]) & (predictions <= bands[i + 1])

            assignments[mask] = i + 1

        return assignments

    def _calculate_band_statistics(self, predictions: np.ndarray, actuals: np.ndarray,
                                  band_assignments: np.ndarray) -> pd.DataFrame:
        """Calculate statistics for each band."""

        stats = []

        for band in range(1, band_assignments.max() + 1):
            mask = band_assignments == band
            n_obs = mask.sum()

            if n_obs > 0:
                band_preds = predictions[mask]
                band_actuals = actuals[mask]

                stats.append({
                    'band': band,
                    'count': n_obs,
                    'pct_count': n_obs / len(predictions),
                    'bad_count': band_actuals.sum(),
                    'good_count': n_obs - band_actuals.sum(),
                    'bad_rate': band_actuals.mean(),
                    'avg_score': band_preds.mean(),
                    'min_score': band_preds.min(),
                    'max_score': band_preds.max()
                })

        df = pd.DataFrame(stats)

        # Check if we have any bands
        if df.empty:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['band', 'count', 'pct_count', 'bad_count',
                                        'good_count', 'bad_rate', 'avg_score',
                                        'min_score', 'max_score', 'cum_count',
                                        'cum_bad', 'cum_good', 'cum_bad_rate',
                                        'bad_capture', 'good_capture', 'ks'])

        # Calculate cumulative statistics
        df['cum_count'] = df['count'].cumsum()
        df['cum_bad'] = df['bad_count'].cumsum()
        df['cum_good'] = df['good_count'].cumsum()
        df['cum_bad_rate'] = df['cum_bad'] / df['cum_count']

        # Calculate KS for each band
        total_bad = actuals.sum()
        total_good = len(actuals) - total_bad

        df['bad_capture'] = df['cum_bad'] / total_bad if total_bad > 0 else 0
        df['good_capture'] = df['cum_good'] / total_good if total_good > 0 else 0
        df['ks'] = np.abs(df['bad_capture'] - df['good_capture'])

        # Calculate PSI (if reference provided)
        df['psi'] = 0  # Placeholder - would need reference distribution

        return df

    def _check_monotonicity(self, band_stats: pd.DataFrame) -> bool:
        """Check if bands have monotonic bad rates."""

        bad_rates = band_stats['bad_rate'].values

        # Check for monotonic decrease (higher band = lower risk)
        is_decreasing = all(bad_rates[i] >= bad_rates[i+1]
                           for i in range(len(bad_rates)-1))

        return is_decreasing

    def _calculate_herfindahl_index(self, band_stats: pd.DataFrame) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.

        HHI = sum(share_i^2) where share_i is the proportion in band i

        Values:
        - < 0.15: Low concentration
        - 0.15-0.25: Moderate concentration
        - > 0.25: High concentration
        """

        shares = band_stats['pct_count'].values
        hhi = np.sum(shares ** 2)

        return hhi

    def _calculate_entropy(self, band_stats: pd.DataFrame) -> float:
        """
        Calculate entropy of distribution across bands.

        Higher entropy = more uniform distribution
        """

        shares = band_stats['pct_count'].values
        # Remove zero shares to avoid log(0)
        shares = shares[shares > 0]

        entropy = -np.sum(shares * np.log(shares))

        return entropy

    def _calculate_gini_coefficient(self, band_stats: pd.DataFrame) -> float:
        """
        Calculate Gini coefficient for inequality in distribution.

        0 = perfect equality, 1 = perfect inequality
        """

        # Check if band_stats is empty
        if band_stats.empty or len(band_stats) == 0:
            return 0.0

        # Sort by bad rate (ascending)
        sorted_stats = band_stats.sort_values('bad_rate')

        # Calculate Lorenz curve
        cum_count = np.cumsum(sorted_stats['pct_count'].values)
        cum_bad = np.cumsum(sorted_stats['bad_count'].values)

        # Check if we have any bads
        if len(cum_bad) == 0 or cum_bad[-1] == 0:
            return 0.0

        # Normalize
        cum_count = np.concatenate([[0], cum_count])
        cum_bad = cum_bad / cum_bad[-1]
        cum_bad = np.concatenate([[0], cum_bad])

        # Calculate area under Lorenz curve
        area = np.trapz(cum_bad, cum_count)

        # Gini = 1 - 2 * area
        gini = 1 - 2 * area

        return abs(gini)

    def _hosmer_lemeshow_test(self, predictions: np.ndarray, actuals: np.ndarray,
                              n_groups: int = 10) -> Tuple[float, float]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.

        H0: Model is well calibrated
        High p-value = good calibration
        """

        # Create groups based on predicted probabilities
        quantiles = np.percentile(predictions, np.linspace(0, 100, n_groups + 1))
        groups = np.digitize(predictions, quantiles[1:-1])

        # Calculate observed and expected for each group
        observed = []
        expected = []
        counts = []

        for g in range(n_groups):
            mask = groups == g
            count = mask.sum()
            if count > 0:
                obs_events = actuals[mask].sum()
                exp_events = predictions[mask].sum()

                observed.append(obs_events)
                expected.append(exp_events)
                counts.append(count)

        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        counts = np.array(counts, dtype=float)

        if len(observed) == 0:
            return 0.0, 1.0

        # Calculate test statistic
        # Avoid division by zero
        expected = np.maximum(expected, 0.5)
        expected_non_events = np.maximum(counts - expected, 0.5)

        chi_square = np.sum(
            (observed - expected) ** 2 / expected +
            ((counts - observed) - expected_non_events) ** 2 / expected_non_events
        )

        # Degrees of freedom = max(unique groups - 2, 1)
        df = max(len(counts) - 2, 1)

        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        return chi_square, p_value


    def _perform_binomial_tests(self, band_stats: pd.DataFrame) -> pd.DataFrame:
        '''Return detailed binomial test diagnostics per band.'''

        annotated_df, summary = self._annotate_band_statistics(band_stats)
        if isinstance(summary, dict) and summary:
            summary_with_table = dict(summary)
            summary_with_table['table'] = annotated_df
            existing_summary = getattr(self, 'band_summary_', None)
            if isinstance(existing_summary, dict):
                merged = dict(existing_summary)
                merged.update(summary_with_table)
                self.band_summary_ = merged
            else:
                self.band_summary_ = summary_with_table

        detail_columns = [
            'band',
            'band_label',
            'bin_range',
            'count',
            'weight_pct',
            'mean_pd',
            'observed_dr',
            'dr_pd_diff',
            'ci_lower_observed',
            'ci_upper_observed',
            'binomial_p_value',
            'binomial_result',
            'binomial_pass',
            'predicted_within_ci',
            'binomial_distance',
            'hhi_contrib',
            'ks',
            'bad_capture',
            'good_capture',
            'expected_defaults',
            'bad_minus_expected',
        ]
        existing_cols = [col for col in detail_columns if col in annotated_df.columns]
        if not existing_cols:
            return annotated_df
        return annotated_df[existing_cols]

    def _calculate_ks_stat(self, band_stats: pd.DataFrame) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""

        return band_stats['ks'].max()

    def _calculate_concentration_ratio(self, band_stats: pd.DataFrame, top_pct: float) -> float:
        """
        Calculate concentration ratio.

        What percentage of bads are captured in top X% of population?
        """

        # Sort by bad rate (descending - worst risks first)
        sorted_stats = band_stats.sort_values('bad_rate', ascending=False)

        # Find how many bads are in top X% of population
        cum_pct = 0
        cum_bads = 0
        total_bads = sorted_stats['bad_count'].sum()

        for _, row in sorted_stats.iterrows():
            if cum_pct >= top_pct:
                break

            cum_pct += row['pct_count']
            cum_bads += row['bad_count']

        return cum_bads / total_bads if total_bads > 0 else 0

    def assign_bands(self, predictions: np.ndarray, bands: np.ndarray) -> np.ndarray:
        """Assign predictions to pre-defined bands."""

        return self._assign_to_bands(predictions, bands)

    def plot_risk_bands(self, band_stats: pd.DataFrame):
        """Plot risk band distribution and bad rates."""

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Band distribution
        ax = axes[0, 0]
        ax.bar(band_stats['band'], band_stats['count'])
        ax.set_xlabel('Risk Band')
        ax.set_ylabel('Count')
        ax.set_title('Risk Band Distribution')

        # 2. Bad rate by band
        ax = axes[0, 1]
        ax.plot(band_stats['band'], band_stats['bad_rate'], 'o-', color='red')
        ax.set_xlabel('Risk Band')
        ax.set_ylabel('Bad Rate')
        ax.set_title('Bad Rate by Risk Band')
        ax.grid(True, alpha=0.3)

        # 3. Cumulative capture
        ax = axes[1, 0]
        ax.plot(band_stats['cum_count'] / band_stats['cum_count'].max(),
               band_stats['cum_bad'] / band_stats['cum_bad'].max(),
               'b-', label='Bad Capture')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('Population %')
        ax.set_ylabel('Bad Capture %')
        ax.set_title('Cumulative Bad Capture')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. KS by band
        ax = axes[1, 1]
        ax.plot(band_stats['band'], band_stats['ks'], 'o-', color='green')
        ax.set_xlabel('Risk Band')
        ax.set_ylabel('KS')
        ax.set_title('KS Statistic by Band')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def export_sql(self, band_stats: pd.DataFrame) -> str:
        """Generate SQL code for risk band assignment."""

        sql = "-- Risk Band Assignment SQL\n"
        sql += "CASE\n"

        for _, row in band_stats.iterrows():
            band = row['band']
            min_score = row['min_score']
            max_score = row['max_score']

            if band == 1:
                sql += f"  WHEN risk_score <= {max_score:.6f} THEN {band}\n"
            elif band == len(band_stats):
                sql += f"  WHEN risk_score > {min_score:.6f} THEN {band}\n"
            else:
                sql += f"  WHEN risk_score > {min_score:.6f} AND risk_score <= {max_score:.6f} THEN {band}\n"

        sql += "  ELSE NULL\n"
        sql += "END AS risk_band"

        return sql

    def export_python(self, band_stats: pd.DataFrame) -> str:
        """Generate Python code for risk band assignment."""

        code = "# Risk Band Assignment Python Code\n"
        code += "def assign_risk_band(risk_score):\n"
        code += "    \"\"\"\n"
        code += "    Assign risk score to risk band.\n"
        code += "    \n"
        code += "    Parameters:\n"
        code += "    -----------\n"
        code += "    risk_score : float\n"
        code += "        Risk score (probability)\n"
        code += "    \n"
        code += "    Returns:\n"
        code += "    --------\n"
        code += "    int : Risk band (1 to n)\n"
        code += "    \"\"\"\n"

        for _, row in band_stats.iterrows():
            band = row['band']
            min_score = row['min_score']
            max_score = row['max_score']

            if band == 1:
                code += f"    if risk_score <= {max_score:.6f}:\n"
                code += f"        return {band}\n"
            elif band == len(band_stats):
                code += f"    elif risk_score > {min_score:.6f}:\n"
                code += f"        return {band}\n"
            else:
                code += f"    elif {min_score:.6f} < risk_score <= {max_score:.6f}:\n"
                code += f"        return {band}\n"

        code += "    else:\n"
        code += "        return None\n"

        return code