"""
Optimal Risk Band Analyzer with comprehensive metrics
Includes: Herfindahl Index, Hosmer-Lemeshow, Binomial Tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
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
            Banding method ('quantile', 'equal_width', 'optimal')

        Returns:
        --------
        pd.DataFrame : Band statistics
        """

        print(f"    Optimizing {n_bands} risk bands (method={method})...")

        # Create initial bands
        if method == 'quantile':
            bands = self._create_quantile_bands(predictions, n_bands)
        elif method == 'equal_width':
            bands = self._create_equal_width_bands(predictions, n_bands)
        elif method == 'optimal':
            bands = self._optimize_bands_iterative(predictions, actuals, n_bands)
        else:
            bands = self._create_quantile_bands(predictions, n_bands)

        # Assign bands
        band_assignments = self._assign_to_bands(predictions, bands)

        # Calculate statistics
        band_stats = self._calculate_band_statistics(
            predictions, actuals, band_assignments
        )

        # Check monotonicity
        is_monotonic = self._check_monotonicity(band_stats)
        if not is_monotonic:
            print("      Warning: Bands are not monotonic in bad rate")

        self.bands_ = bands
        self.band_stats_ = band_stats

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

        metrics = {}

        # Herfindahl-Hirschman Index
        metrics['herfindahl_index'] = self._calculate_herfindahl_index(band_stats)
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

        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 0:
                obs_events = actuals[mask].sum()
                exp_events = predictions[mask].sum()

                observed.append(obs_events)
                expected.append(exp_events)

        observed = np.array(observed)
        expected = np.array(expected)

        # Calculate test statistic
        # Avoid division by zero
        expected = np.maximum(expected, 0.5)
        n_per_group = np.array([np.sum(groups == g) for g in range(n_groups)])
        expected_non_events = n_per_group - expected

        chi_square = np.sum(
            (observed - expected) ** 2 / expected +
            ((n_per_group - observed) - expected_non_events) ** 2 / expected_non_events
        )

        # Degrees of freedom = n_groups - 2
        df = n_groups - 2

        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        return chi_square, p_value

    def _perform_binomial_tests(self, band_stats: pd.DataFrame) -> Dict:
        """
        Perform binomial test for each band.

        Tests if observed bad rate significantly differs from expected.
        """

        results = {}
        total_count = band_stats['count'].sum()
        if total_count == 0:
            return []

        overall_bad_rate = band_stats['bad_count'].sum() / total_count

        for _, row in band_stats.iterrows():
            band = row['band']
            n_obs = int(row['count'])
            n_bad = int(row['bad_count'])

            if n_obs > 0:
                # Binomial test: is bad rate different from overall?
                try:
                    from scipy.stats import binomtest
                    result = binomtest(
                        k=n_bad,
                        n=n_obs,
                        p=overall_bad_rate,
                        alternative='two-sided'
                    )
                    p_value = result.pvalue
                except ImportError:
                    # Fallback for older scipy
                    p_value = stats.binom_test(
                        n_bad,
                        n_obs,
                        overall_bad_rate,
                        alternative='two-sided'
                    )

                results[band] = {
                    'p_value': p_value,
                    'observed_rate': row['bad_rate'],
                    'expected_rate': overall_bad_rate,
                    'significant': p_value < 0.05
                }

        return results

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