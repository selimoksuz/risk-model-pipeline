# PSI Monitoring

Population Stability Index (PSI) is a critical metric for monitoring model and feature stability over time in production environments.

## Table of Contents
- [What is PSI?](#what-is-psi)
- [Mathematical Formula](#mathematical-formula)
- [Calculation Steps](#calculation-steps)
- [Interpretation Guidelines](#interpretation-guidelines)
- [Implementation in Pipeline](#implementation-in-pipeline)
- [Monitoring Strategies](#monitoring-strategies)
- [Troubleshooting High PSI](#troubleshooting-high-psi)

---

## What is PSI?

PSI measures the shift in distribution of a variable between two samples (typically training vs production/OOT). It quantifies population stability and helps detect:

- **Data drift**: Changes in feature distributions
- **Population shift**: Changes in customer behavior
- **Model degradation**: When to retrain models
- **Feature reliability**: Which features remain stable

### Why PSI Matters
- **Early warning system** for model performance issues
- **Regulatory compliance** in credit risk (Basel III requirements)
- **Quality assurance** for data pipelines
- **Feature selection** criterion for robust models

## Mathematical Formula

### Basic PSI Formula
```
PSI = Σ [(% Actual - % Expected) × ln(% Actual / % Expected)]
```

Where:
- **% Expected**: Proportion in reference population (training)
- **% Actual**: Proportion in comparison population (production/OOT)

### Detailed Calculation
For each bin i:
```
PSI_i = (P_actual_i - P_expected_i) × ln(P_actual_i / P_expected_i)
```

Total PSI:
```
PSI = Σ PSI_i for all bins
```

### Alternative Formula (Symmetric)
```
PSI = Σ [(A_i - E_i) × ln(A_i / E_i)]
```
Where A_i and E_i are proportions in actual and expected distributions.

## Calculation Steps

### Step 1: Create Bins
Define consistent bins for the variable:
```python
# For numeric variables
bins = pd.qcut(train_data[feature], q=10, duplicates='drop')

# For categorical variables
bins = feature.unique()
```

### Step 2: Calculate Distributions
```python
# Training distribution (Expected)
train_dist = train_data[feature].value_counts(normalize=True)

# Production distribution (Actual)
prod_dist = prod_data[feature].value_counts(normalize=True)
```

### Step 3: Calculate PSI Components
```python
def calculate_psi(expected, actual, bins):
    psi_components = []
    
    for bin in bins:
        e = expected.get(bin, 0.0001)  # Avoid division by zero
        a = actual.get(bin, 0.0001)
        
        psi_i = (a - e) * np.log(a / e)
        psi_components.append(psi_i)
    
    return sum(psi_components), psi_components
```

### Step 4: Interpret Results
```python
total_psi, components = calculate_psi(train_dist, prod_dist, bins)

if total_psi < 0.10:
    status = "Stable"
elif total_psi < 0.25:
    status = "Monitor"
else:
    status = "Investigate"
```

## Interpretation Guidelines

### PSI Thresholds

| PSI Value | Interpretation | Population Shift | Recommended Action |
|-----------|---------------|------------------|-------------------|
| PSI < 0.10 | Stable | No significant change | Continue monitoring |
| 0.10 ≤ PSI < 0.25 | Slight shift | Small population change | Investigate cause, monitor closely |
| PSI ≥ 0.25 | Significant shift | Major population change | Model retraining likely needed |

### Industry-Specific Guidelines

#### Credit Risk Models
```python
config = Config(
    psi_threshold=0.25  # Industry standard for credit
)
```
- **Regulatory view**: PSI > 0.25 requires documentation
- **Best practice**: Monitor at PSI > 0.10

#### Fraud Detection
```python
config = Config(
    psi_threshold=0.20  # More sensitive for fraud
)
```
- **Real-time monitoring**: Check PSI daily
- **Alert threshold**: PSI > 0.15

#### Marketing Models
```python
config = Config(
    psi_threshold=0.30  # More tolerant for marketing
)
```
- **Seasonal adjustments**: Expected shifts
- **Campaign effects**: Temporary spikes acceptable

## Implementation in Pipeline

### Configuration
```python
from risk_pipeline import Config

config = Config(
    # PSI thresholds
    psi_threshold=0.25,          # Maximum PSI for feature selection
    psi_warning_threshold=0.10,  # Warning level for monitoring
    
    # PSI calculation settings
    psi_bins=10,                 # Number of bins for PSI
    psi_min_bin_size=0.05,      # Minimum 5% in each bin
    
    # Monitoring frequency
    psi_check_frequency='daily'  # daily, weekly, monthly
)
```

### Feature Selection with PSI
```python
def select_stable_features(train_data, oot_data, config):
    """Select features with PSI below threshold"""
    
    stable_features = []
    psi_report = []
    
    for feature in train_data.columns:
        psi = calculate_psi(
            train_data[feature], 
            oot_data[feature]
        )
        
        psi_report.append({
            'feature': feature,
            'psi': psi,
            'stable': psi < config.psi_threshold
        })
        
        if psi < config.psi_threshold:
            stable_features.append(feature)
    
    return stable_features, pd.DataFrame(psi_report)
```

### Automated Monitoring
```python
class PSIMonitor:
    def __init__(self, baseline_data, config):
        self.baseline = baseline_data
        self.config = config
        self.history = []
    
    def check_stability(self, current_data):
        """Check PSI for all features"""
        alerts = []
        
        for feature in self.baseline.columns:
            psi = calculate_psi(
                self.baseline[feature],
                current_data[feature]
            )
            
            if psi > self.config.psi_threshold:
                alerts.append({
                    'feature': feature,
                    'psi': psi,
                    'severity': 'HIGH'
                })
            elif psi > self.config.psi_warning_threshold:
                alerts.append({
                    'feature': feature,
                    'psi': psi,
                    'severity': 'MEDIUM'
                })
            
            self.history.append({
                'timestamp': datetime.now(),
                'feature': feature,
                'psi': psi
            })
        
        return alerts
```

## Monitoring Strategies

### 1. Real-time Monitoring
```python
# Stream processing for immediate detection
def realtime_psi_check(new_batch, baseline, threshold=0.25):
    for feature in new_batch.columns:
        psi = quick_psi(new_batch[feature], baseline[feature])
        if psi > threshold:
            send_alert(f"PSI Alert: {feature} = {psi:.3f}")
```

### 2. Scheduled Monitoring
```python
# Daily/Weekly/Monthly checks
def scheduled_psi_report():
    report = []
    
    # Compare last 30 days vs training
    recent_data = get_recent_data(days=30)
    training_data = get_training_data()
    
    for feature in features:
        psi = calculate_psi(training_data[feature], recent_data[feature])
        trend = calculate_psi_trend(feature, days=90)
        
        report.append({
            'feature': feature,
            'current_psi': psi,
            'trend': trend,
            'status': get_status(psi)
        })
    
    return pd.DataFrame(report)
```

### 3. Sliding Window Monitoring
```python
# Compare consecutive time windows
def sliding_window_psi(data, window_size=30, step=7):
    windows = []
    
    for i in range(0, len(data) - window_size, step):
        window = data[i:i + window_size]
        baseline = data[max(0, i - window_size):i]
        
        if len(baseline) > 0:
            psi = calculate_psi(baseline, window)
            windows.append({
                'start': i,
                'end': i + window_size,
                'psi': psi
            })
    
    return pd.DataFrame(windows)
```

### 4. Cohort-based Monitoring
```python
# Monitor PSI by customer segments
def cohort_psi_monitoring(data, cohort_column='customer_segment'):
    cohorts = data[cohort_column].unique()
    
    results = []
    for cohort in cohorts:
        cohort_data = data[data[cohort_column] == cohort]
        
        for feature in numerical_features:
            psi = calculate_psi(
                train_cohort[feature],
                current_cohort[feature]
            )
            
            results.append({
                'cohort': cohort,
                'feature': feature,
                'psi': psi
            })
    
    return pd.DataFrame(results)
```

## Troubleshooting High PSI

### Common Causes and Solutions

#### 1. Seasonal Effects
**Symptoms**: Regular PSI spikes at certain times
```python
# Solution: Adjust for seasonality
def seasonal_adjusted_psi(data, feature, season_column='month'):
    # Calculate PSI per season
    seasonal_psi = {}
    for season in data[season_column].unique():
        season_data = data[data[season_column] == season]
        seasonal_psi[season] = calculate_psi(
            train_seasonal[feature],
            current_seasonal[feature]
        )
    return seasonal_psi
```

#### 2. Data Collection Changes
**Symptoms**: Sudden PSI jump after specific date
```python
# Detection: Check for structural breaks
def detect_structural_break(time_series_psi):
    from ruptures import Pelt
    algo = Pelt(model="rbf").fit(time_series_psi)
    breaks = algo.predict(pen=10)
    return breaks
```

#### 3. Population Drift
**Symptoms**: Gradual PSI increase over time
```python
# Solution: Adaptive retraining
def adaptive_retraining_trigger(psi_history, threshold=0.20):
    recent_avg = np.mean(psi_history[-30:])
    if recent_avg > threshold:
        return True, "Retrain recommended"
    return False, "Model stable"
```

#### 4. Feature Engineering Issues
**Symptoms**: High PSI in derived features
```python
# Solution: Check feature dependencies
def check_feature_dependencies(high_psi_features):
    dependencies = {}
    for feature in high_psi_features:
        # Check if derived from other features
        base_features = get_base_features(feature)
        base_psi = [calculate_psi(f) for f in base_features]
        dependencies[feature] = {
            'base_features': base_features,
            'base_psi': base_psi
        }
    return dependencies
```

### PSI Decomposition Analysis
```python
def psi_decomposition(feature, train_data, prod_data):
    """Decompose PSI to understand contribution of each bin"""
    
    # Create bins
    bins = pd.qcut(train_data[feature], q=10, duplicates='drop')
    
    # Calculate distributions
    train_dist = train_data[feature].value_counts(normalize=True)
    prod_dist = prod_data[feature].value_counts(normalize=True)
    
    # Calculate PSI per bin
    decomposition = []
    for bin in bins:
        e = train_dist.get(bin, 0.0001)
        a = prod_dist.get(bin, 0.0001)
        psi_contrib = (a - e) * np.log(a / e)
        
        decomposition.append({
            'bin': bin,
            'train_%': e * 100,
            'prod_%': a * 100,
            'difference': (a - e) * 100,
            'psi_contribution': psi_contrib,
            'psi_%': (psi_contrib / total_psi) * 100
        })
    
    return pd.DataFrame(decomposition)
```

## Visualization and Reporting

### PSI Dashboard Components
```python
import plotly.graph_objects as go

def create_psi_dashboard(psi_data):
    """Create interactive PSI monitoring dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PSI Trend', 'Feature Heatmap', 
                       'Distribution Shift', 'Alert Summary')
    )
    
    # 1. PSI Trend over time
    fig.add_trace(
        go.Scatter(x=psi_data['date'], y=psi_data['psi'],
                  mode='lines+markers'),
        row=1, col=1
    )
    
    # 2. Feature PSI heatmap
    fig.add_trace(
        go.Heatmap(z=psi_matrix, x=features, y=dates),
        row=1, col=2
    )
    
    # 3. Distribution comparison
    fig.add_trace(
        go.Histogram(x=train_dist, name='Training'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=prod_dist, name='Production'),
        row=2, col=1
    )
    
    # 4. Alert summary
    fig.add_trace(
        go.Bar(x=alert_counts.index, y=alert_counts.values),
        row=2, col=2
    )
    
    return fig
```

### Automated Reporting
```python
def generate_psi_report(data, config):
    """Generate comprehensive PSI monitoring report"""
    
    report = {
        'summary': {
            'date': datetime.now(),
            'total_features': len(features),
            'stable_features': sum(psi < 0.10),
            'warning_features': sum(0.10 <= psi < 0.25),
            'unstable_features': sum(psi >= 0.25)
        },
        'details': [],
        'recommendations': []
    }
    
    for feature in features:
        psi = calculate_psi(train[feature], current[feature])
        
        report['details'].append({
            'feature': feature,
            'psi': psi,
            'status': get_status(psi),
            'trend': calculate_trend(feature),
            'last_stable': get_last_stable_date(feature)
        })
        
        if psi > config.psi_threshold:
            report['recommendations'].append(
                f"Feature '{feature}' shows significant drift (PSI={psi:.3f}). "
                f"Consider removing from model or investigating cause."
            )
    
    return report
```

## Best Practices

### 1. Consistent Binning
```python
# Save binning rules from training
binning_rules = {}
for feature in numeric_features:
    _, bins = pd.qcut(train[feature], q=10, retbins=True)
    binning_rules[feature] = bins

# Apply same bins to production
prod_binned = pd.cut(prod[feature], bins=binning_rules[feature])
```

### 2. Handle Edge Cases
```python
def safe_psi_calculation(expected, actual, epsilon=0.0001):
    """Handle zero probabilities and missing categories"""
    
    # Add small epsilon to avoid log(0)
    expected = expected + epsilon
    actual = actual + epsilon
    
    # Normalize
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    
    # Calculate PSI
    psi = sum((actual - expected) * np.log(actual / expected))
    
    return psi
```

### 3. Document PSI Thresholds
```yaml
# config.yaml
psi_settings:
  thresholds:
    feature_selection: 0.25  # Drop features above this
    monitoring_warning: 0.10  # Alert but don't drop
    model_retrain: 0.30  # Trigger full retrain
  
  exceptions:
    seasonal_features:  # Higher threshold for seasonal
      - holiday_indicator: 0.40
      - month_of_year: 0.35
```

### 4. Combine with Other Metrics
```python
def comprehensive_stability_check(feature, data):
    """Combine PSI with other stability metrics"""
    
    metrics = {
        'psi': calculate_psi(feature),
        'kolmogorov_smirnov': ks_test(feature),
        'jensen_shannon': js_divergence(feature),
        'wasserstein': wasserstein_distance(feature)
    }
    
    # Weighted stability score
    stability_score = (
        0.5 * (1 - metrics['psi']) +
        0.2 * (1 - metrics['kolmogorov_smirnov']) +
        0.2 * (1 - metrics['jensen_shannon']) +
        0.1 * (1 - metrics['wasserstein'])
    )
    
    return stability_score, metrics
```

## Related Topics
- [Feature Selection Parameters](Feature-Selection-Parameters) - PSI threshold configuration
- [Information Value](Information-Value) - Predictive power measurement
- [Model Selection Criteria](Model-Selection-Criteria) - Stability in model selection