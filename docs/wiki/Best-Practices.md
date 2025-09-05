# Best Practices

This guide covers best practices for building robust credit risk models using the Risk Model Pipeline.

## Table of Contents
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Validation](#model-validation)
- [Production Deployment](#production-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Common Pitfalls](#common-pitfalls)

---

## Data Preparation

### 1. Time-Based Splitting

**✅ DO: Use proper time-based splits**
```python
# Good: Out-of-time validation
train_data = data[data['date'] < '2023-01-01']        # Historical
valid_data = data[(data['date'] >= '2023-01-01') & 
                  (data['date'] < '2023-07-01')]      # Validation period
oot_data = data[data['date'] >= '2023-07-01']        # Out-of-time

# Bad: Random splitting (data leakage risk)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # ❌
```

**Why**: Credit risk models must predict future defaults. Random splits leak future information into training.

### 2. Target Definition

**✅ DO: Define clear performance windows**
```python
# Good: Clear observation and performance windows
def create_target(data):
    """
    Observation window: Application date to +3 months
    Performance window: +3 months to +15 months
    Target: 90+ days past due in performance window
    """
    data['target'] = (
        (data['months_3_to_15_dpd_max'] >= 90) | 
        (data['months_3_to_15_status'] == 'default')
    ).astype(int)
    return data
```

**Why**: Consistent target definition ensures model stability and regulatory compliance.

### 3. Sample Selection

**✅ DO: Document exclusions clearly**
```python
# Good: Clear exclusion criteria
def prepare_sample(data):
    # Document exclusions
    exclusions = {
        'fraud': data[data['fraud_flag'] == 1],
        'incomplete': data[data['application_complete'] == 0],
        'test_accounts': data[data['is_test'] == 1]
    }
    
    # Apply exclusions
    clean_data = data[
        (data['fraud_flag'] == 0) &
        (data['application_complete'] == 1) &
        (data['is_test'] == 0)
    ]
    
    # Log exclusions
    for reason, excluded in exclusions.items():
        print(f"Excluded {len(excluded)} records due to {reason}")
    
    return clean_data
```

### 4. Missing Value Analysis

**✅ DO: Analyze missing patterns before imputation**
```python
# Good: Understand missing patterns
def analyze_missing(data):
    missing_report = pd.DataFrame({
        'feature': data.columns,
        'missing_count': data.isnull().sum(),
        'missing_pct': data.isnull().mean() * 100,
        'missing_corr_with_target': [
            data[col].isnull().astype(int).corr(data['target'])
            for col in data.columns
        ]
    })
    
    # Flag suspicious patterns
    missing_report['suspicious'] = (
        missing_report['missing_corr_with_target'].abs() > 0.1
    )
    
    return missing_report
```

## Feature Engineering

### 1. Domain Knowledge Features

**✅ DO: Create business-relevant features**
```python
# Good: Domain-specific ratios
def create_credit_features(data):
    # Debt service ratios
    data['debt_to_income'] = data['total_debt'] / data['monthly_income']
    data['payment_to_income'] = data['monthly_payment'] / data['monthly_income']
    
    # Utilization metrics
    data['credit_utilization'] = data['current_balance'] / data['credit_limit']
    data['available_credit'] = data['credit_limit'] - data['current_balance']
    
    # Behavioral indicators
    data['payment_consistency'] = data['on_time_payments'] / data['total_payments']
    data['credit_inquiries_6m'] = data['inquiries_last_6_months']
    
    # Stability indicators
    data['employment_years'] = data['employment_months'] / 12
    data['address_stability'] = data['months_at_address'] / data['age_months']
    
    return data
```

### 2. WOE Transformation

**✅ DO: Use proper WOE settings**
```python
# Good: Robust WOE configuration
config = Config(
    # Binning parameters
    n_bins=10,                    # Start with 10, merge if needed
    min_bin_size=0.05,           # Minimum 5% in each bin
    
    # WOE parameters
    woe_monotonic=True,          # Enforce monotonic relationship
    handle_missing='separate',    # Missing as separate category
    
    # Stability checks
    psi_threshold=0.25,          # Monitor distribution shifts
)
```

### 3. Feature Selection Strategy

**✅ DO: Use multi-stage selection**
```python
# Good: Progressive feature selection
def select_features(data, config):
    # Stage 1: Statistical filters
    features = filter_by_iv(data, min_iv=0.02)
    features = filter_by_psi(features, max_psi=0.25)
    
    # Stage 2: Correlation analysis
    features = remove_correlated(features, threshold=0.90)
    features = check_vif(features, max_vif=5.0)
    
    # Stage 3: Business filters
    features = add_mandatory_features(features, ['bureau_score', 'income'])
    features = remove_prohibited_features(features, ['race', 'gender'])
    
    # Stage 4: Model-based selection
    features = boruta_selection(features)
    features = forward_selection(features, max_features=30)
    
    return features
```

## Model Development

### 1. Algorithm Selection

**✅ DO: Compare multiple algorithms**
```python
# Good: Test different model types
config = Config(
    model_type="ensemble",  # Test all available models
    models_to_train=["lightgbm", "xgboost", "catboost"],
    ensemble_method="voting"
)

# Evaluate each model type
for model_type in ["lightgbm", "xgboost", "catboost"]:
    config.model_type = model_type
    pipeline = DualPipeline(config)
    results = pipeline.fit(X_train, y_train, X_valid, y_valid)
    print(f"{model_type}: Gini={results['gini_oot']:.3f}")
```

### 2. Hyperparameter Optimization

**✅ DO: Use systematic optimization**
```python
# Good: Optuna with proper settings
config = Config(
    use_optuna=True,
    n_trials=200,                    # Sufficient trials
    optuna_timeout=3600,             # 1 hour timeout
    optuna_direction="maximize",     # Maximize Gini
    
    # Search space
    optuna_params={
        'n_estimators': (100, 1000),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    }
)
```

### 3. Model Selection Criteria

**✅ DO: Balance performance and stability**
```python
# Good: Multi-criteria selection
config = Config(
    # Selection method
    model_selection_method="balanced",
    
    # Stability constraints
    max_train_oot_gap=0.15,         # Max 15% performance drop
    model_stability_weight=0.3,      # 30% weight on stability
    
    # Performance constraints  
    min_gini_threshold=0.5,          # Minimum acceptable Gini
)

# Custom selection logic
def select_best_model(results):
    # Filter by minimum performance
    candidates = results[results['gini_oot'] > 0.5]
    
    # Filter by stability
    candidates = candidates[
        candidates['train_oot_gap'] < 0.15
    ]
    
    # Score by weighted criteria
    candidates['score'] = (
        0.5 * candidates['gini_oot'] +
        0.3 * (1 - candidates['train_oot_gap']) +
        0.2 * candidates['gini_valid']
    )
    
    return candidates.nlargest(1, 'score').index[0]
```

## Model Validation

### 1. Performance Validation

**✅ DO: Validate across multiple metrics**
```python
# Good: Comprehensive validation
def validate_model(model, X_test, y_test):
    predictions = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        # Discrimination
        'gini': 2 * roc_auc_score(y_test, predictions) - 1,
        'auc': roc_auc_score(y_test, predictions),
        'ks': calculate_ks(y_test, predictions),
        
        # Calibration
        'brier_score': brier_score_loss(y_test, predictions),
        'calibration_slope': calculate_calibration_slope(y_test, predictions),
        
        # Classification
        'precision_at_5pct': precision_at_k(y_test, predictions, k=0.05),
        'recall_at_5pct': recall_at_k(y_test, predictions, k=0.05),
        
        # Stability
        'psi': calculate_psi(train_scores, predictions),
        'feature_stability': check_feature_stability(model)
    }
    
    return metrics
```

### 2. Segmentation Analysis

**✅ DO: Validate across segments**
```python
# Good: Segment-level validation
segments = {
    'new_customers': data['months_on_book'] < 12,
    'prime': data['bureau_score'] > 700,
    'subprime': data['bureau_score'] <= 700,
    'high_income': data['income'] > data['income'].median(),
    'low_income': data['income'] <= data['income'].median()
}

for segment_name, segment_mask in segments.items():
    segment_data = data[segment_mask]
    segment_gini = calculate_gini(
        y_true=segment_data['target'],
        y_pred=segment_data['predictions']
    )
    print(f"{segment_name}: Gini = {segment_gini:.3f}")
```

### 3. Stability Testing

**✅ DO: Test temporal stability**
```python
# Good: Rolling window validation
def rolling_validation(model, data, window_size=3):
    months = data['month'].unique()
    stability_results = []
    
    for i in range(len(months) - window_size):
        window_months = months[i:i+window_size]
        window_data = data[data['month'].isin(window_months)]
        
        gini = calculate_gini(
            window_data['target'],
            model.predict_proba(window_data)[:, 1]
        )
        
        stability_results.append({
            'period': f"{window_months[0]}-{window_months[-1]}",
            'gini': gini,
            'n_samples': len(window_data)
        })
    
    # Check stability
    gini_std = pd.DataFrame(stability_results)['gini'].std()
    if gini_std > 0.05:
        print(f"Warning: High Gini volatility ({gini_std:.3f})")
    
    return stability_results
```

## Production Deployment

### 1. Model Serialization

**✅ DO: Save complete pipeline**
```python
# Good: Save entire pipeline with preprocessing
import joblib
import json

def save_model_package(pipeline, config, metadata, output_dir):
    # Save model
    joblib.dump(pipeline, f"{output_dir}/model.pkl")
    
    # Save configuration
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save metadata
    metadata = {
        'model_version': '1.0.0',
        'training_date': str(datetime.now()),
        'performance': {
            'gini_train': 0.75,
            'gini_oot': 0.72
        },
        'features': pipeline.feature_names_,
        'thresholds': pipeline.thresholds_
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save preprocessing rules
    preprocessing = {
        'binning_rules': pipeline.binning_rules_,
        'woe_tables': pipeline.woe_tables_,
        'imputation_values': pipeline.imputation_values_
    }
    
    joblib.dump(preprocessing, f"{output_dir}/preprocessing.pkl")
```

### 2. Input Validation

**✅ DO: Validate inputs in production**
```python
# Good: Comprehensive input validation
class ModelAPI:
    def __init__(self, model_path):
        self.model = joblib.load(f"{model_path}/model.pkl")
        self.config = json.load(open(f"{model_path}/config.json"))
        self.metadata = json.load(open(f"{model_path}/metadata.json"))
    
    def predict(self, input_data):
        # 1. Schema validation
        self._validate_schema(input_data)
        
        # 2. Range validation
        self._validate_ranges(input_data)
        
        # 3. Missing value check
        self._check_missing(input_data)
        
        # 4. PSI check
        psi = self._calculate_psi(input_data)
        if psi > 0.25:
            logging.warning(f"High PSI detected: {psi:.3f}")
        
        # 5. Make prediction
        prediction = self.model.predict_proba(input_data)[:, 1]
        
        # 6. Validate output
        if prediction < 0 or prediction > 1:
            raise ValueError(f"Invalid prediction: {prediction}")
        
        return prediction
    
    def _validate_schema(self, data):
        expected_features = self.metadata['features']
        actual_features = data.columns.tolist()
        
        missing = set(expected_features) - set(actual_features)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        extra = set(actual_features) - set(expected_features)
        if extra:
            logging.warning(f"Extra features will be ignored: {extra}")
```

### 3. Performance Monitoring

**✅ DO: Track model performance**
```python
# Good: Comprehensive monitoring
class ModelMonitor:
    def __init__(self, model, baseline_data):
        self.model = model
        self.baseline = baseline_data
        self.metrics_history = []
    
    def monitor_batch(self, new_data, labels=None):
        metrics = {
            'timestamp': datetime.now(),
            'n_samples': len(new_data)
        }
        
        # Score distribution
        scores = self.model.predict_proba(new_data)[:, 1]
        metrics['score_mean'] = scores.mean()
        metrics['score_std'] = scores.std()
        
        # PSI monitoring
        for feature in self.model.feature_names_:
            psi = calculate_psi(
                self.baseline[feature],
                new_data[feature]
            )
            metrics[f'psi_{feature}'] = psi
        
        # Performance (if labels available)
        if labels is not None:
            metrics['gini'] = calculate_gini(labels, scores)
            metrics['precision_5pct'] = precision_at_k(labels, scores, 0.05)
        
        self.metrics_history.append(metrics)
        
        # Check alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics):
        # PSI alerts
        psi_features = [k for k in metrics if k.startswith('psi_')]
        high_psi = [k for k in psi_features if metrics[k] > 0.25]
        
        if high_psi:
            alert(f"High PSI detected: {high_psi}")
        
        # Performance alerts
        if 'gini' in metrics and metrics['gini'] < 0.5:
            alert(f"Low Gini: {metrics['gini']:.3f}")
```

## Monitoring & Maintenance

### 1. Drift Detection

**✅ DO: Monitor multiple types of drift**
```python
# Good: Comprehensive drift monitoring
class DriftMonitor:
    def __init__(self, reference_data):
        self.reference = reference_data
        
    def detect_drift(self, current_data):
        drift_report = {}
        
        # 1. Covariate drift (feature distribution)
        for feature in self.reference.columns:
            drift_report[feature] = {
                'psi': calculate_psi(self.reference[feature], 
                                   current_data[feature]),
                'ks_stat': ks_2samp(self.reference[feature], 
                                   current_data[feature]).statistic,
                'mean_shift': (current_data[feature].mean() - 
                             self.reference[feature].mean())
            }
        
        # 2. Concept drift (feature-target relationship)
        if 'target' in current_data:
            for feature in self.reference.columns:
                ref_corr = self.reference[feature].corr(
                    self.reference['target']
                )
                curr_corr = current_data[feature].corr(
                    current_data['target']
                )
                drift_report[feature]['concept_drift'] = abs(
                    curr_corr - ref_corr
                )
        
        # 3. Prediction drift
        ref_scores = self.model.predict_proba(self.reference)[:, 1]
        curr_scores = self.model.predict_proba(current_data)[:, 1]
        
        drift_report['prediction_drift'] = {
            'mean_shift': curr_scores.mean() - ref_scores.mean(),
            'std_shift': curr_scores.std() - ref_scores.std(),
            'psi': calculate_psi(ref_scores, curr_scores)
        }
        
        return drift_report
```

### 2. Model Retraining

**✅ DO: Have clear retraining triggers**
```python
# Good: Systematic retraining decision
def should_retrain(monitor_results, config):
    triggers = {
        'performance_degradation': False,
        'high_psi': False,
        'concept_drift': False,
        'time_based': False
    }
    
    # Performance trigger
    recent_gini = monitor_results['gini'].tail(30).mean()
    if recent_gini < config.min_gini_threshold * 0.9:
        triggers['performance_degradation'] = True
    
    # PSI trigger
    recent_psi = monitor_results['psi'].tail(30).mean()
    if recent_psi > config.psi_threshold:
        triggers['high_psi'] = True
    
    # Concept drift trigger
    if monitor_results['concept_drift'].max() > 0.1:
        triggers['concept_drift'] = True
    
    # Time trigger (e.g., quarterly)
    last_training = monitor_results['last_training_date']
    if (datetime.now() - last_training).days > 90:
        triggers['time_based'] = True
    
    # Decision
    if any(triggers.values()):
        print(f"Retraining triggered by: {[k for k, v in triggers.items() if v]}")
        return True
    
    return False
```

### 3. A/B Testing

**✅ DO: Test new models safely**
```python
# Good: Gradual rollout with monitoring
class ABTestController:
    def __init__(self, model_a, model_b, split_ratio=0.1):
        self.model_a = model_a  # Current production
        self.model_b = model_b  # Challenger
        self.split_ratio = split_ratio
        self.results = []
    
    def route_request(self, request_id, features):
        # Deterministic routing based on ID
        use_model_b = hash(request_id) % 100 < self.split_ratio * 100
        
        if use_model_b:
            score = self.model_b.predict_proba(features)[:, 1]
            model_used = 'B'
        else:
            score = self.model_a.predict_proba(features)[:, 1]
            model_used = 'A'
        
        # Log for analysis
        self.results.append({
            'request_id': request_id,
            'model': model_used,
            'score': score,
            'timestamp': datetime.now()
        })
        
        return score
    
    def analyze_test(self):
        results_df = pd.DataFrame(self.results)
        
        # Compare score distributions
        score_stats = results_df.groupby('model')['score'].agg([
            'mean', 'std', 'min', 'max'
        ])
        
        # Statistical test
        scores_a = results_df[results_df['model'] == 'A']['score']
        scores_b = results_df[results_df['model'] == 'B']['score']
        ks_stat, p_value = ks_2samp(scores_a, scores_b)
        
        return {
            'score_statistics': score_stats,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
```

## Common Pitfalls

### 1. Data Leakage

**❌ AVOID: Future information in features**
```python
# Bad: Using future information
data['days_until_default'] = (data['default_date'] - data['application_date']).days
# This uses default_date which is only known in the future!

# Good: Use only past information
data['days_since_last_payment'] = (data['application_date'] - data['last_payment_date']).days
```

### 2. Overfitting

**❌ AVOID: Too complex models**
```python
# Bad: Overly complex model
config = Config(
    max_depth=20,           # Too deep
    n_estimators=5000,      # Too many trees
    min_samples_split=2,    # Too fine splits
    regularization=0        # No regularization
)

# Good: Regularized model
config = Config(
    max_depth=6,            # Reasonable depth
    n_estimators=300,       # Sufficient trees
    min_samples_split=100,  # Minimum samples
    reg_alpha=1.0,          # L1 regularization
    reg_lambda=1.0          # L2 regularization
)
```

### 3. Silent Failures

**❌ AVOID: Ignoring errors**
```python
# Bad: Silent failure
try:
    prediction = model.predict(data)
except:
    prediction = 0.5  # Default score - dangerous!

# Good: Proper error handling
try:
    prediction = model.predict(data)
except Exception as e:
    logging.error(f"Prediction failed: {e}")
    # Return error response, don't hide failure
    return {
        'status': 'error',
        'message': str(e),
        'timestamp': datetime.now()
    }
```

### 4. Inconsistent Preprocessing

**❌ AVOID: Different preprocessing in train/production**
```python
# Bad: Inconsistent binning
# Training
train_bins = pd.qcut(data['income'], q=10)

# Production (different!)
prod_bins = pd.cut(data['income'], bins=10)

# Good: Save and reuse binning rules
# Training
quantiles = data['income'].quantile([0, 0.1, 0.2, ..., 1.0])
joblib.dump(quantiles, 'income_bins.pkl')

# Production
quantiles = joblib.load('income_bins.pkl')
prod_bins = pd.cut(data['income'], bins=quantiles)
```

### 5. Ignored Business Constraints

**❌ AVOID: Purely technical optimization**
```python
# Bad: Ignoring business logic
selected_features = features_by_iv[:100]  # Top 100 by IV

# Good: Combine technical and business criteria
technical_features = features_by_iv[:100]
business_features = ['bureau_score', 'income', 'employment_status']
prohibited_features = ['race', 'religion', 'political_affiliation']

selected_features = (
    set(technical_features) | 
    set(business_features)
) - set(prohibited_features)
```

## Recommended Reading

### Internal Documentation
- [Feature Selection Parameters](Feature-Selection-Parameters)
- [Model Selection Criteria](Model-Selection-Criteria)
- [WOE Methodology](WOE-Methodology)
- [PSI Monitoring](PSI-Monitoring)

### External Resources
- [Basel III Requirements for Credit Risk](https://www.bis.org/basel_framework/)
- [GDPR Compliance for ML Models](https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/)
- [Model Risk Management (SR 11-7)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)

## Checklist for Production Models

### Pre-deployment
- [ ] Model performance meets business thresholds
- [ ] Stability validated on out-of-time data
- [ ] Documentation complete and reviewed
- [ ] Regulatory compliance verified
- [ ] Fallback strategy defined
- [ ] Monitoring framework in place

### Post-deployment
- [ ] Daily PSI monitoring active
- [ ] Weekly performance tracking
- [ ] Monthly segment analysis
- [ ] Quarterly model review
- [ ] Annual model revalidation
- [ ] Incident response plan tested