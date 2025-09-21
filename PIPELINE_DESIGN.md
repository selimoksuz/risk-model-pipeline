# Risk Model Pipeline - Yeni Tasarım

## 1. TEK PIPELINE YAPISI

```python
class RiskModelPipeline:
    """
    Tek ve kapsamlı pipeline sınıfı.
    Config ile hangi özelliklerin aktif olacağı belirlenir.
    """
    
    def __init__(self, config: Config):
        self.config = config
        # Tüm modüller başlatılır ama config'e göre kullanılır
    
    def fit(self, 
            train_df: pd.DataFrame,
            calibration_df: pd.DataFrame = None,  # Stage1 kalibrasyon
            stage2_calibration_df: pd.DataFrame = None,  # Stage2 kalibrasyon
            variable_dictionary: pd.DataFrame = None,  # Değişken açıklamaları
            external_score_df: pd.DataFrame = None):  # Skorlama için
        """
        Ana training fonksiyonu
        """
        pass
    
    def score(self, df: pd.DataFrame, model_name: str = 'best'):
        """
        Skorlama fonksiyonu
        Config'te default olarak score=False gelir
        """
        if not self.config.enable_scoring:
            raise ValueError("Scoring is disabled in config")
        pass
```

## 2. CONFIG YAPISI

```python
class Config:
    def __init__(self):
        # DATA
        self.target_column = 'target'
        self.id_column = None
        self.time_column = None
        
        # SPLITTING
        self.create_test_split = True  # Test split oluştur
        self.test_size = 0.2
        self.stratify_test = True  # Event rate'i koru
        self.oot_months = 3  # Son 3 ay OOT
        
        # SCORING
        self.enable_scoring = False  # Default kapalı
        
        # FEATURE ENGINEERING
        self.calculate_woe_all = True  # Tüm değişkenler için WOE
        self.calculate_univariate_gini = True  # Tüm değişkenler için uni gini
        self.check_woe_degradation = True  # WOE'li gini düşüyorsa kontrol
        
        # WOE OPTIMIZATION
        self.woe_optimization_metric = 'iv'  # veya 'gini'
        self.woe_max_bins = 10
        self.woe_min_bin_size = 0.05
        self.woe_monotonic = True  # Numerik için monotonluk
        self.woe_merge_insignificant = True  # Kategorik için merge
        
        # SELECTION PIPELINE
        self.selection_steps = [
            'psi',           # 1. PSI filtresi
            'vif',           # 2. VIF filtresi
            'correlation',   # 3. Korelasyon clustering
            'iv',            # 4. IV filtresi
            'boruta',        # 5. Boruta (LightGBM based)
            'stepwise'       # 6. Forward/Backward/Stepwise
        ]
        
        # SELECTION METHODS
        self.stepwise_method = 'forward'  # 'forward', 'backward', 'stepwise', 'forward_1se'
        self.boruta_estimator = 'lightgbm'  # veya 'randomforest'
        
        # MODELS
        self.algorithms = [
            'logistic',
            'gam',  # Generalized Additive Model
            'catboost',
            'lightgbm',
            'xgboost',
            'randomforest',
            'extratrees'
        ]
        
        # DUAL PIPELINE
        self.enable_dual = True  # WOE + RAW birlikte
        
        # CALIBRATION
        self.calibration_method = 'isotonic'  # veya 'sigmoid'
        self.stage2_calibration = True
        
        # RISK BANDS
        self.optimize_risk_bands = True
        self.n_risk_bands = 10
        self.risk_band_tests = ['binomial', 'hosmer_lemeshow', 'herfindahl']
        
        # REPORTING
        self.calculate_shap = True
        self.include_variable_dictionary = True
```

## 3. DATA FLOW

```
1. INPUT VALIDATION
   - train_df zorunlu
   - calibration_df opsiyonel
   - stage2_calibration_df opsiyonel
   - variable_dictionary opsiyonel

2. DATA SPLITTING
   - Train içinden test ayrılır (stratified)
   - Time column varsa OOT ayrılır (son 3 ay)
   - Time column yoksa random OOT

3. VARIABLE CLASSIFICATION
   - Numeric / Categorical ayrımı
   - Numeric: imputation + outlier handling
   - Categorical: rare category handling

4. WOE CALCULATION (TÜM DEĞİŞKENLER)
   - Her değişken için optimal binning
   - IV/Gini maximization
   - Monotonluk kontrolü (numeric)
   - Insignificant merge (categorical)

5. UNIVARIATE ANALYSIS
   - Raw univariate Gini
   - WOE univariate Gini
   - WOE degradation check

6. FEATURE SELECTION PIPELINE
   a. PSI Filtering (>0.25 drop)
   b. VIF Filtering (>5 drop)
   c. Correlation Clustering (>0.95 keep best)
   d. IV Filtering (<0.02 drop)
   e. Boruta Selection (LightGBM based)
   f. Stepwise Selection (forward/backward/stepwise)
   g. Noise Sentinel Check

7. MODEL TRAINING
   - Dual pipeline (WOE + RAW) if enabled
   - All algorithms with Optuna HPO
   - Cross-validation

8. CALIBRATION
   - Stage 1: Event rate calibration
   - Stage 2: Recent predictions calibration (if data provided)

9. RISK BANDS
   - PSI optimal bands
   - Event rate convergent
   - Statistical tests (binomial, Hosmer-Lemeshow, Herfindahl)

10. REPORTING
    - Model comparison
    - Variable importance (SHAP + native)
    - WOE bins with univariate Gini
    - Risk bands with tests
    - Calibration curves
    - Variable dictionary integration
```

## 4. KEY FEATURES

### 4.1 WOE Optimization
```python
def optimize_woe_bins(self, X, y, variable_type='numeric'):
    """
    IV veya Gini'yi maximize eden optimal binning
    """
    if variable_type == 'numeric':
        # Monotonluk korunarak bin sayısı artırılır
        # IV artışı durduğunda durulur
        pass
    else:  # categorical
        # IV/Gini'ye katkısı olmayan kategoriler merge edilir
        # Chi-square test ile anlamlılık kontrolü
        pass
```

### 4.2 Selection Methods
```python
def forward_selection(self, X, y, max_features=20):
    """Adım adım en iyi değişkeni ekle"""
    pass

def backward_selection(self, X, y, min_features=5):
    """Tüm değişkenlerle başla, en kötüyü çıkar"""
    pass

def stepwise_selection(self, X, y):
    """Forward + Backward kombinasyonu"""
    pass

def forward_selection_1se(self, X, y):
    """1 standard error rule ile forward selection"""
    pass
```

### 4.3 Calibration Stages
```python
def calibrate_stage1(self, y_true, y_pred, calibration_df=None):
    """
    Stage 1: Event rate calibration
    - calibration_df varsa onunla
    - Yoksa long-run average ile
    """
    pass

def calibrate_stage2(self, y_pred_stage1, stage2_df):
    """
    Stage 2: Recent predictions calibration
    - Lower/upper bound adjustment
    """
    pass
```

### 4.4 Risk Band Optimization
```python
def optimize_risk_bands(self, scores, y_true):
    """
    Optimal risk bantları:
    - PSI minimization
    - Event rate monotonicity
    - Volume distribution balance
    """
    bands = self.find_optimal_cuts(scores, y_true)
    
    # Statistical tests
    self.binomial_test(bands)
    self.hosmer_lemeshow_test(bands)
    self.herfindahl_index(bands)
    
    return bands
```

## 5. NOTEBOOK STRUCTURE

```python
# 1. Setup
from risk_pipeline import RiskModelPipeline, Config

# 2. Configuration
config = Config()
config.target_column = 'target'
config.enable_dual = True
config.stepwise_method = 'forward'
# ... diğer ayarlar

# 3. Initialize Pipeline
pipeline = RiskModelPipeline(config)

# 4. Fit Pipeline
results = pipeline.fit(
    train_df=train_data,
    calibration_df=calibration_data,  # Opsiyonel
    stage2_calibration_df=recent_data,  # Opsiyonel
    variable_dictionary=var_dict  # Opsiyonel
)

# 5. View Results
print(results['best_model'])
print(results['selected_features'])
print(results['risk_bands'])

# 6. Score New Data
config.enable_scoring = True
scores = pipeline.score(new_data, model_name='best')

# 7. Step-by-step örnek
# Her adımı ayrı ayrı çalıştırma örnekleri
```

## 6. IMPLEMENTATION STEPS

1. **Pipeline sınıfını basitleştir** - Tek class
2. **Config'i güncelle** - Tüm parametreler
3. **WOE optimization ekle** - IV/Gini based
4. **Selection metodlarını implement et** - Forward/Backward/Stepwise
5. **Calibration stage'leri ekle** - Stage1 + Stage2
6. **Risk band optimization** - Statistical tests
7. **Reporting'i güncelle** - Variable dictionary support
8. **Notebook hazırla** - Step-by-step + Full run

## 7. AVANTAJLAR

✅ **Tek pipeline** - Karmaşıklık azaldı
✅ **Config driven** - Her şey config'ten kontrol
✅ **Modüler** - İstenmeyen özellikler kapatılabilir
✅ **Esnek** - Her türlü veri yapısına uyum
✅ **Production ready** - Scoring default kapalı
✅ **Comprehensive** - Tüm özellikler tek yerde