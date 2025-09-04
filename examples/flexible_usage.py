#!/usr/bin/env python3
"""
Risk Model Pipeline - Esnek Kullanım Örnekleri

Bu script, paketi farklı senaryolarda nasıl kullanacağınızı gösterir:
1. Sadece model eğitimi
2. Model + kalibrasyon
3. Sadece skorlama
4. Sonradan kalibrasyon ekleme
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')

from risk_pipeline.pipeline import Config, RiskModelPipeline
# from risk_pipeline.utils.scoring import load_model_artifacts, score_data
# from risk_pipeline.utils.pipeline_runner import run_pipeline_from_dataframe

# ==============================================================================
# SENARYO 1: SADECE MODEL EĞİTİMİ (Kalibrasyon yok, Skorlama yok)
# ==============================================================================
def train_only_example():
    """Sadece model eğitimi yap"""
    print("=" * 60)
    print("SENARYO 1: SADECE MODEL EĞİTİMİ")
    print("=" * 60)
    
    # Veri hazırla
    train_df = pd.DataFrame({
        'app_id': range(1000),
        'app_dt': pd.date_range('2024-01-01', periods=1000),
        'target': np.random.binomial(1, 0.2, 1000),
        'age': np.random.randint(18, 70, 1000),
        'income': np.random.lognormal(10, 0.5, 1000)
    })
    
    # Model eğit - kalibrasyon YOK
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        calibration_data_path=None,  # Kalibrasyon YOK
        hpo_trials=2,  # Hızlı test
        hpo_timeout_sec=10
    )
    
    pipeline = RiskModelPipeline(config)
    results = pipeline.run(train_df)
    
    print(f"\nModel eğitildi!")
    print(f"Best Model: {pipeline.best_model_name_}")
    print(f"Features: {pipeline.final_vars_}")
    
    return pipeline, results

# ==============================================================================
# SENARYO 2: MODEL + KALİBRASYON (Skorlama yok)
# ==============================================================================
def train_with_calibration_example():
    """Model eğit ve kalibre et"""
    print("\n" + "=" * 60)
    print("SENARYO 2: MODEL + KALİBRASYON")
    print("=" * 60)
    
    # Training data
    train_df = pd.DataFrame({
        'app_id': range(1000),
        'app_dt': pd.date_range('2024-01-01', periods=1000),
        'target': np.random.binomial(1, 0.2, 1000),
        'age': np.random.randint(18, 70, 1000),
        'income': np.random.lognormal(10, 0.5, 1000)
    })
    
    # Calibration data as DataFrame (no CSV needed!)
    cal_df = pd.DataFrame({
        'app_id': range(2000, 2300),
        'app_dt': pd.date_range('2024-06-01', periods=300),
        'target': np.random.binomial(1, 0.25, 300),
        'age': np.random.randint(18, 70, 300),
        'income': np.random.lognormal(10, 0.5, 300)
    })
    
    # Model + Kalibrasyon (DataFrame support!)
    config = Config(
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        calibration_df=cal_df,  # DataFrame directly! No CSV needed
        calibration_method="isotonic",
        hpo_trials=2,
        hpo_timeout_sec=10
    )
    
    pipeline = RiskModelPipeline(config)
    results = pipeline.run(train_df)
    
    print(f"\nModel eğitildi ve kalibre edildi!")
    print(f"Calibrator: {hasattr(pipeline, 'calibrator_') and pipeline.calibrator_ is not None}")
    print(f"Note: Calibration provided as DataFrame, no CSV file needed!")
    
    return pipeline, results

# ==============================================================================
# SENARYO 3: SADECE SKORLAMA (Önceden eğitilmiş model ile)
# ==============================================================================
def score_only_example(pipeline, results):
    """Sadece skorlama yap"""
    print("\n" + "=" * 60)
    print("SENARYO 3: SADECE SKORLAMA")
    print("=" * 60)
    
    # Skorlama verisi (bazıları targetsız)
    scoring_df = pd.DataFrame({
        'app_id': range(3000, 3500),
        'app_dt': pd.date_range('2024-08-01', periods=500),
        'target': [np.nan] * 300 + list(np.random.binomial(1, 0.3, 200)),  # %60 targetsız
        'age': np.random.randint(18, 70, 500),
        'income': np.random.lognormal(10, 0.5, 500)
    })
    
    print(f"Scoring data: {len(scoring_df)} rows")
    print(f"  With target: {(~scoring_df['target'].isna()).sum()}")
    print(f"  Without target: {scoring_df['target'].isna().sum()}")
    
    # Model ve artifacts'ları hazırla
    import joblib
    import json
    from datetime import datetime
    
    # Normalde bunlar dosyadan yüklenir
    model = pipeline.models_[pipeline.best_model_name_]
    final_features = pipeline.final_vars_
    woe_mapping = {}
    for var_name, var_info in pipeline.woe_map.items():
        woe_mapping[var_name] = {
            'var': var_info.var if hasattr(var_info, 'var') else var_name,
            'bins': []
        }
        if hasattr(var_info, 'bins'):
            for bin_info in var_info.bins:
                woe_mapping[var_name]['bins'].append({
                    'range': list(bin_info['range']) if isinstance(bin_info['range'], (list, tuple)) else bin_info['range'],
                    'woe': float(bin_info.get('woe', 0))
                })
    
    calibrator = pipeline.calibrator_ if hasattr(pipeline, 'calibrator_') else None
    
    # Skorlama
    scoring_results = score_data(
        scoring_df=scoring_df,
        model=model,
        final_features=final_features,
        woe_mapping=woe_mapping,
        calibrator=calibrator,
        training_scores=None,  # PSI hesaplamak istemiyorsak None
        feature_mapping=None
    )
    
    print(f"\nSkorlama tamamlandı!")
    print(f"Total scored: {scoring_results['n_total']}")
    
    if 'with_target' in scoring_results:
        wt = scoring_results['with_target']
        print(f"\nWith Target Metrics:")
        print(f"  AUC: {wt['auc']:.3f}")
        print(f"  Gini: {wt['gini']:.3f}")
        print(f"  Default Rate: {wt['default_rate']:.3f}")
    
    if 'without_target' in scoring_results:
        wot = scoring_results['without_target']
        print(f"\nWithout Target:")
        print(f"  Score Mean: {wot['score_stats']['mean']:.3f}")
        print(f"  Score Std: {wot['score_stats']['std']:.3f}")
    
    return scoring_results

# ==============================================================================
# SENARYO 4: SONRADAN KALİBRASYON EKLEME
# ==============================================================================
def add_calibration_later_example(pipeline):
    """Eğitilmiş modele sonradan kalibrasyon ekle"""
    print("\n" + "=" * 60)
    print("SENARYO 4: SONRADAN KALİBRASYON EKLEME")
    print("=" * 60)
    
    from sklearn.isotonic import IsotonicRegression
    
    # Kalibrasyon verisi
    cal_df = pd.DataFrame({
        'app_id': range(4000, 4200),
        'app_dt': pd.date_range('2024-09-01', periods=200),
        'target': np.random.binomial(1, 0.22, 200),
        'age': np.random.randint(18, 70, 200),
        'income': np.random.lognormal(10, 0.5, 200)
    })
    
    # WOE dönüşümü uygula
    from risk_pipeline.core.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    # Apply WOE transformation using the feature engineer
    cal_woe = apply_woe(cal_df, pipeline.woe_map)
    X_cal = cal_woe[pipeline.final_vars_]
    y_cal = cal_df['target'].values
    
    # Model tahminleri
    model = pipeline.models_[pipeline.best_model_name_]
    try:
        raw_scores = model.predict_proba(X_cal)[:, 1]
    except:
        raw_scores = model.predict(X_cal)
    
    # Kalibratör eğit
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_scores.reshape(-1, 1), y_cal)
    
    print(f"Kalibratör eğitildi!")
    print(f"Calibrator type: {type(calibrator).__name__}")
    
    # Pipeline'a ekle
    pipeline.calibrator_ = calibrator
    
    return calibrator

# ==============================================================================
# SENARYO 5: HEPSİ BİR ARADA (Helper Function)
# ==============================================================================
def all_in_one_example():
    """Tek fonksiyonda her şey"""
    print("\n" + "=" * 60)
    print("SENARYO 5: HEPSİ BİR ARADA")
    print("=" * 60)
    
    # Training data
    train_df = pd.DataFrame({
        'app_id': range(1000),
        'app_dt': pd.date_range('2024-01-01', periods=1000),
        'target': np.random.binomial(1, 0.2, 1000),
        'age': np.random.randint(18, 70, 1000),
        'income': np.random.lognormal(10, 0.5, 1000)
    })
    
    # Tek fonksiyonda pipeline + skorlama
    results = run_pipeline_from_dataframe(
        df=train_df,
        id_col="app_id",
        time_col="app_dt",
        target_col="target",
        
        # Opsiyonel
        calibration_data_path=None,  # İsterseniz ekleyin
        
        # Performans
        hpo_trials=2,
        hpo_timeout_sec=10,
        
        # Çıktılar
        output_folder="outputs",
        output_excel="report.xlsx"
    )
    
    print(f"\nPipeline tamamlandı!")
    print(f"Best Model: {results['best_model']}")
    print(f"Features: {results['final_features']}")
    print(f"Output: {results.get('output_folder', 'outputs')}")
    
    return results

# ==============================================================================
# ANA PROGRAM
# ==============================================================================
def main():
    """Tüm senaryoları çalıştır"""
    
    print("\n" + "="*70)
    print("RISK MODEL PIPELINE - ESNEK KULLANIM ÖRNEKLERİ")
    print("="*70)
    
    # 1. Sadece model eğitimi
    pipeline1, results1 = train_only_example()
    
    # 2. Model + Kalibrasyon
    pipeline2, results2 = train_with_calibration_example()
    
    # 3. Sadece skorlama (1. modeli kullan)
    scoring_results = score_only_example(pipeline1, results1)
    
    # 4. Sonradan kalibrasyon ekle
    calibrator = add_calibration_later_example(pipeline1)
    
    # 5. Hepsi bir arada
    all_results = all_in_one_example()
    
    print("\n" + "="*70)
    print("TÜM SENARYOLAR BAŞARIYLA ÇALIŞTIRILDI!")
    print("="*70)
    print("\nÖzet:")
    print("✓ Model eğitimi (kalibrasyon olmadan)")
    print("✓ Model eğitimi (kalibrasyon ile)")
    print("✓ Sadece skorlama")
    print("✓ Sonradan kalibrasyon ekleme")
    print("✓ Tek fonksiyonda her şey")
    print("\nPaket notebook'ta veya script'te esnek kullanıma hazır!")

if __name__ == "__main__":
    main()