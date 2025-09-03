#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerçekçi veri ile end-to-end pipeline testi
- Yüksek Gini (~0.70) 
- PSI'a takılan değişkenler
- Korele değişkenler
- Ama sonuçta model için değişken kalacak
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
from src.risk_pipeline.pipeline16 import RiskModelPipeline, Config

warnings.filterwarnings('ignore')

print("="*80)
print("GERÇEKÇI PIPELINE TESTİ - YÜKSEK GINI")
print("="*80)

# Set seed for reproducibility
np.random.seed(2024)

def create_realistic_credit_data(n_samples=5000):
    """
    Kredi riski için gerçekçi sentetik veri oluştur
    - Yüksek ayırıcı güce sahip değişkenler (Gini ~0.70)
    - Bazı değişkenler PSI'a takılacak
    - Bazı değişkenler korele olacak
    """
    
    # Zaman serisi oluştur (train/test/oot için)
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    
    # Core risk faktörleri (yüksek ayırıcı güç)
    # 1. Kredi skoru - EN GÜÇLÜ tahmin edici
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    credit_score_norm = (credit_score - 300) / 550  # 0-1 normalize
    
    # 2. Gelir (income) - İkinci güçlü tahmin edici
    income = np.random.lognormal(10.5, 0.6, n_samples)
    income_norm = np.log1p(income) / 15  # normalize
    
    # 3. Yaş - Orta düzey tahmin edici
    age = np.random.beta(5, 3, n_samples) * 52 + 18  # 18-70 yaş, ortalama 40
    age_norm = (age - 18) / 52
    
    # 4. İstihdam süresi (employment_years) - İyi tahmin edici
    employment_years = np.random.exponential(7, n_samples).clip(0, 40)
    emp_norm = employment_years / 40
    
    # 5. Borç/Gelir oranı (debt_to_income) - Güçlü tahmin edici
    debt_to_income = np.random.beta(2, 5, n_samples) * 100
    dti_norm = debt_to_income / 100
    
    # Default olasılığını hesapla (lojistik model)
    # Yüksek Gini için güçlü ilişki kuruyoruz
    default_logit = (
        -2.5 +  # intercept (ortalama %15-25 default rate için)
        2.5 * (1 - credit_score_norm) +    # Kredi skoru düşükse risk yüksek
        1.5 * (1 - income_norm) +           # Gelir düşükse risk yüksek
        1.0 * dti_norm +                    # Borç oranı yüksekse risk yüksek
        0.8 * (1 - emp_norm) +              # İstihdam süresi azsa risk yüksek
        0.6 * (1 - age_norm) +              # Genç yaşta risk yüksek
        np.random.normal(0, 0.2, n_samples) # Daha az gürültü (daha yüksek Gini)
    )
    
    default_prob = 1 / (1 + np.exp(-default_logit))
    target = np.random.binomial(1, default_prob, n_samples)
    
    # Ek değişkenler (bazıları korele, bazıları PSI'a takılacak)
    
    # 6. Kredi limiti (income ile korele)
    credit_limit = income * np.random.uniform(2, 5, n_samples) + np.random.normal(0, 5000, n_samples)
    credit_limit = credit_limit.clip(1000, 200000)
    
    # 7. Mevcut kredi sayısı (age ile hafif korele)
    num_credits = np.random.poisson(1 + age/30, n_samples).clip(0, 10)
    
    # 8. Son 6 ayda kredi başvurusu (risk göstergesi)
    recent_inquiries = np.random.poisson(0.5 + 3*(1-credit_score_norm), n_samples).clip(0, 10)
    
    # 9. Kredi kullanım oranı (debt_to_income ile korele)
    credit_utilization = np.minimum(debt_to_income * 1.2 + np.random.normal(0, 10, n_samples), 100).clip(0, 100)
    
    # 10. Ödeme gecikmesi sayısı (kredi skoru ile ters korele)
    delinquency_count = np.random.poisson(0.1 + 5*(1-credit_score_norm), n_samples).clip(0, 20)
    
    # 11. Ev sahipliği (income ve age ile ilişkili)
    home_owner_prob = 0.2 + 0.3*income_norm + 0.3*age_norm
    home_ownership = np.random.binomial(1, home_owner_prob.clip(0, 1), n_samples)
    
    # 12. PSI drift için değişken (zaman içinde değişecek)
    # OOT'de farklı dağılım gösterecek
    time_index = np.arange(n_samples) / n_samples
    drift_feature = np.random.normal(50 + 20*time_index, 10, n_samples)
    
    # 13. Noise değişken (çok düşük bilgi değeri)
    noise_feature = np.random.randn(n_samples) * 100
    
    # Kategorik değişkenler
    
    # 14. Eğitim seviyesi (income ile ilişkili)
    education_probs = np.column_stack([
        0.1 + 0.3*(1-income_norm),  # Lise
        0.3 + 0.1*income_norm,       # Lisans
        0.2 + 0.3*income_norm,       # Y.Lisans
        0.1 + 0.2*income_norm        # Doktora
    ])
    education_probs = education_probs / education_probs.sum(axis=1, keepdims=True)
    education = np.array(['Lise', 'Lisans', 'Y.Lisans', 'Doktora'])[
        np.array([np.random.choice(4, p=p) for p in education_probs])
    ]
    
    # 15. İstihdam türü (default ile ilişkili)
    emp_type_probs = np.column_stack([
        0.5 + 0.2*credit_score_norm,    # Maaşlı
        0.2 + 0.1*(1-credit_score_norm), # Serbest
        np.full(n_samples, 0.1),        # Emekli
        0.1 + 0.1*(1-credit_score_norm)  # İşsiz
    ])
    emp_type_probs = emp_type_probs / emp_type_probs.sum(axis=1, keepdims=True)
    employment_type = np.array(['Maasli', 'Serbest', 'Emekli', 'Issiz'])[
        np.array([np.random.choice(4, p=p) for p in emp_type_probs])
    ]
    
    # 16. Şehir kategorisi
    city_tier = np.random.choice(['Buyuksehir', 'Sehir', 'Ilce', 'Koy'], 
                                 n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # 17. Medeni durum
    marital_status = np.random.choice(['Bekar', 'Evli', 'Bosanmis'], 
                                      n_samples, p=[0.3, 0.6, 0.1])
    
    # 18. Sektör
    sector = np.random.choice(['Finans', 'Teknoloji', 'Saglik', 'Egitim', 'Diger'],
                             n_samples, p=[0.15, 0.20, 0.15, 0.10, 0.40])
    
    # 19. Kredi tipi (başvurulan)
    loan_type = np.random.choice(['Ihtiyac', 'Konut', 'Tasit', 'KMH'],
                                 n_samples, p=[0.4, 0.2, 0.2, 0.2])
    
    # 20. Müşteri segmenti (birkaç değişkenin kombinasyonu)
    segment_score = credit_score_norm*0.4 + income_norm*0.3 + age_norm*0.3
    customer_segment = pd.cut(segment_score, bins=4, 
                              labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'app_id': [f'APP_{i:08d}' for i in range(n_samples)],
        'app_dt': dates,
        'target': target,
        
        # Numerik değişkenler (güçlü tahmin ediciler)
        'kredi_skoru': credit_score,
        'aylik_gelir': income,
        'yas': age.astype(int),
        'istihdam_suresi_yil': employment_years,
        'borc_gelir_orani': debt_to_income,
        
        # Korele değişkenler
        'kredi_limiti': credit_limit,
        'kredi_kullanim_orani': credit_utilization,
        
        # Risk göstergeleri
        'mevcut_kredi_sayisi': num_credits,
        'son_6ay_basvuru': recent_inquiries,
        'gecikme_sayisi': delinquency_count,
        
        # Diğer numerik
        'ev_sahibi': home_ownership,
        'drift_feature': drift_feature,  # PSI'a takılacak
        'noise_feature': noise_feature,  # Düşük IV, elenecek
        
        # Kategorik değişkenler
        'egitim': education,
        'istihdam_tipi': employment_type,
        'sehir_kategori': city_tier,
        'medeni_durum': marital_status,
        'sektor': sector,
        'kredi_tipi': loan_type,
        'musteri_segmenti': customer_segment
    })
    
    # Eksik değerler ekle (gerçekçilik için)
    missing_cols = ['istihdam_suresi_yil', 'kredi_kullanim_orani', 'sektor']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

# Ana veri seti oluştur
print("\n1. Gerçekçi kredi riski verisi oluşturuluyor...")
df = create_realistic_credit_data(5000)

print(f"   Veri boyutu: {df.shape}")
print(f"   Target oranı: {df['target'].mean():.2%}")
print(f"   Değişken sayısı: {df.shape[1] - 3}")  # id, date, target hariç

# Target ile korelasyon kontrol
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
correlations = df[numeric_cols].corrwith(df['target']).abs().sort_values(ascending=False)
print(f"\n   En yüksek korelasyonlar (target ile):")
for col, corr in correlations.head(5).items():
    print(f"     - {col}: {corr:.3f}")

# Kalibrasyon verisi (farklı dağılım)
print("\n2. Kalibrasyon verisi oluşturuluyor...")
cal_df = create_realistic_credit_data(1000)
# Kalibrasyon verisinde biraz farklı default rate (max %100 olabilir)
cal_target_prob = min(cal_df['target'].mean() * 1.1, 0.95)
cal_df['target'] = np.random.binomial(1, cal_target_prob, len(cal_df))
print(f"   Kalibrasyon boyutu: {cal_df.shape}")
print(f"   Kalibrasyon target oranı: {cal_df['target'].mean():.2%}")

# Veri sözlüğü
print("\n3. Veri sözlüğü hazırlanıyor...")
data_dict = pd.DataFrame({
    'alan_adi': [
        'kredi_skoru', 'aylik_gelir', 'yas', 'istihdam_suresi_yil', 'borc_gelir_orani',
        'kredi_limiti', 'kredi_kullanim_orani', 'mevcut_kredi_sayisi', 'son_6ay_basvuru',
        'gecikme_sayisi', 'ev_sahibi', 'drift_feature', 'noise_feature',
        'egitim', 'istihdam_tipi', 'sehir_kategori', 'medeni_durum', 'sektor', 
        'kredi_tipi', 'musteri_segmenti'
    ],
    'alan_aciklamasi': [
        'Kredi risk skoru (300-850 arası)',
        'Aylık gelir tutarı (TL)',
        'Müşteri yaşı (yıl)',
        'Toplam istihdam süresi (yıl)',
        'Borç/Gelir oranı (%)',
        'Toplam kredi limiti (TL)',
        'Kredi kullanım oranı (%)',
        'Mevcut aktif kredi sayısı',
        'Son 6 ayda yapılan başvuru sayısı',
        'Toplam ödeme gecikme sayısı',
        'Ev sahibi olma durumu (1/0)',
        'Zaman bağımlı özellik (drift test)',
        'Gürültü değişkeni (düşük bilgi)',
        'En yüksek eğitim seviyesi',
        'İstihdam durumu',
        'Yerleşim yeri kategorisi',
        'Medeni durum',
        'Çalışılan sektör',
        'Başvurulan kredi türü',
        'Müşteri segmenti'
    ]
})
print(f"   Tanımlanan değişken sayısı: {len(data_dict)}")

# Pipeline yapılandırması
print("\n4. Pipeline yapılandırılıyor...")
cfg = Config(
    # Temel ayarlar
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    
    # Veri bölme
    use_test_split=True,
    test_size_row_frac=0.2,
    oot_window_months=3,  # Son 3 ay OOT
    
    # Veri sözlüğü ve kalibrasyon
    data_dictionary_df=data_dict,
    calibration_df=cal_df,
    calibration_method='isotonic',
    
    # Model ayarları
    cv_folds=5,
    random_state=2024,
    n_jobs=2,
    
    # Hyperparameter optimizasyon
    hpo_timeout_sec=10,  # 10 saniye (çok hızlı test için)
    hpo_trials=5,        # 5 deneme (çok hızlı test için)
    
    # Feature engineering eşikleri
    rare_threshold=0.01,      # %1'den az görülen kategoriler
    psi_threshold=0.25,       # PSI > 0.25 ise değişken elenir (daha toleranslı)
    iv_min=0.01,             # IV < 0.01 ise değişken elenir (daha düşük eşik)
    rho_threshold=0.95,      # Korelasyon > 0.95 ise birini ele (daha toleranslı)
    
    # Çıktılar
    output_folder='outputs_realistic',
    output_excel_path='realistic_model_report.xlsx',
    log_file='outputs_realistic/pipeline.log',
    write_parquet=True,
    write_csv=True
)

print("   Yapılandırma hazır!")
print(f"   - PSI eşiği: {cfg.psi_threshold}")
print(f"   - IV minimum: {cfg.iv_min}")
print(f"   - Korelasyon eşiği: {cfg.rho_threshold}")

# Pipeline çalıştır
print("\n5. Pipeline çalıştırılıyor...")
print("="*60)

pipeline = RiskModelPipeline(cfg)
pipeline.run(df)

# Sonuçları kontrol et
print("\n" + "="*60)
print("SONUÇLAR")
print("="*60)

if pipeline.best_model_name_:
    print(f"\n✅ EN İYİ MODEL: {pipeline.best_model_name_}")
    print(f"✅ FİNAL DEĞİŞKEN SAYISI: {len(pipeline.final_vars_)}")
    
    if pipeline.final_vars_:
        print(f"\nFinal Değişkenler:")
        for i, var in enumerate(pipeline.final_vars_, 1):
            desc = data_dict[data_dict['alan_adi'] == var]['alan_aciklamasi'].values
            desc_str = desc[0] if len(desc) > 0 else "Açıklama yok"
            print(f"  {i}. {var}: {desc_str}")
    
    # Model performansı
    if pipeline.models_summary_ is not None and not pipeline.models_summary_.empty:
        best = pipeline.models_summary_[pipeline.models_summary_['model_name'] == pipeline.best_model_name_].iloc[0]
        
        print(f"\n📊 Model Performansı:")
        print(f"  Train:")
        print(f"    - AUC: {best.get('AUC_train', 'N/A'):.3f}" if best.get('AUC_train') else "    - AUC: N/A")
        print(f"    - Gini: {best.get('Gini_train', 'N/A'):.3f}" if best.get('Gini_train') else "    - Gini: N/A")
        
        if best.get('AUC_test'):
            print(f"  Test:")
            print(f"    - AUC: {best.get('AUC_test', 'N/A'):.3f}")
            print(f"    - Gini: {best.get('Gini_test', 'N/A'):.3f}")
        
        if best.get('AUC_OOT'):
            print(f"  OOT:")
            print(f"    - AUC: {best.get('AUC_OOT', 'N/A'):.3f}")
            print(f"    - Gini: {best.get('Gini_OOT', 'N/A'):.3f}")
            print(f"    - KS: {best.get('KS_OOT', 'N/A'):.3f}")
    
    # PSI kontrolü
    if hasattr(pipeline, 'psi_df_') and pipeline.psi_df_ is not None:
        print(f"\n📈 PSI Analizi:")
        high_psi = pipeline.psi_df_[pipeline.psi_df_['PSI'] > cfg.psi_threshold]
        if not high_psi.empty:
            print(f"  PSI > {cfg.psi_threshold} olan değişkenler (elendi):")
            for _, row in high_psi.iterrows():
                print(f"    - {row['variable']}: PSI={row['PSI']:.3f}")
        else:
            print(f"  Tüm değişkenler PSI < {cfg.psi_threshold}")
    
    # Korelasyon kontrolü
    if hasattr(pipeline, 'corr_dropped_') and pipeline.corr_dropped_:
        print(f"\n🔗 Korelasyon nedeniyle elenen değişkenler:")
        for item in pipeline.corr_dropped_:
            print(f"    - {item.get('dropped', 'N/A')} (corr={item.get('corr', 'N/A'):.3f} with {item.get('kept', 'N/A')})")
    
    # IV kontrolü
    if hasattr(pipeline, 'iv_filter_log_') and pipeline.iv_filter_log_:
        print(f"\n📊 IV Analizi:")
        for item in pipeline.iv_filter_log_:
            if item.get('reason', '').startswith('Low IV'):
                print(f"    - {item.get('variable', 'N/A')}: IV={item.get('iv', 'N/A'):.4f} (elendi)")
    
else:
    print("⚠️ UYARI: Model seçilemedi!")
    print(f"Final değişkenler: {pipeline.final_vars_}")

# Excel raporu kontrolü
excel_path = os.path.join(cfg.output_folder, cfg.output_excel_path)
if os.path.exists(excel_path):
    print(f"\n📁 Excel raporu oluşturuldu: {excel_path}")
    excel_file = pd.ExcelFile(excel_path)
    print(f"   Sheet sayısı: {len(excel_file.sheet_names)}")
    
    # WOE tablosu örneği
    if 'best_model_woe_df' in excel_file.sheet_names:
        woe_df = pd.read_excel(excel_path, sheet_name='best_model_woe_df')
        if not woe_df.empty:
            first_var = woe_df['variable'].iloc[0]
            print(f"\n📊 WOE Örneği ({first_var}):")
            # iv column may not exist in woe_df
            display_cols = ['group', 'count', 'event_rate', 'woe']
            if 'iv' in woe_df.columns:
                display_cols.append('iv')
            var_woe = woe_df[woe_df['variable'] == first_var][display_cols].head()
            print(var_woe.to_string(index=False))

print("\n" + "="*80)
print("TEST TAMAMLANDI!")
print("="*80)