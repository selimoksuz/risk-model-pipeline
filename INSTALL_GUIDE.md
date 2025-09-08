# Installation Guide - Dependency Hell'den Kurtulma Rehberi 🚀

## 🎯 Hızlı Başlangıç (Önerilen)

### Seçenek 1: Minimal Kurulum (Sadece Core)
```bash
# Sadece pipeline'ı çalıştırmak için gerekli olanlar
pip install -r requirements-core.txt
pip install -e .
```

### Seçenek 2: Docker ile Garantili Ortam (EN GÜVENLİ)
```bash
# Hiç dependency problemi yaşamayın!
docker-compose up risk-pipeline
```

### Seçenek 3: Tam Kurulum (Tüm Özellikler)
```bash
# Tüm özellikleri istiyorsanız
pip install -r requirements-exact.txt
pip install -e .
```

## 🔍 Ortamınızı Kontrol Edin

```bash
# Otomatik kontrol ve düzeltme
python check_environment.py
```

## 📦 Kurulum Seçenekleri

### 1. **Minimal (Core Only)**
- ✅ Pipeline çalışır
- ✅ Model eğitimi yapılır  
- ✅ Excel raporları üretilir
- ❌ Görselleştirme yok
- ❌ SHAP analizi yok

```bash
pip install risk-model-pipeline
```

### 2. **Görselleştirme Desteği ile**
```bash
pip install risk-model-pipeline[viz]
```

### 3. **ML Özellikleri ile**
```bash
pip install risk-model-pipeline[ml]
```

### 4. **Full Paket**
```bash
pip install risk-model-pipeline[all]
```

## 🐳 Docker ile Çalıştırma (Dependency Hell YOK!)

```bash
# İmajı build et
docker build -t risk-pipeline .

# Jupyter notebook olarak çalıştır
docker run -p 8888:8888 -v $(pwd)/data:/app/data risk-pipeline

# CLI olarak çalıştır
docker run -v $(pwd)/data:/app/data risk-pipeline python run_pipeline.py
```

## 🔧 Sorun Giderme

### Problem: "ImportError: cannot import name 'X'"
```bash
# Çözüm 1: Environment'ı sıfırla
pip uninstall risk-model-pipeline -y
pip install --no-cache-dir -r requirements-exact.txt
pip install -e .

# Çözüm 2: Docker kullan
docker-compose up
```

### Problem: "Version conflict"
```bash
# Exact versiyonları kullan
pip install -r requirements-exact.txt --force-reinstall
```

### Problem: "Matplotlib/Seaborn patladı"
```python
# Kodda safe import kullanın:
from risk_pipeline.utils.safe_imports import safe_matplotlib_import

plt, PLOT_AVAILABLE = safe_matplotlib_import()
if PLOT_AVAILABLE:
    # Görselleştirme yap
    plt.plot([1, 2, 3])
else:
    print("Görselleştirme mevcut değil, devam ediliyor...")
```

## 🏗️ Farklı Ortamlar için Setup

### Anaconda Kullanıcıları
```bash
# Yeni environment oluştur
conda create -n risk-pipeline python=3.9
conda activate risk-pipeline

# Core paketleri conda ile kur
conda install pandas numpy scikit-learn -c conda-forge

# Geri kalanı pip ile
pip install -r requirements-core.txt
```

### Virtualenv Kullanıcıları
```bash
# Yeni environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Exact versiyonları kur
pip install -r requirements-exact.txt
```

### Google Colab
```python
# Colab'da çalıştırmak için
!pip install git+https://github.com/selimoksuz/risk-model-pipeline.git
```

## 📊 Özellik Matrisi

| Özellik | Core | +viz | +ml | Docker |
|---------|------|------|-----|--------|
| Model Eğitimi | ✅ | ✅ | ✅ | ✅ |
| WOE Dönüşümü | ✅ | ✅ | ✅ | ✅ |
| Excel Rapor | ✅ | ✅ | ✅ | ✅ |
| Grafikler | ❌ | ✅ | ✅ | ✅ |
| SHAP | ❌ | ❌ | ✅ | ✅ |
| HPO (Optuna) | ❌ | ❌ | ✅ | ✅ |
| Garantili Çalışma | ⚠️ | ⚠️ | ⚠️ | ✅ |

## 🆘 Hala Sorun mu Var?

1. **Environment Report oluştur:**
   ```bash
   python check_environment.py
   ```

2. **Issue aç:** https://github.com/selimoksuz/risk-model-pipeline/issues

3. **Docker kullan:** Kesin çözüm!

## 💡 Pro İpuçları

1. **Production için:** Docker veya requirements-exact.txt kullanın
2. **Development için:** requirements-core.txt + ihtiyaç duyulan extras
3. **CI/CD için:** Docker image kullanın
4. **Notebook için:** `pip install risk-model-pipeline[notebook,viz]`

---

📌 **Not:** Dependency hell'den tamamen kurtulmak için Docker kullanmanızı şiddetle tavsiye ederiz!