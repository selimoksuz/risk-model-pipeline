# -*- coding: utf-8 -*-
"""
Risk Model Pipeline -- Orchestrated, Robust, Logged, Resource-Aware (PSI v2 FAST)

Bu tek dosya; 16 PARSEL akisini orkestrasyonla calistiran, WOE->PSI (vektorize)->FS->Model->Rapor
boru hattinin DERLENMIS ve DUZELTILMIS surumudur.

- Dataclasses icin MUTABLE DEFAULT hatasi (orchestrator alani) `default_factory` ile giderildi.
- PSI asamasinda 'np.bincount' hatasina yol acan -1 kodlari onlemek icin numeric bin araliklari
  `_normalize_numeric_edges` ile KOMSU ve BITISIK hale getirildi (bosluk kalmiyor).
- `_apply_bins` ve `_bin_labels_for_variable` eksiksizdir; unseen->OTHER, MISSING ayrimi yapilir.

Kullanim (ornek en altta):
    cfg  = Config(...)
    pipe = RiskModelPipeline(cfg)
    pipe.run(df)            # 16 Parsel orchestrated
    # pipe.export_reports() # Raporlar run sonunda da yazilir
"""

import os, sys, json, time, gc, warnings, uuid
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import UTF-8 fix for Windows console
try:
    from .utf8_fix import setup_utf8_console, safe_print
    setup_utf8_console()
except:
    # Fallback if utf8_fix is not available
    def safe_print(msg, file=None):
        try:
            print(msg, file=file, flush=True)
        except:
            print(str(msg).encode('ascii', 'ignore').decode('ascii'), file=file, flush=True)
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

# ---- BLAS/OpenMP oversubscription onleme ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Guard against oversubscription in BLAS backends
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Ensure UTF-8 console output on Windows
os.environ.setdefault('PYTHONUTF8','1')
try:
    import sys as _sys_utf8
    if hasattr(_sys_utf8.stdout, 'reconfigure'):
        _sys_utf8.stdout.reconfigure(encoding='utf-8')
        _sys_utf8.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pygam import LogisticGAM
from boruta import BorutaPy
from .stages import fit_calibrator, apply_calibrator
from .model.ensemble import soft_voting_ensemble
from .reporting.shap_utils import compute_shap_values, summarize_shap

warnings.filterwarnings("ignore")
try:
    import psutil
except Exception:
    psutil = None


# ========================= Logging helpers =========================
def now_str() -> str: return time.strftime("%H:%M:%S")

def sys_metrics() -> str:
    if psutil is None: return ""
    try:
        vm = psutil.virtual_memory(); cpu = psutil.cpu_percent(interval=0.1)
        return f" | CPU={cpu:.0f}% RAM={vm.percent:.0f}%"
    except Exception:
        return ""

class Timer:
    def __init__(self, label: str, logger=print):
        self.label = label; self.t0=None; self.logger=logger
    def __enter__(self):
        self.t0=time.time(); self.logger(f"[{now_str()}] >> {self.label} basliyor{sys_metrics()}"); return self
    def __exit__(self, exc_type, exc, tb):
        dt=time.time()-self.t0; status="OK" if exc is None else f"FAIL: {exc}"
        self.logger(f"[{now_str()}] â--  {self.label} bitti ({dt:.2f}s) â€” {status}{sys_metrics()}")


# ========================= Time helpers =========================
def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    try:
        if ts is None or pd.isna(ts): return pd.NaT
        if getattr(ts, "tzinfo", None): ts = ts.tz_localize(None)
        return pd.Timestamp(ts).to_period("M").to_timestamp()
    except Exception:
        try:
            ts2 = pd.to_datetime(ts, errors="coerce")
            return ts2.to_period("M").to_timestamp()
        except Exception:
            return pd.NaT

def safe_month_shift(anchor: pd.Timestamp, k: int) -> pd.Timestamp:
    base = month_floor(anchor)
    if pd.isna(base): return pd.NaT
    try:
        return month_floor(base - pd.DateOffset(months=k))
    except Exception:
        try:
            return (base.to_period("M") - k).to_timestamp()
        except Exception:
            return pd.NaT


# ========================= Metrics & WOE helpers =========================
def gini_from_auc(auc: float) -> float: return 2*auc - 1

def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, scores)
    ks_vals = tpr - fpr
    i = int(np.argmax(ks_vals))
    return float(ks_vals[i]), float(thr[i] if i < len(thr) else np.nan)

def ks_table(y_true: np.ndarray, scores: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "s": scores})
    df["band"] = pd.qcut(df["s"].rank(method="first"), q=n_bands, labels=False, duplicates="drop")
    out=[]
    for b in range(int(df["band"].max()), -1, -1):
        mask = (df["band"] >= b)
        tp = int((df.loc[mask,"y"]==1).sum()); fn = int((df.loc[~mask,"y"]==1).sum())
        tn = int((df.loc[~mask,"y"]==0).sum()); fp = int((df.loc[mask,"y"]==0).sum())
        tpr = tp/max(tp+fn,1); fpr = fp/max(fp+tn,1)
        out.append({"score_band":b,"TPR":tpr,"FPR":fpr,"KS":tpr-fpr,"cutpoint":float(df.loc[df["band"]==b,"s"].min())})
    return pd.DataFrame(out)

def psi_value(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1); return float(np.sum((p - q) * np.log(p / q)))

def jeffreys_counts(event: int, nonevent: int, alpha: float = 0.5) -> Tuple[float, float]:
    return event + alpha, nonevent + alpha

def woe_from_counts(event: int, nonevent: int, total_event: int, total_nonevent: int, alpha: float = 0.5):
    e_s, ne_s = jeffreys_counts(event, nonevent, alpha)
    te_s, tne_s = jeffreys_counts(total_event, total_nonevent, alpha)
    rate = e_s/(e_s+ne_s); de = e_s/te_s; dne = ne_s/tne_s
    woe = float(np.log(max(de,1e-12)/max(dne,1e-12)))
    return woe, float(rate), float(de - dne)

def compute_iv(rows: List[Dict[str, Any]]) -> float:
    iv=0.0
    for r in rows:
        e_s, ne_s = jeffreys_counts(r["event"], r["non_event"], 0.5)
        te_s, tne_s = jeffreys_counts(r["total_event"], r["total_nonevent"], 0.5)
        de=e_s/te_s; dne=ne_s/tne_s; iv+=(de-dne)*r["woe"]
    return float(iv)


# ========================= Orchestrasyon & Config =========================
@dataclass
class Orchestrator:
    enable_validate: bool = True
    enable_classify: bool = True
    enable_missing_policy: bool = True
    enable_split: bool = True
    enable_woe: bool = True
    enable_psi: bool = True
    enable_transform: bool = True
    enable_corr_cluster: bool = True
    enable_fs: bool = True
    enable_final_corr: bool = True
    enable_noise: bool = True
    enable_model: bool = True
    enable_best_select: bool = True
    enable_report: bool = True
    enable_dictionary: bool = False
    halt_on_critical: bool = True
    continue_on_optional: bool = True

@dataclass
class Config:
    # columns
    id_col: str = "app_id"; time_col: str = "app_dt"; target_col: str = "target"
    # split & cv
    use_test_split: bool = False
    # fraction of pre-OOT rows to allocate to TEST when splitting by months
    test_size_row_frac: float = 0.2
    stratify_by: Optional[List[str]] = None
    oot_window_months: int = 3
    oot_anchor_mode: str = "last_complete_month"
    oot_anchor_date: Optional[str] = None
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = max(1, (os.cpu_count() or 2)//2)
    # woe & psi
    rare_threshold: float = 0.02
    psi_threshold: float = 0.20
    psi_warn_low: float = 0.10
    psi_eps: float = 1e-6
    jeffreys_alpha: float = 0.5
    min_bins_numeric: int = 5
    min_count_auto_frac: float = 0.005
    min_count_auto_floor: int = 50
    min_bin_share_target: float = 0.02
    max_bin_share_target: float = 0.35
    max_abs_woe_warn: float = 3.0
    monotonic_enabled: bool = False
    # new thresholds & calibration
    calibration_data_path: Optional[str] = None
    calibration_df: Optional[pd.DataFrame] = field(default=None)  # NEW: DataFrame support
    data_dictionary_path: Optional[str] = None  # Path to Excel with variable descriptions
    data_dictionary_df: Optional[pd.DataFrame] = field(default=None)  # DataFrame with alan_adi, alan_aciklamasi columns
    calibration_method: str = "isotonic"
    iv_min: float = 0.02
    iv_high_flag: float = 0.50
    cluster_top_k: int = 2
    rho_threshold: float = 0.8
    vif_threshold: float = 5.0
    psi_threshold_feature: float = 0.25
    psi_threshold_score: float = 0.10
    shap_sample: int = 25000
    ensemble: bool = False
    ensemble_top_k: int = 3
    try_mlp: bool = False
    hpo_method: str = "random"
    hpo_timeout_sec: int = 1200
    hpo_trials: int = 60
    # psi logging
    psi_verbose: bool = True
    psi_log_every: int = 5
    psi_sample_months: int = 6
    # outputs
    output_excel_path: str = "model_report.xlsx"
    output_folder: str = "outputs"
    log_file: Optional[str] = None
    write_parquet: bool = True
    write_csv: bool = False
    dictionary_path: Optional[str] = None
    use_benchmarks: bool = True
    use_noise_sentinel: bool = True
    ks_bands: int = 10
    # DUAL PIPELINE SETTINGS
    enable_dual_pipeline: bool = False  # Enable dual pipeline (WOE + Raw)
    raw_imputation_strategy: str = "median"  # Imputation for raw pipeline: median, mean, zero
    raw_outlier_method: str = "iqr"  # Outlier method for raw pipeline: iqr, zscore, percentile
    raw_outlier_threshold: float = 3.0  # Threshold for outlier removal
    # orchestrator & run id
    run_id: str = ""
    orchestrator: Orchestrator = field(default_factory=Orchestrator)  # <<< mutable default FIX

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class NumericBin:
    left: float; right: float; count: int; event: int; non_event: int; event_rate: float; woe: float

@dataclass
class CategoricalGroup:
    members: List[Any]; count: int; event: int; non_event: int
    event_rate: float; woe: float; label: str

@dataclass
class VariableWOE:
    name: str; var_type: str
    numeric_bins: Optional[List[NumericBin]] = None
    categorical_groups: Optional[List[CategoricalGroup]] = None
    total_event: int = 0; total_nonevent: int = 0; iv: float = 0.0


# ========================= Pipeline (16 Parsel, tek dosya) =========================
class RiskModelPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.artifacts: Dict[str, Any] = {"run_id": cfg.run_id, "active_steps": [], "pool": {}}

        self.df_: Optional[pd.DataFrame] = None
        self.train_idx_: Optional[np.ndarray] = None
        self.test_idx_: Optional[np.ndarray] = None
        self.oot_idx_: Optional[np.ndarray] = None
        self.woe_map: Dict[str, VariableWOE] = {}

        self.psi_summary_: Optional[pd.DataFrame] = None
        self.psi_dropped_: Optional[pd.DataFrame] = None
        self.models_: Dict[str, Any] = {}
        self.models_summary_: Optional[pd.DataFrame] = None
        self.best_model_name_: Optional[str] = None
        self.best_model_vars_df_: Optional[pd.DataFrame] = None
        self.best_model_woe_df_: Optional[pd.DataFrame] = None
        self.top20_iv_df_: Optional[pd.DataFrame] = None
        self.top50_uni_: Optional[pd.DataFrame] = None
        self.final_vars_df_: Optional[pd.DataFrame] = None
        self.ks_info_traincv_: Optional[pd.DataFrame] = None
        self.ks_info_test_: Optional[pd.DataFrame] = None
        self.ks_info_oot_: Optional[pd.DataFrame] = None
        self.confusion_oot_: Optional[pd.DataFrame] = None
        self.oot_scores_df_: Optional[pd.DataFrame] = None

        # Dual pipeline attributes
        self.raw_fit_params_: Optional[Dict] = None
        self.raw_models_: Dict[str, Any] = {}
        self.raw_models_summary_: Optional[pd.DataFrame] = None
        self.raw_baseline_vars_: List[str] = []
        self.raw_final_vars_: List[str] = []
        self.woe_models_: Dict[str, Any] = {}
        self.woe_models_summary_: Optional[pd.DataFrame] = None
        self.woe_best_model_name_: Optional[str] = None
        self.woe_final_vars_: List[str] = []

        self.cluster_reps_: List[str] = []; self.baseline_vars_: List[str] = []; self.final_vars_: List[str] = []
        self.high_iv_flags_: List[str] = []
        self.iv_filter_log_: List[Dict[str, Any]] = []
        self.corr_dropped_: List[Dict[str, Any]] = []
        self.shap_summary_: Optional[Dict[str, float]] = None
        self.calibrator_: Optional[Any] = None
        self.calibration_report_: Optional[Dict[str, Any]] = None

        self.logger = self.setup_logger(cfg)
        self.log_fh = None
        if self.cfg.log_file:
            os.makedirs(os.path.dirname(self.cfg.log_file) or ".", exist_ok=True)
            # Use UTF-8 with BOM for better Windows Notepad compatibility
            self.log_fh = open(self.cfg.log_file, "w", encoding="utf-8-sig")
    
    def setup_logger(self, cfg: Config):
        """Setup logger for the pipeline"""
        return print  # Simple logger that uses print function

    def _log(self, msg: str):
        # Use safe_print for proper UTF-8 handling
        safe_print(msg)
        if self.log_fh:
            try:
                self.log_fh.write(str(msg) + "\n")
                self.log_fh.flush()
            except Exception:
                try:
                    self.log_fh.write(str(msg).encode("ascii", "ignore").decode("ascii") + "\n")
                    self.log_fh.flush()
                except Exception:
                    pass
    def _activate(self, name: str): self.artifacts["active_steps"].append(name)
    
    def _load_data_dictionary(self) -> Dict[str, str]:
        """Load data dictionary from DataFrame or Excel file"""
        var_descriptions = {}
        
        # Try DataFrame first
        if self.cfg.data_dictionary_df is not None:
            df = self.cfg.data_dictionary_df
            if 'alan_adi' in df.columns and 'alan_aciklamasi' in df.columns:
                var_descriptions = dict(zip(df['alan_adi'], df['alan_aciklamasi']))
                self._log(f"   - Data dictionary loaded from DataFrame: {len(var_descriptions)} variables")
        
        # Try file path
        elif self.cfg.data_dictionary_path and os.path.exists(self.cfg.data_dictionary_path):
            try:
                df = pd.read_excel(self.cfg.data_dictionary_path)
                if 'alan_adi' in df.columns and 'alan_aciklamasi' in df.columns:
                    var_descriptions = dict(zip(df['alan_adi'], df['alan_aciklamasi']))
                    self._log(f"   - Data dictionary loaded from {self.cfg.data_dictionary_path}: {len(var_descriptions)} variables")
            except Exception as e:
                self._log(f"   - Warning: Could not load data dictionary: {e}")
        
        return var_descriptions

    # ==================== ORCHESTRATED RUN ====================
    def run(self, df: pd.DataFrame):
        self.df_ = df

        # Parsel 2 â€” Giris dogrulama & sabitleme
        if self.cfg.orchestrator.enable_validate:
            with Timer("2) Giris dogrulama & sabitleme", self._log):
                self._activate("validate")
                self._validate_and_freeze(self.df_)
                self._downcast_inplace(self.df_)

        # Parsel 3 â€” Degisken siniflamasi
        if self.cfg.orchestrator.enable_classify:
            with Timer("3) Degisken siniflamasi", self._log):
                self._activate("classify")
                self.var_catalog_ = self._classify_variables(self.df_)
                self._log(f"   - numeric={int((self.var_catalog_.variable_group=='numeric').sum())}, "
                          f"categorical={int((self.var_catalog_.variable_group=='categorical').sum())}")

        # Parsel 4 â€” Eksik & Nadir kural objesi
        with Timer("4) Eksik & Nadir deger politikasi", self._log):
            self._activate("missing_policy")
            self.policy_ = {"rare_threshold": self.cfg.rare_threshold,
                            "unknown_to": "OTHER", "missing_label": "MISSING", "other_label": "OTHER"}

        # Parsel 5 â€” Zaman bolmesi
        if self.cfg.orchestrator.enable_split:
            with Timer("5) Zaman bolmesi (Train/Test/OOT)", self._log):
                self._activate("split")
                self.train_idx_, self.test_idx_, self.oot_idx_, anchor = self._split_time(self.df_)
                self.artifacts["pool"]["anchor"] = str(anchor) if anchor is not None else None
                self._log(f"   - Train={len(self.train_idx_)}, Test={0 if self.test_idx_ is None else len(self.test_idx_)}, OOT={len(self.oot_idx_)}")
        else:
            idx_all = np.arange(len(self.df_))
            self.train_idx_, self.test_idx_, self.oot_idx_ = idx_all, None, np.array([], dtype=int)

        # Parsel 6 â€” WOE binleme
        if self.cfg.orchestrator.enable_woe:
            with Timer("6) WOE binleme (yalniz Train; adaptif)", self._log):
                self._activate("woe")
                self.woe_map = self._fit_woe_mapping(self.df_.iloc[self.train_idx_], self.var_catalog_, self.policy_)
                self._log(f"   - WOE hazir: {len(self.woe_map)} degisken")
                self._log("   - Not: WOE haritasi SADECE TRAIN'de ogrenildi; TEST/OOT icin ayni harita uygulanir (leakage yok)")

        # Parsel 7 â€” PSI v2 FAST
        if self.cfg.orchestrator.enable_psi:
            with Timer("7) PSI (vektorize)", self._log):
                self._activate("psi")
                psi_keep = self._psi_screening(self.df_, self.train_idx_, self.test_idx_, self.oot_idx_)
                if not psi_keep:
                    iv_sorted = sorted([(k, v.iv) for k, v in self.woe_map.items()], key=lambda t: t[1], reverse=True)
                    psi_keep = [iv_sorted[0][0]] if iv_sorted else []
                self._log(f"   - PSI sonrasi kalan: {len(psi_keep)}")
        else:
            psi_keep = list(self.woe_map.keys())

        # IV filter after PSI
        psi_keep = self._iv_filter(psi_keep)

        # Parsel 8 â€” Transform
        X_tr = X_te = X_oot = y_tr = y_te = y_oot = None
        if self.cfg.orchestrator.enable_transform:
            with Timer("8) WOE transform (Train/Test/OOT)", self._log):
                X_tr, y_tr = self._transform(self.df_.iloc[self.train_idx_], psi_keep)
                if self.test_idx_ is not None and len(self.test_idx_)>0:
                    X_te, y_te = self._transform(self.df_.iloc[self.test_idx_], psi_keep)
                X_oot, y_oot = self._transform(self.df_.iloc[self.oot_idx_], psi_keep)
                self._log(f"   - X_train={X_tr.shape}, X_test={None if X_te is None else X_te.shape}, X_oot={X_oot.shape}")

        # Parsel 9 â€” Korelasyon & cluster temsilcileri
        if self.cfg.orchestrator.enable_corr_cluster:
            with Timer("9) Korelasyon & cluster", self._log):
                self.cluster_reps_ = self._corr_and_cluster(X_tr, psi_keep) or psi_keep[:min(10, len(psi_keep))]
                self._log(f"   - cluster temsilcisi={len(self.cluster_reps_)}")
        else:
            self.cluster_reps_ = psi_keep

        # Parsel 10 â€” FS
        if self.cfg.orchestrator.enable_fs:
            with Timer("10) Feature selection (Forward+1SE)", self._log):
                self.baseline_vars_ = self._feature_selection(X_tr, y_tr, self.cluster_reps_, psi_keep) or psi_keep[:min(5, len(psi_keep))]
                self._log(f"   - baseline degisken={len(self.baseline_vars_)}")
        else:
            self.baseline_vars_ = psi_keep

        # Parsel 11 â€” Nihai korelasyon filtresi
        pre_final = self.baseline_vars_
        if self.cfg.orchestrator.enable_final_corr:
            with Timer("11) Nihai korelasyon filtresi", self._log):
                pre_final = self._final_corr_filter(X_tr[self.baseline_vars_], y_tr) or self.baseline_vars_
                self._log(f"   - corr sonrasi={len(pre_final)}")

        # Parsel 12 â€” Noise sentinel
        if self.cfg.orchestrator.enable_noise and self.cfg.use_noise_sentinel:
            with Timer("12) Gurultu (noise) sentineli", self._log):
                self.final_vars_ = self._noise_sentinel_check(X_tr, y_tr, pre_final) or pre_final
                self._log(f"   - final degisken={len(self.final_vars_)}")
        else:
            self.final_vars_ = pre_final

        # Parsel 13 â€" Modelleme
        if self.cfg.orchestrator.enable_model:
            with Timer("13) Modelleme & degerlendirme (WOE)", self._log):
                self._train_and_evaluate_models(X_tr, y_tr, X_te, y_te, X_oot, y_oot)
        
        # DUAL PIPELINE: Run raw variable pipeline if enabled
        if self.cfg.enable_dual_pipeline:
            self._log("\n" + "="*80)
            self._log("DUAL PIPELINE: RAW VARIABLES (Ham Degiskenler)")
            self._log("="*80)
            
            # Store WOE results
            self.woe_models_ = self.models_.copy()
            self.woe_models_summary_ = self.models_summary_.copy() if self.models_summary_ is not None else None
            self.woe_best_model_name_ = self.best_model_name_
            self.woe_final_vars_ = self.final_vars_.copy() if self.final_vars_ else []
            
            # Prepare raw data
            X_tr_raw = X_te_raw = X_oot_raw = None
            with Timer("8b) Raw transform (Train/Test/OOT)", self._log):
                # Use original features for raw pipeline
                X_tr_raw, y_tr = self._transform_raw(self.df_.iloc[self.train_idx_], psi_keep)
                if self.test_idx_ is not None and len(self.test_idx_)>0:
                    X_te_raw, y_te = self._transform_raw(self.df_.iloc[self.test_idx_], psi_keep, 
                                                         fit_params=self.raw_fit_params_)
                X_oot_raw, y_oot = self._transform_raw(self.df_.iloc[self.oot_idx_], psi_keep, 
                                                       fit_params=self.raw_fit_params_)
                self._log(f"   - X_train_raw={X_tr_raw.shape}, X_test_raw={None if X_te_raw is None else X_te_raw.shape}, X_oot_raw={X_oot_raw.shape}")
            
            # Feature selection for raw pipeline
            if self.cfg.orchestrator.enable_fs:
                with Timer("10b) Feature selection RAW (Forward+1SE)", self._log):
                    self.raw_baseline_vars_ = self._feature_selection(X_tr_raw, y_tr, psi_keep, psi_keep) or psi_keep[:min(5, len(psi_keep))]
                    self._log(f"   - raw baseline degisken={len(self.raw_baseline_vars_)}")
            else:
                self.raw_baseline_vars_ = psi_keep
            
            # Final correlation filter for raw
            raw_pre_final = self.raw_baseline_vars_
            if self.cfg.orchestrator.enable_final_corr:
                with Timer("11b) Nihai korelasyon filtresi RAW", self._log):
                    raw_pre_final = self._final_corr_filter(X_tr_raw[self.raw_baseline_vars_], y_tr) or self.raw_baseline_vars_
                    self._log(f"   - raw corr sonrasi={len(raw_pre_final)}")
            
            # Noise sentinel for raw
            if self.cfg.orchestrator.enable_noise and self.cfg.use_noise_sentinel:
                with Timer("12b) Gurultu sentineli RAW", self._log):
                    self.raw_final_vars_ = self._noise_sentinel_check(X_tr_raw, y_tr, raw_pre_final) or raw_pre_final
                    self._log(f"   - raw final degisken={len(self.raw_final_vars_)}")
            else:
                self.raw_final_vars_ = raw_pre_final
            
            # Clear models for raw pipeline
            self.models_ = {}
            self.final_vars_ = self.raw_final_vars_
            
            # Train raw models
            if self.cfg.orchestrator.enable_model:
                with Timer("13b) Modelleme & degerlendirme (RAW)", self._log):
                    self._train_and_evaluate_models(X_tr_raw, y_tr, X_te_raw, y_te, X_oot_raw, y_oot)
            
            # Store raw results
            self.raw_models_ = self.models_.copy()
            self.raw_models_summary_ = self.models_summary_.copy() if self.models_summary_ is not None else None
            
            # Combine results from both pipelines
            self._combine_dual_pipeline_results()

        # Parsel 14 â€” Best select
        if self.cfg.orchestrator.enable_best_select:
            with Timer("14) En iyi model secimi", self._log):
                self._select_best_model(); self._log(f"   - best={self.best_model_name_}")

        if self.cfg.shap_sample and self.best_model_name_:
            try:
                mdl = self.models_.get(self.best_model_name_)
                shap_vals = compute_shap_values(mdl, X_tr[self.final_vars_], self.cfg.shap_sample, self.cfg.random_state)
                self.shap_summary_ = summarize_shap(shap_vals, self.final_vars_)
            except Exception:
                self.shap_summary_ = None

        if self.cfg.calibration_data_path or self.cfg.calibration_df is not None:
            with Timer("14b) Kalibrasyon", self._log):
                self._calibrate_model()

        # Parsel 15 â€” Rapor tablolari & export
        if self.cfg.orchestrator.enable_report:
            with Timer("15) Rapor tablolari", self._log):
                self._build_report_tables(psi_keep)
                self._build_top50_univariate(X_tr, y_tr)
                self._persist_artifacts(X_oot, y_oot)
            with Timer("15b) Export (Excel/Parquet)", self._log):
                self.export_reports()

        # Parsel 16 â€” Dictionary (opsiyonel)
        if self.cfg.orchestrator.enable_dictionary:
            with Timer("16) Dictionary entegrasyonu", self._log):
                self._integrate_dictionary()

        self._finalize_meta()
        self._log(f"[{now_str()}] >> RUN tamam - run_id={self.cfg.run_id}{sys_metrics()}")
        if self.log_fh:
            self.log_fh.close()
        return self

    # ---------------- core: validate / downcast / classify / split ----------------
    def _validate_and_freeze(self, df: pd.DataFrame):
        for c in [self.cfg.id_col, self.cfg.time_col, self.cfg.target_col]:
            if c not in df.columns: raise ValueError(f"Zorunlu kolon eksik: {c}")
        if not set(pd.Series(df[self.cfg.target_col]).dropna().unique()).issubset({0,1}):
            raise ValueError("target_col yalniz {0,1} olmali.")
        try:
            df["snapshot_month"] = pd.to_datetime(df[self.cfg.time_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        except Exception:
            df["snapshot_month"] = df[self.cfg.time_col].apply(month_floor)

    def _downcast_inplace(self, df: pd.DataFrame):
        for c in df.columns:
            s = df[c]
            try:
                if pd.api.types.is_integer_dtype(s):   df[c]=pd.to_numeric(s, downcast="integer")
                elif pd.api.types.is_float_dtype(s):   df[c]=pd.to_numeric(s, downcast="float")
            except Exception: pass

    def _classify_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        from .stages import classify_variables
        return classify_variables(df, id_col=self.cfg.id_col, time_col=self.cfg.time_col, target_col=self.cfg.target_col)
    def _split_time(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[pd.Timestamp]]:
        from .stages import time_based_split
        train_idx, test_idx, oot_idx = time_based_split(
            df,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            use_test_split=self.cfg.use_test_split,
            oot_window_months=self.cfg.oot_window_months,
            test_size_row_frac=self.cfg.test_size_row_frac,
        )
        anchor = None
        return train_idx.values, (test_idx.values if test_idx is not None else None), oot_idx.values, anchor
    def _get_categories(self, cats_obj) -> List[Any]:
        try:
            return list(cats_obj.categories)
        except AttributeError:
            try:
                return list(cats_obj.cat.categories)
            except Exception:
                vals = pd.Series(cats_obj).dropna().unique().tolist()
                return sorted(vals)

    def _make_unique(self, labels: List[str]) -> List[str]:
        uniq=[]; seen=set()
        for lab in labels:
            base=str(lab)
            if base not in seen: uniq.append(base); seen.add(base)
            else:
                k=2
                while f"{base}__{k}" in seen: k+=1
                uniq.append(f"{base}__{k}"); seen.add(f"{base}__{k}")
        return uniq

    def _bin_labels_for_variable(self, vw: "VariableWOE") -> List[str]:
        """Bir degiskenin WOE bin/grup etiketlerini (sirali ve benzersiz) verir."""
        if vw.var_type == "numeric":
            labels = [f"[{b.left},{b.right})"
                      for b in vw.numeric_bins
                      if not (np.isnan(b.left) and np.isnan(b.right))]
            labels.append("MISSING")
        else:
            labels = [g.label for g in vw.categorical_groups if g.label != "MISSING"]
            if "OTHER" not in labels:
                labels.append("OTHER")
            labels.append("MISSING")
        return self._make_unique(labels)

    def _apply_bins(self, df: pd.DataFrame, var: str, vw: "VariableWOE") -> pd.Series:
        """
        Train'de fit edilmis mapping'i kullanarak df[var]'i WOE bin/grup etiketlerine cevirir.
        Cikis pandas.Categorical (sabit kategori seti).
        """
        s = df[var]
        out = pd.Series(index=s.index, dtype=object)

        if vw.var_type == "numeric":
            miss = s.isna()
            out.loc[miss] = "MISSING"
            for b in vw.numeric_bins:
                if np.isnan(b.left) and np.isnan(b.right):
                    continue
                m = (~miss) & (s >= b.left) & (s <= b.right)
                out.loc[m] = f"[{b.left},{b.right})"
        else:
            miss = s.isna()
            out.loc[miss] = "MISSING"
            assigned = miss.copy()
            for g in vw.categorical_groups:
                if g.label == "MISSING":
                    continue
                m = (~miss) & (s.isin(g.members))
                out.loc[m] = g.label
                assigned |= m
            out.loc[~assigned] = "OTHER"

        out = out.astype("category")
        labels = self._bin_labels_for_variable(vw)
        try:
            out = out.cat.set_categories(labels)
        except Exception:
            out = out.cat.set_categories(self._make_unique(labels))
        return out

    # ----------------------- WOE FIT -----------------------
    def _fit_woe_mapping(self, train_df: pd.DataFrame, var_catalog: pd.DataFrame, policy: Dict[str, Any]) -> Dict[str, VariableWOE]:
        y=train_df[self.cfg.target_col].values
        total_e=int((y==1).sum()); total_ne=int((y==0).sum()); n_total=int(len(y))
        min_count_auto=max(self.cfg.min_count_auto_floor,int(self.cfg.min_count_auto_frac*n_total))
        mapping={}
        for _,row in var_catalog.iterrows():
            var=row["variable"]; vtype=row["variable_group"]; s=train_df[var]
            vw=VariableWOE(name=var,var_type=vtype,total_event=total_e,total_nonevent=total_ne)
            if vtype=="numeric":
                vw.numeric_bins=self._bin_numeric_adaptive(
                    s,y,self.cfg.min_bins_numeric,min_count_auto,
                    self.cfg.min_bin_share_target,self.cfg.max_bin_share_target,
                    self.cfg.jeffreys_alpha,self.cfg.max_abs_woe_warn,self.cfg.monotonic_enabled
                )
                rows=[{"event":b.event,"non_event":b.non_event,"woe":b.woe,"total_event":total_e,"total_nonevent":total_ne} for b in vw.numeric_bins]
                vw.iv=compute_iv(rows)
            else:
                vw.categorical_groups=self._group_categorical_adaptive(
                    s,y,policy["rare_threshold"],policy["missing_label"],policy["other_label"],
                    min_count_auto,self.cfg.min_bin_share_target,self.cfg.max_bin_share_target,
                    self.cfg.jeffreys_alpha,self.cfg.max_abs_woe_warn
                )
                rows=[{"event":g.event,"non_event":g.non_event,"woe":g.woe,"total_event":total_e,"total_nonevent":total_ne} for g in vw.categorical_groups]
                vw.iv=compute_iv(rows)
            mapping[var]=vw
        return mapping

    # ---- numeric bin helpers ----
    def _numeric_bins_from_bin_index(self, df: pd.DataFrame, alpha: float) -> List[NumericBin]:
        y=df["y"].values; x=df["x"].values; cats=df["bin"]
        out=[]; total_e=int((y==1).sum()); total_ne=int((y==0).sum())
        categories=self._get_categories(cats)
        for cat in categories:
            mask=(cats==cat); sub=df.loc[mask,"x"]
            l=float(sub.min()) if len(sub) else -np.inf; r=float(sub.max()) if len(sub) else np.inf
            cnt=int(mask.sum()); e=int(df.loc[mask,"y"].sum()); ne=cnt-e
            w,er,_=woe_from_counts(e,ne,total_e,total_ne,alpha)
            out.append(NumericBin(l,r,cnt,e,ne,er,w))
        miss=int(df["x"].isna().sum())
        if miss>0:
            e=int(df.loc[df["x"].isna(),"y"].sum()); ne=miss-e
            w,er,_=woe_from_counts(e,ne,alpha=alpha,total_event=total_e,total_nonevent=total_ne)
            out.append(NumericBin(float("nan"),float("nan"),miss,e,ne,er,w))
        return self._normalize_numeric_edges(out)

    def _numeric_bins_from_edges(self, df: pd.DataFrame, edges: List[float], alpha: float) -> List[NumericBin]:
        try: e=sorted(set([float(x) for x in edges if x is not None and np.isfinite(x)]))
        except Exception: e=edges
        if len(e)<2:
            x=df["x"].values; y=df["y"].values
            total_e=int((y==1).sum()); total_ne=int((y==0).sum())
            fin=np.isfinite(x); cnt=int(fin.sum()); e_cnt=int(df.loc[fin,"y"].sum()); ne_cnt=cnt-e_cnt
            w,er,_=woe_from_counts(e_cnt,ne_cnt,total_e,total_ne,alpha)
            out=[NumericBin(-np.inf,np.inf,cnt,e_cnt,ne_cnt,er,w)]
            miss=int((~fin).sum())
            if miss>0:
                e_m=int(df.loc[~fin,"y"].sum()); ne_m=miss-e_m
                w_m,er_m,_=woe_from_counts(e_m,ne_m,total_e,total_ne,alpha)
                out.append(NumericBin(float("nan"),float("nan"),miss,e_m,ne_m,er_m,w_m))
            return out
        x=df["x"].values; y=df["y"].values
        try: cats=pd.cut(x,bins=e,include_lowest=True,duplicates="drop")
        except Exception:
            e=np.unique(np.asarray(e,dtype=float)); cats=pd.cut(x,bins=e,include_lowest=True,duplicates="drop")
        total_e=int((y==1).sum()); total_ne=int((y==0).sum())
        out=[]; categories=self._get_categories(cats)
        for cat in categories:
            mask=(cats==cat); cnt=int(mask.sum()); e_cnt=int(df.loc[mask,"y"].sum()); ne_cnt=cnt-e_cnt
            try: l=float(cat.left); r=float(cat.right)
            except Exception:
                l=float(df.loc[mask,"x"].min()) if cnt>0 else -np.inf
                r=float(df.loc[mask,"x"].max()) if cnt>0 else  np.inf
            w,er,_=woe_from_counts(e_cnt,ne_cnt,total_e,total_ne,alpha)
            out.append(NumericBin(l,r,cnt,e_cnt,ne_cnt,er,w))
        miss_mask=~np.isfinite(x); miss=int(miss_mask.sum())
        if miss>0:
            e_m=int(df.loc[miss_mask,"y"].sum()); ne_m=miss-e_m
            w_m,er_m,_=woe_from_counts(e_m,ne_m,total_e,total_ne,alpha)
            out.append(NumericBin(float("nan"),float("nan"),miss,e_m,ne_m,er_m,w_m))
        return self._normalize_numeric_edges(out)

    def _normalize_numeric_edges(self, bins: List[NumericBin]) -> List[NumericBin]:
        """
        Bin araliklarini bitisik hale getirir (bosluk birakmaz):
        - Ilk bin left = -inf
        - Son bin right = +inf
        - Aradaki tum binlerde next.left = prev.right
        - MISSING bin(ler) (NaN-NaN) sona alinir.
        """
        finite = [b for b in bins if not (np.isnan(b.left) and np.isnan(b.right))]
        finite = sorted(finite, key=lambda b: (b.left, b.right))
        if finite:
            finite[0].left = -np.inf
            for i in range(1, len(finite)):
                finite[i].left = finite[i-1].right  # KOMSULUK: bosluk kalmasin
            finite[-1].right = np.inf
        miss = [b for b in bins if np.isnan(b.left) and np.isnan(b.right)]
        return finite + miss

    def _edges_from_pairs(self, pairs: List[Tuple[float, float]]) -> List[float]:
        edges=[pairs[0][0]]
        for _,r in pairs: edges.append(r)
        return edges

    def _merge_numeric_bins(self, df, bins, i, j, alpha):
        b1,b2=bins[i],bins[j]; l=min(b1.left,b2.left); r=max(b1.right,b2.right)
        mask=(df["x"]>=l)&(df["x"]<=r)
        cnt=int(mask.sum()); e=int(df.loc[mask,"y"].sum()); ne=cnt-e
        te=int((df["y"]==1).sum()); tne=int((df["y"]==0).sum())
        w,er,_=woe_from_counts(e,ne,te,tne,alpha)
        merged=NumericBin(l,r,cnt,e,ne,er,w)
        return bins[:i]+[merged]+bins[j+1:]

    def _numeric_adaptive_optimize(self, df, bins, min_count_auto, min_share, max_share, alpha, max_abs_woe, monotonic):
        n=len(df); max_iter=20
        for _ in range(max_iter):
            changed=False
            for i in range(len(bins)-1):
                b=bins[i]; share=b.count/max(n,1)
                if share<min_share or b.count<min_count_auto:
                    bins=self._merge_numeric_bins(df,bins,i,i+1,alpha); changed=True; break
            if changed: bins=self._normalize_numeric_edges(bins); continue
            for i in range(len(bins)):
                b=bins[i]
                if np.isnan(b.left) and np.isnan(b.right): continue
                share=b.count/max(n,1)
                if share>max_share or abs(b.woe)>max_abs_woe:
                    df_in=df[(df["x"]>=b.left)&(df["x"]<=b.right)]
                    if len(df_in)>=2*min_count_auto:
                        median=df_in["x"].median()
                        if np.isfinite(median) and (b.left<median<b.right):
                            pairs=[]
                            for j,bb in enumerate(bins):
                                if j<i: pairs.append((bb.left,bb.right))
                                elif j==i: pairs+=[(bb.left,median),(median,bb.right)]
                                else: pairs.append((bb.left,bb.right))
                            bins=self._numeric_bins_from_edges(df,self._edges_from_pairs(pairs),alpha); changed=True; break
            if changed: bins=self._normalize_numeric_edges(bins); continue
            if monotonic:
                finite=[b for b in bins if not (np.isnan(b.left) and np.isnan(b.right))]
                if len(finite)>=3:
                    w=[b.woe for b in finite]
                    mono_up=all(w[i]<=w[i+1] for i in range(len(w)-1))
                    mono_dn=all(w[i]>=w[i+1] for i in range(len(w)-1))
                    if not(mono_up or mono_dn):
                        best_idx,best_cost=None,float("inf")
                        for i in range(len(bins)-1):
                            if any(np.isnan([bins[i].left,bins[i].right,bins[i+1].left,bins[i+1].right])): continue
                            cost=abs(bins[i].woe-bins[i+1].woe)
                            if cost<best_cost: best_cost=cost; best_idx=i
                        if best_idx is not None:
                            bins=self._merge_numeric_bins(df,bins,best_idx,best_idx+1,alpha); changed=True
            if changed: bins=self._normalize_numeric_edges(bins); continue
            finite=[b for b in bins if not (np.isnan(b.left) and np.isnan(b.right))]
            if len(finite)<self.cfg.min_bins_numeric and len(finite)>=1:
                lengths=[bb.right-bb.left for bb in finite]; idx=int(np.argmax(lengths)); b=finite[idx]
                df_in=df[(df["x"]>=b.left)&(df["x"]<=b.right)]
                if len(df_in)>=2*min_count_auto:
                    median=df_in["x"].median()
                    pairs=[]; idx_all=bins.index(b)
                    for j,bb in enumerate(bins):
                        if j<idx_all: pairs.append((bb.left,bb.right))
                        elif j==idx_all: pairs+=[(bb.left,median),(median,bb.right)]
                        else: pairs.append((bb.left,bb.right))
                    bins=self._numeric_bins_from_edges(df,self._edges_from_pairs(pairs),alpha); changed=True
            if not changed: break
        return bins

    def _bin_numeric_adaptive(self, x, y, min_bins, min_count_auto, min_share, max_share, alpha, max_abs_woe, monotonic):
        df=pd.DataFrame({"x":x,"y":y})
        try: df["bin"]=pd.qcut(df["x"].rank(method="first"), q=max(min_bins,5), duplicates="drop")
        except Exception: df["bin"]=pd.cut(df["x"], bins=max(min_bins,5), include_lowest=True, duplicates="drop")
        bins=self._numeric_bins_from_bin_index(df,alpha)
        bins=self._numeric_adaptive_optimize(df,bins,min_count_auto,min_share,max_share,alpha,max_abs_woe,monotonic)
        return bins

    def _group_categorical_adaptive(self, x, y, rare_threshold, missing_label, other_label,
                                    min_count_auto, min_share, max_share, alpha, max_abs_woe) -> List[CategoricalGroup]:
        df=pd.DataFrame({"x":x,"y":y}); n=len(df); te=int((y==1).sum()); tne=int((y==0).sum()); groups=[]
        miss_mask=df["x"].isna(); miss_cnt=int(miss_mask.sum())
        if miss_cnt>0:
            e=int(df.loc[miss_mask,"y"].sum()); ne=miss_cnt-e
            w,er,_=woe_from_counts(e,ne,te,tne,alpha)
            groups.append(CategoricalGroup([missing_label],miss_cnt,e,ne,er,w,"MISSING"))
        df_nm=df.loc[~miss_mask].copy()
        level_counts=df_nm["x"].value_counts(dropna=True)
        rare=set(level_counts[level_counts/max(len(df_nm),1)<rare_threshold].index.tolist())
        regular=set(level_counts.index)-rare
        def grp(levels,label):
            m=df_nm["x"].isin(levels); cnt=int(m.sum()); e=int(df_nm.loc[m,"y"].sum()); ne=cnt-e
            w,er,_=woe_from_counts(e,ne,te,tne,alpha)
            return CategoricalGroup(list(levels),cnt,e,ne,er,w,label)
        regs=[grp([lv],f"G_{str(lv)[:16]}") for lv in regular]
        if rare: groups.append(grp(list(rare),other_label))
        regs=sorted(regs,key=lambda g:g.event_rate)
        max_iter=20
        for _ in range(max_iter):
            changed=False
            for i in range(len(regs)-1):
                g=regs[i]; share=g.count/max(n,1)
                if share<min_share or g.count<min_count_auto or abs(g.woe)>max_abs_woe:
                    nxt=regs[i+1]; regs=regs[:i]+[grp(g.members+nxt.members,f"G_{len(g.members)+len(nxt.members)}")]+regs[i+2:]; changed=True; break
            if changed: continue
            for i in range(len(regs)):
                g=regs[i]; share=g.count/max(n,1)
                if share>max_share and len(g.members)>=2:
                    stats=[]
                    for lv in g.members:
                        m=(df_nm["x"]==lv); cnt=int(m.sum()); e=int(df_nm.loc[m,"y"].sum()); ne=cnt-e
                        w,er,_=woe_from_counts(e,ne,te,tne,alpha)
                        stats.append((lv,er))
                    stats=sorted(stats,key=lambda t:t[1])
                    left=[t[0] for t in stats[:len(stats)//2]]; right=[t[0] for t in stats[len(stats)//2:]]
                    regs=regs[:i]+[grp(left,f"G_{len(left)}"),grp(right,f"G_{len(right)}")]+regs[i+1:]; changed=True; break
            if not changed: break
        groups.extend(regs)
        return groups

    # ----------------------- PSI v2 FAST -----------------------
    def _fmt_psi_line(self, var: str, scope: str, label: str, psi_val: float, status: str) -> str:
        return f"      - {var:<30s} | {scope:<17s} | {label:<12s} | PSI={psi_val:6.3f} | {status}"

    def _eta(self, start_ts: float, done: int, total: int) -> str:
        if done==0: return "ETA: hesaplaniyor..."
        elapsed=time.time()-start_ts; per_item=elapsed/done; rem=max(total-done,0)*per_item
        return f"ETA: ~{int(rem):d}s (kalan {total-done}/{total})"

    def _psi_screening(self, df, train_idx, test_idx, oot_idx) -> List[str]:
        cfg = self.cfg
        variables = list(self.woe_map.keys())
        mapping = {"variables": {}}
        for v in variables:
            vw = self.woe_map.get(v)
            if vw is None:
                continue
            if vw.var_type == "numeric":
                mapping["variables"][v] = {
                    "type": "numeric",
                    "bins": [{"left": b.left, "right": b.right, "woe": b.woe} for b in (vw.numeric_bins or [])],
                }
            else:
                mapping["variables"][v] = {
                    "type": "categorical",
                    "groups": [{"label": g.label, "members": list(map(str, g.members)), "woe": g.woe} for g in (vw.categorical_groups or [])],
                }
        from .stages import apply_woe as _apply_woe, feature_psi as _feature_psi
        tr_df = df.iloc[train_idx]
        te_df = df.iloc[test_idx] if (test_idx is not None and len(test_idx) > 0) else None
        oot_df = df.iloc[oot_idx]
        tr_woe = _apply_woe(tr_df, mapping)
        te_woe = _apply_woe(te_df, mapping) if te_df is not None else None
        oot_woe = _apply_woe(oot_df, mapping)
        psi_rows = []
        dropped_vars = set(); warn_vars = set(); keep = []
        psi_te = _feature_psi(tr_woe[variables], te_woe[variables], sample=None, bins=10) if te_woe is not None else {}
        psi_oot = _feature_psi(tr_woe[variables], oot_woe[variables], sample=None, bins=10)
        for v in variables:
            if v in psi_te:
                val = float(psi_te[v]); status = "KEEP"
                if val > cfg.psi_threshold_feature: status = "DROP"; dropped_vars.add(v)
                elif val > cfg.psi_warn_low: status = "WARN"; warn_vars.add(v)
                psi_rows.append({"variable": v, "compare_scope": "train_vs_test", "compare_label": "TEST_ALL", "psi_value": val, "status": status, "notes": ""})
            val = float(psi_oot.get(v, 0.0)); status = "KEEP"
            if val > cfg.psi_threshold_feature: status = "DROP"; dropped_vars.add(v)
            elif val > cfg.psi_warn_low: status = "WARN"; warn_vars.add(v)
            psi_rows.append({"variable": v, "compare_scope": "train_vs_oot", "compare_label": "OOT_ALL", "psi_value": val, "status": status, "notes": ""})
        for v in variables:
            if v not in dropped_vars:
                keep.append(v)
        psi_df = pd.DataFrame(psi_rows).sort_values(["variable", "compare_scope", "compare_label"]).reset_index(drop=True)
        self.psi_summary_ = psi_df.copy()
        dropped = psi_df.query("status==\"DROP\"").groupby("variable")["psi_value"].max().reset_index()
        if not dropped.empty and "psi_value" in dropped.columns:
            dropped = dropped.rename(columns={"psi_value": "max_psi"})
        self.psi_dropped_ = dropped.copy()
        keep_final = sorted(list(set(keep) - set(dropped["variable"].tolist())))
        self._log(f"   * PSI özet: KEEP={len(keep_final)} | DROP={len(dropped)} | WARN={len(warn_vars)}")
        del psi_df, dropped
        gc.collect()
        return keep_final
    # ----------------------- IV filter -----------------------
    def _iv_filter(self, vars_in: List[str]) -> List[str]:
        if not vars_in:
            return []
        # Build IV DataFrame from current WOE map
        import pandas as pd
        iv_rows = []
        for v in vars_in:
            iv_val = self.woe_map.get(v).iv if v in self.woe_map else 0.0
            iv_rows.append({"variable": v, "iv": float(iv_val)})
        iv_df = pd.DataFrame(iv_rows)
        # Use stages.selection to pick by IV threshold
        try:
            from .stages import iv_rank_select
            kept = iv_rank_select(iv_df, min_iv=self.cfg.iv_min, max_features=None)
        except Exception:
            kept = [r["variable"] for r in iv_rows if r["iv"] >= self.cfg.iv_min]
        # Build decision log and high-iv flags
        self.iv_filter_log_ = []
        self.high_iv_flags_ = []
        for v in vars_in:
            iv = self.woe_map.get(v).iv if v in self.woe_map else 0.0
            decision = "keep" if v in kept else "drop_low_iv"
            if iv > self.cfg.iv_high_flag:
                self.high_iv_flags_.append(v)
                if decision == "keep":
                    decision = "flag_high_iv"
            self.iv_filter_log_.append({"variable": v, "iv": float(iv), "decision": decision})
        if self.high_iv_flags_:
            self._log(f"   - High IV flags: {','.join(self.high_iv_flags_)}")
        return kept

    # ----------------------- Transform -----------------------
    def _transform(self, df: pd.DataFrame, keep_vars: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        # Build a lightweight mapping from in-memory woe_map, then delegate to stages.apply_woe
        mapping = {"variables": {}}
        for v in keep_vars:
            vw = self.woe_map.get(v)
            if vw is None:
                continue
            if vw.var_type == "numeric":
                mapping["variables"][v] = {
                    "type": "numeric",
                    "bins": [{"left": b.left, "right": b.right, "woe": b.woe} for b in (vw.numeric_bins or [])],
                }
            else:
                mapping["variables"][v] = {
                    "type": "categorical",
                    "groups": [{"label": g.label, "members": list(map(str, g.members)), "woe": g.woe} for g in (vw.categorical_groups or [])],
                }
        try:
            from .stages import apply_woe as _apply_woe
            X_all = _apply_woe(df, mapping)
            X = X_all[[c for c in keep_vars if c in X_all.columns]].copy()
        except Exception:
            # Fallback: empty frame if mapping application fails
            X = pd.DataFrame(index=df.index, columns=keep_vars)
        y = df[self.cfg.target_col].values
        return X, y
    
    # ----------------------- Raw Transform (for dual pipeline) -----------------------
    def _transform_raw(self, df: pd.DataFrame, keep_vars: List[str], fit_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform data using raw variables with imputation and outlier handling"""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Select raw features
        X = df[keep_vars].copy()
        y = df[self.cfg.target_col].values
        
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if fit_params is None:
            # Fit mode: learn imputation and outlier parameters
            fit_params = {}
            
            # Impute numeric columns
            if numeric_cols:
                imputer = SimpleImputer(strategy=self.cfg.raw_imputation_strategy)
                X_num_imputed = pd.DataFrame(
                    imputer.fit_transform(X[numeric_cols]),
                    columns=numeric_cols,
                    index=X.index
                )
                fit_params['numeric_imputer'] = imputer
                
                # Handle outliers
                if self.cfg.raw_outlier_method == "iqr":
                    Q1 = X_num_imputed.quantile(0.25)
                    Q3 = X_num_imputed.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.cfg.raw_outlier_threshold * IQR
                    upper_bound = Q3 + self.cfg.raw_outlier_threshold * IQR
                    fit_params['outlier_bounds'] = (lower_bound, upper_bound)
                    X_num_imputed = X_num_imputed.clip(lower=lower_bound, upper=upper_bound, axis=1)
                elif self.cfg.raw_outlier_method == "zscore":
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_num_imputed)
                    fit_params['scaler'] = scaler
                    X_scaled = np.clip(X_scaled, -self.cfg.raw_outlier_threshold, self.cfg.raw_outlier_threshold)
                    X_num_imputed = pd.DataFrame(
                        scaler.inverse_transform(X_scaled),
                        columns=numeric_cols,
                        index=X.index
                    )
                elif self.cfg.raw_outlier_method == "percentile":
                    lower_p = 1
                    upper_p = 99
                    lower_bound = X_num_imputed.quantile(lower_p/100)
                    upper_bound = X_num_imputed.quantile(upper_p/100)
                    fit_params['outlier_bounds'] = (lower_bound, upper_bound)
                    X_num_imputed = X_num_imputed.clip(lower=lower_bound, upper=upper_bound, axis=1)
                
                X[numeric_cols] = X_num_imputed
            
            # Impute categorical columns (mode)
            if categorical_cols:
                for col in categorical_cols:
                    mode_val = X[col].mode()[0] if not X[col].mode().empty else "MISSING"
                    fit_params[f'mode_{col}'] = mode_val
                    X[col] = X[col].fillna(mode_val)
            
            self.raw_fit_params_ = fit_params
        else:
            # Transform mode: use learned parameters
            if numeric_cols and 'numeric_imputer' in fit_params:
                X_num_imputed = pd.DataFrame(
                    fit_params['numeric_imputer'].transform(X[numeric_cols]),
                    columns=numeric_cols,
                    index=X.index
                )
                
                # Apply outlier clipping
                if 'outlier_bounds' in fit_params:
                    lower_bound, upper_bound = fit_params['outlier_bounds']
                    X_num_imputed = X_num_imputed.clip(lower=lower_bound, upper=upper_bound, axis=1)
                elif 'scaler' in fit_params:
                    X_scaled = fit_params['scaler'].transform(X_num_imputed)
                    X_scaled = np.clip(X_scaled, -self.cfg.raw_outlier_threshold, self.cfg.raw_outlier_threshold)
                    X_num_imputed = pd.DataFrame(
                        fit_params['scaler'].inverse_transform(X_scaled),
                        columns=numeric_cols,
                        index=X.index
                    )
                
                X[numeric_cols] = X_num_imputed
            
            # Apply categorical imputation
            if categorical_cols:
                for col in categorical_cols:
                    if f'mode_{col}' in fit_params:
                        X[col] = X[col].fillna(fit_params[f'mode_{col}'])
        
        return X, y

    # ----------------------- Corr & cluster -----------------------
    def _corr_and_cluster(self, X: pd.DataFrame, keep_vars: List[str]) -> List[str]:
        if not keep_vars:
            return []
        nz = [v for v in keep_vars if X[v].std(skipna=True) > 0]
        if not nz:
            return keep_vars[:1]
        corr = X[nz].corr(method="spearman").fillna(0.0)
        reps = []
        picked = set()
        top_k = self.cfg.cluster_top_k or 1
        top_k = 1 if top_k < 1 else int(top_k)
        for v in nz:
            if v in picked:
                continue
            group = [g for g in nz if abs(corr.loc[v, g]) > 0.90]
            for g in group:
                picked.add(g)
            group_sorted = sorted(group, key=lambda g: self.woe_map.get(g, VariableWOE(g, "", iv=0.0)).iv, reverse=True)
            reps.extend(group_sorted[:top_k])
        del corr
        gc.collect()
        return reps

    # ----------------------- FS -----------------------
    def _feature_selection(self, X, y, candidate_vars, all_vars) -> List[str]:
        try:
            rf = RandomForestClassifier(
                n_estimators=max(200, 100 * self.cfg.n_jobs),
                n_jobs=self.cfg.n_jobs,
                random_state=self.cfg.random_state,
                class_weight="balanced_subsample",
            )
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=self.cfg.random_state)
            boruta.fit(X[all_vars].values, y)
            boruta_vars = [all_vars[i] for i, keep in enumerate(boruta.support_) if keep]
            self._log(f"   - Boruta: {len(boruta_vars)}/{len(all_vars)} kaldi")
        except Exception as e:
            boruta_vars = candidate_vars or all_vars[:min(10, len(all_vars))]
            self._log("   - Boruta kullanilamadi, aday/kesit ile devam ediliyor")
        if not boruta_vars:
            # Boruta başarısız olursa daha fazla değişken bırak
            boruta_vars = all_vars[:min(20, len(all_vars))]  # 10'dan 20'ye çıkarıldı
        selected = self._forward_1se_selection(X[boruta_vars], y, "KS", self.cfg.cv_folds)
        
        # Minimum değişken sayısı kontrolü (performans kaybını önlemek için)
        min_features = 3  # Reduced to 3 to ensure GAM can run
        if len(selected) < min_features and len(boruta_vars) >= min_features:
            # Çok az değişken seçildiyse, en iyi IV'lı değişkenleri ekle
            iv_scores = {}
            for var in boruta_vars:
                if var not in selected and var in self.woe_map:
                    iv_scores[var] = self.woe_map[var].iv
            
            # IV'ye göre sırala ve eksik değişkenleri ekle
            sorted_vars = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
            for var, iv in sorted_vars:
                if len(selected) < min_features:
                    selected.append(var)
                else:
                    break
            self._log(f"   - Minimum {min_features} değişken için eklendi")
        
        self._log(f"   - Forward+1SE secti: {len(selected)}")
        gc.collect()
        return selected

    def _forward_1se_selection(self, X: pd.DataFrame, y: np.ndarray, maximize_metric="KS", cv_folds=5) -> List[str]:
        remaining=list(X.columns); selected=[]; best_seq=[]
        def cv_score(cols: List[str])->float:
            if not cols: return -np.inf
            skf=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=self.cfg.random_state)
            sc=[]
            for tr,va in skf.split(X[cols],y):
                m=LogisticRegression(penalty="l2",solver="lbfgs",max_iter=300,class_weight="balanced")
                m.fit(X.iloc[tr][cols],y[tr]); p=self._proba_1d(m, X.iloc[va][cols])
                ks,_=ks_statistic(y[va],p); auc=roc_auc_score(y[va],p); sc.append(ks if maximize_metric.upper()=="KS" else auc)
            return float(np.mean(sc))
        while remaining:
            best_v,best_s=None,-np.inf
            for v in remaining:
                s=cv_score(selected+[v])
                if s>best_s: best_s,best_v=s,v
            if best_v is None: break
            selected.append(best_v); remaining.remove(best_v); best_seq.append((selected.copy(),best_s))
        if not best_seq: return selected
        scores=[s for _,s in best_seq]; best=np.max(scores); std=np.std(scores); target=best-std
        cand_sets=[s for s,sc in best_seq if sc>=target]
        k=min(len(s) for s in cand_sets) if cand_sets else len(selected)
        pick=[s for s in cand_sets if len(s)==k]
        return pick[0] if pick else selected

    # ----------------------- final corr filter -----------------------
    def _final_corr_filter(self, X: pd.DataFrame, y: np.ndarray, thr: Optional[float] = None) -> List[str]:
        if X.shape[1] == 0:
            return []
        rho_thr = self.cfg.rho_threshold if thr is None else thr
        try:
            from .stages import drop_correlated
            kept, dropped_df = drop_correlated(X, threshold=rho_thr)
            self.corr_dropped_ = []
            if dropped_df is not None and not dropped_df.empty:
                for _, r in dropped_df.iterrows():
                    self.corr_dropped_.append({"variable": str(r.get("var2")), "kept_with": str(r.get("var1")), "rho": float(r.get("rho", 0.0))})
            final_keep = kept
        except Exception:
            # fallback to keep-all if correlation computation fails
            final_keep = list(X.columns)
            self.corr_dropped_ = []
        # optional VIF
        if self.cfg.vif_threshold and len(final_keep) > 1:
            vifs = {}
            Xk = X[final_keep]
            for col in final_keep:
                X_other = Xk.drop(columns=[col])
                y_ = Xk[col].values
                if X_other.shape[1] == 0:
                    vifs[col] = 1.0
                    continue
                beta, *_ = np.linalg.lstsq(X_other.values, y_, rcond=None)
                y_hat = X_other.values @ beta
                ss_res = np.sum((y_ - y_hat) ** 2)
                ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif = 1.0 / (1.0 - r2) if (1 - r2) > 1e-6 else np.inf
                vifs[col] = vif
            final_keep = [c for c in final_keep if vifs.get(c, 0.0) <= self.cfg.vif_threshold]
            dropped_vif = [c for c in vifs if vifs[c] > self.cfg.vif_threshold]
            for c in dropped_vif:
                self._log(f"   - {c} VIF>{self.cfg.vif_threshold:.2f} nedeniyle elendi")
                self.corr_dropped_.append({"variable": c, "kept_with": None, "reason": f"vif>{self.cfg.vif_threshold}"})
        return final_keep

    # ----------------------- noise sentinel -----------------------
    def _noise_sentinel_check(self, X, y, pre_final) -> List[str]:
        if not pre_final: return pre_final
        np.random.seed(self.cfg.random_state)
        Xn=X.copy(); Xn["__noise_g"]=np.random.normal(size=len(X)).astype("float32")
        Xn["__noise_p"]=np.random.permutation(X[pre_final[0]].values)
        sel=self._forward_1se_selection(Xn[pre_final+["__noise_g","__noise_p"]],y,"KS",self.cfg.cv_folds)
        if "__noise_g" in sel or "__noise_p" in sel:
            sel=self._forward_1se_selection(X[pre_final],y,"KS",self.cfg.cv_folds)
        del Xn; gc.collect()
        return [v for v in sel if not v.startswith("__noise_")]

    # ----------------------- Calibration -----------------------
    def _calibrate_model(self):
        # Support both DataFrame and file path
        cal_df = None
        
        # First check if DataFrame is provided
        if self.cfg.calibration_df is not None:
            cal_df = self.cfg.calibration_df.copy()
            self._log(f"   - Using calibration DataFrame: {cal_df.shape}")
        # Otherwise check for file path
        elif self.cfg.calibration_data_path:
            path = self.cfg.calibration_data_path
            try:
                if str(path).lower().endswith(".parquet") or str(path).lower().endswith(".parq"):
                    cal_df = pd.read_parquet(path)
                else:
                    cal_df = pd.read_csv(path)
                self._log(f"   - Loaded calibration data from {path}: {cal_df.shape}")
            except Exception as e:
                self._log(f"   - calibration data load failed: {e}")
                self.calibrator_ = None
                return
        else:
            # No calibration data provided
            self.calibrator_ = None
            return
        if self.cfg.id_col in cal_df.columns:
            cal_ids = set(cal_df[self.cfg.id_col].astype(str))
            train_ids = set(self.df_.iloc[self.train_idx_][self.cfg.id_col].astype(str)) if self.train_idx_ is not None else set()
            oot_ids = set(self.df_.iloc[self.oot_idx_][self.cfg.id_col].astype(str)) if self.oot_idx_ is not None else set()
            test_ids = set(self.df_.iloc[self.test_idx_][self.cfg.id_col].astype(str)) if self.test_idx_ is not None and len(self.test_idx_)>0 else set()
            if cal_ids & (train_ids | oot_ids | test_ids):
                self._log("   - calibration data overlaps with train/test/oot; skipping calibration")
                self.calibrator_ = None
                return
        # Apply WOE transform using the same mapping as training data
        from .stages import apply_woe
        cal_df_woe = apply_woe(cal_df, self.woe_map)
        
        # Check which final variables are available after WOE transformation
        available_vars = [var for var in self.final_vars_ if var in cal_df_woe.columns]
        if not available_vars:
            self._log(f"   - No final variables available in calibration data after WOE transform. Expected: {self.final_vars_}, Available: {list(cal_df_woe.columns)}")
            self.calibrator_ = None
            return
        
        if len(available_vars) < len(self.final_vars_):
            self._log(f"   - Warning: Only {len(available_vars)}/{len(self.final_vars_)} final variables available in calibration data")
        
        X_cal = cal_df_woe[available_vars]
        y_cal = cal_df[self.cfg.target_col].values
        
        mdl = self.models_.get(self.best_model_name_)
        if mdl is None:
            self.calibrator_ = None
            return
        raw = self._proba_1d(mdl, X_cal)
        try:
            self.calibrator_ = fit_calibrator(raw, y_cal, method=self.cfg.calibration_method)
            cal_scores = apply_calibrator(self.calibrator_, raw)
            brier = brier_score_loss(y_cal, cal_scores)
            self.calibration_report_ = {"brier": float(brier)}
            self._log(f"   - Calibration successful: Brier score = {brier:.4f}")
        except Exception as e:
            self._log(f"   - calibration failed: {e}")
            self.calibrator_ = None

    # ----------------------- utils: robust proba -----------------------
    def _proba_1d(self, mdl, Xdf: pd.DataFrame) -> np.ndarray:
        try:
            proba = mdl.predict_proba(Xdf)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                return proba.ravel()
            if proba.shape[1] == 2:
                return proba[:, 1]
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba.max(axis=1)
        except Exception:
            if hasattr(mdl, "predict_mu"):
                return np.asarray(mdl.predict_mu(Xdf)).ravel()
            try:
                scores = np.asarray(mdl.decision_function(Xdf)).ravel()
                return 1.0 / (1.0 + np.exp(-scores))
            except Exception:
                return np.asarray(mdl.predict(Xdf)).ravel()

    # ----------------------- modelleme -----------------------
    def _hyperparameter_tune(self, base_estimator, param_dist, X, y) -> Any:
        method = (self.cfg.hpo_method or "random").lower()
        timeout = self.cfg.hpo_timeout_sec
        n_trials = self.cfg.hpo_trials
        start = time.time()
        best_params = {}
        best_score = -np.inf
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        
        # Special handling for GAM with low feature count
        from pygam import LogisticGAM
        if isinstance(base_estimator, LogisticGAM) and X.shape[1] < 3:
            # Return base model without tuning for GAM with <3 features
            self._log(f"   - GAM: Skipping HPO (only {X.shape[1]} features)")
            return base_estimator

        if method == "optuna":
            try:
                import optuna

                def objective(trial):
                    params = {}
                    for p, space in param_dist.items():
                        if isinstance(space, list):
                            params[p] = trial.suggest_categorical(p, space)
                        else:
                            params[p] = trial.suggest_categorical(p, list(space))
                    mdl = clone(base_estimator)
                    mdl.set_params(**params)
                    sc = []
                    for tr, va in skf.split(X, y):
                        mdl.fit(X.iloc[tr], y[tr])
                        p = self._proba_1d(mdl, X.iloc[va])
                        ks, _ = ks_statistic(y[va], p)
                        sc.append(ks)
                    return float(np.mean(sc)) if sc else -np.inf

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, timeout=timeout)
                best_params = study.best_trial.params if study.best_trial else {}
            except Exception:
                method = "random"

        if method == "random":
            sampler = ParameterSampler(param_dist, n_iter=n_trials, random_state=self.cfg.random_state)
            for params in sampler:
                if time.time() - start > timeout:
                    break
                mdl = clone(base_estimator)
                mdl.set_params(**params)
                sc = []
                for tr, va in skf.split(X, y):
                    mdl.fit(X.iloc[tr], y[tr])
                    p = self._proba_1d(mdl, X.iloc[va])
                    ks, _ = ks_statistic(y[va], p)
                    sc.append(ks)
                    if time.time() - start > timeout:
                        break
                if sc:
                    score = float(np.mean(sc))
                    if score > best_score:
                        best_score = score
                        best_params = params
                if time.time() - start > timeout:
                    break

        mdl = clone(base_estimator)
        mdl.set_params(**best_params)
        mdl.fit(X, y)
        return mdl

    def _train_and_evaluate_models(self, Xtr, ytr, Xte, yte, Xoot, yoot):
        if not self.final_vars_:
            if len(Xtr.columns) > 0:
                self.final_vars_ = [Xtr.columns[0]]
            else:
                self._log("HATA: Model egitimi icin hic degisken bulunamadi!")
                self.final_vars_ = []
                return
        models = {
            "Logit_L2": (
                LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced"),
                {"C": np.logspace(-3, 3, 7)},
            ),
            "RandomForest": (
                RandomForestClassifier(
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                    class_weight="balanced_subsample",
                ),
                {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 5, 20],
                },
            ),
            "ExtraTrees": (
                ExtraTreesClassifier(
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                    class_weight="balanced",
                ),
                {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 5, 20],
                },
            ),
            "XGBoost": (
                XGBClassifier(
                    eval_metric="logloss",
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                    tree_method="hist",
                    verbosity=0,
                ),
                {
                    "n_estimators": [200, 500],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.7, 1.0],
                },
            ),
            "LightGBM": (
                LGBMClassifier(
                    class_weight="balanced",
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                    verbosity=-1,  # Suppress warnings
                    min_child_samples=10,  # Avoid overfitting warnings
                ),
                {
                    "n_estimators": [300, 500],
                    "num_leaves": [31, 63],
                    "max_depth": [-1, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.7, 1.0],
                },
            ),
            "GAM": (
                LogisticGAM(max_iter=200),  # Prevent convergence warnings
                {"lam": np.logspace(-3, 3, 7)},
            ),
        }
        if self.cfg.try_mlp:
            models["MLP"] = (
                MLPClassifier(random_state=self.cfg.random_state, max_iter=200),
                {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]},
            )
        rows = []
        ks_tables = {"traincv": None, "test": None, "oot": None}
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        for name, (base_mdl, params) in models.items():
            # Skip GAM if we have less than 3 features (GAM requires at least 3)
            if name == "GAM" and len(self.final_vars_) < 3:
                self._log(f"   - Skipping GAM (requires >=3 features, have {len(self.final_vars_)})")
                continue
            print(f"[{now_str()}]   - {name} tuning{sys_metrics()}")
            mdl = self._hyperparameter_tune(base_mdl, params, Xtr[self.final_vars_], ytr)
            print(f"[{now_str()}]   - {name} CV basliyor{sys_metrics()}")
            sc = []
            for tr, va in skf.split(Xtr[self.final_vars_], ytr):
                m = clone(mdl)
                m.fit(Xtr.iloc[tr][self.final_vars_], ytr[tr])
                p = self._proba_1d(m, Xtr.iloc[va][self.final_vars_])
                ks, _ = ks_statistic(ytr[va], p)
                auc = roc_auc_score(ytr[va], p)
                sc.append((ks, auc))
            ks_cv = float(np.mean([s[0] for s in sc]))
            auc_cv = float(np.mean([s[1] for s in sc]))
            gini_cv = gini_from_auc(auc_cv)
            mdl.fit(Xtr[self.final_vars_], ytr)
            p_tr = self._proba_1d(mdl, Xtr[self.final_vars_])
            ks_tables["traincv"] = ks_table(ytr, p_tr, n_bands=self.cfg.ks_bands)
            ks_te = auc_te = gini_te = None
            if Xte is not None and yte is not None and Xte.shape[0] > 0:
                p_te = self._proba_1d(mdl, Xte[self.final_vars_])
                ks_te, _ = ks_statistic(yte, p_te)
                auc_te = roc_auc_score(yte, p_te)
                gini_te = gini_from_auc(auc_te)
                ks_tables["test"] = ks_table(yte, p_te, n_bands=self.cfg.ks_bands)
            p_oot = self._proba_1d(mdl, Xoot[self.final_vars_])
            ks_oot, thr_oot = ks_statistic(yoot, p_oot)
            auc_oot = roc_auc_score(yoot, p_oot)
            gini_oot = gini_from_auc(auc_oot)
            ks_tables["oot"] = ks_table(yoot, p_oot, n_bands=self.cfg.ks_bands)
            rows.append(
                {
                    "model_name": name,
                    "KS_TrainCV": ks_cv,
                    "AUC_TrainCV": auc_cv,
                    "Gini_TrainCV": gini_cv,
                    "KS_Test": ks_te,
                    "AUC_Test": auc_te,
                    "Gini_Test": gini_te,
                    "KS_OOT": ks_oot,
                    "AUC_OOT": auc_oot,
                    "Gini_OOT": gini_oot,
                    "KS_OOT_threshold": thr_oot,
                }
            )
            self.models_[name] = mdl
        if self.cfg.ensemble and rows:
            top_names = [r["model_name"] for r in sorted(rows, key=lambda r: r["KS_OOT"], reverse=True)][: self.cfg.ensemble_top_k]
            top_models = [self.models_[n] for n in top_names if n in self.models_]
            if len(top_models) >= 2:
                class _EnsembleWrapper:
                    def __init__(self, models):
                        self.models = models
                    def predict_proba(self, X):
                        p = soft_voting_ensemble(self.models, X=X)
                        return np.vstack([1 - p, p]).T

                print(f"[{now_str()}]   - Ensemble tuning{sys_metrics()}")
                sc = []
                for tr, va in skf.split(Xtr[self.final_vars_], ytr):
                    mdl_list = [clone(m).fit(Xtr.iloc[tr][self.final_vars_], ytr[tr]) for m in top_models]
                    p = soft_voting_ensemble(mdl_list, X=Xtr.iloc[va][self.final_vars_])
                    ks, _ = ks_statistic(ytr[va], p)
                    auc = roc_auc_score(ytr[va], p)
                    sc.append((ks, auc))
                ks_cv = float(np.mean([s[0] for s in sc])) if sc else None
                auc_cv = float(np.mean([s[1] for s in sc])) if sc else None
                gini_cv = gini_from_auc(auc_cv) if auc_cv is not None else None
                p_tr = soft_voting_ensemble(top_models, X=Xtr[self.final_vars_])
                ks_tables["traincv"] = ks_table(ytr, p_tr, n_bands=self.cfg.ks_bands)
                ks_te = auc_te = gini_te = None
                if Xte is not None and yte is not None and Xte.shape[0] > 0:
                    p_te = soft_voting_ensemble(top_models, X=Xte[self.final_vars_])
                    ks_te, _ = ks_statistic(yte, p_te)
                    auc_te = roc_auc_score(yte, p_te)
                    gini_te = gini_from_auc(auc_te)
                    ks_tables["test"] = ks_table(yte, p_te, n_bands=self.cfg.ks_bands)
                p_oot = soft_voting_ensemble(top_models, X=Xoot[self.final_vars_])
                ks_oot, thr_oot = ks_statistic(yoot, p_oot)
                auc_oot = roc_auc_score(yoot, p_oot)
                gini_oot = gini_from_auc(auc_oot)
                ks_tables["oot"] = ks_table(yoot, p_oot, n_bands=self.cfg.ks_bands)
                rows.append(
                    {
                        "model_name": "Ensemble",
                        "KS_TrainCV": ks_cv,
                        "AUC_TrainCV": auc_cv,
                        "Gini_TrainCV": gini_cv,
                        "KS_Test": ks_te,
                        "AUC_Test": auc_te,
                        "Gini_Test": gini_te,
                        "KS_OOT": ks_oot,
                        "AUC_OOT": auc_oot,
                        "Gini_OOT": gini_oot,
                        "KS_OOT_threshold": thr_oot,
                    }
                )
                self.models_["Ensemble"] = _EnsembleWrapper(top_models)

        self.models_summary_ = pd.DataFrame(rows).sort_values("KS_OOT", ascending=False).reset_index(drop=True)
        self.ks_info_traincv_, self.ks_info_test_, self.ks_info_oot_ = (
            ks_tables["traincv"],
            ks_tables["test"],
            ks_tables["oot"],
        )
    
    def _combine_dual_pipeline_results(self):
        """Combine results from both WOE and RAW pipelines"""
        if not self.cfg.enable_dual_pipeline:
            return
        
        # Combine model summaries
        if self.woe_models_summary_ is not None and self.raw_models_summary_ is not None:
            woe_summary = self.woe_models_summary_.copy()
            woe_summary['pipeline'] = 'WOE'
            woe_summary['model_name'] = woe_summary['model_name'].apply(lambda x: f"WOE_{x}")
            
            raw_summary = self.raw_models_summary_.copy()
            raw_summary['pipeline'] = 'RAW'
            raw_summary['model_name'] = raw_summary['model_name'].apply(lambda x: f"RAW_{x}")
            
            self.models_summary_ = pd.concat([woe_summary, raw_summary], ignore_index=True)
            
            # Combine models
            combined_models = {}
            for name, model in self.woe_models_.items():
                combined_models[f"WOE_{name}"] = model
            for name, model in self.raw_models_.items():
                combined_models[f"RAW_{name}"] = model
            self.models_ = combined_models
            
            # Log summary
            self._log("\n" + "="*80)
            self._log("DUAL PIPELINE SUMMARY")
            self._log("="*80)
            self._log(f"WOE Pipeline: {len(self.woe_final_vars_)} variables, {len(self.woe_models_)} models")
            self._log(f"RAW Pipeline: {len(self.raw_final_vars_)} variables, {len(self.raw_models_)} models")
            
            # Best models from each pipeline
            if self.woe_models_summary_ is not None and not self.woe_models_summary_.empty:
                best_woe = self.woe_models_summary_.nlargest(1, 'Gini_OOT').iloc[0]
                self._log(f"Best WOE Model: {best_woe['model_name']} - Gini OOT: {best_woe['Gini_OOT']:.4f}")
            
            if self.raw_models_summary_ is not None and not self.raw_models_summary_.empty:
                best_raw = self.raw_models_summary_.nlargest(1, 'Gini_OOT').iloc[0]
                self._log(f"Best RAW Model: {best_raw['model_name']} - Gini OOT: {best_raw['Gini_OOT']:.4f}")

    def _select_best_model(self):
        df=self.models_summary_
        if df is None or df.empty: self.best_model_name_=None; return
        top=df.sort_values(["KS_OOT","AUC_OOT","Gini_OOT"], ascending=[False, False, False]).iloc[0]
        self.best_model_name_=str(top["model_name"])

    # ----------------------- Rapor tablolari + univariate top50 -----------------------
    def _build_report_tables(self, psi_keep: List[str]):
        iv_rows = [{"variable": v, "IV": self.woe_map[v].iv, "variable_group": self.woe_map[v].var_type} for v in psi_keep]
        if iv_rows:
            self.top20_iv_df_ = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).head(20).reset_index(drop=True)
        else:
            self.top20_iv_df_ = pd.DataFrame(columns=["variable", "IV", "variable_group"])
        self.high_iv_flags_df_ = pd.DataFrame({"variable": self.high_iv_flags_}) if self.high_iv_flags_ else None
        self.iv_decisions_df_ = pd.DataFrame(self.iv_filter_log_)
        self.corr_dropped_df_ = pd.DataFrame(self.corr_dropped_)

        bm = self.best_model_name_
        mdl = self.models_.get(bm) if bm else None
        if bm is None or mdl is None:
            self.best_model_vars_df_ = None
        else:
            # Load data dictionary for variable descriptions
            var_descriptions = self._load_data_dictionary()
            
            if hasattr(mdl, "coef_"):
                coefs = mdl.coef_[0]
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_, "coef_or_importance": coefs}) \
                    .assign(
                        sign=lambda d: np.sign(d["coef_or_importance"]).astype(int),
                        variable_group=lambda d: d["variable"].map(lambda v: self.woe_map[v].var_type if v in self.woe_map else None),
                        description=lambda d: d["variable"].map(lambda v: var_descriptions.get(v, ""))
                    ).sort_values("coef_or_importance", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
            elif hasattr(mdl, "feature_importances_"):
                imps = mdl.feature_importances_
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_, "coef_or_importance": imps}) \
                    .assign(
                        sign=lambda d: np.where(d["coef_or_importance"] >= 0, 1, -1),
                        variable_group=lambda d: d["variable"].map(lambda v: self.woe_map[v].var_type if v in self.woe_map else None),
                        description=lambda d: d["variable"].map(lambda v: var_descriptions.get(v, ""))
                    ).sort_values("coef_or_importance", ascending=False).reset_index(drop=True)
            else:
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_})
                self.best_model_vars_df_["description"] = self.best_model_vars_df_["variable"].map(lambda v: var_descriptions.get(v, ""))

        # Load data dictionary for WOE report
        var_descriptions = self._load_data_dictionary()
        
        w_rows = []
        for v in self.final_vars_:
            vw = self.woe_map[v]
            var_rows = []
            var_desc = var_descriptions.get(v, "")
            
            if vw.var_type == "numeric":
                for b in vw.numeric_bins:
                    if np.isnan(b.left) and np.isnan(b.right):
                        label = "MISSING"
                        bin_from = None
                        bin_to = None
                    else:
                        label = f"[{b.left},{b.right})"
                        bin_from = b.left
                        bin_to = b.right
                    
                    var_rows.append({
                        "variable": v, 
                        "variable_description": var_desc,
                        "variable_group": "numeric",
                        "group": label, 
                        "bin_from": bin_from,
                        "bin_to": bin_to,
                        "values": label,
                        "count": b.count, 
                        "event": b.event, 
                        "non_event": b.non_event,
                        "event_rate": b.event_rate, 
                        "woe": b.woe
                    })
            else:
                for g in vw.categorical_groups:
                    vals = g.label if g.label in ("MISSING", "OTHER") else "{"+",".join(map(str, g.members))+"}"
                    var_rows.append({
                        "variable": v, 
                        "variable_description": var_desc,
                        "variable_group": "categorical",
                        "group": g.label, 
                        "bin_from": None,
                        "bin_to": None,
                        "values": vals,
                        "count": g.count, 
                        "event": g.event, 
                        "non_event": g.non_event,
                        "event_rate": g.event_rate, 
                        "woe": g.woe
                    })
            
            # Sort by event_rate (default rate) for monotonic ordering
            var_rows.sort(key=lambda x: x["event_rate"])
            w_rows.extend(var_rows)
        
        self.best_model_woe_df_ = pd.DataFrame(w_rows) if w_rows else None

        self.final_vars_df_ = pd.DataFrame({
            "variable": self.final_vars_,
            "variable_group": [self.woe_map[v].var_type if v in self.woe_map else None for v in self.final_vars_]
        })

    def _build_top50_univariate(self, X_tr: Optional[pd.DataFrame], y_tr: Optional[np.ndarray]):
        if X_tr is None or y_tr is None:
            return
        rows = []
        for v in X_tr.columns:
            s = X_tr[v].astype("float32").values
            try:
                auc = roc_auc_score(y_tr, s)
            except Exception:
                auc = np.nan
            gini = 2 * auc - 1 if np.isfinite(auc) else np.nan
            rows.append({"variable": v, "AUC_uni": auc, "Gini_uni": gini})
        if rows:
            self.top50_uni_ = pd.DataFrame(rows).sort_values("Gini_uni", ascending=False).head(50).reset_index(drop=True)
        else:
            self.top50_uni_ = pd.DataFrame(columns=["variable", "AUC_uni", "Gini_uni"])

    def _persist_artifacts(self, X_oot: Optional[pd.DataFrame], y_oot: Optional[np.ndarray]):
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        mapping = {"variables": {}, "jeffreys_alpha": self.cfg.jeffreys_alpha, "run_id": self.cfg.run_id}
        for v, vw in self.woe_map.items():
            if vw.var_type == "numeric":
                mapping["variables"][v] = {
                    "type": "numeric",
                    "bins": [{"left": b.left, "right": b.right, "woe": b.woe} for b in vw.numeric_bins]
                }
            else:
                mapping["variables"][v] = {
                    "type": "categorical",
                    "groups": [{"label": g.label, "members": list(map(str, g.members)), "woe": g.woe} for g in vw.categorical_groups]
                }
        # Persist JSON mapping (always) and final_vars list for future scoring
        try:
            with open(os.path.join(self.cfg.output_folder, f"woe_mapping_{self.cfg.run_id}.json"), "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.cfg.output_folder, f"final_vars_{self.cfg.run_id}.json"), "w", encoding="utf-8") as f:
                json.dump({"final_vars": self.final_vars_}, f, ensure_ascii=False, indent=2)
            if self.shap_summary_:
                with open(os.path.join(self.cfg.output_folder, f"shap_summary_{self.cfg.run_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(self.shap_summary_, f, ensure_ascii=False, indent=2)
                if self.cfg.write_csv:
                    pd.DataFrame(
                        sorted(self.shap_summary_.items(), key=lambda t: t[1], reverse=True),
                        columns=["variable", "shap"]
                    ).head(20).to_csv(
                        os.path.join(self.cfg.output_folder, f"shap_top20_{self.cfg.run_id}.csv"), index=False
                    )
        except Exception:
            pass

        if self.best_model_name_:
            mdl = self.models_.get(self.best_model_name_)
            if mdl is not None:
                # Persist trained best model for future scoring
                try:
                    import joblib
                    joblib.dump(mdl, os.path.join(self.cfg.output_folder, f"best_model_{self.cfg.run_id}.joblib"))
                except Exception:
                    try:
                        import pickle
                        with open(os.path.join(self.cfg.output_folder, f"best_model_{self.cfg.run_id}.pkl"), "wb") as f:
                            pickle.dump(mdl, f)
                    except Exception:
                        pass
                # If OOT present, persist OOT scores too
                if X_oot is not None and y_oot is not None and X_oot.shape[0] > 0:
                    prob = self._proba_1d(mdl, X_oot[self.final_vars_])
                    df_scores = pd.DataFrame({"prob": prob, "target": y_oot})
                    self.oot_scores_df_ = df_scores
                    try:
                        if self.cfg.write_parquet:
                            df_scores.to_parquet(os.path.join(self.cfg.output_folder, f"oot_scores_{self.cfg.run_id}.parquet"), index=False)
                        elif self.cfg.write_csv:
                            df_scores.to_csv(os.path.join(self.cfg.output_folder, f"oot_scores_{self.cfg.run_id}.csv"), index=False)
                    except Exception:
                        pass
                _, thr = ks_statistic(y_oot, prob)
                pred = (prob >= thr).astype(int)
                try:
                    from sklearn.metrics import confusion_matrix
                    tn, fp, fn, tp = confusion_matrix(y_oot, pred).ravel()
                except Exception:
                    tn = int(((pred == 0) & (y_oot == 0)).sum())
                    fp = int(((pred == 1) & (y_oot == 0)).sum())
                    fn = int(((pred == 0) & (y_oot == 1)).sum())
                    tp = int(((pred == 1) & (y_oot == 1)).sum())
                self.confusion_oot_ = pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": thr}])

        def save(df, name):
            if df is None or (hasattr(df, 'empty') and df.empty):
                return
            try:
                if self.cfg.write_parquet:
                    df.to_parquet(os.path.join(self.cfg.output_folder, f"{name}_{self.cfg.run_id}.parquet"), index=False)
                elif self.cfg.write_csv:
                    df.to_csv(os.path.join(self.cfg.output_folder, f"{name}_{self.cfg.run_id}.csv"), index=False)
            except Exception:
                pass

        save(self.high_iv_flags_df_, "high_iv_flags")
        save(self.iv_decisions_df_, "iv_decisions")
        save(self.corr_dropped_df_, "corr_dropped")

    # ----------------------- Dictionary entegrasyonu -----------------------
    def _integrate_dictionary(self):
        if not self.cfg.dictionary_path:
            return
        try:
            dic = pd.read_excel(self.cfg.dictionary_path)
            dic.columns = [c.strip().lower() for c in dic.columns]
            if "alanadi" not in dic.columns or "aciklama" not in dic.columns:
                return
            dic["__key"] = dic["alanadi"].astype(str).str.strip().str.casefold()
            dic = dic[["__key", "aciklama"]].rename(columns={"aciklama": "definition"})

            def enrich(df):
                if df is None or df.empty:
                    return df
                tmp = df.copy()
                tmp["__key"] = tmp["variable"].astype(str).str.strip().str.casefold()
                tmp = tmp.merge(dic, on="__key", how="left").drop(columns=["__key"])
                return tmp

            self.final_vars_df_ = enrich(self.final_vars_df_)
            self.best_model_vars_df_ = enrich(self.best_model_vars_df_)
            if self.best_model_woe_df_ is not None and not self.best_model_woe_df_.empty:
                tmp = self.best_model_woe_df_.copy()
                tmp["__key"] = tmp["variable"].astype(str).str.strip().str.casefold()
                self.best_model_woe_df_ = tmp.merge(dic, on="__key", how="left").drop(columns=["__key"])
        except Exception:
            # sozluk yoksa veya okunamadiysa akis devam etsin
            pass

    # ----------------------- Meta & Export -----------------------
    def _finalize_meta(self):
        self.artifacts["pool"]["config"] = asdict(self.cfg)
        self.artifacts["pool"]["final_vars"] = self.final_vars_
        self.artifacts["pool"]["best_model_name"] = self.best_model_name_
        if self.shap_summary_ is not None:
            self.artifacts["pool"]["shap_summary"] = self.shap_summary_
        if self.high_iv_flags_:
            self.artifacts["pool"]["high_iv_flags"] = self.high_iv_flags_
        if self.corr_dropped_:
            self.artifacts["pool"]["corr_dropped"] = self.corr_dropped_
        if self.iv_filter_log_:
            self.artifacts["pool"]["iv_decisions"] = self.iv_filter_log_

    def export_reports(self):
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        xlsx = os.path.join(self.cfg.output_folder, self.cfg.output_excel_path)

        # Assemble sheets as DataFrames for single-file multi-sheet report
        sheets = {}
        if self.final_vars_df_ is not None:
            sheets["final_vars"] = self.final_vars_df_
        sheets["best_name"] = pd.DataFrame({"best_name": [self.best_model_name_]})
        if self.models_summary_ is not None:
            sheets["models_summary"] = self.models_summary_
            if self.best_model_name_ is not None:
                try:
                    sheets["best_model"] = self.models_summary_.query("model_name==@self.best_model_name_")
                except Exception:
                    sheets["best_model"] = self.models_summary_.head(1)
        if self.best_model_vars_df_ is not None:
            sheets["best_model_vars_df"] = self.best_model_vars_df_
        if self.best_model_woe_df_ is not None:
            sheets["best_model_woe_df"] = self.best_model_woe_df_
        if self.top20_iv_df_ is not None:
            sheets["top20_iv_df"] = self.top20_iv_df_
        if getattr(self, "top50_uni_", None) is not None:
            sheets["top50_univariate"] = self.top50_uni_
        if self.ks_info_traincv_ is not None:
            sheets["ks_info_traincv"] = self.ks_info_traincv_
        if self.ks_info_test_ is not None:
            sheets["ks_info_test"] = self.ks_info_test_
        if self.ks_info_oot_ is not None:
            sheets["ks_info_oot"] = self.ks_info_oot_
        if self.psi_summary_ is not None:
            sheets["psi_summary"] = self.psi_summary_
        if self.psi_dropped_ is not None:
            sheets["psi_dropped_features"] = self.psi_dropped_

        # Flatten WOE mapping as a sheet
        try:
            rows = []
            for v, vw in self.woe_map.items():
                if vw.var_type == "numeric" and vw.numeric_bins is not None:
                    for b in vw.numeric_bins:
                        rows.append({
                            "variable": v,
                            "type": "numeric",
                            "item_type": "bin",
                            "left": b.left,
                            "right": b.right,
                            "label": None,
                            "members": None,
                            "woe": b.woe,
                        })
                elif vw.var_type == "categorical" and vw.categorical_groups is not None:
                    for g in vw.categorical_groups:
                        rows.append({
                            "variable": v,
                            "type": "categorical",
                            "item_type": "group",
                            "left": None,
                            "right": None,
                            "label": g.label,
                            "members": ", ".join(map(str, g.members)) if g.members is not None else None,
                            "woe": g.woe,
                        })
            if rows:
                sheets["woe_mapping"] = pd.DataFrame(rows)
        except Exception:
            pass

        if self.oot_scores_df_ is not None and not self.oot_scores_df_.empty:
            sheets["oot_scores"] = self.oot_scores_df_

        # run_meta as key/value
        try:
            meta_df = pd.DataFrame({
                "key": list(self.artifacts["pool"].keys()),
                "value": [
                    json.dumps(self.artifacts["pool"][k], ensure_ascii=False)
                    if isinstance(self.artifacts["pool"][k], (dict, list))
                    else self.artifacts["pool"][k]
                    for k in self.artifacts["pool"].keys()
                ],
            })
            sheets["run_meta"] = meta_df
        except Exception:
            pass

        # Write multi-sheet Excel via stages.report
        try:
            from .stages import write_multi_sheet
            write_multi_sheet(xlsx, sheets)
        except Exception:
            # Fallback to previous writer-based method
            try:
                with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
                    for name, df in sheets.items():
                        try:
                            df.to_excel(wr, sheet_name=name[:31], index=False)
                        except Exception:
                            pass
            except Exception:
                alt = os.path.join(self.cfg.output_folder, f"model_report_{self.cfg.run_id}.xlsx")
                with pd.ExcelWriter(alt, engine="xlsxwriter") as wr:
                    for name, df in sheets.items():
                        try:
                            df.to_excel(wr, sheet_name=name[:31], index=False)
                        except Exception:
                            pass
                self._log(f"[warn] Excel locked: wrote to {os.path.basename(alt)} instead")

        # Optional Parquet/CSV export (disabled by default)
        def save(df, name):
            try:
                if df is None or df.empty:
                    return
                if self.cfg.write_parquet:
                    df.to_parquet(os.path.join(self.cfg.output_folder, f"{name}.parquet"), index=False)
                elif self.cfg.write_csv:
                    df.to_csv(os.path.join(self.cfg.output_folder, f"{name}.csv"), index=False)
            except Exception:
                pass
        save(self.final_vars_df_, "final_vars")
        save(self.best_model_vars_df_, "best_model_vars_df")
        save(self.best_model_woe_df_, "best_model_woe_df")
        save(self.top20_iv_df_, "top20_iv_df")
        save(self.top50_uni_, "top50_univariate")
        save(self.models_summary_, "models_summary")
        save(self.psi_summary_, "psi_summary")
        save(self.psi_dropped_, "psi_dropped_features")
        save(self.ks_info_traincv_, "ks_info_traincv")
        save(self.ks_info_test_, "ks_info_test")
        save(self.ks_info_oot_, "ks_info_oot")
        save(self.confusion_oot_, "confusion_oot")

    def _write_excel(self, wr):
        # Helper to convert written sheet to an Excel table
        def _as_table(writer, df: pd.DataFrame, sheet: str, table_name: str):
            if df is None or df.empty:
                return
            try:
                ws = writer.sheets[sheet]
                nrows, ncols = df.shape
                try:
                    # openpyxl path
                    from openpyxl.utils import get_column_letter
                    from openpyxl.worksheet.table import Table, TableStyleInfo
                    ref = f"A1:{get_column_letter(ncols)}{nrows+1}"
                    t = Table(displayName=table_name, ref=ref)
                    t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
                    ws.add_table(t)
                except Exception:
                    # xlsxwriter path
                    ws.add_table(0, 0, nrows, ncols-1, {"name": table_name, "style": "Table Style Medium 9"})
            except Exception:
                pass

        # dataset kolonu YOK; kesitler sheet adiyla ayrisir
        if self.final_vars_df_ is not None:
            self.final_vars_df_.to_excel(wr, sheet_name="final_vars", index=False)
            _as_table(wr, self.final_vars_df_, "final_vars", "tbl_final_vars")
        pd.DataFrame({"best_name": [self.best_model_name_]}).to_excel(wr, sheet_name="best_name", index=False)
        if self.models_summary_ is not None and self.best_model_name_ is not None:
            df_best = self.models_summary_.query("model_name==@self.best_model_name_")
            df_best.to_excel(wr, sheet_name="best_model", index=False)
            _as_table(wr, df_best, "best_model", "tbl_best_model")
        if self.models_summary_ is not None:
            self.models_summary_.to_excel(wr, sheet_name="models_summary", index=False)
            _as_table(wr, self.models_summary_, "models_summary", "tbl_models_summary")
        if self.best_model_vars_df_ is not None:
            self.best_model_vars_df_.to_excel(wr, sheet_name="best_model_vars_df", index=False)
            _as_table(wr, self.best_model_vars_df_, "best_model_vars_df", "tbl_best_model_vars_df")
        if self.best_model_woe_df_ is not None:
            self.best_model_woe_df_.to_excel(wr, sheet_name="best_model_woe_df", index=False)
            _as_table(wr, self.best_model_woe_df_, "best_model_woe_df", "tbl_best_model_woe_df")
        if self.top20_iv_df_ is not None:
            self.top20_iv_df_.to_excel(wr, sheet_name="top20_iv_df", index=False)
            _as_table(wr, self.top20_iv_df_, "top20_iv_df", "tbl_top20_iv_df")
        if self.top50_uni_ is not None:
            self.top50_uni_.to_excel(wr, sheet_name="top50_univariate", index=False)
            _as_table(wr, self.top50_uni_, "top50_univariate", "tbl_top50_univariate")
        if self.ks_info_traincv_ is not None:
            self.ks_info_traincv_.to_excel(wr, sheet_name="ks_info_traincv", index=False)
            _as_table(wr, self.ks_info_traincv_, "ks_info_traincv", "tbl_ks_info_traincv")
        if self.ks_info_test_ is not None:
            self.ks_info_test_.to_excel(wr, sheet_name="ks_info_test", index=False)
            _as_table(wr, self.ks_info_test_, "ks_info_test", "tbl_ks_info_test")
        if self.ks_info_oot_ is not None:
            self.ks_info_oot_.to_excel(wr, sheet_name="ks_info_oot", index=False)
            _as_table(wr, self.ks_info_oot_, "ks_info_oot", "tbl_ks_info_oot")
        if self.psi_summary_ is not None:
            self.psi_summary_.to_excel(wr, sheet_name="psi_summary", index=False)
            _as_table(wr, self.psi_summary_, "psi_summary", "tbl_psi_summary")
        if self.psi_dropped_ is not None:
            self.psi_dropped_.to_excel(wr, sheet_name="psi_dropped_features", index=False)
            _as_table(wr, self.psi_dropped_, "psi_dropped_features", "tbl_psi_dropped")
        # woe_mapping as a flattened table
        try:
            rows=[]
            for v, vw in self.woe_map.items():
                if vw.var_type == "numeric" and vw.numeric_bins is not None:
                    for b in vw.numeric_bins:
                        rows.append({
                            "variable": v,
                            "type": "numeric",
                            "item_type": "bin",
                            "left": b.left,
                            "right": b.right,
                            "label": None,
                            "members": None,
                            "woe": b.woe,
                        })
                elif vw.var_type == "categorical" and vw.categorical_groups is not None:
                    for g in vw.categorical_groups:
                        rows.append({
                            "variable": v,
                            "type": "categorical",
                            "item_type": "group",
                            "left": None,
                            "right": None,
                            "label": g.label,
                            "members": ", ".join(map(str, g.members)) if g.members is not None else None,
                            "woe": g.woe,
                        })
            if rows:
                woe_df = pd.DataFrame(rows)
                woe_df.to_excel(wr, sheet_name="woe_mapping", index=False)
                _as_table(wr, woe_df, "woe_mapping", "tbl_woe_mapping")
        except Exception:
            pass
        # oot_scores
        if self.oot_scores_df_ is not None and not self.oot_scores_df_.empty:
            self.oot_scores_df_.to_excel(wr, sheet_name="oot_scores", index=False)
            _as_table(wr, self.oot_scores_df_, "oot_scores", "tbl_oot_scores")
        # run_meta
        meta_df = pd.DataFrame({
            "key": list(self.artifacts["pool"].keys()),
            "value": [
                json.dumps(self.artifacts["pool"][k], ensure_ascii=False)
                if isinstance(self.artifacts["pool"][k], (dict, list))
                else self.artifacts["pool"][k]
                for k in self.artifacts["pool"]
            ]
        })
        meta_df.to_excel(wr, sheet_name="run_meta", index=False)
        _as_table(wr, meta_df, "run_meta", "tbl_run_meta")




