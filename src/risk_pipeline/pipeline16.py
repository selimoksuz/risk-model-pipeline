# -*- coding: utf-8 -*-
"""
Risk Model Pipeline — Orchestrated, Robust, Logged, Resource-Aware (PSI v2 FAST)

Bu tek dosya; 16 PARSEL akışını orkestrasyonla çalıştıran, WOE→PSI (vektörize)→FS→Model→Rapor
boru hattının DERLENMİŞ ve DÜZELTİLMİŞ sürümüdür.

- Dataclasses için MUTABLE DEFAULT hatası (orchestrator alanı) `default_factory` ile giderildi.
- PSI aşamasında 'np.bincount' hatasına yol açan -1 kodları önlemek için numeric bin aralıkları
  `_normalize_numeric_edges` ile KOMŞU ve BİTİŞİK hale getirildi (boşluk kalmıyor).
- `_apply_bins` ve `_bin_labels_for_variable` eksiksizdir; unseen→OTHER, MISSING ayrımı yapılır.

Kullanım (örnek en altta):
    cfg  = Config(...)
    pipe = RiskModelPipeline(cfg)
    pipe.run(df)            # 16 Parsel orchestrated
    # pipe.export_reports() # Raporlar run sonunda da yazılır
"""

import os, sys, json, time, gc, warnings, uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

# ---- BLAS/OpenMP oversubscription önleme ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

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
        self.t0=time.time(); self.logger(f"[{now_str()}] ▶ {self.label} başlıyor{sys_metrics()}"); return self
    def __exit__(self, exc_type, exc, tb):
        dt=time.time()-self.t0; status="OK" if exc is None else f"FAIL: {exc}"
        self.logger(f"[{now_str()}] ■ {self.label} bitti ({dt:.2f}s) — {status}{sys_metrics()}")


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
    # psi logging
    psi_verbose: bool = True
    psi_log_every: int = 5
    psi_sample_months: int = 6
    # outputs
    output_excel_path: str = "model_report.xlsx"
    output_folder: str = "outputs"
    write_parquet: bool = True
    write_csv: bool = False
    dictionary_path: Optional[str] = None
    use_benchmarks: bool = True
    use_noise_sentinel: bool = True
    ks_bands: int = 10
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

        self.cluster_reps_: List[str] = []; self.baseline_vars_: List[str] = []; self.final_vars_: List[str] = []

    def _log(self, msg: str): print(msg)
    def _activate(self, name: str): self.artifacts["active_steps"].append(name)

    # ==================== ORCHESTRATED RUN ====================
    def run(self, df: pd.DataFrame):
        self.df_ = df

        # Parsel 2 — Giriş doğrulama & sabitleme
        if self.cfg.orchestrator.enable_validate:
            with Timer("2) Giriş doğrulama & sabitleme", self._log):
                self._activate("validate")
                self._validate_and_freeze(self.df_)
                self._downcast_inplace(self.df_)

        # Parsel 3 — Değişken sınıflaması
        if self.cfg.orchestrator.enable_classify:
            with Timer("3) Değişken sınıflaması", self._log):
                self._activate("classify")
                self.var_catalog_ = self._classify_variables(self.df_)
                self._log(f"   - numeric={int((self.var_catalog_.variable_group=='numeric').sum())}, "
                          f"categorical={int((self.var_catalog_.variable_group=='categorical').sum())}")

        # Parsel 4 — Eksik & Nadir kural objesi
        with Timer("4) Eksik & Nadir değer politikası", self._log):
            self._activate("missing_policy")
            self.policy_ = {"rare_threshold": self.cfg.rare_threshold,
                            "unknown_to": "OTHER", "missing_label": "MISSING", "other_label": "OTHER"}

        # Parsel 5 — Zaman bölmesi
        if self.cfg.orchestrator.enable_split:
            with Timer("5) Zaman bölmesi (Train/Test/OOT)", self._log):
                self._activate("split")
                self.train_idx_, self.test_idx_, self.oot_idx_, anchor = self._split_time(self.df_)
                self.artifacts["pool"]["anchor"] = str(anchor) if anchor is not None else None
                self._log(f"   - Train={len(self.train_idx_)}, Test={0 if self.test_idx_ is None else len(self.test_idx_)}, OOT={len(self.oot_idx_)}")
        else:
            idx_all = np.arange(len(self.df_))
            self.train_idx_, self.test_idx_, self.oot_idx_ = idx_all, None, np.array([], dtype=int)

        # Parsel 6 — WOE binleme
        if self.cfg.orchestrator.enable_woe:
            with Timer("6) WOE binleme (yalnız Train; adaptif)", self._log):
                self._activate("woe")
                self.woe_map = self._fit_woe_mapping(self.df_.iloc[self.train_idx_], self.var_catalog_, self.policy_)
                self._log(f"   - WOE hazır: {len(self.woe_map)} değişken")

        # Parsel 7 — PSI v2 FAST
        if self.cfg.orchestrator.enable_psi:
            with Timer("7) PSI (vektörize)", self._log):
                self._activate("psi")
                psi_keep = self._psi_screening(self.df_, self.train_idx_, self.test_idx_, self.oot_idx_)
                if not psi_keep:
                    iv_sorted = sorted([(k, v.iv) for k, v in self.woe_map.items()], key=lambda t: t[1], reverse=True)
                    psi_keep = [iv_sorted[0][0]] if iv_sorted else []
                self._log(f"   - PSI sonrası kalan: {len(psi_keep)}")
        else:
            psi_keep = list(self.woe_map.keys())

        # Parsel 8 — Transform
        X_tr = X_te = X_oot = y_tr = y_te = y_oot = None
        if self.cfg.orchestrator.enable_transform:
            with Timer("8) WOE transform (Train/Test/OOT)", self._log):
                X_tr, y_tr = self._transform(self.df_.iloc[self.train_idx_], psi_keep)
                if self.test_idx_ is not None and len(self.test_idx_)>0:
                    X_te, y_te = self._transform(self.df_.iloc[self.test_idx_], psi_keep)
                X_oot, y_oot = self._transform(self.df_.iloc[self.oot_idx_], psi_keep)
                self._log(f"   - X_train={X_tr.shape}, X_test={None if X_te is None else X_te.shape}, X_oot={X_oot.shape}")

        # Parsel 9 — Korelasyon & cluster temsilcileri
        if self.cfg.orchestrator.enable_corr_cluster:
            with Timer("9) Korelasyon & cluster", self._log):
                self.cluster_reps_ = self._corr_and_cluster(X_tr, psi_keep) or psi_keep[:min(10, len(psi_keep))]
                self._log(f"   - cluster temsilcisi={len(self.cluster_reps_)}")
        else:
            self.cluster_reps_ = psi_keep

        # Parsel 10 — FS
        if self.cfg.orchestrator.enable_fs:
            with Timer("10) Feature selection (Forward+1SE)", self._log):
                self.baseline_vars_ = self._feature_selection(X_tr, y_tr, self.cluster_reps_, psi_keep) or psi_keep[:min(5, len(psi_keep))]
                self._log(f"   - baseline değişken={len(self.baseline_vars_)}")
        else:
            self.baseline_vars_ = psi_keep

        # Parsel 11 — Nihai korelasyon filtresi
        pre_final = self.baseline_vars_
        if self.cfg.orchestrator.enable_final_corr:
            with Timer("11) Nihai korelasyon filtresi", self._log):
                pre_final = self._final_corr_filter(X_tr[self.baseline_vars_], y_tr) or self.baseline_vars_
                self._log(f"   - corr sonrası={len(pre_final)}")

        # Parsel 12 — Noise sentinel
        if self.cfg.orchestrator.enable_noise and self.cfg.use_noise_sentinel:
            with Timer("12) Gürültü (noise) sentineli", self._log):
                self.final_vars_ = self._noise_sentinel_check(X_tr, y_tr, pre_final) or pre_final
                self._log(f"   - final değişken={len(self.final_vars_)}")
        else:
            self.final_vars_ = pre_final

        # Parsel 13 — Modelleme
        if self.cfg.orchestrator.enable_model:
            with Timer("13) Modelleme & değerlendirme", self._log):
                self._train_and_evaluate_models(X_tr, y_tr, X_te, y_te, X_oot, y_oot)

        # Parsel 14 — Best select
        if self.cfg.orchestrator.enable_best_select:
            with Timer("14) En iyi model seçimi", self._log):
                self._select_best_model(); self._log(f"   - best={self.best_model_name_}")

        # Parsel 15 — Rapor tabloları & export
        if self.cfg.orchestrator.enable_report:
            with Timer("15) Rapor tabloları", self._log):
                self._build_report_tables(psi_keep)
                self._build_top50_univariate(X_tr, y_tr)
                self._persist_artifacts(X_oot, y_oot)
            with Timer("15b) Export (Excel/Parquet)", self._log):
                self.export_reports()

        # Parsel 16 — Dictionary (opsiyonel)
        if self.cfg.orchestrator.enable_dictionary:
            with Timer("16) Dictionary entegrasyonu", self._log):
                self._integrate_dictionary()

        self._finalize_meta()
        self._log(f"[{now_str()}] ■ RUN tamam — run_id={self.cfg.run_id}{sys_metrics()}")
        return self

    # ---------------- core: validate / downcast / classify / split ----------------
    def _validate_and_freeze(self, df: pd.DataFrame):
        for c in [self.cfg.id_col, self.cfg.time_col, self.cfg.target_col]:
            if c not in df.columns: raise ValueError(f"Zorunlu kolon eksik: {c}")
        if not set(pd.Series(df[self.cfg.target_col]).dropna().unique()).issubset({0,1}):
            raise ValueError("target_col yalnız {0,1} olmalı.")
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
        fixed={self.cfg.id_col,self.cfg.time_col,self.cfg.target_col,"snapshot_month"}
        candidates=[c for c in df.columns if c not in fixed]
        rows=[]
        for c in candidates:
            s=df[c]; vtype="numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
            uniques=s.nunique(dropna=True); ratio=uniques/max(len(s),1)
            is_hcc=(uniques>100) or (ratio>0.2)
            rows.append({"variable":c,"variable_group":vtype,"is_hcc":bool(is_hcc)})
        return pd.DataFrame(rows)

    def _split_time(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[pd.Timestamp]]:
        sm=df["snapshot_month"].dropna()
        if len(sm)==0: return np.arange(len(df)), None, np.array([],dtype=int), None
        try:
            if self.cfg.oot_anchor_mode=="fixed_date":
                if not self.cfg.oot_anchor_date: raise ValueError("fixed_date fakat tarih yok")
                anchor=month_floor(pd.Timestamp(self.cfg.oot_anchor_date))
            else:
                anchor=month_floor(pd.Timestamp(sm.max()))
        except Exception:
            anchor=month_floor(pd.Timestamp(sm.max()))
        n=int(max(0,self.cfg.oot_window_months))
        if n<1:
            oot_idx=np.array([],dtype=int); pre_idx=np.arange(len(df))
        else:
            oot_months={m for k in range(n) if not pd.isna((m:=safe_month_shift(anchor,k)))}
            if not oot_months: oot_months={anchor}
            try:
                snaps=df["snapshot_month"].apply(month_floor); is_oot=snaps.isin(oot_months).values
            except Exception:
                is_oot=df["snapshot_month"].isin(oot_months).values
            oot_idx=np.where(is_oot)[0]; pre_idx=np.where(~is_oot)[0]
        test_idx=None; train_idx=pre_idx
        if self.cfg.use_test_split and len(pre_idx)>0:
            try:
                pre=df.iloc[pre_idx].copy(); months=sorted(pd.Series(pre["snapshot_month"]).dropna().unique())
                if len(months)>=3:
                    cut=max(1,int(len(months)*0.8))
                    tr_m=set(months[:cut]); te_m=set(months[cut:])
                    tr_mask=pre["snapshot_month"].isin(tr_m).values; te_mask=pre["snapshot_month"].isin(te_m).values
                    tr_local=pre_idx[np.where(tr_mask)[0]]; te_local=pre_idx[np.where(te_mask)[0]]
                    if len(tr_local)>=1 and len(te_local)>=1: train_idx,test_idx=tr_local,te_local
            except Exception:
                train_idx,test_idx=pre_idx,None
        return train_idx,test_idx,oot_idx,anchor

    # --------------- helpers for categorical/numeric labels ----------------
    def _get_categories(self, cats_obj) -> List[Any]:
        try: return list(cats_obj.categories)
        except AttributeError:
            try: return list(cats_obj.cat.categories)
            except Exception:
                vals=pd.Series(cats_obj).dropna().unique().tolist(); return sorted(vals)

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
        """Bir değişkenin WOE bin/grup etiketlerini (sıralı ve benzersiz) verir."""
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
        Train'de fit edilmiş mapping'i kullanarak df[var]'ı WOE bin/grup etiketlerine çevirir.
        Çıkış pandas.Categorical (sabit kategori seti).
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
        Bin aralıklarını bitişik hale getirir (boşluk bırakmaz):
        - İlk bin left = -inf
        - Son bin right = +inf
        - Aradaki tüm binlerde next.left = prev.right
        - MISSING bin(ler) (NaN-NaN) sona alınır.
        """
        finite = [b for b in bins if not (np.isnan(b.left) and np.isnan(b.right))]
        finite = sorted(finite, key=lambda b: (b.left, b.right))
        if finite:
            finite[0].left = -np.inf
            for i in range(1, len(finite)):
                finite[i].left = finite[i-1].right  # KOMŞULUK: boşluk kalmasın
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
        return f"      ▪ {var:<30s} | {scope:<17s} | {label:<12s} | PSI={psi_val:6.3f} | {status}"

    def _eta(self, start_ts: float, done: int, total: int) -> str:
        if done==0: return "ETA: hesaplanıyor..."
        elapsed=time.time()-start_ts; per_item=elapsed/done; rem=max(total-done,0)*per_item
        return f"ETA: ~{int(rem):d}s (kalan {total-done}/{total})"

    def _psi_screening(self, df, train_idx, test_idx, oot_idx) -> List[str]:
        cfg=self.cfg; base_df=df.iloc[train_idx]; psi_rows=[]
        def probs_from_codes(codes: np.ndarray, K: int) -> np.ndarray:
            cnt=np.bincount(codes, minlength=K).astype(np.float64); s=cnt.sum()
            return cnt/s if s>0 else np.full(K,1.0/max(K,1),dtype=np.float64)
        def psi_codes(base_codes: np.ndarray, comp_codes: np.ndarray, K: int, eps: float)->float:
            p=np.clip(probs_from_codes(base_codes,K),eps,1.0); q=np.clip(probs_from_codes(comp_codes,K),eps,1.0)
            # Correct PSI formula uses log(p/q)
            return float(np.sum((p-q)*np.log(p/q)))
        train_months=base_df["snapshot_month"].to_numpy()
        uniq_months=np.unique(train_months[~pd.isna(train_months)])
        sample_months=uniq_months[:cfg.psi_sample_months] if cfg.psi_sample_months and len(uniq_months)>cfg.psi_sample_months else uniq_months
        keep,drop,warn=[],[],[]
        variables=list(self.woe_map.keys()); total=len(variables); t0=time.time()
        if cfg.psi_verbose:
            self._log(f"   • PSI başlayacak: {total} değişken | Train-ays={len(uniq_months)} | Test={'yok' if (test_idx is None or len(test_idx)==0) else 'var'} | OOT={len(oot_idx)}")
        for i,var in enumerate(variables,1):
            vw=self.woe_map[var]; v_start=time.time()
            full_cat=self._apply_bins(df,var,vw)
            codes_all=full_cat.cat.codes.to_numpy(np.int32); K=len(full_cat.cat.categories)
            tr_codes=codes_all[train_idx]
            te_codes=codes_all[test_idx] if (test_idx is not None and len(test_idx)>0) else None
            oot_codes=codes_all[oot_idx]
            sys_drop=False; any_warn=False
            if cfg.psi_verbose:
                self._log(f"    - {var}: Train-ay karşılaştırmaları ({len(uniq_months)} ay; örnek log={len(sample_months)})")
            for m in uniq_months:
                mask_m=(train_months==m); comp=tr_codes[mask_m]; psi_val=psi_value(probs_from_codes(tr_codes,K), probs_from_codes(comp,K), cfg.psi_eps)
                status="KEEP"
                if psi_val>cfg.psi_threshold: status="DROP"; sys_drop=True
                elif psi_val>cfg.psi_warn_low: status="WARN"; any_warn=True
                psi_rows.append({"variable":var,"compare_scope":"train_vs_train_month","compare_label":str(m)[:10],
                                 "psi_value":psi_val,"status":status,"notes":""})
                if cfg.psi_verbose and (m in sample_months):
                    self._log(self._fmt_psi_line(var,"Train-v-TrainM",str(m)[:10],psi_val,status))
            if te_codes is not None:
                psi_val=psi_value(probs_from_codes(tr_codes,K), probs_from_codes(te_codes,K), cfg.psi_eps); status="KEEP"
                if psi_val>cfg.psi_threshold: status="DROP"; sys_drop=True
                elif psi_val>cfg.psi_warn_low: status="WARN"; any_warn=True
                psi_rows.append({"variable":var,"compare_scope":"train_vs_test","compare_label":"TEST_ALL",
                                 "psi_value":psi_val,"status":status,"notes":""})
                if cfg.psi_verbose:
                    self._log(self._fmt_psi_line(var,"Train-v-Test  ","TEST_ALL",psi_val,status))
            psi_val=psi_value(probs_from_codes(tr_codes,K), probs_from_codes(oot_codes,K), cfg.psi_eps); status="KEEP"
            if psi_val>cfg.psi_threshold: status="DROP"; sys_drop=True
            elif psi_val>cfg.psi_warn_low: status="WARN"; any_warn=True
            psi_rows.append({"variable":var,"compare_scope":"train_vs_oot","compare_label":"OOT_ALL",
                             "psi_value":psi_val,"status":status,"notes":""})
            if cfg.psi_verbose:
                self._log(self._fmt_psi_line(var,"Train-v-OOT   ","OOT_ALL",psi_val,status))
            if sys_drop:
                drop.append(var); 
                if cfg.psi_verbose: self._log(f"      ✖ {var} → DROP (PSI eşik aşıldı)")
            else:
                keep.append(var)
                if cfg.psi_verbose:
                    if any_warn: warn.append(var); self._log(f"      ! {var} → KEEP (WARN bölgeleri var)")
                    else: self._log(f"      ✓ {var} → KEEP")
            if (i % max(1,cfg.psi_log_every))==0 or i==total:
                self._log(f"    progress: {i}/{total} tamamlandı — {self._eta(t0,i,total)}")
            if cfg.psi_verbose: self._log(f"      (süre: {time.time()-v_start:.2f}s)")
            del full_cat, codes_all, tr_codes, te_codes, oot_codes
        psi_df=pd.DataFrame(psi_rows).sort_values(["variable","compare_scope","compare_label"]).reset_index(drop=True)
        self.psi_summary_=psi_df.copy()
        dropped=psi_df.query("status=='DROP'").groupby("variable")["psi_value"].max().reset_index()
        # Ensure the column is named consistently before further use
        dropped = dropped.rename(columns={"psi_value":"max_psi"})
        self.psi_dropped_ = dropped.copy()
        keep_final=sorted(list(set(keep)-set(dropped["variable"].tolist())))
        self._log(f"   • PSI özet: KEEP={len(keep_final)} | DROP={len(dropped)} | WARN={len(set(warn)-set(dropped['variable']))}")
        if cfg.psi_verbose and len(dropped)>0:
            top_drop=dropped.sort_values("max_psi",ascending=False).head(10)
            self._log("   • En yüksek PSI ile DROP edilenler (ilk 10):")
            for _,r in top_drop.iterrows(): self._log(f"      - {r['variable']}: max_psi={r['max_psi']:.3f}")
        del psi_df,dropped,keep,drop,warn; gc.collect()
        return keep_final

    # ----------------------- Transform -----------------------
    def _transform(self, df: pd.DataFrame, keep_vars: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        X={}; y=df[self.cfg.target_col].values
        for v in keep_vars:
            vw=self.woe_map[v]; s=df[v]
            if vw.var_type=="numeric":
                w=pd.Series(index=s.index,dtype="float32"); miss=s.isna()
                miss_bin=next((b for b in vw.numeric_bins if np.isnan(b.left) and np.isnan(b.right)),None)
                w.loc[miss]=float(miss_bin.woe if miss_bin else 0.0)
                for b in vw.numeric_bins:
                    if np.isnan(b.left) and np.isnan(b.right): continue
                    m=(~miss)&(s>=b.left)&(s<=b.right); w.loc[m]=float(b.woe)
                X[v]=w.values
            else:
                w=pd.Series(index=s.index,dtype="float32"); miss=s.isna(); assigned=miss.copy()
                g_miss=next((g for g in vw.categorical_groups if g.label=="MISSING"),None)
                w.loc[miss]=float(g_miss.woe if g_miss else 0.0)
                for g in vw.categorical_groups:
                    if g.label=="MISSING": continue
                    m=(~miss)&(s.isin(g.members)); w.loc[m]=float(g.woe); assigned|=m
                g_other=next((g for g in vw.categorical_groups if g.label=="OTHER"),None)
                w.loc[~assigned]=float(g_other.woe if g_other else 0.0); X[v]=w.values
        X=pd.DataFrame(X,index=df.index); return X,y

    # ----------------------- Corr & cluster -----------------------
    def _corr_and_cluster(self, X: pd.DataFrame, keep_vars: List[str]) -> List[str]:
        if not keep_vars: return []
        nz=[v for v in keep_vars if X[v].std(skipna=True)>0]
        if not nz: return keep_vars[:1]
        corr=X[nz].corr(method="spearman").fillna(0.0)
        reps=[]; picked=set()
        for v in nz:
            if v in picked: continue
            group=[g for g in nz if abs(corr.loc[v,g])>0.90]
            for g in group: picked.add(g)
            reps.append(v)
        del corr; gc.collect()
        return reps

    # ----------------------- FS -----------------------
    def _feature_selection(self, X, y, candidate_vars, all_vars) -> List[str]:
        if not candidate_vars: candidate_vars=all_vars[:min(10,len(all_vars))]
        rf=RandomForestClassifier(n_estimators=max(200,100*self.cfg.n_jobs),min_samples_leaf=20,
                                  n_jobs=self.cfg.n_jobs,random_state=self.cfg.random_state,
                                  class_weight="balanced_subsample")
        rf.fit(X[candidate_vars],y)
        from sklearn.inspection import permutation_importance
        perm=permutation_importance(rf,X[candidate_vars],y,n_repeats=5,random_state=self.cfg.random_state,n_jobs=self.cfg.n_jobs)
        imp_mean=pd.Series(perm.importances_mean,index=candidate_vars)
        thr=float(np.median(imp_mean.values)-np.std(imp_mean.values))
        stable_vars=imp_mean[imp_mean>thr].index.tolist() or candidate_vars
        selected=self._forward_1se_selection(X[stable_vars],y,"KS",self.cfg.cv_folds)
        del rf,perm,imp_mean; gc.collect()
        return selected

    def _forward_1se_selection(self, X: pd.DataFrame, y: np.ndarray, maximize_metric="KS", cv_folds=5) -> List[str]:
        remaining=list(X.columns); selected=[]; best_seq=[]
        def cv_score(cols: List[str])->float:
            if not cols: return -np.inf
            skf=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=self.cfg.random_state)
            sc=[]
            for tr,va in skf.split(X[cols],y):
                m=LogisticRegression(penalty="l2",solver="lbfgs",max_iter=300,class_weight="balanced")
                m.fit(X.iloc[tr][cols],y[tr]); p=m.predict_proba(X.iloc[va][cols])[:,1]
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
    def _final_corr_filter(self, X: pd.DataFrame, y: np.ndarray, thr: float = 0.7) -> List[str]:
        if X.shape[1]==0: return []
        vars_=list(X.columns); corr=X.corr(method="spearman").abs().fillna(0.0)
        keep=[]; dropped=set()
        for i,v in enumerate(vars_):
            if v in dropped: continue
            highly=[u for u in vars_[i+1:] if corr.loc[v,u]>thr]
            cand=[v]+highly; best_v,best_s=None,-np.inf
            for c in cand:
                try:
                    m=LogisticRegression(penalty="l2",solver="lbfgs",max_iter=300,class_weight="balanced")
                    m.fit(X[[c]],y); p=m.predict_proba(X[[c]])[:,1]; ks,_=ks_statistic(y,p)
                    if ks>best_s: best_s,best_v=ks,c
                except Exception: continue
            keep.append(best_v if best_v else v)
            for c in cand:
                if c!=best_v: dropped.add(c)
        del corr; gc.collect()
        return list(dict.fromkeys(keep))

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

    # ----------------------- modelleme -----------------------
    def _train_and_evaluate_models(self, Xtr, ytr, Xte, yte, Xoot, yoot):
        if not self.final_vars_: self.final_vars_=[Xtr.columns[0]]
        models={"Logit_L2": LogisticRegression(penalty="l2",solver="lbfgs",max_iter=500,class_weight="balanced")}
        if self.cfg.use_benchmarks:
            models.update({
                "RandomForest": RandomForestClassifier(n_estimators=max(300,150*self.cfg.n_jobs),min_samples_leaf=20,
                                                       n_jobs=self.cfg.n_jobs,random_state=self.cfg.random_state,
                                                       class_weight="balanced_subsample"),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=max(400,200*self.cfg.n_jobs),min_samples_leaf=20,
                                                   n_jobs=self.cfg.n_jobs,random_state=self.cfg.random_state,
                                                   class_weight="balanced"),
            })
        rows=[]; ks_tables={"traincv":None,"test":None,"oot":None}
        skf=StratifiedKFold(n_splits=self.cfg.cv_folds,shuffle=True,random_state=self.cfg.random_state)
        for name,mdl in models.items():
            print(f"[{now_str()}]   - {name} CV başlıyor{sys_metrics()}")
            sc=[]
            for tr,va in skf.split(Xtr[self.final_vars_],ytr):
                mdl.fit(Xtr.iloc[tr][self.final_vars_],ytr[tr])
                p=mdl.predict_proba(Xtr.iloc[va][self.final_vars_])[:,1]
                ks,_=ks_statistic(ytr[va],p); auc=roc_auc_score(ytr[va],p); sc.append((ks,auc))
            ks_cv=float(np.mean([s[0] for s in sc])); auc_cv=float(np.mean([s[1] for s in sc])); gini_cv=gini_from_auc(auc_cv)
            mdl.fit(Xtr[self.final_vars_],ytr); p_tr=mdl.predict_proba(Xtr[self.final_vars_])[:,1]
            ks_tables["traincv"]=ks_table(ytr,p_tr,n_bands=self.cfg.ks_bands)
            ks_te=auc_te=gini_te=None
            if Xte is not None and yte is not None and Xte.shape[0]>0:
                p_te=mdl.predict_proba(Xte[self.final_vars_])[:,1]
                ks_te,_=ks_statistic(yte,p_te); auc_te=roc_auc_score(yte,p_te); gini_te=gini_from_auc(auc_te)
                ks_tables["test"]=ks_table(yte,p_te,n_bands=self.cfg.ks_bands)
            p_oot=mdl.predict_proba(Xoot[self.final_vars_])[:,1]
            ks_oot,thr_oot=ks_statistic(yoot,p_oot); auc_oot=roc_auc_score(yoot,p_oot); gini_oot=gini_from_auc(auc_oot)
            ks_tables["oot"]=ks_table(yoot,p_oot,n_bands=self.cfg.ks_bands)
            rows.append({"model_name":name,"KS_TrainCV":ks_cv,"AUC_TrainCV":auc_cv,"Gini_TrainCV":gini_cv,
                         "KS_Test":ks_te,"AUC_Test":auc_te,"Gini_Test":gini_te,
                         "KS_OOT":ks_oot,"AUC_OOT":auc_oot,"Gini_OOT":gini_oot,"KS_OOT_threshold":thr_oot})
            self.models_[name]=mdl
        self.models_summary_=pd.DataFrame(rows).sort_values("KS_OOT",ascending=False).reset_index(drop=True)
        self.ks_info_traincv_,self.ks_info_test_,self.ks_info_oot_=ks_tables["traincv"],ks_tables["test"],ks_tables["oot"]

    def _select_best_model(self):
        df=self.models_summary_
        if df is None or df.empty: self.best_model_name_=None; return
        top=df.sort_values(["KS_OOT","AUC_OOT","Gini_OOT"], ascending=[False, False, False]).iloc[0]
        self.best_model_name_=str(top["model_name"])

    # ----------------------- Rapor tabloları + univariate top50 -----------------------
    def _build_report_tables(self, psi_keep: List[str]):
        iv_rows = [{"variable": v, "IV": self.woe_map[v].iv, "variable_group": self.woe_map[v].var_type} for v in psi_keep]
        self.top20_iv_df_ = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).head(20).reset_index(drop=True)

        bm = self.best_model_name_
        mdl = self.models_.get(bm) if bm else None
        if bm is None or mdl is None:
            self.best_model_vars_df_ = None
        else:
            if hasattr(mdl, "coef_"):
                coefs = mdl.coef_[0]
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_, "coef_or_importance": coefs}) \
                    .assign(
                        sign=lambda d: np.sign(d["coef_or_importance"]).astype(int),
                        variable_group=lambda d: d["variable"].map(lambda v: self.woe_map[v].var_type if v in self.woe_map else None)
                    ).sort_values("coef_or_importance", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
            elif hasattr(mdl, "feature_importances_"):
                imps = mdl.feature_importances_
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_, "coef_or_importance": imps}) \
                    .assign(
                        sign=lambda d: np.where(d["coef_or_importance"] >= 0, 1, -1),
                        variable_group=lambda d: d["variable"].map(lambda v: self.woe_map[v].var_type if v in self.woe_map else None)
                    ).sort_values("coef_or_importance", ascending=False).reset_index(drop=True)
            else:
                self.best_model_vars_df_ = pd.DataFrame({"variable": self.final_vars_})

        w_rows = []
        for v in self.final_vars_:
            vw = self.woe_map[v]
            if vw.var_type == "numeric":
                for b in vw.numeric_bins:
                    label = "MISSING" if (np.isnan(b.left) and np.isnan(b.right)) else f"[{b.left},{b.right})"
                    w_rows.append({
                        "variable": v, "variable_group": "numeric",
                        "group": label, "values": label,
                        "count": b.count, "event": b.event, "non_event": b.non_event,
                        "event_rate": b.event_rate, "woe": b.woe
                    })
            else:
                for g in vw.categorical_groups:
                    vals = g.label if g.label in ("MISSING", "OTHER") else "{"+",".join(map(str, g.members))+"}"
                    w_rows.append({
                        "variable": v, "variable_group": "categorical",
                        "group": g.label, "values": vals,
                        "count": g.count, "event": g.event, "non_event": g.non_event,
                        "event_rate": g.event_rate, "woe": g.woe
                    })
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
        self.top50_uni_ = pd.DataFrame(rows).sort_values("Gini_uni", ascending=False).head(50).reset_index(drop=True)

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
        # Persist JSON mapping only if requested via write_csv flag (as a lightweight toggle)
        try:
            if self.cfg.write_csv:
                with open(os.path.join(self.cfg.output_folder, f"woe_mapping_{self.cfg.run_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if self.best_model_name_ and X_oot is not None and y_oot is not None and X_oot.shape[0] > 0:
            mdl = self.models_.get(self.best_model_name_)
            if mdl is not None:
                prob = mdl.predict_proba(X_oot[self.final_vars_])[:, 1]
                df_scores = pd.DataFrame({"prob": prob, "target": y_oot})
                self.oot_scores_df_ = df_scores
                # Optional external persist
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
            # sözlük yoksa veya okunamadıysa akış devam etsin
            pass

    # ----------------------- Meta & Export -----------------------
    def _finalize_meta(self):
        self.artifacts["pool"]["config"] = asdict(self.cfg)
        self.artifacts["pool"]["final_vars"] = self.final_vars_
        self.artifacts["pool"]["best_model_name"] = self.best_model_name_

    def export_reports(self):
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        xlsx = os.path.join(self.cfg.output_folder, self.cfg.output_excel_path)
        # Excel
        try:
            with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
                self._write_excel(wr)
        except Exception:
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as wr:
                self._write_excel(wr)
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

        # dataset kolonu YOK; kesitler sheet adıyla ayrışır
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
