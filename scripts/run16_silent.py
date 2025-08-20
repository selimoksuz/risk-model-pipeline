import os
import pandas as pd
from risk_pipeline.pipeline16 import Config, RiskModelPipeline, Orchestrator

# Clean outputs each run to avoid accumulation
out_dir = 'outputs'
if os.path.isdir(out_dir):
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except Exception:
            pass

csv_path = 'data/input.csv'
df = pd.read_csv(csv_path)

# Orchestrator (disable PSI to avoid potential small-sample or branch-specific issues)
orch = Orchestrator(
    enable_validate=True,
    enable_classify=True,
    enable_missing_policy=True,
    enable_split=True,
    enable_woe=True,
    enable_psi=False,
    enable_transform=True,
    enable_corr_cluster=True,
    enable_fs=True,
    enable_final_corr=True,
    enable_noise=True,
    enable_model=False,
    enable_best_select=False,
    enable_report=True,
    enable_dictionary=False,
)

cfg = Config(
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    use_test_split=True,            # internal TEST (month-based 80/20)
    oot_window_months=3,            # last 3 months as OOT
    output_folder=out_dir,
    output_excel_path='model_report.xlsx',
    psi_verbose=False,
)

pipe = RiskModelPipeline(cfg)
# Minimize console logging noise
pipe._log = lambda *args, **kwargs: None

pipe.run(df)
print(f"Done. Best={pipe.best_model_name_} | Reports -> {cfg.output_folder}")
