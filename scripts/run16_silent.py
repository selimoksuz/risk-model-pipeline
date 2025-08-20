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

# Orchestrator with PSI enabled
orch = Orchestrator(
    enable_validate=True,
    enable_classify=True,
    enable_missing_policy=True,
    enable_split=True,
    enable_woe=True,
    enable_psi=True,       # enable PSI for requested behavior
    enable_transform=True,
    enable_corr_cluster=True,
    enable_fs=True,
    enable_final_corr=True,
    enable_noise=True,
    enable_model=True,
    enable_best_select=True,
    enable_report=True,
    enable_dictionary=False,
)

cfg = Config(
    id_col='app_id',
    time_col='app_dt',
    target_col='target',
    use_test_split=True,            # enable internal TEST split (80/20 by months)
    oot_window_months=3,            # last 3 months as OOT
    output_folder=out_dir,
    output_excel_path='model_report.xlsx',
    psi_verbose=True,
    write_parquet=False,
    write_csv=False,
    run_id='latest',                # overwrite artifacts based on run_id-stamped names
)

pipe = RiskModelPipeline(cfg)
# Keep logs minimal to avoid encoding issues
pipe._log = lambda *args, **kwargs: None

pipe.run(df)
print(f"Done. Best={pipe.best_model_name_} | Reports -> {cfg.output_folder}")
