from pathlib import Path
from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.data.sample import load_credit_risk_sample

sample = load_credit_risk_sample()
OUTPUT_DIR = Path('debug_output')
OUTPUT_DIR.mkdir(exist_ok=True)

cfg = Config(
    target_column='target',
    id_column='customer_id',
    time_column='app_dt',
    create_test_split=True,
    train_ratio=0.6,
    test_ratio=0.2,
    oot_ratio=0.2,
    stratify_test=True,
    oot_months=2,
    enable_tsfresh_features=True,
    enable_dual=False,
    enable_scoring=False,
    output_folder=str(OUTPUT_DIR),
    algorithms=['gam'],
    use_optuna=True,
    n_trials=1,
    optuna_timeout=60,
    random_state=42,
)

try:
    UnifiedRiskPipeline(cfg).fit(sample.development)
except Exception as exc:
    import traceback
    traceback.print_exc()
