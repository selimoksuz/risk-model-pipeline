import importlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
PACKAGE_PATH = SRC_PATH / "risk_pipeline"
INIT_FILE = PACKAGE_PATH / "__init__.py"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Remove any already-imported copies of the package
for module_name in list(sys.modules.keys()):
    if module_name.startswith("risk_pipeline"):
        del sys.modules[module_name]

importlib.invalidate_caches()

spec = importlib.util.spec_from_file_location(
    "risk_pipeline",
    INIT_FILE,
    submodule_search_locations=[str(PACKAGE_PATH)]
)
module = importlib.util.module_from_spec(spec)
sys.modules["risk_pipeline"] = module
spec.loader.exec_module(module)

import numpy as np
import pandas as pd
import pytest

from risk_pipeline.core.config import Config


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_samples = 200

    features = pd.DataFrame(
        {
            "feature_1": rng.normal(0, 1, size=n_samples),
            "feature_2": rng.normal(0, 1, size=n_samples),
            "feature_3": rng.normal(0, 1, size=n_samples),
        }
    )

    linear_term = features["feature_1"] * 0.8 + features["feature_2"] * -0.4
    noise = rng.normal(0, 0.5, size=n_samples)
    target = (linear_term + noise > 0).astype(int)

    dataset = features.copy()
    dataset["app_id"] = np.arange(n_samples)
    dataset["app_dt"] = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    dataset["target"] = target
    return dataset


@pytest.fixture
def minimal_config(tmp_path) -> Config:
    return Config(
        target_column="target",
        id_column="app_id",
        time_column="app_dt",
        create_test_split=True,
        test_size=0.3,
        stratify_test=True,
        oot_size=0.2,
        calculate_woe_all=False,
        calculate_univariate_gini=False,
        selection_steps=[],
        use_noise_sentinel=False,
        algorithms=["logistic"],
        use_optuna=False,
        enable_dual=False,
        optimize_risk_bands=False,
        calculate_shap=False,
        report_components=[],
        save_reports=False,
        save_plots=False,
        save_models=False,
        risk_band_tests=[],
        output_folder=str(tmp_path / "outputs"),
    )
