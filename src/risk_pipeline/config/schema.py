from pydantic import BaseModel, Field
from typing import Optional, List

class RunConfig(BaseModel):
    experiment_name: str = Field(default="default_experiment")
    seed: int = 42
    target: str = "target"
    id_col: str = "app_id"
    date_col: str = "app_dt"
    oot_start: Optional[str] = None
    oot_end: Optional[str] = None
    enable_psi: bool = True
    enable_fs: bool = True
    enable_model: bool = True
    enable_calibration: bool = True
    enable_report: bool = True

class PathConfig(BaseModel):
    data_input: str = "data/input.csv"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"

class Config(BaseModel):
    run: RunConfig = RunConfig()
    path: PathConfig = PathConfig()
