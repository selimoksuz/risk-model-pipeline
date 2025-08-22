from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List

from ..api import score_df


def build_scored_frame(
    df: pd.DataFrame,
    *,
    mapping: Dict[str, Any],
    final_vars: List[str],
    model: Any,
    id_col: str = "app_id",
    calibrator: Any | None = None,
) -> pd.DataFrame:
    """Thin wrapper around api.score_df for clearer layering."""
    return score_df(df, mapping, final_vars, model, id_col=id_col, calibrator=calibrator)

