from __future__ import annotations

import pandas as pd


def classify_variables(df: pd.DataFrame, *, id_col: str, time_col: str, target_col: str) -> pd.DataFrame:
    """Return a catalog DataFrame with variable names and groups (numeric/categorical).

    The id/time/target columns are excluded from catalog. Output has columns:
    - variable: original column name
    - variable_group: 'numeric' or 'categorical'
    """
    exclude = {id_col, time_col, target_col}
    vars_ = [c for c in df.columns if c not in exclude]
    if not vars_:
        return pd.DataFrame(columns=["variable", "variable_group"])
    num = set(df[vars_].select_dtypes(include=["number"]).columns)
    out = []
    for c in vars_:
        grp = "numeric" if c in num else "categorical"
        out.append({"variable": c, "variable_group": grp})
    return pd.DataFrame(out)
