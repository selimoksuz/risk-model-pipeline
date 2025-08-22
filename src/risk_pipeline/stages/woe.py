from __future__ import annotations

import pandas as pd
from typing import Dict, Any


def apply_woe(df_in: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """Apply WOE mapping to a DataFrame using the serialized mapping structure
    saved by the pipeline (numeric bins and categorical groups). Returns a new
    DataFrame with the same index containing WOE-transformed variables for all
    variables present in the mapping and the input frame.
    """
    out = {}
    for v, info in mapping.get("variables", {}).items():
        if v not in df_in.columns:
            continue
        if info.get("type") == "numeric":
            s = df_in[v]
            w = pd.Series(index=s.index, dtype="float32")
            miss = s.isna()
            miss_woe = 0.0
            for b in info.get("bins", []):
                left = b.get("left"); right = b.get("right"); woe = b.get("woe", 0.0)
                if left is None or right is None or (pd.isna(left) and pd.isna(right)):
                    miss_woe = float(woe)
                    continue
                m = (~miss) & (s >= left) & (s <= right)
                w.loc[m] = float(woe)
            w.loc[miss] = float(miss_woe)
            out[v] = w.values
        else:
            s = df_in[v].astype(object)
            w = pd.Series(index=s.index, dtype="float32")
            miss = s.isna()
            assigned = miss.copy()
            miss_woe = 0.0
            other_woe = 0.0
            groups = info.get("groups", [])
            for g in groups:
                lab = g.get("label")
                woe = float(g.get("woe", 0.0))
                if lab == "MISSING":
                    miss_woe = woe; continue
                if lab == "OTHER":
                    other_woe = woe; continue
                members = set(map(str, g.get("members", [])))
                m = (~miss) & (s.astype(str).isin(members))
                w.loc[m] = woe
                assigned |= m
            w.loc[miss] = float(miss_woe)
            w.loc[~assigned] = float(other_woe)
            out[v] = w.values
    return pd.DataFrame(out, index=df_in.index)

