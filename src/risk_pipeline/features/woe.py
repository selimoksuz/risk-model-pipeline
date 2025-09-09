import numpy as np
import pandas as pd


def simple_woe(series: pd.Series, target: pd.Series):
    # placeholder: to be replaced with your full WOE binning
    df = pd.DataFrame({"x": series, "y": target}).copy()
    df["bin"] = pd.qcut(df["x"].rank(method="first"), q=10, duplicates="drop")
    grouped = df.groupby("bin")["y"].agg(["sum", "count"])
    grouped["non_event"] = grouped["count"] - grouped["sum"]
    # smoothing to avoid div-by-zero
    eps = 0.5
    rate_e = (grouped["sum"] + eps) / (grouped["sum"].sum() + eps * len(grouped))
    rate_ne = (grouped["non_event"] + eps) / (grouped["non_event"].sum() + eps * len(grouped))
    grouped["woe"] = np.log(rate_e / rate_ne)
    return grouped.reset_index()
