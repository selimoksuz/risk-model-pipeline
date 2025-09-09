from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return pd.read_csv(p)
