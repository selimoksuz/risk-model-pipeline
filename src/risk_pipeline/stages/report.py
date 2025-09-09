from __future__ import annotations

from typing import Dict

import pandas as pd


def write_multi_sheet(xlsx_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """Write multiple DataFrames to a single Excel file as separate sheets.
    Overwrites existing file.
    """
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is None:
                continue
            try:
                df.to_excel(w, sheet_name=name[:31], index=False)
            except Exception:
                # sheet name too long or other issues: fallback to sanitized name
                df.to_excel(w, sheet_name="sheet", index=False)
