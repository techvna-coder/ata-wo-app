# core/io_excel.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List
from .mapping import INCOLS, MAP_ALIASES, ORDER_OUTCOLS

def load_wo_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}

    # Map columns by alias (mở rộng alias tiếng Việt/không dấu/viết thường)
    out: Dict[str, pd.Series] = {}
    for std_col in INCOLS:
        found = None
        if std_col in MAP_ALIASES:
            cand_alias: List[str] = list(MAP_ALIASES[std_col]) + [
                # alias mở rộng
                "mo ta", "mô tả", "hanh dong", "hành động", "khac phuc", "khắc phục",
                "defect", "description", "symptom", "rectification", "action", "repair", "corrective"
            ]
            seen = set()
            for alias in cand_alias:
                al = alias.strip().lower()
                if al in seen:
                    continue
                seen.add(al)
                if al in cols_lower:
                    found = cols_lower[al]
                    break
        if found:
            out[std_col] = df[found].astype(str)
        else:
            out[std_col] = pd.Series([""] * len(df), dtype=str)

    # Fallback: nếu không map được Defect_Text & Rectification_Text, dùng cột text dài nhất
    if out["Defect_Text"].str.len().sum() == 0 and out["Rectification_Text"].str.len().sum() == 0:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        best = None
        best_len = -1
        for c in text_cols:
            L = df[c].fillna("").str.len().sum()
            if L > best_len:
                best, best_len = c, L
        if best:
            base_text = df[best].astype(str)
            out["Defect_Text"] = base_text
            out["Rectification_Text"] = base_text

    # Ensure all required columns exist
    for k in INCOLS:
        if k not in out:
            out[k] = pd.Series([""] * len(df), dtype=str)

    return pd.DataFrame(out)
