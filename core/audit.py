# core/audit.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

MANIFEST = Path("data_store/manifest.json")
INGEST_DIR = Path("data_store/ingest")
WO_PARQUET = Path("data_store/wo_training.parquet")
ATA_PARQUET = Path("data_store/ata_map.parquet")

# Các pattern dùng lại cho nhận diện cột
DESC_PATTERNS = [
    r"^W/?O\s*Description$", r"\b(description|defect|symptom)\b",
    r"\bmô\s*tả\b", r"\bmo\s*ta\b",
]
ACTION_PATTERNS = [
    r"^W/?O\s*Action$", r"\b(rectification|action|repair|corrective|rectify)\b",
    r"\bhành\s*động\b", r"\bhanh\s*dong\b", r"\bkhắc\s*phục\b", r"\bkhac\s*phuc\b",
]
ATA_FINAL_PATTERNS = [
    r"\bATA\s*0?4\s*(Corrected|Final)\b", r"\bATA\s*final\b", r"\bATA04_Final\b",
    r"\bATA\s*Corrected\b", r"\bATA\s*04\s*Corrected\b",
]
ATA_ENTERED_PATTERNS = [
    r"^ATA$", r"\bATA\s*0?4\b", r"\bATA\s*04\b", r"\bATA04_Entered\b",
    r"\bATA\s*Code\b", r"\bATA_Code\b",
]

def _find_col(df: pd.DataFrame, pats: List[str]) -> Optional[str]:
    for pat in pats:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                return c
    return None

def classify_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Phân loại 1 DataFrame là ATA map hay WO, và xác định các cột liên quan."""

    cols_lower = [c.lower() for c in df.columns]

    # 🩵 NEW: nhận diện dạng 'A321 Data ATA map.xlsx' có 1 cột duy nhất
    if len(df.columns) == 1:
        col0 = df.columns[0]
        sample = " ".join(df[col0].head(10).astype(str).tolist())
        # Nếu có pattern ATA code như '79-21-42' xuất hiện nhiều
        if len(re.findall(r"\b\d{2}-\d{2}(?:-\d{2})?\b", sample)) >= 3:
            return {
                "kind": "ATA_MAP",
                "desc_col": col0,
                "action_col": None,
                "ata_final_col": None,
                "ata_entered_col": None,
                "columns": list(df.columns),
            }

    # 🩶 logic cũ (vẫn giữ nguyên)
    is_ata_map = (
        any(re.search(r"ata.*0?4|^ata$|code", c) for c in cols_lower)
        and any(re.search(r"name|title|system|mô tả|mo ta|description", c) for c in cols_lower)
    )

    desc_col = _find_col(df, DESC_PATTERNS)
    act_col  = _find_col(df, ACTION_PATTERNS)
    ata_final_col   = _find_col(df, ATA_FINAL_PATTERNS)
    ata_entered_col = _find_col(df, ATA_ENTERED_PATTERNS)

    is_wo = bool(desc_col and (ata_final_col or ata_entered_col))

    kind = "unknown"
    if is_wo:
        kind = "WO"
    elif is_ata_map:
        kind = "ATA_MAP"

    return {
        "kind": kind,
        "desc_col": desc_col,
        "action_col": act_col,
        "ata_final_col": ata_final_col,
        "ata_entered_col": ata_entered_col,
        "columns": list(df.columns),
    }
def load_manifest() -> Dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"by_id": {}}

def list_ingested_files() -> List[Dict[str, Any]]:
    m = load_manifest()
    out = []
    for fid, rec in m.get("by_id", {}).items():
        out.append({"file_id": fid, **rec})
    # sắp xếp theo tên
    out.sort(key=lambda x: x.get("name",""))
    return out

def audit_store() -> Dict[str, Any]:
    """Tạo thống kê nhanh cho wo_training.parquet và ata_map.parquet."""
    report: Dict[str, Any] = {}

    # WO training
    if WO_PARQUET.exists():
        dfw = pd.read_parquet(WO_PARQUET)
        report["wo_training"] = {
            "exists": True,
            "rows": int(len(dfw)),
            "distinct_ata04": int(dfw["ata04"].nunique()),
            "sample": dfw.head(10),
            "top_ata": dfw["ata04"].value_counts().head(20).rename_axis("ATA04").reset_index(name="count"),
        }
    else:
        report["wo_training"] = {"exists": False}

    # ATA map
    if ATA_PARQUET.exists():
        dfa = pd.read_parquet(ATA_PARQUET)
        report["ata_map"] = {
            "exists": True,
            "rows": int(len(dfa)),
            "sample": dfa.head(10),
        }
        # Nếu có wo_training, tính mức độ phủ tên gọi
        if report["wo_training"]["exists"]:
            dfw = pd.read_parquet(WO_PARQUET)
            cov = (
                dfw[["ata04"]]
                .drop_duplicates()
                .merge(dfa[["ATA04"]].drop_duplicates(), left_on="ata04", right_on="ATA04", how="left", indicator=True)
            )
            coverage = (cov["_merge"] == "both").mean() if len(cov) else 0.0
            report["ata_map"]["coverage_on_training"] = float(round(coverage * 100, 2))
    else:
        report["ata_map"] = {"exists": False}

    return report

def classify_all_ingested(limit_preview_rows: int = 5) -> List[Dict[str, Any]]:
    """Đọc từng file trong ingest/ và chạy classify_dataframe (đọc nhẹ nhàng)."""
    out = []
    paths = sorted(INGEST_DIR.glob("*.xls*"))
    for p in paths:
        try:
            df = pd.read_excel(p, dtype=str, nrows=200)  # đọc nhẹ 200 hàng đầu để map cột
            info = classify_dataframe(df)
            row = {"path": str(p.name), **info}
            if info["kind"] == "WO":
                # preview thêm vài dòng outlook
                prev = df[[c for c in [info["desc_col"], info["action_col"], info["ata_final_col"], info["ata_entered_col"]] if c]].head(limit_preview_rows)
                row["preview"] = prev
            out.append(row)
        except Exception as e:
            out.append({"path": str(p.name), "kind": "error", "error": str(e)})
    return out
