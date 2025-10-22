# core/audit_enhanced.py
"""
Enhanced audit with support for A321 ATA map format.
Replaces core/audit.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

MANIFEST = Path("data_store/manifest.json")
INGEST_DIR = Path("data_store/ingest")
WO_PARQUET = Path("data_store/wo_training.parquet")
ATA_PARQUET = Path("data_store/ata_map.parquet")

# Patterns for column detection
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
    """Find column matching patterns."""
    for pat in pats:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                return c
    return None


def _is_a321_ata_map(df: pd.DataFrame) -> bool:
    """
    Detect if DataFrame is A321 ATA map format.
    
    Characteristics:
    - Has UI markers (expand_more, chevron_right)
    - First column contains ATA codes with descriptions
    - Format: "XX - (ENGINE) - SYSTEM" or "XX-YY - ..."
    """
    # Check column names for UI markers
    first_col_name = df.columns[0]
    if any(marker in first_col_name for marker in ['xpand_more', 'expand_more', 'chevron']):
        return True
    
    # Check first few values for ATA pattern with UI markers
    first_col = df.iloc[:, 0].astype(str)
    sample = first_col.head(10)
    
    # Count lines with UI markers + ATA codes
    ata_with_ui = 0
    for val in sample:
        val_clean = val.replace('\xa0', ' ')
        if any(marker in val for marker in ['expand_more', 'chevron_right', 'xpand_more']):
            if re.search(r'\d{2}\s*-\s*\(.+?\)', val_clean):
                ata_with_ui += 1
    
    return ata_with_ui >= 3  # At least 3 lines match pattern


def classify_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced classifier supporting A321 ATA map format.
    
    Returns dict with:
        - kind: "WO" | "ATA_MAP" | "A321_ATA_MAP" | "unknown"
        - Additional metadata
    """
    # Check for A321 format first
    if _is_a321_ata_map(df):
        return {
            "kind": "A321_ATA_MAP",
            "format": "hierarchical_with_ui_markers",
            "columns": list(df.columns),
            "processor": "ata_map_loader.py",
            "notes": "Use specialized loader for this format",
        }
    
    # Standard ATA map check
    cols_lower = [c.lower() for c in df.columns]
    is_ata_map = (
        any(re.search(r"ata.*0?4|^ata$|code", c) for c in cols_lower)
        and any(re.search(r"name|title|system|mô tả|mo ta|description", c) for c in cols_lower)
    )
    
    # WO check
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
    """Load sync manifest."""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"by_id": {}}


def list_ingested_files() -> List[Dict[str, Any]]:
    """List all ingested files from manifest."""
    m = load_manifest()
    out = []
    for fid, rec in m.get("by_id", {}).items():
        out.append({"file_id": fid, **rec})
    out.sort(key=lambda x: x.get("name",""))
    return out


def audit_store() -> Dict[str, Any]:
    """Audit data store status."""
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
    """
    Classify all ingested files with enhanced A321 detection.
    """
    out = []
    paths = sorted(INGEST_DIR.glob("*.xls*"))
    
    for p in paths:
        try:
            df = pd.read_excel(p, dtype=str, nrows=200)
            info = classify_dataframe(df)
            row = {"path": str(p.name), **info}
            
            # Add preview for different types
            if info["kind"] == "WO":
                prev_cols = [c for c in [info["desc_col"], info["action_col"], 
                                         info["ata_final_col"], info["ata_entered_col"]] if c]
                prev = df[prev_cols].head(limit_preview_rows)
                row["preview"] = prev
            
            elif info["kind"] == "A321_ATA_MAP":
                # Preview A321 format
                first_col = df.columns[0]
                sample_values = df[first_col].head(limit_preview_rows).tolist()
                
                # Clean for display
                cleaned = []
                for val in sample_values:
                    val_str = str(val).replace('\xa0', ' ')
                    val_str = re.sub(r'^(expand_more|chevron_right|xpand_more)\s*', '', val_str)
                    cleaned.append(val_str[:80])
                
                row["preview_text"] = cleaned
            
            out.append(row)
            
        except Exception as e:
            out.append({"path": str(p.name), "kind": "error", "error": str(e)})
    
    return out


# Backward compatibility
def classify_dataframe_old(df: pd.DataFrame) -> Dict[str, Any]:
    """Old function name for compatibility."""
    return classify_dataframe(df)
