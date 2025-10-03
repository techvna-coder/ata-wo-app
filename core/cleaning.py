# core/cleaning.py
from __future__ import annotations
import regex as re

# Các cụm meta/audit không liên quan kỹ thuật
_META_PATTERNS = [
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*",                     # form NRC
    r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",               # S.O-A321-3348-2019-HEAT
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",                   # mã nhân sự kiểu VAE03251
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",        # 05.AUG.2025
    r"\b\d{1,2}[:.]\d{2}\b.*",                     # 09:24, 11:22
    r"^\s*\d+\s*WORKSTEP\S*\b.*",                  # "1 WORKSTEP ..."
]

# Giữ lại các tham chiếu kỹ thuật/triệu chứng
# (không xoá các dòng chứa AMM/TSM/FIM/ECAM/EICAS/CAS/fault/leak...)
_KEEP_HINT = re.compile(
    r"\b(AMM|TSM|FIM|ECAM|EICAS|CAS|FAULT|FAIL|LEAK|OVERHEAT|VIBRAT|SMOKE|WARNING|CAUTION|ADVISORY)\b",
    re.I,
)

_meta_re = re.compile("|".join(_META_PATTERNS), re.I)

_whitespace_re = re.compile(r"[ \t]+")

def clean_wo_text(s: str) -> str:
    """Làm sạch mô tả/rectification: bỏ audit/meta, giữ symptom/citation kỹ thuật."""
    if not s:
        return ""
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    for ln in lines:
        if not ln:
            continue
        # Nếu dòng chứa hint kỹ thuật → giữ lại nguyên vẹn
        if _KEEP_HINT.search(ln):
            kept.append(ln)
            continue
        # Nếu là meta/audit → loại bỏ
        if _meta_re.search(ln):
            continue
        kept.append(ln)
    # rút gọn khoảng trắng
    out = " ".join(kept)
    out = _whitespace_re.sub(" ", out)
    return out.strip()
