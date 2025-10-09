# core/cleaning.py
# =============================================================================
# Mục tiêu:
#   - Làm sạch văn bản WO theo từng dòng.
#   - Chỉ loại bỏ header/footer/signature thuần META, KHÔNG chứa thông tin kỹ thuật.
#   - Giữ lại các dòng có tín hiệu kỹ thuật (AMM/TSM/FIM/ESPM, ECAM/EICAS/CAS,
#     động từ thao tác kỹ thuật REPLACED/RESET/CB/TESTED/SWAP/NCD, v.v.).
#   - Xoá các mẩu META "inline" (BY <id>, ON <date>, hh:mm) nhưng giữ phần kỹ thuật.
#
# API:
#   - clean_text_block(s: str) -> str  (giữ tương thích mã cũ)
#   - clean_wo_text(s: str)    -> str  (alias, giống clean_text_block)
# =============================================================================

from __future__ import annotations

# Ưu tiên dùng 'regex' (tương thích PCRE hơn); fallback sang 're' nếu thiếu.
try:
    import regex as re
except Exception:  # pragma: no cover
    import re  # type: ignore

from typing import Iterable

# =============================================================================
# 1) Tín hiệu kỹ thuật → nếu một dòng chứa 1 trong các mẫu sau: GIỮ LẠI dòng
# =============================================================================
_KEEP_HINT = re.compile(
    r"\b("
    r"AMM|TSM|FIM|ESPM"                     # manuals
    r"|ECAM|EICAS|CAS|FWS|ACARS"            # systems/warning
    r"|FOUND|DAMAGE|DAMAGED|CRACK|LEAK|FAIL|FAULT|INOP|OVERHEAT|SMOKE|VIBRAT"  # symptoms
    r"|REPLACE|REPLACED|REPAIR|RECTIFY|ADJUST|CALIBRAT|CALIBRATE|RIG|RIGGED"   # actions
    r"|C/O|REF|TASKCARD"                    # abbreviations
    r"|MISSING|BROKEN|WORN|ERODED|CORRODED|CONTAMINATED"                        # conditions
    r"|CHECK\s+SATIS|WITHIN\s+LIMITS|NO\s+ABNORMAL"                             # findings
    r"|NCD|RESET|CB|CIRCUIT\s*BREAKER|PULLED|PUSHED|SWAP|INTERCHANGED|EXCHANGED"
    r"|\d{2}-\d{2}(?:-\d{2})?(?:-\d+)?\b"   # ATA code 02-05[-xx[-yyy]]
    r")\b",
    re.IGNORECASE,
)

# =============================================================================
# 2) Dòng META thuần (nhật ký, chữ ký, mốc thời gian) → CHỈ xóa khi KHÔNG có hint
# =============================================================================
_PURE_META_PATTERNS: Iterable[str] = [
    # nhật ký thao tác/quy trình
    r"^\s*\d+\s+WORKSTEP\s+ADDED\s+BY\s+[A-Z]{3}\d+\s+ON\s+\d{2}\.[A-Z]{3}\.\d{4}[,\s]+\d{2}:\d{2}\s*$",
    r"^\s*WORKSTEP\s+ADDED\s+BY\s+[A-Z]{3}\d+\s+ON\s+\d{2}\.[A-Z]{3}\.\d{4}(?:[,\s]+\d{2}:\d{2})?\s*$",
    r"^\s*FINDING\s*\(NRC\)\b.*$",
    r"^\s*(DESCRIPTION|ACTION\s+PERFORMED|PERFORMED)\s+SIGN\s+[A-Z]{3}\d+\s*$",
    r"^\s*RELEASED\s+BY\b.*$",
    r"^\s*NOTE\s*:\b.*$",
    r"^\s*BY\s+[A-Z]{3}\d{5}\s*$",

    # mốc thời gian hành chính
    r"^\s*(?:ON\s+)?\d{1,2}\.[A-Z]{3}\.\d{4}\s*$",
    r"^\s*\d{1,2}:\d{2}\s*$",

    # nhãn biểu mẫu nội bộ
    r"^\s*(FORM|SHEET|DOC|TASKCARD)\s*[:#]?\s*[\w\-]+(?:\s*/\s*ITEM\s*\d+)?\s*$",
]
_PURE_META_RE = re.compile("|".join(_PURE_META_PATTERNS), re.IGNORECASE)

# =============================================================================
# 3) Mẫu META "inline" sẽ bị xoá trong lòng một dòng còn lại (không xoá cả dòng)
# =============================================================================
# Ví dụ: " ... ACTION PERFORMED BY VAE03251 ON 03.AUG.2025, 09:15 ..." -> xoá " BY VAE03251", " ON 03.AUG.2025", " 09:15"
_INLINE_META_SUBS = [
    (re.compile(r"\s+BY\s+[A-Z]{3}\d{5}\b", re.IGNORECASE), ""),                       # BY VAE03251
    (re.compile(r"\s+ON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b", re.IGNORECASE), ""),            # ON 03.AUG.2025
    (re.compile(r"\s+\d{1,2}:\d{2}\b"), ""),                                           # 09:15
    (re.compile(r"\s+DESCRIPTION\s+SIGN\s+[A-Z]{3}\d+\b", re.IGNORECASE), ""),         # DESCRIPTION SIGN VAE03251
    (re.compile(r"\s+ACTION\s+PERFORMED\s+BY\s+[A-Z]{3}\d+\b", re.IGNORECASE), ""),    # ACTION PERFORMED BY VAE03251
    (re.compile(r"\s+PERFORMED\s+SIGN\s+[A-Z]{3}\d+\b", re.IGNORECASE), ""),           # PERFORMED SIGN VAE03251
]

_WS_RE = re.compile(r"[ \t]+")


def _has_keep_hint(line: str) -> bool:
    """Dòng có chứa tín hiệu kỹ thuật?"""
    return bool(_KEEP_HINT.search(line or ""))


def _is_pure_meta_line(line: str) -> bool:
    """
    Dòng thuần META (nhật ký/chữ ký/mốc thời gian)?
    Quy tắc: nếu CÓ hint kỹ thuật → KHÔNG coi là META (giữ lại).
    """
    if _has_keep_hint(line):
        return False
    return bool(_PURE_META_RE.match(line or ""))


def _strip_inline_meta(line: str) -> str:
    """Xoá các mẩu metadata nằm trong lòng một dòng, giữ phần kỹ thuật."""
    out = line
    for pat, repl in _INLINE_META_SUBS:
        out = pat.sub(repl, out)
    return out.strip()


def _normalize_spaces(s: str) -> str:
    """Gộp nhiều khoảng trắng thành một và tỉa hai đầu."""
    if not s:
        return ""
    s = _WS_RE.sub(" ", s)
    return s.strip()


def clean_text_block(s: str) -> str:
    """
    Làm sạch khối văn bản WO:
      - Tách theo dòng; bỏ dòng rỗng.
      - Nếu dòng có hint kỹ thuật → GIỮ.
      - Nếu dòng là META thuần (không có hint) → BỎ.
      - Các dòng còn lại: xoá META "inline" (BY/ON/time/…); giữ phần còn lại nếu đủ nghĩa.
      - Ghép về một chuỗi và chuẩn hoá khoảng trắng.
    """
    if not s:
        return ""

    # Chuẩn hoá xuống dòng
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [ln.strip() for ln in re.split(r"\n+", s)]
    lines = [ln for ln in raw_lines if ln]

    kept: list[str] = []
    for ln in lines:
        # 1) Giữ nguyên nếu có tín hiệu kỹ thuật
        if _has_keep_hint(ln):
            kept.append(ln)
            continue

        # 2) Loại nếu là META thuần
        if _is_pure_meta_line(ln):
            continue

        # 3) Còn lại: xoá inline meta và giữ nếu còn nội dung có nghĩa
        cleaned = _strip_inline_meta(ln)
        if cleaned and len(cleaned) > 5:
            kept.append(cleaned)

    return _normalize_spaces(" ".join(kept))


def clean_wo_text(s: str) -> str:
    """
    Alias cho clean_text_block để tương thích phiên bản mới.
    """
    return clean_text_block(s)


# ====================== Tự kiểm thử cơ bản ======================
if __name__ == "__main__":
    test_wo = """
    1 WORKSTEP ADDED BY VAE03251 ON 01.AUG.2025, 06:30
    FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1
    FOUND AIR OUTLET AT F/O (FIN: 4501HM) AND THIRD OCCUPANT INDIVIDUAL AIR-OUTLET (FIN 4503HM) DAMAGED (MISSING FLAP)
    DESCRIPTION SIGN VAE03251
    ACTION PERFORMED BY VAE03251 ON 03.AUG.2025, 09:15
    FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1
    C/O REPLACE INDIVIDUAL AIR-OUTLET AT F/O SIDE
    REF AMM 21-21-44-000-001-A (AUG 01 2025)
    AMM 21-21-44-400-001-A (AUG 01 2025)
    PERFORMED SIGN VAE03251
    """
    cleaned = clean_text_block(test_wo)
    print("=== CLEANED OUTPUT ===")
    print(cleaned)
    print("\n=== CHECKS ===")
    print("AMM refs:", "✓" if "AMM 21-21-44" in cleaned else "✗")
    print("FOUND/DAMAGED:", "✓" if ("FOUND" in cleaned and "DAMAGED" in cleaned) else "✗")
    print("REPLACE:", "✓" if "REPLACE" in cleaned else "✗")
