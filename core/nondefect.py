# core/nondefect.py
from __future__ import annotations

import regex as re

# ------------------------------------------------------------
# Nguồn mặc định: constants (nếu có), sau đó cộng thêm bộ mở rộng
# ------------------------------------------------------------
try:
    # Nếu đã khai báo trong core/constants.py, ta lấy làm nền
    from .constants import NONDEFECT_PATTERNS as CONST_NONDEFECT
    from .constants import DEFECT_KEYWORDS as CONST_DEFECT
except Exception:
    CONST_NONDEFECT = []
    CONST_DEFECT = []

# Bổ sung các cụm "meta/audit" thường gặp trong WO đã làm sạch nhưng không phải hỏng hóc kỹ thuật
EXTRA_NONDEFECT = [
    # Hoạt động khai thác/dịch vụ không phải defect
    r"\b(clean(ing)?|lubrication|servic(e|ing)|cabin tidy|interior clean|disinfection|water service|lavatory service)\b",
    r"\b(first aid kit|tyre wear|tire wear|scheduled (check|maintenance)|software (load|upgrade|update))\b",
    r"\b(gse|ground support|paint touch(-|\s)?up|refuel(l)?ing|defuel(l)?ing|galley restock)\b",

    # Dấu vết tác nghiệp/nhật ký (không mang ý nghĩa kỹ thuật)
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b",
    r"\bFINDING\s*\(NRC\)\b",
    r"\bPART\s+REQUIREMENT\b",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b",   # ví dụ: S.O-A321-3348-2019-HEAT
    r"\bBY\s+[A-Z]{3}\d{5}\b",       # mã nhân sự dạng VAE03251
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b",  # 05.AUG.2025
    r"\b\d{1,2}[:.]\d{2}\b",               # 09:24, 11:22
    r"^\s*\d+\s*WORKSTEP\S*\b",
]

# Tăng cường các từ khoá “có khả năng là hỏng hóc kỹ thuật”
EXTRA_DEFECT = [
    r"\bfail(ure)?\b",
    r"\bfault\b",
    r"\bleak\b",
    r"\boverheat(ing)?\b",
    r"\bvibrat(ion|e|ing)?\b",
    r"\bsmoke\b",
    r"\bwarning\b",
    r"\bcaution\b",
    r"\badvisory\b",
    r"\bECAM\b",
    r"\bEICAS\b",
    r"\bCAS\b",
    r"\bmsg\b",            # message
    r"\bno\s*go\b",
    r"\binop\b",           # inoperative
    r"\bdegrad(ed|ation)\b",
    r"\bintermittent\b",
    r"\bshort\b",          # electrical short
    r"\btrip(ped)?\b",     # cb tripped
    r"\bspurious\b",
    r"\bdiscrepancy\b",
]

# Hợp nhất (không trùng lặp)
NONDEFECT_PATTERNS = list({*CONST_NONDEFECT, *EXTRA_NONDEFECT})
DEFECT_KEYWORDS = list({*CONST_DEFECT, *EXTRA_DEFECT})

# Biên dịch regex
_nondef = re.compile("|".join(NONDEFECT_PATTERNS), flags=re.I) if NONDEFECT_PATTERNS else None
_defect = re.compile("|".join(DEFECT_KEYWORDS), flags=re.I) if DEFECT_KEYWORDS else None


def is_technical_defect(desc: str | None, action: str | None) -> bool:
    """
    Phân loại nhanh một WO có phải “hỏng hóc kỹ thuật” hay không.

    Quy tắc:
    1) Nếu có bất kỳ từ khoá hỏng hóc (DEFECT_KEYWORDS) → True.
    2) Nếu KHÔNG có từ khoá hỏng hóc và có khớp NONDEFECT_PATTERNS (dịch vụ, meta/audit) → False.
    3) Nếu thiếu thông tin → False; còn lại → True thận trọng.

    Lưu ý: Bộ lọc meta/audit (WORKSTEP..., NRC...) không phủ định khi trong dòng vẫn có từ khoá hỏng hóc.
    """
    text = f"{desc or ''} {action or ''}".strip()
    if not text:
        return False

    # Ưu tiên phát hiện “dấu hiệu kỹ thuật” trước
    if _defect and _defect.search(text):
        return True

    # Nếu không có dấu hiệu kỹ thuật mà có pattern phi kỹ thuật → loại
    if _nondef and _nondef.search(text):
        return False

    # Mặc định: thận trọng coi là kỹ thuật (để không bỏ sót)
    return True
