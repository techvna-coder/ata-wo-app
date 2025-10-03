# core/nondefect.py
from __future__ import annotations

import regex as re
from typing import Tuple, Dict, Any

# ============================================================
# 1) TỪ KHÓA / MẪU NHẬN DIỆN
# ============================================================

# a) Dấu hiệu hỏng hóc mạnh (chỉ cần xuất hiện là nghiêng về DEFECT)
DEFECT_STRONG = [
    r"\bfail(ure|ed)?\b", r"\bfault(s)?\b", r"\bmalfunction\b", r"\bdefect(s)?\b",
    r"\bleak(s|ing)?\b", r"\boverheat(ing)?\b", r"\bvibrat(e|ion|ing)\b", r"\bsmoke\b",
    r"\bwarning\b", r"\bcaution\b", r"\badvisory\b",
    r"\bintermittent\b", r"\bspurious\b", r"\bno\s*go\b", r"\binop(erative)?\b",
    r"\bdegrad(ed|ation)\b", r"\bshort( circuit)?\b", r"\btrip(ped)?\b",
    # Mã/cảnh báo hệ thống:
    r"\bECAM\b", r"\bEICAS\b", r"\bCAS\b", r"\bACARS\b", r"\bFWS\b",
]

# b) Hành động khắc phục/can thiệp kỹ thuật (nếu có → xu hướng DEFECT)
CORRECTIVE_ACTIONS = [
    r"\breplace(d|ment)?\b", r"\breplaced\b", r"\binstall(ed|ation)?\b", r"\bremove(d|al)?\b",
    r"\brepair(ed|ing)?\b", r"\brectif(y|ied|ication)\b", r"\btroubleshoot(ing)?\b",
    r"\badjust(ed|ment)?\b", r"\bmodif(y|ication|ied)\b", r"\bcalibrat(e|ion|ed)\b",
    r"\bclean(ed|ing)\b(?=.*(contaminat|oil|fluid|debris|filt|connector|sensor))",
    r"\bservic(e|ing)\b(?=.*(hyd|oxy|fuel|oil|fluid|water))",
    r"\bSB\b", r"\bService Bulletin\b", r"\bAD\b", r"\bMEL\b", r"\bdeferr(al|ed)\b",
    r"\bNFF\b(?=.*then.*replace)",  # ít gặp nhưng bảo thủ
]

# c) Hoạt động routine/servicing/inspection (nghiêng về NON-DEFECT nếu không có dấu hiệu hỏng hóc mạnh)
ROUTINE_PATTERNS = [
    r"\b(clean(ing)?|lubricat(e|ion|ed)|servic(e|ing)|disinfection|interior clean|cabin tidy)\b",
    r"\b(remove|install)\b(?=.*(cover|cap|plug|pin|tag|blanket|protector))",  # lắp/tháo bảo vệ
    r"\bperiodic\b|\bscheduled\b|\broutine\b|\bA-check\b|\bC-check\b|\bIL\b|\b4D\b",
    r"\binspect(ion|ed|ing)?\b(?!.*(found|leak|fault|fail|damage|crack|burn|smoke))",
    r"\bops(\.|eration(al)?)?\s*check(s|ed)?\s*(ok|good|sat(isfactory)?)\b",
    r"\bleak\s*check\s*(ok|good|sat(isfactory)?)\b(?!.*(before|prior))",
    r"\bfunctional\s*check\s*(ok|good|sat(isfactory)?)\b",
    r"\bno\s*(abnormalit(y|ies)|defect|fault|finding|discrepanc(y|ies))\b",
    r"\bfound\s*(serviceable|normal)\b",
]

# d) “Meta / Audit trail” – nên bỏ qua khi đánh giá
META_PATTERNS = [
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*", r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",         # S.O-A321-3348-2019-HEAT
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",             # VAE03251
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",  # 05.AUG.2025
    r"\b\d{1,2}[:.]\d{2}\b.*",               # 09:24, 11:22
    r"^\s*\d+\s*WORKSTEP\S*\b.*",
]

# e) Tham chiếu tài liệu – giữ lại để nguồn khác khai thác, nhưng không dùng phân loại đơn thuần
DOC_CITATIONS = [
    r"\b(AMM|TSM|FIM)\b\s*[-:]?\s*\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
]

# Biên dịch regex
_re_defect_strong = re.compile("|".join(DEFECT_STRONG), re.I)
_re_corrective     = re.compile("|".join(CORRECTIVE_ACTIONS), re.I)
_re_routine        = re.compile("|".join(ROUTINE_PATTERNS), re.I)
_re_meta           = re.compile("|".join(META_PATTERNS), re.I)
_re_docs           = re.compile("|".join(DOC_CITATIONS), re.I)

# ============================================================
# 2) HỖ TRỢ LÀM SẠCH NHẸ
# ============================================================

_ws = re.compile(r"[ \t]+")

def _strip_meta(s: str) -> str:
    """Loại meta/audit trail nhưng giữ lại dòng có tín hiệu kỹ thuật."""
    if not s:
        return ""
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    for ln in lines:
        if not ln:
            continue
        # nếu có tín hiệu kỹ thuật hoặc citation → giữ
        if _re_defect_strong.search(ln) or _re_docs.search(ln):
            kept.append(ln)
            continue
        # nếu là meta → bỏ
        if _re_meta.search(ln):
            continue
        kept.append(ln)
    out = " ".join(kept)
    return _ws.sub(" ", out).strip()

# ============================================================
# 3) PHÂN LOẠI
# ============================================================

def _score_patterns(text: str, rx: re.Pattern) -> int:
    return len(list(rx.finditer(text))) if text and rx else 0

def classify_nondefect_reason(desc: str | None, action: str | None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Trả về (is_defect, reason, details).
    Quy tắc:
      1) Có symptom/cảnh báo mạnh → DEFECT.
      2) Nếu có ACTION khắc phục thực sự → DEFECT (dù có 'ops/leak check sat' hậu kiểm).
      3) Nếu KHÔNG có (1)(2) và phù hợp routine/inspection/no-finding → NON-DEFECT.
      4) Mặc định thiên về DEFECT nếu còn mơ hồ (để không bỏ sót).
    """
    raw_text = f"{desc or ''} {action or ''}".strip()
    text = _strip_meta(raw_text)

    if not text:
        return False, "Thiếu dữ liệu mô tả/hành động.", {"raw": raw_text}

    # 1) Dấu hiệu hỏng hóc mạnh
    strong_hits = _score_patterns(text, _re_defect_strong)
    if strong_hits > 0:
        return True, "Phát hiện symptom/cảnh báo mạnh (fault/leak/ECAM/EICAS/CAS…).", {
            "strong_hits": strong_hits
        }

    # 2) Hành động khắc phục
    corrective_hits = _score_patterns(text, _re_corrective)
    if corrective_hits > 0:
        return True, "Có hành động khắc phục thực sự (replace/repair/adjust/rectify…).", {
            "corrective_hits": corrective_hits
        }

    # 3) Routine / Inspection / No-finding
    routine_hits = _score_patterns(text, _re_routine)
    if routine_hits > 0:
        return False, "Routine/inspection/no-finding – không có symptom/hành động khắc phục.", {
            "routine_hits": routine_hits
        }

    # 4) Mặc định nghiêng về DEFECT (thận trọng)
    return True, "Không có keyword routine/no-finding; ưu tiên thận trọng coi là defect.", {}


def is_technical_defect(desc: str | None, action: str | None) -> bool:
    """API rút gọn cho các chỗ gọi hiện tại."""
    is_def, _, _ = classify_nondefect_reason(desc, action)
    return is_def
