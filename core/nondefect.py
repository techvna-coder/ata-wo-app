# core/nondefect.py
from __future__ import annotations

import regex as re
from typing import Tuple, Dict, Any, Optional

# ============================================================
# 1) TỪ KHÓA / MẪU NHẬN DIỆN
# ============================================================

# a) Dấu hiệu hỏng hóc mạnh → DEFECT ngay khi xuất hiện
DEFECT_STRONG = [
    r"\bfail(ure|ed)?\b", r"\bfault(s)?\b", r"\bmalfunction\b", r"\bdefect(s)?\b",
    r"\bleak(s|ing)?\b", r"\boverheat(ing)?\b", r"\bvibrat(e|ion|ing)\b", r"\bsmoke\b",
    r"\bwarning\b", r"\bcaution\b", r"\badvisory\b",
    r"\bintermittent\b", r"\bspurious\b", r"\bno\s*go\b", r"\binop(erative)?\b",
    r"\bdegrad(ed|ation)\b", r"\bshort( circuit)?\b", r"\btrip(ped)?\b",
    # Mã/cảnh báo hệ thống:
    r"\bECAM\b", r"\bEICAS\b", r"\bCAS\b", r"\bACARS\b", r"\bFWS\b",
]

# b) Hành động khắc phục/can thiệp kỹ thuật → DEFECT
CORRECTIVE_ACTIONS = [
    r"\breplace(d|ment)?\b", r"\breplaced\b", r"\binstall(ed|ation)?\b", r"\bremove(d|al)?\b",
    r"\brepair(ed|ing)?\b", r"\brectif(y|ied|ication)\b", r"\btroubleshoot(ing)?\b",
    r"\badjust(ed|ment)?\b", r"\bmodif(y|ication|ied)\b", r"\bcalibrat(e|ion|ed)\b",
    r"\bSB\b", r"\bService Bulletin\b", r"\bAD\b", r"\bMEL\b", r"\bdeferr(al|ed)\b",
]

# c) Hoạt động routine/định kỳ (không phát hiện hỏng hóc) → NON-DEFECT
ROUTINE_PATTERNS = [
    r"\b(clean(ing)?|lubricat(e|ion|ed)|servic(e|ing)|disinfection|interior clean|cabin tidy)\b",
    r"\bperiodic\b|\bscheduled\b|\broutine\b|\bA-check\b|\bC-check\b|\bIL\b|\b4D\b",
    r"\binspect(ion|ed|ing)?\b(?!.*(found|leak|fault|fail|damage|crack|burn|smoke))",
    r"\bops(\.|eration(al)?)?\s*test(s|ed)?\b(?!.*(fail|fault|abnormal))",
    r"\bops(\.|eration(al)?)?\s*check(s|ed)?\s*(ok|good|sat(isfactory)?)\b",
    r"\bleak\s*check\s*(ok|good|sat(isfactory)?)\b(?!.*(before|prior))",
    r"\bfunctional\s*check\s*(ok|good|sat(isfactory)?)\b",
    r"\bno\s*(abnormalit(y|ies)|defect|fault|finding|discrepanc(y|ies))\b",
    r"\bfound\s*(serviceable|normal)\b",
    # tháo/gắn để gửi đi kiểm tra/overhaul theo lịch
    r"\b(send|sent)\s+for\s+(inspection|bench\s*test|overhaul|shop\s*visit)\b",
    r"\bremove(d)?\s+for\s+(inspection|overhaul|bench\s*test)\b",
]

# d) Cosmetic → NON-DEFECT (không ảnh hưởng độ tin cậy)
COSMETIC_PATTERNS = [
    r"\bpaint\s*(peel|chip|flake|touch[- ]?up)\b",
    r"\b(name\s*plate|placard|decal|stencil)\s*(miss(ing)?|faded|worn)\b",
    r"\bcosmetic\b", r"\bappearance\b",
]

# e) Hao mòn tự nhiên → NON-DEFECT (khi không kèm symptom lỗi)
NATURAL_WEAR_PATTERNS = [
    r"\b(tyre|tire)\s*(wear|worn|tread)\b",
    r"\bbrake\s*wear\b",
    r"\berosion\b",
]

# f) Meta/Audit trail – loại khỏi ngữ cảnh (không dùng để phân loại)
META_PATTERNS = [
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*", r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",
    r"\b\d{1,2}[:.]\d{2}\b.*",
    r"^\s*\d+\s*WORKSTEP\S*\b.*",
]

# g) Tham chiếu tài liệu – giữ lại (không quyết định defect/non-defect)
DOC_CITATIONS = [
    r"\b(AMM|TSM|FIM)\b\s*[-:]?\s*\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
]

# h) Nhận diện Type = scheduled
SCHEDULED_TYPE = [
    r"\bSCHEDULED\b", r"\bSCHEDULED\s*W/?O\b", r"\bROUTINE\b", r"\bCHECK\b",
    r"\bA-?CHECK\b", r"\bC-?CHECK\b",
]

# ------------------------------------------------------------
# Compile regex
# ------------------------------------------------------------
_re_defect_strong = re.compile("|".join(DEFECT_STRONG), re.I)
_re_corrective     = re.compile("|".join(CORRECTIVE_ACTIONS), re.I)
_re_routine        = re.compile("|".join(ROUTINE_PATTERNS), re.I)
_re_cosmetic       = re.compile("|".join(COSMETIC_PATTERNS), re.I)
_re_wear           = re.compile("|".join(NATURAL_WEAR_PATTERNS), re.I)
_re_meta           = re.compile("|".join(META_PATTERNS), re.I)
_re_docs           = re.compile("|".join(DOC_CITATIONS), re.I)
_re_scheduled_type = re.compile("|".join(SCHEDULED_TYPE), re.I)

_ws = re.compile(r"[ \t]+")

# ============================================================
# 2) LÀM SẠCH NHẸ – loại meta/audit, giữ symptom/citation
# ============================================================
def _strip_meta(s: str) -> str:
    if not s:
        return ""
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    for ln in lines:
        if not ln:
            continue
        if _re_defect_strong.search(ln) or _re_docs.search(ln):
            kept.append(ln); continue
        if _re_meta.search(ln):
            continue
        kept.append(ln)
    out = " ".join(kept)
    return _ws.sub(" ", out).strip()

def _has(rx: re.Pattern, text: str) -> bool:
    return bool(rx.search(text)) if text and rx else False

def _count(rx: re.Pattern, text: str) -> int:
    return len(list(rx.finditer(text))) if text and rx else 0

# ============================================================
# 3) PHÂN LOẠI CHI TIẾT
# ============================================================
def classify_nondefect_reason(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Trả về (is_defect, reason, details)
      - is_defect: True → liên quan độ tin cậy hệ thống/thiết bị; False → non-defect
      - reason   : lý do ngắn gọn
      - details  : số lần trúng pattern để debug
    """
    raw_text = f"{desc or ''} {action or ''}".strip()
    text = _strip_meta(raw_text)

    if not text:
        # không có thông tin → coi non-defect để tránh nhiễu thống kê (có thể đổi thành True nếu muốn thận trọng hơn)
        return False, "Thiếu dữ liệu mô tả/hành động.", {"raw_empty": True}

    # 1) symptom/cảnh báo mạnh → DEFECT
    strong = _count(_re_defect_strong, text)
    if strong > 0:
        return True, "Symptom/cảnh báo mạnh (fault/failure/leak/ECAM/EICAS/CAS…).", {"strong_hits": strong}

    # 2) nếu Type là SCHEDULED/định kỳ và nội dung là routine/ops test/inspection/send for overhaul mà không có symptom → NON-DEFECT
    is_scheduled = bool(_re_scheduled_type.search(wo_type or "")) if wo_type else False
    routine_hits = _count(_re_routine, text)
    if is_scheduled and routine_hits > 0:
        return False, "WO định kỳ (Type=SCHEDULED) với nội dung routine/inspection/ops test/overhaul không có symptom.", {
            "scheduled": True, "routine_hits": routine_hits
        }

    # 3) cosmetic → NON-DEFECT khi không có symptom
    cosmetic_hits = _count(_re_cosmetic, text)
    if cosmetic_hits > 0:
        return False, "Cosmetic (paint/nameplate/placard…), không ảnh hưởng độ tin cậy.", {
            "cosmetic_hits": cosmetic_hits
        }

    # 4) hao mòn tự nhiên → NON-DEFECT khi không có symptom
    wear_hits = _count(_re_wear, text)
    if wear_hits > 0:
        return False, "Hao mòn tự nhiên (tyre/brake wear/erosion) – không phải hỏng hóc hệ thống.", {
            "wear_hits": wear_hits
        }

    # 5) hành động khắc phục thực sự → DEFECT,
    #    ngoại lệ: nếu là scheduled và câu mô tả là remove/send for overhaul/inspection (định kỳ) → NON-DEFECT
    corr_hits = _count(_re_corrective, text)
    if corr_hits > 0:
        if is_scheduled and routine_hits > 0:
            return False, "Tháo/lắp/phục vụ gửi đi inspection/overhaul theo lịch – không có symptom.", {
                "scheduled": True, "routine_hits": routine_hits, "corrective_hits": corr_hits
            }
        return True, "Có hành động khắc phục (repair/replace/adjust/rectify/troubleshoot…).", {
            "corrective_hits": corr_hits
        }

    # 6) routine/no-finding nói chung → NON-DEFECT
    if routine_hits > 0:
        return False, "Routine/inspection/no-finding – không có symptom/hành động khắc phục.", {
            "routine_hits": routine_hits
        }

    # 7) mặc định: coi là DEFECT (thận trọng để không bỏ sót case ảnh hưởng độ tin cậy)
    return True, "Không khớp routine/cosmetic/wear; mặc định coi là defect để bảo toàn độ tin cậy.", {}

def is_technical_defect(desc: Optional[str], action: Optional[str], wo_type: Optional[str] = None) -> bool:
    is_def, _, _ = classify_nondefect_reason(desc, action, wo_type=wo_type)
    return is_def
