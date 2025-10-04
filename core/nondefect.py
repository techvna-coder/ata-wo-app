# core/nondefect.py
from __future__ import annotations

import regex as re
from typing import Tuple, Dict, Any, Optional

# ============================================================
# 1) TỪ KHÓA / MẪU NHẬN DIỆN
# ============================================================

# a) Dấu hiệu hỏng hóc mạnh (không gồm từ "defect" chung chung)
DEFECT_STRONG = [
    r"\bfail(ure|ed)?\b",
    r"\bfault(s)?\b",
    r"\bmalfunction\b",
    r"\bleak(s|ing)?\b",
    r"\boverheat(ing)?\b",
    r"\bvibrat(e|ion|ing)\b",
    r"\bsmoke\b",
    r"\bwarning\b",
    r"\bcaution\b",
    r"\badvisory\b",
    r"\bintermittent\b",
    r"\bspurious\b",
    r"\bno\s*go\b",
    r"\binop(erative)?\b",
    r"\bdegrad(ed|ation)\b",
    r"\bshort( circuit)?\b",
    r"\btrip(ped)?\b",
    # Mã/cảnh báo hệ thống:
    r"\bECAM\b",
    r"\bEICAS\b",
    r"\bCAS\b",
    r"\bACARS\b",
    r"\bFWS\b",
]

# b) "defect" chỉ tính khi có NGỮ CẢNH đi kèm found/detected/observed/noted/occurred/present
DEFECT_WITH_CONTEXT = [
    r"\bdefect(s)?\b(?=.*\b(found|detected|observed|reported|noted|occurred|present)\b)",
]

# c) Hành động khắc phục/can thiệp kỹ thuật (nghiêng về DEFECT)
CORRECTIVE_ACTIONS = [
    r"\breplace(d|ment)?\b",
    r"\breplaced\b",
    r"\binstall(ed|ation)?\b",
    r"\bremove(d|al)?\b",
    r"\brepair(ed|ing)?\b",
    r"\brectif(y|ied|ication)\b",
    r"\btroubleshoot(ing)?\b",
    r"\badjust(ed|ment)?\b",
    r"\bmodif(y|ication|ied)\b",
    r"\bcalibrat(e|ion|ed)\b",
    r"\bSB\b",
    r"\bService Bulletin\b",
    r"\bAD\b",
    r"\bMEL\b",
    r"\bdeferr(al|ed)\b",
]

# d) Hoạt động định kỳ/routine (không phát hiện hỏng hóc) → NON-DEFECT
ROUTINE_PATTERNS = [
    # cleaning/servicing chung
    r"\b(clean(ing)?|lubricat(e|ion|ed)|servic(e|ing)|disinfection|interior clean|cabin tidy)\b",

    # scheduled/routine/check loại hình
    r"\b(periodic|scheduled|routine)\b",
    r"\b(A-?check|C-?check|IL|4D|transit\s*check|daily\s*check|weekly\s*check|preflight\s*check|walkaround)\b",

    # inspection loại hình (+ biến thể thuật ngữ)
    r"\b(general|detailed|special)\s+visual\s+inspection\b",   # GVI/DVI/SDI
    r"\b(zonal|zone)\s+(visual\s+)?inspection\b",              # ZVI/ZI
    r"\b(boroscope|borescope)\s+inspection\b",
    r"\bvisual\s+inspection\b",                                # fallback

    # form diễn đạt inspection/check/ops/functional/leak
    r"\binspect(ion|ed|ing)?\b(?!.*\b(found|leak|fault|fail|damage|crack|burn|smoke)\b)",
    r"\bperform\s+check(s)?\b",
    r"\b(checks?|ops(\.|eration(al)?)?\s*(test|check)|functional\s*check|leak\s*check)\b\s*(ok|good|sat(isfactory)?)?\b",
    r"\bno\s*(abnormalit(y|ies)|defect|fault|finding|discrepanc(y|ies))\b",
    r"\bfound\s*(serviceable|normal)\b",

    # gửi đi kiểm tra/overhaul theo lịch
    r"\b(send|sent)\s+for\s+(inspection|bench\s*test|overhaul|shop\s*visit|nrc)\b",
    r"\bremove(d)?\s+for\s+(inspection|overhaul|bench\s*test)\b",

    # tham chiếu AMM inspection (05-xx/20x/200-xxx thường là task inspection)
    r"\bref(er)?\s+amm\b.*\b(05|06|20[0-9]|200)\b[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
    r"\bamm\s+task\b",
    r"\bamm\b.*\b200[- ]?\d{3}\b",  # AMM xx-xx-200-xxx

    # cụm câu điều kiện/routine hay gặp
    r"\b(checks\s+scheduled|make\s+sure|refer\s+to)\b",
]

# e) Cosmetic → NON-DEFECT
COSMETIC_PATTERNS = [
    r"\bpaint\s*(peel|chip|flake|touch[- ]?up|scratch|flake)\b",
    r"\b(name\s*plate|placard|decal|stencil|label|marking)\s*(miss(ing)?|faded|worn|illegible)\b",
    r"\bcosmetic\b",
    r"\bappearance\b",
]

# f) Hao mòn tự nhiên → NON-DEFECT
NATURAL_WEAR_PATTERNS = [
    r"\b(tyre|tire)\s*(wear|worn|tread|scuff)\b",
    r"\bbrake\s*wear\b",
    r"\berosion\b",
    r"\bweather(ing)?\b",
]

# g) Meta/Audit trail – mở rộng để bỏ qua NOTE/FILE LOCATED/DIRECTORY & audit trail
META_PATTERNS = [
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*",
    r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",
    r"^NOTE:\b.*",
    r"^FILE\s+LOCATED\b.*",
    r"^(DIRECTORY|DIRECTORY://)\b.*",
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",
    r"\b\d{1,2}[:.]\d{2}\b.*",
    r"^\s*\d+\s*WORKSTEP\S*\b.*",
]

# h) Tham chiếu tài liệu – giữ lại (không quyết định defect/non-defect)
DOC_CITATIONS = [
    r"\b(AMM|TSM|FIM)\b\s*[-:]?\s*\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
]

# i) Nhận diện Type scheduled
SCHEDULED_TYPE = [
    r"\bSCHEDULED\b",
    r"\bSCHEDULED\s*W/?O\b",
    r"\bROUTINE\b",
    r"\bCHECK\b",
    r"\bA-?CHECK\b",
    r"\bC-?CHECK\b",
]

# ------------------------------------------------------------
# Compile regex
# ------------------------------------------------------------
_re_defect_strong   = re.compile("|".join(DEFECT_STRONG), re.I)
_re_defect_context  = re.compile("|".join(DEFECT_WITH_CONTEXT), re.I)
_re_corrective      = re.compile("|".join(CORRECTIVE_ACTIONS), re.I)
_re_routine         = re.compile("|".join(ROUTINE_PATTERNS), re.I)
_re_cosmetic        = re.compile("|".join(COSMETIC_PATTERNS), re.I)
_re_wear            = re.compile("|".join(NATURAL_WEAR_PATTERNS), re.I)
_re_meta            = re.compile("|".join(META_PATTERNS), re.I | re.M)
_re_docs            = re.compile("|".join(DOC_CITATIONS), re.I)
_re_scheduled_type  = re.compile("|".join(SCHEDULED_TYPE), re.I)

_ws = re.compile(r"[ \t]+")

# ============================================================
# 2) LÀM SẠCH NHẸ – loại meta/audit, giữ symptom/citation
# ============================================================
def _strip_meta(s: str) -> str:
    if not s:
        return ""
    # Tách dòng, bỏ các dòng meta; nếu dòng meta lại chứa symptom/citation thì giữ
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    for ln in lines:
        if not ln:
            continue
        if _re_meta.search(ln):
            if _re_defect_strong.search(ln) or _re_docs.search(ln):
                kept.append(ln)
            continue
        kept.append(ln)
    out = " ".join(kept)
    return _ws.sub(" ", out).strip()

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
        # Có thể chỉnh về True nếu muốn cực kỳ thận trọng
        return False, "Thiếu dữ liệu mô tả/hành động.", {"raw_empty": True}

    # 1) Symptom/cảnh báo mạnh (KHÔNG tính 'defect' chung chung)
    strong = _count(_re_defect_strong, text)
    if strong > 0:
        return True, "Symptom/cảnh báo mạnh (fault/failure/leak/ECAM/EICAS/CAS…).", {"strong_hits": strong}

    # 2) Nếu Type = SCHEDULED và nội dung routine → NON-DEFECT
    is_scheduled = bool(_re_scheduled_type.search(wo_type or "")) if wo_type else False
    routine_hits = _count(_re_routine, text)
    if is_scheduled and routine_hits > 0:
        return False, "WO định kỳ (Type=SCHEDULED) với nội dung routine/inspection/ops test/overhaul, không có symptom.", {
            "scheduled": True, "routine_hits": routine_hits
        }

    # 3) Cosmetic → NON-DEFECT
    cosmetic_hits = _count(_re_cosmetic, text)
    if cosmetic_hits > 0:
        return False, "Cosmetic (paint/nameplate/placard/label…), không ảnh hưởng độ tin cậy.", {
            "cosmetic_hits": cosmetic_hits
        }

    # 4) Hao mòn tự nhiên → NON-DEFECT
    wear_hits = _count(_re_wear, text)
    if wear_hits > 0:
        return False, "Hao mòn tự nhiên (tyre/brake wear/erosion/weathering) – không phải hỏng hóc hệ thống.", {
            "wear_hits": wear_hits
        }

    # 5) Hành động khắc phục → DEFECT
    corr_hits = _count(_re_corrective, text)
    if corr_hits > 0:
        # Ngoại lệ: scheduled + remove/send for overhaul dạng routine (không symptom)
        if is_scheduled and routine_hits > 0:
            return False, "Tháo/lắp/gửi đi inspection/overhaul theo lịch – không có symptom.", {
                "scheduled": True, "routine_hits": routine_hits, "corrective_hits": corr_hits
            }
        return True, "Có hành động khắc phục (repair/replace/adjust/rectify/troubleshoot…).", {
            "corrective_hits": corr_hits
        }

    # 6) "defect" có NGỮ CẢNH (found/detected/observed/noted/...) → DEFECT
    defect_ctx = _count(_re_defect_context, text)
    if defect_ctx > 0:
        return True, "Có 'defect' với ngữ cảnh (found/detected/observed/noted…).", {
            "defect_with_context": defect_ctx
        }

    # 7) Routine/no-finding nói chung → NON-DEFECT
    if routine_hits > 0:
        return False, "Routine/inspection/no-finding – không có symptom/hành động khắc phục.", {
            "routine_hits": routine_hits
        }

    # 8) Mặc định: DEFECT (thận trọng)
    return True, "Không khớp routine/cosmetic/wear; mặc định coi là defect để bảo toàn độ tin cậy.", {}

def is_technical_defect(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None
) -> bool:
    is_def, _, _ = classify_nondefect_reason(desc, action, wo_type=wo_type)
    return is_def
