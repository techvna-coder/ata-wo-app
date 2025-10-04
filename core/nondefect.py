# core/nondefect.py
from __future__ import annotations

import regex as re
from typing import Tuple, Dict, Any, Optional

# ============================================================
# 1) PATTERNS
# ============================================================

# Symptom/cảnh báo mạnh → DEFECT (không gồm "defect" chung chung)
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
    r"\bECAM\b",
    r"\bEICAS\b",
    r"\bCAS\b",
    r"\bACARS\b",
    r"\bFWS\b",
]

# "defect/damage" chỉ tính khi có NGỮ CẢNH found/detected/observed/noted/occurred/present
DEFECT_WITH_CONTEXT = [
    r"\b(defect(s)?|damage(d)?)\b(?=.*\b(found|detected|observed|reported|noted|occurred|present)\b)",
]

# Hành động khắc phục/can thiệp kỹ thuật → DEFECT (trừ remove/send for overhaul theo lịch)
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

# Routine/định kỳ → NON-DEFECT khi không có symptom/defect-with-context
ROUTINE_PATTERNS = [
    # cleaning/servicing
    r"\b(clean(ing)?|lubricat(e|ion|ed)|servic(e|ing)|disinfection|interior clean|cabin tidy)\b",

    # scheduled kinds
    r"\b(periodic|scheduled|routine)\b",
    r"\b(A-?check|C-?check|IL|4D|transit\s*check|daily\s*check|weekly\s*check|preflight\s*check|walkaround)\b",

    # inspections
    r"\b(general|detailed|special)\s+visual\s+inspection\b",   # GVI/DVI/SDI
    r"\b(zonal|zone)\s+(visual\s+)?inspection\b",              # ZVI/ZI
    r"\b(boroscope|borescope)\s+inspection\b",
    r"\bvisual\s+inspection\b",

    # forms: inspection/check/ops/functional/leak (cho phép "check satisf/ok")
    r"\binspect(ion|ed|ing)?\b",
    r"\bperform\s+check(s)?\b",
    r"\b(checks?|ops(\.|eration(al)?)?\s*(test|check)|functional\s*check|leak\s*check)\b(\s*(ok|good|sat(isfactory)?)\b)?",
    r"\bno\s*(abnormalit(y|ies)|defect|fault|finding|discrepanc(y|ies))\b",
    r"\bfound\s*(serviceable|normal)\b",

    # send/remove for scheduled shop visit/overhaul/bench test
    r"\b(send|sent)\s+for\s+(inspection|bench\s*test|overhaul|shop\s*visit|nrc)\b",
    r"\bremove(d)?\s+for\s+(inspection|overhaul|bench\s*test)\b",

    # AMM inspection references (xx-xx-200-xxx thường là inspection)
    r"\bref(er)?\s+amm\b.*\b(05|06|20[0-9]|200)\b[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
    r"\bamm\s+task\b",
    r"\bamm\b.*\b200[- ]?\d{3}\b",

    # common routine phrases
    r"\b(checks\s+scheduled|make\s+sure|refer\s+to)\b",
]

# Cosmetic → NON-DEFECT
COSMETIC_PATTERNS = [
    r"\bpaint\s*(peel|chip|flake|touch[- ]?up|scratch|flake)\b",
    r"\b(name\s*plate|placard|decal|stencil|label|marking)\s*(miss(ing)?|faded|worn|illegible)\b",
    r"\bcosmetic\b",
    r"\bappearance\b",
]

# Natural wear → NON-DEFECT
NATURAL_WEAR_PATTERNS = [
    r"\b(tyre|tire)\s*(wear|worn|tread|scuff)\b",
    r"\bbrake\s*wear\b",
    r"\berosion\b",
    r"\bweather(ing)?\b",
]

# Meta/audit – mở rộng thêm “refer note/see note” và ngoặc có ‘note/file/directory’
META_LINE_PATTERNS = [
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*",
    r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",
    r"^\s*NOTE:\b.*",
    r"^\s*FILE\s+LOCATED\b.*",
    r"^\s*(DIRECTORY|DIRECTORY://)\b.*",
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",
    r"\b\d{1,2}[:.]\d{2}\b.*",
    r"^\s*\d+\s*WORKSTEP\S*\b.*",
    r".*\brefer\s+note\b.*",
    r".*\bsee\s+note\b.*",
]
# Ngoặc chứa các từ khóa meta (xoá trong lòng câu)
PAREN_META = r"\((?=[^)]*(note|file\s+located|directory://|refer\s+note|see\s+note))[^)]*\)"

# Nhận diện Type scheduled
SCHEDULED_TYPE = [
    r"\bSCHEDULED\b",
    r"\bSCHEDULED\s*W/?O\b",
    r"\bROUTINE\b",
    r"\bCHECK\b",
    r"\bA-?CHECK\b",
    r"\bC-?CHECK\b",
]

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------
_re_defect_strong   = re.compile("|".join(DEFECT_STRONG), re.I)
_re_defect_context  = re.compile("|".join(DEFECT_WITH_CONTEXT), re.I)
_re_corrective      = re.compile("|".join(CORRECTIVE_ACTIONS), re.I)
_re_routine         = re.compile("|".join(ROUTINE_PATTERNS), re.I)
_re_cosmetic        = re.compile("|".join(COSMETIC_PATTERNS), re.I)
_re_wear            = re.compile("|".join(NATURAL_WEAR_PATTERNS), re.I)
_re_meta_line       = re.compile("|".join(META_LINE_PATTERNS), re.I | re.M)
_re_paren_meta      = re.compile(PAREN_META, re.I)
_re_docs            = re.compile(r"\b(AMM|TSM|FIM)\b\s*[-:]?\s*\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?", re.I)
_re_scheduled_type  = re.compile("|".join(SCHEDULED_TYPE), re.I)

_ws = re.compile(r"[ \t]+")

# ============================================================
# 2) CLEANING – bỏ meta ở dòng & trong ngoặc
# ============================================================
def _strip_meta(text: str) -> str:
    if not text:
        return ""
    # Xoá các cụm trong ngoặc có chứa note/file located/directory://...
    cleaned = _re_paren_meta.sub(" ", str(text))
    # Tách dòng, bỏ dòng meta; giữ dòng nếu chứa symptom mạnh hoặc citation
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", cleaned)]
    kept = []
    for ln in lines:
        if not ln:
            continue
        if _re_meta_line.search(ln):
            if _re_defect_strong.search(ln) or _re_docs.search(ln):
                kept.append(ln)
            continue
        kept.append(ln)
    out = " ".join(kept)
    return _ws.sub(" ", out).strip()

def _count(rx: re.Pattern, text: str) -> int:
    return len(list(rx.finditer(text))) if text and rx else 0

# ============================================================
# 3) CLASSIFY
# ============================================================
def classify_nondefect_reason(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Trả về (is_defect, reason, details)
      - is_defect: True → liên quan độ tin cậy; False → non-defect
    """
    raw_text = f"{desc or ''} {action or ''}".strip()
    text = _strip_meta(raw_text)

    if not text:
        return False, "Thiếu dữ liệu mô tả/hành động.", {"raw_empty": True}

    # 1) Symptom/cảnh báo mạnh
    strong = _count(_re_defect_strong, text)
    if strong > 0:
        return True, "Symptom/cảnh báo mạnh (fault/failure/leak/ECAM/EICAS/CAS…).", {"strong_hits": strong}

    # 2) Thông tin scheduled
    is_scheduled = bool(_re_scheduled_type.search(wo_type or "")) if wo_type else False

    # 3) Đếm nhóm pattern
    routine_hits  = _count(_re_routine, text)
    cosmetic_hits = _count(_re_cosmetic, text)
    wear_hits     = _count(_re_wear, text)
    corr_hits     = _count(_re_corrective, text)
    defect_ctx    = _count(_re_defect_context, text)

    # 4) Luật ưu tiên: Scheduled + Routine + không symptom/defect_ctx
    if is_scheduled and routine_hits > 0 and corr_hits == 0 and defect_ctx == 0:
        return False, "WO định kỳ (Type=SCHEDULED) với nội dung inspection/check/ops test/overhaul, không có symptom.", {
            "scheduled": True, "routine_hits": routine_hits
        }

    # 5) Cosmetic / Natural wear (không symptom)
    if cosmetic_hits > 0:
        return False, "Cosmetic (paint/nameplate/placard/label…), không ảnh hưởng độ tin cậy.", {
            "cosmetic_hits": cosmetic_hits
        }
    if wear_hits > 0:
        return False, "Hao mòn tự nhiên (tyre/brake wear/erosion/weathering) – không phải hỏng hóc hệ thống.", {
            "wear_hits": wear_hits
        }

    # 6) Hành động khắc phục → DEFECT
    if corr_hits > 0:
        # Ngoại lệ: nếu scheduled + routine (remove/send for overhaul theo lịch) → NON-DEFECT
        if is_scheduled and routine_hits > 0 and defect_ctx == 0:
            return False, "Tháo/lắp/gửi đi inspection/overhaul theo lịch – không có symptom.", {
                "scheduled": True, "routine_hits": routine_hits, "corrective_hits": corr_hits
            }
        return True, "Có hành động khắc phục (repair/replace/adjust/rectify/troubleshoot…).", {
            "corrective_hits": corr_hits
        }

    # 7) "defect/damage" có NGỮ CẢNH → DEFECT
    if defect_ctx > 0:
        return True, "Có 'defect/damage' với ngữ cảnh (found/detected/observed/noted…).", {
            "defect_with_context": defect_ctx
        }

    # 8) Routine/no-finding nói chung → NON-DEFECT
    if routine_hits > 0:
        return False, "Routine/inspection/no-finding – không có symptom/hành động khắc phục.", {
            "routine_hits": routine_hits
        }

    # 9) Mặc định: DEFECT (thận trọng)
    return True, "Không khớp routine/cosmetic/wear; mặc định coi là defect để bảo toàn độ tin cậy.", {}

def is_technical_defect(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None
) -> bool:
    is_def, _, _ = classify_nondefect_reason(desc, action, wo_type=wo_type)
    return is_def
