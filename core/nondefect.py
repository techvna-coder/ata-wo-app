# core/nondefect.py
from __future__ import annotations

import unicodedata
import regex as re
from typing import Tuple, Dict, Any, Optional

# ============================================================
# 0) CẤU HÌNH ĐIỂM & NGƯỠNG (có thể tinh chỉnh theo fleet)
# ============================================================

WEIGHTS = {
    # Tín hiệu nghiêng Defect
    "strong_symptom":      4.0,   # fault/failure/leak/overheat/vibration/smoke/INOP/short/trip/crack + ECAM/EICAS/CAS/ACARS/FWS
    "warn_ctx":            3.0,   # ECAM/EICAS/CAS warning/caution, hoặc warning|caution + light/message/indication
    "defect_with_context": 3.2,   # "defect|damage|crack" <-> "found|detected|observed|noted|present|occurred" (hai chiều)
    "corrective_strong":   3.8,   # replace/repair/rectify/troubleshoot/modify + SB/AD (đã bỏ adjust/calibrate/rig/shim)
    "corrective_weak":     0.8,   # remove/install (có thể routine)

    # Tín hiệu nghiêng Non-Defect
    "scheduled_routine":   3.7,   # Type=SCHEDULED + routine/inspection/ops/functional/leak check OK, AMM 05/200-xxx
    "routine_generic":     1.9,   # routine/periodic/visual/zonal/borescope/perform check/ops test
    "rigging_servicing":   2.4,   # rig/rigging/adjust/calibrate/shim/grease/close-up/open cowls (không symptom)
    "no_finding":          2.8,   # no abnormality/defect/fault/finding/discrepancy; serviceable/normal/check satis/within limits/NIL
    "cosmetic":            3.0,   # paint peel/chip/scratch; nameplate/placard/decal/label missing/faded/worn
    "natural_wear":        2.5,   # tyre/tire wear, brake wear, erosion/weathering
    "scheduled_overhaul":  2.5,   # send/remove for bench test/overhaul/shop visit (theo lịch)
}

THRESHOLDS = {
    "decision_margin": 1.0,   # nếu |defect - nondefect| < margin → REVIEW (trừ luật ưu tiên)
}

# ============================================================
# 1) MẪU NHẬN DIỆN (Anh + Việt)
# ============================================================

# Triệu chứng/cảnh báo mạnh
PAT_STRONG_SYMPTOM = [
    r"\bfail(ure|ed)?\b", r"\bfault(s)?\b", r"\bmalfunction\b",
    r"\bleak(s|ing)?\b", r"\boverheat(ing)?\b", r"\bvibrat(e|ion|ing)\b",
    r"\bsmoke\b", r"\bintermittent\b", r"\bspurious\b",
    r"\bno\s*go\b", r"\binop(erative)?\b", r"\bdegrad(ed|ation)\b",
    r"\bshort( circuit)?\b", r"\btrip(ped)?\b",
    r"\bcrack(s|ed)?\b",
    r"\bECAM\b", r"\bEICAS\b", r"\bCAS\b", r"\bACARS\b", r"\bFWS\b",
    # Việt – triệu chứng
    r"\brò\b", r"\brò\s*ri\b", r"\brỉ\b", r"\brò\s*dị\b",
    r"\brò\s*rỉ\b", r"\brò\s*khí\b", r"\brò\s*nhiên\s*liệu\b",
    r"\bquá\s*nhiệt\b", r"\brung\b", r"\brung\s*lắc\b",
    r"\bhỏng\b", r"\blỗi\b", r"\bbáo\s*động\b", r"\bcảnh\s*báo\b",
]

# Cảnh báo có ngữ cảnh hệ thống
PAT_WARN_CTX = [
    r"\b(ECAM|EICAS|CAS)\s+(warning|caution)\b",
    r"\bwarning\s*(light|message|caption|indication)\b",
    r"\bcaution\s*(light|message|caption|indication)\b",
]

# "defect/damage/crack" có NGỮ CẢNH (HAI CHIỀU) – loại nhiễu “NOTE/FILE LOCATED…”
PAT_DEFECT_WITH_CONTEXT = [
    r"\b(defect(s)?|damage(d)?|crack(s)?|cracked)\b(?!.{0,400}?\b(NOTE|directory|file located)\b).*?\b(found|detected|observed|reported|noted|occurred|present)\b",
    r"\b(found|detected|observed|reported|noted|occurred|present)\b(?!.{0,400}?\b(NOTE|directory|file located)\b).*?\b(defect(s)?|damage(d)?|crack(s)?|cracked)\b",
    # Việt
    r"\b(hư\s*hỏng|hỏng\s*hóc|hư\s*hại|nứt)\b.*?\b(phát\s*hiện|ghi\s*nhận|quan\s*sát|báo\s*cáo)\b",
    r"\b(phát\s*hiện|ghi\s*nhận|quan\s*sát|báo\s*cáo)\b.*?\b(hư\s*hỏng|hỏng\s*hóc|hư\s*hại|nứt)\b",
]

# Hành động khắc phục mạnh (đÃ bỏ adjust/calibrate/rig/shim)
PAT_CORRECTIVE_STRONG = [
    r"\breplac(e|ed|ing|ement)\b",
    r"\brepair(ed|ing)?\b", r"\brectif(y|ied|ication)\b",
    r"\btroubleshoot(ing)?\b",
    r"\bmodif(y|ication|ied)\b",
    r"\bMEL\b", r"\bdeferr(al|ed)\b",
    r"\b(embod(y|ied)|incorporat(e|ed)|accomplish(ed)?|comply\s+with|implement(ed)?|appl(ied|y))\b\s+(SB|Service\s+Bulletin)\b",
    r"\b(embod(y|ied)|incorporat(e|ed)|accomplish(ed)?|comply\s+with|implement(ed)?|appl(ied|y))\b\s+(AD|Airworthiness\s+Directive)\b",
    r"\bC/O\s+REPLAC(ED|E|ING|EMENT)\b",
    # Việt
    r"\bsửa\s*chữa\b", r"\bthay\b", r"\bthay\s*mới\b",
]

# Hành động yếu (có thể routine)
PAT_CORRECTIVE_WEAK = [
    r"\bremove(d|al)?\b", r"\binstall(ed|ation)?\b",
    # Việt
    r"\btháo\b", r"\bgắn\b", r"\blắp\b", r"\btháo\s*lắp\b",
]

# Routine/định kỳ/inspection/check/ops test
PAT_ROUTINE_GENERIC = [
    r"\b(clean(ing)?|lubricat(e|ion|ed)|servic(e|ing)|disinfection|interior clean|cabin tidy)\b",
    r"\b(periodic|scheduled|routine)\b",
    r"\b(A-?check|C-?check|IL|4D|transit\s*check|daily\s*check|weekly\s*check|preflight\s*check|walkaround)\b",
    r"\b(general|detailed|special)\s+visual\s+inspection\b",
    r"\b(zonal|zone)\s+(visual\s+)?inspection\b",
    r"\b(boroscope|borescope)\s+inspection\b",
    r"\bvisual\s+inspection\b",
    r"\binspect(ion|ed|ing)?\b",
    r"\bperform\s+check(s)?\b",
    r"\b(checks?|ops(\.|eration(al)?)?\s*(test|check)|functional\s*check|leak\s*check)\b",
    r"\bref(er)?\s+amm\b.*\b(05|06|20[0-9]|200)\b[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?",
    r"\bamm\s+task\b", r"\bamm\b.*\b200[- ]?\d{3}\b",
    r"\b(checks\s+scheduled|make\s+sure|refer\s+to)\b",
    # Việt
    r"\bkiểm\s*tra\s*(định\s*kỳ|tổng\s*quát|khu\s*vực|chung)\b",
    r"\bkiểm\s*tra\s*trực\s*quan\b", r"\bkiểm\s*tra\s*thử\s*nghiệm\b",
    r"\bkiểm\s*tra\s*rò\s*rỉ\b", r"\bkiểm\s*tra\s*chức\s*năng\b",
    r"\btheo\s*AMM\b", r"\btham\s*chiếu\s*AMM\b",
]

# Rigging/servicing/adjust/calibrate/shim/grease/close-up…
PAT_RIGGING_SERVICING = [
    r"\brig(ging)?\b", r"\bshims?\b", r"\bshim(ming)?\b",
    r"\badjust(ed|ment|ing)?\b", r"\bcalibrat(e|ion|ed|ing)\b",
    r"\bapply\s+(a\s+)?thin\s+film\s+of\s+the\s+grease\b", r"\bgrease\b", r"\blubricat(e|ion|ed)\b",
    r"\bclose[-\s]?up\b", r"\bopen\s+the\s+fan\s+cowls\b",
    r"\b(check\s+of\s+the\s+rigging|rigging\s+check)\b",
    # Việt
    r"\bchỉnh\s*căn\b", r"\bđiều\s*chỉnh\b", r"\bbôi\s*mỡ\b", r"\bbôi\s*trơn\b",
]

# Không phát hiện bất thường (no-finding)
PAT_NO_FINDING = [
    r"\bno\s*(abnormalit(y|ies)|defect|fault|finding|discrepanc(y|ies))\b",
    r"\bfound\s*(serviceable|normal)\b",
    r"\bwithin\s+limits?\b", r"\bcheck\s+satis(factory)?\b",
    r"\brecord(?:ed)?\s+(raised\s+)?(a\s+)?nrc(\s+or\s+wo)?\s*:\s*nil\b",
    r"\bno\s*crack\s*(?:found|detected)?\b",
    # Việt
    r"\bkhông\s*phát\s*hiện\s*(bất\s*thường|hỏng\s*hóc|khuyết\s*điểm)\b",
    r"\bkhông\s*có\s*phát\s*hiện\b", r"\bkhông\s*có\s*bất\s*thường\b",
    r"\bđạt\s*yêu\s*cầu\b", r"\bđạt\s*chuẩn\b", r"\bhài\s*lòng\b",
]

# Cosmetic
PAT_COSMETIC = [
    r"\bpaint\s*(peel|chip|flake|touch[- ]?up|scratch|flake)\b",
    r"\b(name\s*plate|placard|decal|stencil|label|marking)\s*(miss(ing)?|faded|worn|illegible)\b",
    r"\bcosmetic\b", r"\bappearance\b",
    # Việt
    r"\bbong\s*sơn\b", r"\btróc\s*sơn\b", r"\bxước\s*sơn\b",
    r"\bmất\s*bảng\s*tên\b", r"\bmờ\s*tem\b", r"\bnhãn\s*mác\b",
]

# Natural wear
PAT_WEAR = [
    r"\b(tyre|tire)\s*(wear|worn|tread|scuff)\b",
    r"\bbrake\s*wear\b",
    r"\berosion\b", r"\bweather(ing)?\b",
    # Việt
    r"\bmòn\s*lốp\b", r"\bmòn\s*phanh\b", r"\bsự\s*bào\s*mòn\b",
]

# Send/remove for scheduled shop/overhaul
PAT_SCHEDULED_OVERHAUL = [
    r"\b(send|sent)\s+for\s+(inspection|bench\s*test|overhaul|shop\s*visit|nrc)\b",
    r"\bremove(d)?\s+for\s+(inspection|overhaul|bench\s*test)\b",
    # Việt
    r"\bgửi\s*đi\s*(kiểm\s*tra|overhaul|xưởng)\b",
    r"\btháo\s*ra\s*để\s*(kiểm\s*tra|overhaul)\b",
]

# “Instructional mentions”: dặn dò tìm nứt/rò… (không phải phát hiện)
PAT_INSTRUCTIONAL = [
    r"\b(look\s+for|check\s+for|inspect\s+for)\s+(crack(s)?|leak(s|age)?|damage|defect(s)?)\b",
    r"\bif\s+(any\s+)?crack\s+found\b", r"\bif\s+there\s+is\s+contamination\b",
    # Việt
    r"\b(tìm\s+kiếm|quan\s*sát|kiểm\s*tra)\s*(nứt|rò\s*rỉ|hư\s*hỏng|khuyết\s*điểm)\b",
    r"\bnếu\s+có\s+nứt\b", r"\bnếu\s+phát\s*hiện\b",
]

# Lines to drop (meta/audit/instruction headers)
PAT_META_LINES = [
    r"^\s*WARNING:\b.*", r"^\s*CAUTION:\b.*",
    r"^\s*STANDARD\s+PRACTICES:.*",
    r"^\s*JOB\s+SET[- ]?UP\b.*",
    r"^\s*PREPARATION:.*",
    r"^\s*CLOSE[- ]?UP:.*",
    r"\b(WORKSTEP\s+ADDED\s+BY|PERFORMED\s*SIGN|DESCRIPTION\s*SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*", r"\bPART\s+REQUIREMENT\b.*",
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",
    r"^\s*NOTE:\b.*", r"^\s*FILE\s+LOCATED\b.*",
    r"^\s*(DIRECTORY|DIRECTORY://)\b.*",
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",
    r"\b\d{1,2}[:.]\d{2}\b.*",
    r"^\s*\d+\s*WORKSTEP\S*\b.*",
    r".*\brefer\s+note\b.*", r".*\bsee\s+note\b.*",
    # Việt
    r"^\s*CHÚ\s*Ý:\b.*", r"^\s*CẢNH\s*BÁO:\b.*",
    r"^\s*THỰC\s*HÀNH\s*TIÊU\s*CHUẨN:.*",
    r"^\s*CHUẨN\s*BỊ:.*",
    r"^\s*KẾT\s*THÚC:.*",
]
# Parenthetical meta to drop
PAREN_META = r"\((?=[^)]*(note|file\s+located|directory://|refer\s+note|see\s+note|chú\s*thích))[^)]*\)"

# Scheduled type detection
PAT_SCHEDULED_TYPE = [
    r"\bSCHEDULED\b", r"\bSCHEDULED\s*W/?O\b",
    r"\bROUTINE\b", r"\bCHECK\b", r"\bA-?CHECK\b", r"\bC-?CHECK\b",
    # Việt
    r"\bđịnh\s*kỳ\b", r"\bkiểm\s*tra\b",
]

# ------------------------------------------------------------
# Compile helpers
# ------------------------------------------------------------
def _rx(or_list): return re.compile("|".join(or_list), re.I)
RX_STRONG_SYMPTOM      = _rx(PAT_STRONG_SYMPTOM)
RX_WARN_CTX            = _rx(PAT_WARN_CTX)
RX_DEFECT_WITH_CONTEXT = _rx(PAT_DEFECT_WITH_CONTEXT)
RX_CORR_STRONG         = _rx(PAT_CORRECTIVE_STRONG)
RX_CORR_WEAK           = _rx(PAT_CORRECTIVE_WEAK)
RX_ROUTINE_GENERIC     = _rx(PAT_ROUTINE_GENERIC)
RX_RIG_SERV            = _rx(PAT_RIGGING_SERVICING)
RX_NO_FINDING          = _rx(PAT_NO_FINDING)
RX_COSMETIC            = _rx(PAT_COSMETIC)
RX_WEAR                = _rx(PAT_WEAR)
RX_SCHED_OVH           = _rx(PAT_SCHEDULED_OVERHAUL)
RX_META_LINES          = _rx(PAT_META_LINES)
RX_PAREN_META          = re.compile(PAREN_META, re.I)
RX_SCHEDULED_TYPE      = _rx(PAT_SCHEDULED_TYPE)
RX_INSTRUCTIONAL       = _rx(PAT_INSTRUCTIONAL)

WS = re.compile(r"[ \t]+")

# ============================================================
# 2) TIỀN XỬ LÝ
# ============================================================
def _normalize(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def _strip_meta(text: str) -> str:
    if not text:
        return ""
    cleaned = RX_PAREN_META.sub(" ", str(text))
    kept = []
    for ln in re.split(r"[\r\n]+", cleaned):
        ln = ln.strip()
        if not ln:
            continue
        if RX_META_LINES.search(ln):
            # GIỮ nếu dòng meta có cảnh báo/hỏng hóc rõ; KHÔNG còn giữ chỉ vì 'corrective weak/strong'
            if (RX_WARN_CTX.search(ln) or RX_STRONG_SYMPTOM.search(ln)):
                kept.append(ln)
            continue
        kept.append(ln)
    out = " ".join(kept)
    return WS.sub(" ", out).strip()

def _count(rx: re.Pattern, text: str) -> int:
    return len(list(rx.finditer(text))) if text and rx else 0

# ============================================================
# 3) PHÂN LOẠI (SCORING)
# ============================================================
def classify_nondefect_reason(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Trả về (is_defect, reason, details):
      - is_defect: True → liên quan độ tin cậy; False → non-defect
      - reason   : giải thích ngắn
      - details  : chấm điểm & đếm hit từng nhóm để audit
    """
    raw = f"{_normalize(desc)} {_normalize(action)}".strip()
    text = _strip_meta(raw)

    if not text:
        return False, "Thiếu dữ liệu mô tả/hành động.", {"raw_empty": True}

    # Đếm tín hiệu
    hits = {
        "strong_symptom":      _count(RX_STRONG_SYMPTOM, text),
        "warn_ctx":            _count(RX_WARN_CTX, text),
        "defect_with_context": _count(RX_DEFECT_WITH_CONTEXT, text),
        "corrective_strong":   _count(RX_CORR_STRONG, text),
        "corrective_weak":     _count(RX_CORR_WEAK, text),

        "scheduled_routine":   0,  # tính theo Type + routine/overhaul
        "routine_generic":     _count(RX_ROUTINE_GENERIC, text),
        "rigging_servicing":   _count(RX_RIG_SERV, text),
        "no_finding":          _count(RX_NO_FINDING, text),
        "cosmetic":            _count(RX_COSMETIC, text),
        "natural_wear":        _count(RX_WEAR, text),
        "scheduled_overhaul":  _count(RX_SCHED_OVH, text),

        "instructional":       _count(RX_INSTRUCTIONAL, text),
    }

    # Booster NRC: nếu có 'FINDING (NRC)' + (found|damage/defect|crack|replace/repair) → cộng nhẹ vào defect_with_context
    if re.search(r"\bFINDING\s*\(NRC\)", text, flags=re.I):
        if (re.search(r"\bfound\b", text, flags=re.I) or
            re.search(r"\b(damage(d)?|defect(s)?|crack(s)?|cracked)\b", text, flags=re.I) or
            re.search(r"\breplac(e|ed|ing|ement)\b|\brepair(ed|ing)?\b", text, flags=re.I)):
            hits["defect_with_context"] += 1

    # Giảm nhiễu: các “instructional mentions” (look/check for crack/leak…) không phải phát hiện.
    if hits["instructional"] > 0 and hits["defect_with_context"] > 0:
        hits["defect_with_context"] = max(0, hits["defect_with_context"] - 1)

    is_scheduled_type = bool(RX_SCHEDULED_TYPE.search(_normalize(wo_type or "")))
    if is_scheduled_type and (hits["routine_generic"] > 0 or hits["scheduled_overhaul"] > 0):
        hits["scheduled_routine"] = 1

    # Tính điểm
    defect_score = (
        hits["strong_symptom"]      * WEIGHTS["strong_symptom"] +
        hits["warn_ctx"]            * WEIGHTS["warn_ctx"] +
        hits["defect_with_context"] * WEIGHTS["defect_with_context"] +
        hits["corrective_strong"]   * WEIGHTS["corrective_strong"] +
        hits["corrective_weak"]     * WEIGHTS["corrective_weak"]
    )
    nondef_score = (
        hits["scheduled_routine"]   * WEIGHTS["scheduled_routine"] +
        hits["routine_generic"]     * WEIGHTS["routine_generic"] +
        hits["rigging_servicing"]   * WEIGHTS["rigging_servicing"] +
        hits["no_finding"]          * WEIGHTS["no_finding"] +
        hits["cosmetic"]            * WEIGHTS["cosmetic"] +
        hits["natural_wear"]        * WEIGHTS["natural_wear"] +
        hits["scheduled_overhaul"]  * WEIGHTS["scheduled_overhaul"]
    )

    # ===== Luật ưu tiên cứng: Scheduled/Inspection/Rigging/Servicing + No finding/NIL/Within limits → NON-DEFECT
    if (is_scheduled_type or hits["routine_generic"] > 0 or hits["rigging_servicing"] > 0):
        if hits["no_finding"] > 0 and hits["strong_symptom"] == 0:
            # Không ‘replace/repair’ thực sự
            if hits["corrective_strong"] == 0:
                return False, "Scheduled/Inspection/Rigging + NO FINDING/CHECK SATIS → Non-Defect.", {
                    "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "scheduled_no_finding"
                }

    # Luật ưu tiên: Raise NRC nhưng ghi nhận NIL → Non-Defect
    if re.search(r"\braise(d)?\s+(a\s+)?nrc(\s+or\s+wo)?\b", text, flags=re.I) and \
       re.search(r"\brecord(?:ed)?\s+(raised\s+)?(a\s+)?nrc(\s+or\s+wo)?\s*:\s*nil\b", text, flags=re.I):
        return False, "RAISE NRC nhưng RECORD: NIL → Non-Defect.", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "nrc_nil"
        }

    # Luật ưu tiên Defect (thực sự): symptom rõ hoặc corrective mạnh (replace/repair/rectify/modify/TS) không thuộc servicing
    if hits["strong_symptom"] > 0:
        # Trừ khi có kết luận No finding rất rõ (ưu tiên chuẩn hãng)
        if hits["no_finding"] > 0 and (is_scheduled_type or hits["routine_generic"] > 0 or hits["rigging_servicing"] > 0):
            return False, "Có mô tả symptom nhưng kết luận NO FINDING trong work scheduled → Non-Defect.", {
                "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "symptom_but_no_finding"
            }
        return True, "Triệu chứng hỏng hóc rõ ràng.", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "strong_symptom"
        }

    if hits["corrective_strong"] > 0:
        return True, "Hành động khắc phục mạnh (replace/repair/rectify/modify/SB/AD).", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "corrective_strong"
        }

    # Ưu tiên: Scheduled + Routine + không symptom/defect_ctx/corrective_strong → NON-DEFECT
    if is_scheduled_type and (hits["routine_generic"] > 0 or hits["scheduled_overhaul"] > 0 or hits["rigging_servicing"] > 0):
        if (hits["strong_symptom"] == 0 and hits["defect_with_context"] == 0 and hits["corrective_strong"] == 0):
            return False, "WO định kỳ: inspection/ops/rigging/servicing; không có symptom/hành động khắc phục mạnh.", {
                "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "priority_rule": "scheduled_routine"
            }

    # Quyết định theo điểm
    margin = THRESHOLDS["decision_margin"]
    if nondef_score - defect_score >= margin:
        return False, "Tín hiệu routine/rigging/no-finding/cosmetic/wear trội hơn.", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits
        }
    if defect_score - nondef_score >= margin:
        return True, "Tín hiệu hỏng hóc/hành động khắc phục trội hơn.", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits
        }

    # Sát nút: nếu scheduled + no_finding + routine/rigging và không có symptom mạnh → Non-Defect
    if is_scheduled_type and hits["no_finding"] > 0 and (hits["routine_generic"] > 0 or hits["rigging_servicing"] > 0) and hits["strong_symptom"] == 0:
        return False, "Scheduled + No-finding + Routine/Rigging: ưu tiên Non-Defect.", {
            "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "tie_break": "scheduled_no_finding"
        }

    # Mặc định thận trọng
    return True, "Biên độ điểm nhỏ; mặc định coi là Defect để bảo toàn độ tin cậy.", {
        "defect_score": defect_score, "nondef_score": nondef_score, "hits": hits, "tie_break": "safety_first"
    }

def is_technical_defect(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None
) -> bool:
    is_def, _, _ = classify_nondefect_reason(desc, action, wo_type=wo_type)
    return is_def


# ============================================================
# (Tuỳ chọn) Kiểm thử nhanh
# ============================================================
if __name__ == "__main__":
    case1 = """
    CHECK OF THE RIGGING OF THE THRUST REVERSER LATCHES AND BUMPERS ENG #1
    MEASURE THE CLOSING FORCE AT EACH LATCH. CHECK SATIS.
    RECORD RAISED NRC OR WO: NIL
    """
    print("Case1 (rigging, no finding, scheduled) →", is_technical_defect(case1, "", wo_type="SCHEDULED W/O"))  # False

    case2 = "FOUND PACK 2 LEAK; C/O REPLACED DUCT SEAL. CHECK SATIS."
    print("Case2 (leak + replace) →", is_technical_defect(case2, "", wo_type="UNSCHEDULED"))  # True

    case3 = "APPLY A THIN FILM OF THE GREASE ON LATCH. ADJUST LATCH CLOSING FORCE. CHECK SATIS."
    print("Case3 (servicing/adjust) →", is_technical_defect(case3, "", wo_type="SCHEDULED W/O"))  # False
