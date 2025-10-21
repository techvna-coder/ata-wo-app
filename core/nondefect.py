# core/nondefect.py - V3 OPTIMIZED (Rule-based, No LLM)
"""
LOGIC NGHIỆP VỤ:
1. Type = "SCHEDULED W/O" → Công việc chủ động
   - Nếu phát hiện defect (FOUND + symptom + corrective) → Technical Defect (tính độ tin cậy)
   - Nếu routine/inspection/no finding → NON-DEFECT (không tính)

2. Type ≠ "SCHEDULED W/O" → Công việc phản ứng
   - Nếu là technical defect thực sự → Technical Defect (tính độ tin cậy)
   - Nếu là natural wear/cosmetic/cabin items → NON-DEFECT (không tính)

3. Reference: NON DEFECT.xlsx patterns (841 patterns)
"""

from __future__ import annotations
import regex as re
import unicodedata
from typing import Tuple, Dict, Any, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# ===== DEFINITE NON-DEFECT PATTERNS (from NON DEFECT.xlsx) =====

# Category 1: CABIN ITEMS (không liên quan reliability)
CABIN_ITEMS = [
    r"\bbaby\b", r"\bbassinet\b", r"\bcurtain\b", r"\bseat\s*(cover|belt|cushion|pocket)?\b",
    r"\bblanket\b", r"\bpillow\b", r"\bmagazine\b", r"\blife\s*vest\b",
    r"\bgalley\b", r"\blavatory\b", r"\btray\s*table\b", r"\barmrest\b",
    r"\bheadrest\b", r"\bcarpet\b", r"\bsecurity\s*box\b",
    r"\binfant\s*seat\b", r"\bpassenger\s*seat\b",
]

# Category 2: COSMETIC / MINOR (không ảnh hưởng airworthiness)
COSMETIC_PATTERNS = [
    r"\bpaint\s*(chip|peel|scratch|touch[-\s]?up)\b",
    r"\b(scratch|mark|stain|discolor|faded)\b.*\b(paint|surface|panel)\b",
    r"\bcosmetic\b", r"\bappearance\b",
    r"\btow\s*marking\b", r"\bmarking\s*(missing|faded|worn)\b",
    r"\bname\s*plate\b.*\b(missing|faded|illegible)\b",
    r"\bplacard\b.*\b(missing|faded)\b",
]

# Category 3: NATURAL WEAR (hao mòn tự nhiên trong limits)
NATURAL_WEAR = [
    r"\b(tyre|tire)\s*(wear|worn|pressure|tread)\b",
    r"\bbrake\s*(wear|worn|pad)\b",
    r"\bwear\s*(within\s*)?limit(s)?\b",
    r"\bworn\s*to\s*limit\b",
    r"\berosion\b", r"\bweather(ing)?\b",
]

# Category 4: ROUTINE / SCHEDULED (không phát hiện defect)
ROUTINE_PATTERNS = [
    r"\broutine\s*(inspection|check|maintenance)\b",
    r"\bscheduled\s*(inspection|check|maintenance)\b",
    r"\bperiodic\s*(ground\s*)?check\b",
    r"\bgeneral\s*visual\s*inspection\b",
    r"\b(daily|weekly|monthly|transit)\s*check\b",
    r"\bfunctional\s*test\b", r"\bops\s*test\b", r"\bleak\s*check\b",
    r"\bcheck\s*(and\s*)?(record|take\s*note|quantity|condition)\b",
    r"\bmonitor(ing)?\b",
]

# Category 5: SERVICING / LUBRICATION (preventive)
SERVICING_PATTERNS = [
    r"\bservic(e|ing)\b", r"\blubrication\b", r"\bgrease\b",
    r"\bclean(ing)?\b", r"\bdisinfect(ion)?\b",
    r"\breplenish\b", r"\bdrain\s*and\s*replenish\b",
    r"\bvacuum\s*cleaning\b", r"\bdry\s*cleaning\b",
]

# Category 6: ADMINISTRATIVE / DATA (không liên quan kỹ thuật)
ADMINISTRATIVE = [
    r"\brecord\b.*\b(life|data|p/n|s/n|information)\b",
    r"\bupdate\b.*\b(data|software|database)\b",
    r"\bcorrect\b.*\btechlog\b",
    r"\bdata\s*(base|transmission)\b",
    r"\banalysis\b.*\b(fluid|oil)\b",
    r"\bsoftware\b.*\b(load|upload|update)\b",
    r"\bnavigation\s*database\b",
]

# Category 7: EQUIPMENT / GSE (ground support equipment)
EQUIPMENT_PATTERNS = [
    r"\b(gse|ground\s*support\s*equipment)\b",
    r"\bstandby\s*equipment\b",
    r"\btool\b.*\b(missing|prepare|standby)\b",
]

# Category 8: SPECIFIC NON-DEFECTS (từ file)
SPECIFIC_NONDEFECTS = [
    r"\bbonding\s*lead\b.*\b(broken|missing)\b",  # Lead bonding bị đứt (minor)
    r"\banti[-\s]?insecticide\b",
    r"\bautoland\s*(out\s*of\s*date|over\s*due)\b",
    r"\bcap\b.*\b(missing|yellow\s*accu|brake)\b",
    r"\bbattery\s*exp(iry)?\s*date\b",
    r"\bbird\s*strike\b.*\b(minor|no\s*damage)\b",
    r"\bcan\s*not\s*perform\b.*\b(due\s*to|no\s*power)\b",
]

# ===== DEFINITE DEFECT PATTERNS =====

# Strong symptom indicators
STRONG_SYMPTOMS = [
    r"\b(fail|failure|fault|faulty|malfunction)\b",
    r"\b(leak|leaking|leakage)\b",
    r"\b(crack|cracked)\b",
    r"\binop(erative)?\b", r"\bno\s*go\b",
    r"\boverheat(ing)?\b", r"\bsmoke\b", r"\bvibrat(e|ion)\b",
    r"\bshort\s*circuit\b", r"\btrip(ped)?\b",
    r"\b(ECAM|EICAS|CAS)\s*(warning|caution|fault)\b",
]

# Strong corrective indicators
STRONG_CORRECTIVE = [
    r"\breplace(d|ment)?\b",
    r"\brepair(ed)?\b", r"\brectify(ied)?\b",
    r"\btroubleshoot(ing)?\b",
    r"\bC/O\s*REPLAC(E|ED)\b",
    r"\bMEL\b", r"\bCDL\b", r"\bdeferr(al|ed)\b",
]

# Compile patterns
_CABIN_RE = re.compile("|".join(CABIN_ITEMS), re.I)
_COSMETIC_RE = re.compile("|".join(COSMETIC_PATTERNS), re.I)
_WEAR_RE = re.compile("|".join(NATURAL_WEAR), re.I)
_ROUTINE_RE = re.compile("|".join(ROUTINE_PATTERNS), re.I)
_SERVICING_RE = re.compile("|".join(SERVICING_PATTERNS), re.I)
_ADMIN_RE = re.compile("|".join(ADMINISTRATIVE), re.I)
_EQUIPMENT_RE = re.compile("|".join(EQUIPMENT_PATTERNS), re.I)
_SPECIFIC_ND_RE = re.compile("|".join(SPECIFIC_NONDEFECTS), re.I)

_STRONG_SYMPTOM_RE = re.compile("|".join(STRONG_SYMPTOMS), re.I)
_STRONG_CORRECTIVE_RE = re.compile("|".join(STRONG_CORRECTIVE), re.I)

# Helpers
def _normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", t)


# ============================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================

def is_technical_defect(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None
) -> bool:
    """
    Phân loại WO có phải Technical Defect (tính độ tin cậy) không.
    
    Returns:
        True: Technical Defect → tính độ tin cậy
        False: NON-DEFECT → KHÔNG tính độ tin cậy
    """
    is_def, _, _ = classify_technical_defect_v3(desc, action, wo_type)
    return is_def


def classify_technical_defect_v3(
    desc: Optional[str],
    action: Optional[str],
    wo_type: Optional[str] = None
) -> Tuple[bool, str, float]:
    """
    Phân loại chi tiết với reason và confidence.
    
    LOGIC:
    1. SCHEDULED W/O:
       - Phát hiện defect + corrective → DEFECT (tính)
       - Routine/inspection/no finding → NON-DEFECT (không tính)
    
    2. NON-SCHEDULED:
       - Technical defect → DEFECT (tính)
       - Cabin items/cosmetic/natural wear → NON-DEFECT (không tính)
    
    Returns:
        (is_defect: bool, reason: str, confidence: float)
    """
    desc_norm = _normalize(desc)
    action_norm = _normalize(action)
    combined = f"{desc_norm} {action_norm}"
    
    if not combined.strip():
        return False, "Empty work order", 0.50
    
    is_scheduled = bool(wo_type and re.search(r"\bSCHEDULED\s*W/?O\b", wo_type, re.I))
    
    # ===== STAGE 1: DEFINITE NON-DEFECT (Hard Rules) =====
    
    # 1.1: Cabin items (luôn không tính)
    if _CABIN_RE.search(combined):
        return False, "Cabin item (không liên quan reliability)", 0.98
    
    # 1.2: Cosmetic only (không có structural damage)
    if _COSMETIC_RE.search(combined) and not _STRONG_SYMPTOM_RE.search(combined):
        return False, "Cosmetic issue (không ảnh hưởng airworthiness)", 0.95
    
    # 1.3: Natural wear within limits
    if _WEAR_RE.search(combined) and re.search(r"\bwithin\s*limit(s)?\b", combined, re.I):
        return False, "Natural wear within limits", 0.93
    
    # 1.4: Specific non-defects (bonding lead, autoland due, etc.)
    if _SPECIFIC_ND_RE.search(combined):
        return False, "Specific non-defect pattern (từ file NON DEFECT)", 0.92
    
    # 1.5: Administrative tasks
    if _ADMIN_RE.search(combined) and not _STRONG_SYMPTOM_RE.search(combined):
        return False, "Administrative task (record/data/update)", 0.90
    
    # 1.6: Equipment/GSE (không phải aircraft)
    if _EQUIPMENT_RE.search(combined):
        return False, "Equipment/GSE issue", 0.88
    
    # ===== STAGE 2: SCHEDULED W/O LOGIC =====
    
    if is_scheduled:
        # Check if defect was FOUND during scheduled work
        has_found = bool(re.search(r"\bfound\b", combined, re.I))
        has_symptom = _STRONG_SYMPTOM_RE.search(combined)
        has_corrective = _STRONG_CORRECTIVE_RE.search(combined)
        
        # Scheduled + found defect + corrective → DEFECT
        if has_found and (has_symptom or has_corrective):
            return True, "Scheduled work: phát hiện defect cần corrective", 0.92
        
        # Scheduled + routine/servicing → NON-DEFECT
        if _ROUTINE_RE.search(combined) or _SERVICING_RE.search(combined):
            # Unless strong symptom present
            if has_symptom and has_corrective:
                return True, "Scheduled: có symptom + corrective action", 0.85
            return False, "Scheduled routine/servicing (không phát hiện defect)", 0.90
        
        # Scheduled + no finding → NON-DEFECT
        if re.search(r"\bno\s*(abnormal|fault|finding|defect)\b", combined, re.I):
            return False, "Scheduled: no abnormality found", 0.95
        
        # Scheduled but ambiguous → Default NON-DEFECT (safety: không tính nhầm routine)
        return False, "Scheduled work: không rõ defect", 0.70
    
    # ===== STAGE 3: NON-SCHEDULED LOGIC =====
    
    # Servicing/lubrication without symptom → NON-DEFECT
    if _SERVICING_RE.search(combined) and not _STRONG_SYMPTOM_RE.search(combined):
        return False, "Servicing/lubrication preventive", 0.88
    
    # Routine inspection (even unscheduled) without symptom → NON-DEFECT
    if _ROUTINE_RE.search(combined) and not _STRONG_SYMPTOM_RE.search(combined):
        return False, "Routine inspection (không phát hiện defect)", 0.85
    
    # Strong symptom + strong corrective → DEFECT
    if _STRONG_SYMPTOM_RE.search(combined) and _STRONG_CORRECTIVE_RE.search(combined):
        return True, "Technical defect: symptom + corrective action", 0.95
    
    # Strong symptom alone → DEFECT
    if _STRONG_SYMPTOM_RE.search(combined):
        return True, "Technical defect: strong symptom detected", 0.88
    
    # Strong corrective alone → DEFECT
    if _STRONG_CORRECTIVE_RE.search(combined):
        return True, "Technical defect: corrective action performed", 0.82
    
    # ===== STAGE 4: DEFAULT (Conservative) =====
    
    # Default: Nếu không rõ ràng → Coi là DEFECT (safety first cho non-scheduled)
    # Nhưng confidence thấp để có thể review
    return True, "Ambiguous case - default to defect (safety first)", 0.65


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("TECHNICAL DEFECT CLASSIFICATION - V3 TEST")
    print("="*80)
    
    test_cases = [
        # NON-DEFECT cases
        {
            "desc": "BABY BASSINET MISSING",
            "action": "INSTALL NEW BASSINET",
            "type": "",
            "expected": False,
            "reason": "Cabin item"
        },
        {
            "desc": "PAINT CHIP ON NOSE CONE",
            "action": "PAINT TOUCH-UP PER AMM 51-00-00",
            "type": "",
            "expected": False,
            "reason": "Cosmetic"
        },
        {
            "desc": "TYRE WEAR CHECK",
            "action": "TREAD DEPTH WITHIN LIMITS",
            "type": "SCHEDULED W/O",
            "expected": False,
            "reason": "Natural wear within limits"
        },
        {
            "desc": "ROUTINE INSPECTION ENGINE MOUNT",
            "action": "VISUAL INSPECTION. CHECK SATIS. NO ABNORMALITY FOUND.",
            "type": "SCHEDULED W/O",
            "expected": False,
            "reason": "Scheduled routine, no finding"
        },
        {
            "desc": "BONDING LEAD BROKEN AT DOOR",
            "action": "REPLACE BONDING LEAD",
            "type": "",
            "expected": False,
            "reason": "Bonding lead (specific non-defect)"
        },
        
        # DEFECT cases
        {
            "desc": "ECAM CAUTION: PACK 2 FAULT",
            "action": "C/O REPLACED PACK 2 COMPRESSOR",
            "type": "",
            "expected": True,
            "reason": "ECAM fault + replace"
        },
        {
            "desc": "FOUND HYDRAULIC LEAK AT ACTUATOR",
            "action": "REPLACE ACTUATOR SEAL",
            "type": "SCHEDULED W/O",
            "expected": True,
            "reason": "Scheduled: found defect + corrective"
        },
        {
            "desc": "ENGINE VIBRATION DURING TAKEOFF",
            "action": "TROUBLESHOOT ENGINE. REPLACED FAN BLADE.",
            "type": "",
            "expected": True,
            "reason": "Symptom + corrective"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases, 1):
        result, reason, conf = classify_technical_defect_v3(
            tc["desc"], tc["action"], tc.get("type", "")
        )
        
        status = "✓" if result == tc["expected"] else "✗"
        
        if result == tc["expected"]:
            passed += 1
            print(f"\n{status} Test {i}: PASS")
        else:
            failed += 1
            print(f"\n{status} Test {i}: FAIL")
            print(f"   Expected: {tc['expected']} ({tc['reason']})")
            print(f"   Got: {result}")
        
        print(f"   Desc: {tc['desc'][:60]}")
        print(f"   Result: {'DEFECT' if result else 'NON-DEFECT'}")
        print(f"   Confidence: {conf:.2f}")
        print(f"   Reason: {reason}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
