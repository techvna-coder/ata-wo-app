# core/cleaning.py - FINAL OPTIMIZED VERSION
from __future__ import annotations
import regex as re

# ================================================================================
# NGUYÊN TẮC MỚI:
# 1. CHỈ loại bỏ header/footer/signature KHÔNG CÓ thông tin kỹ thuật
# 2. GIỮ LẠI toàn bộ dòng có: Manual refs, Symptoms, Actions, Technical terms
# 3. Loại bỏ metadata đơn thuần (BY xxx ON dd.MMM.yyyy, timestamp)
# ================================================================================

# ========== WHITELIST: Technical keywords PHẢI GIỮ ==========
_TECHNICAL_KEYWORDS = re.compile(
    r"\b("
    # Manuals & Documentation
    r"AMM|TSM|FIM|ESPM|IPC|CMM|WDM|SRM|SB|AD|MEL|MMEL|"
    r"TASKCARD|TASK\s*CARD|"
    
    # Systems & Avionics
    r"ECAM|EICAS|CAS|FWS|ACARS|FCOM|QRH|"
    
    # Symptoms (Defects)
    r"FOUND|DAMAGE(?:D)?|CRACK(?:ED|S)?|LEAK(?:ING|AGE|S)?|"
    r"FAIL(?:ED|URE|S)?|FAULT(?:Y|S)?|INOP(?:ERATIVE)?|"
    r"OVERHEAT(?:ING)?|SMOKE|VIBRAT(?:E|ION|ING)|"
    r"INTERMITTENT|SPURIOUS|NO\s*GO|DEGRAD(?:ED|ATION)|"
    r"SHORT(?:\s*CIRCUIT)?|TRIP(?:PED)?|"
    r"MISSING|BROKEN|WORN|ERODED|CORRODED|CONTAMINATED|"
    
    # Actions (Corrective)
    r"REPLACE(?:D|MENT)?|REPAIR(?:ED|ING)?|RECTIFY(?:IED|ICATION)?|"
    r"TROUBLESHOOT(?:ING)?|ADJUST(?:ED|MENT)?|CALIBRAT(?:E|ED|ION)|"
    r"RIG(?:GING)?|MODIFY(?:IED|ICATION)?|"
    r"C/O|CHANGE\s*OUT|REMOVED?|INSTALL(?:ED|ATION)?|"
    
    # Common abbreviations
    r"REF(?:ERENCE)?|PER|IAW|"
    
    # IPC-specific
    r"FIG(?:URE)?|ITEM|IT(?:EM)?|SHEET|SHT|"
    
    # Findings
    r"CHECK\s+SATIS(?:FACTORY)?|WITHIN\s+LIMITS|NO\s+ABNORMAL(?:ITY|ITIES)?|"
    r"SERVICEABLE|NORMAL|"
    
    # ATA pattern
    r"\d{2}-\d{2}(?:-\d{2})?(?:-\d+)?"
    r")\b",
    re.I
)

# ========== BLACKLIST: Pure meta lines (no technical content) ==========
_PURE_META_PATTERNS = [
    # Signature lines (standalone)
    r"^\s*\d+\s+WORKSTEP\s+ADDED\s+BY\s+[A-Z]{3}\d+\s+ON\s+\d{2}\.[A-Z]{3}\.\d{4}[,\s]+\d{2}:\d{2}\s*$",
    r"^\s*(?:DESCRIPTION|ACTION\s+PERFORMED|PERFORMED)\s+SIGN\s+[A-Z]{3}\d+\s*$",
    r"^\s*BY\s+[A-Z]{3}\d{5}\s*$",
    r"^\s*ON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\s*$",
    
    # Timestamps (standalone)
    r"^\s*\d{1,2}:\d{2}\s*$",
    
    # Part requirement headers (usually empty)
    r"^\s*PART\s+REQUIREMENT\s*$",
]

_pure_meta_re = re.compile("|".join(_PURE_META_PATTERNS), re.I)
_whitespace_re = re.compile(r"[ \t]+")

# ========== INLINE META TO REMOVE ==========
# Remove from within lines but keep the rest
_INLINE_META_PATTERNS = [
    r"\s+BY\s+[A-Z]{3}\d{5}\b",                    # " BY VAE03251"
    r"\s+ON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b",         # " ON 01.AUG.2025"
    r"\s+\d{1,2}:\d{2}\b",                          # " 06:30"
    r",\s*\d{1,2}:\d{2}\b",                         # ", 06:30"
]

_inline_meta_re = re.compile("|".join(_INLINE_META_PATTERNS), re.I)


def clean_wo_text(s: str) -> str:
    """
    Làm sạch THÔNG MINH với WHITELIST approach:
    
    Priority 1: Nếu dòng có technical keyword → GIỮ NGUYÊN
    Priority 2: Nếu dòng là pure meta (signature/timestamp) → BỎ
    Priority 3: Các dòng khác → xóa inline meta nhưng GIỮ phần còn lại
    
    Examples:
        Input:  "FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1"
        Output: "FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1"  ✅
        
        Input:  "FOUND AIR OUTLET DAMAGED BY VAE03251 ON 01.AUG.2025"
        Output: "FOUND AIR OUTLET DAMAGED"  ✅
        
        Input:  "DESCRIPTION SIGN VAE03251"
        Output: ""  (removed - pure meta)
    """
    if not s:
        return ""
    
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    
    for ln in lines:
        if not ln:
            continue
        
        # ========== PRIORITY 1: Technical content → KEEP ==========
        if _TECHNICAL_KEYWORDS.search(ln):
            # Remove inline meta but keep the technical content
            cleaned = _inline_meta_re.sub("", ln)
            cleaned = cleaned.strip()
            if cleaned:
                kept.append(cleaned)
            continue
        
        # ========== PRIORITY 2: Pure meta → DROP ==========
        if _pure_meta_re.match(ln):
            continue
        
        # ========== PRIORITY 3: Other lines → Clean inline meta ==========
        cleaned = _inline_meta_re.sub("", ln)
        cleaned = cleaned.strip()
        
        # Keep if has meaningful content (>5 chars, not just punctuation)
        if cleaned and len(cleaned) > 5 and re.search(r'[a-zA-Z]', cleaned):
            kept.append(cleaned)
    
    # Rút gọn khoảng trắng
    out = " ".join(kept)
    out = _whitespace_re.sub(" ", out)
    return out.strip()


# ============== TEST CASES ==============
if __name__ == "__main__":
    print("="*80)
    print("CLEANING TEST SUITE")
    print("="*80)
    
    test_cases = [
        # Test 1: FINDING (NRC) with technical content → KEEP
        {
            "name": "FINDING (NRC) with TASKCARD",
            "input": "FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1",
            "expect": "FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1",
        },
        
        # Test 2: Defect description with inline meta → KEEP (clean meta)
        {
            "name": "Defect with inline meta",
            "input": "FOUND AIR OUTLET AT F/O DAMAGED BY VAE03251 ON 01.AUG.2025",
            "expect": "FOUND AIR OUTLET AT F/O DAMAGED",
        },
        
        # Test 3: AMM reference → KEEP
        {
            "name": "AMM reference",
            "input": "REF AMM 21-21-44-000-001-A (AUG 01 2025)",
            "expect": "REF AMM 21-21-44-000-001-A (AUG 01 2025)",
        },
        
        # Test 4: IPC reference → KEEP
        {
            "name": "IPC reference",
            "input": "IPC 21-21-45 FIG 401 ITEM 21",
            "expect": "IPC 21-21-45 FIG 401 ITEM 21",
        },
        
        # Test 5: Pure signature → DROP
        {
            "name": "Pure signature",
            "input": "DESCRIPTION SIGN VAE03251",
            "expect": "",
        },
        
        # Test 6: Timestamp only → DROP
        {
            "name": "Timestamp only",
            "input": "06:30",
            "expect": "",
        },
        
        # Test 7: C/O REPLACE action → KEEP
        {
            "name": "C/O REPLACE",
            "input": "C/O REPLACED CABLE ASSY OF DUCT",
            "expect": "C/O REPLACED CABLE ASSY OF DUCT",
        },
        
        # Test 8: Full WO (your example)
        {
            "name": "Full WO Example",
            "input": """1 WORKSTEP ADDED BY VAE03251 ON 01.AUG.2025, 06:30
FINDING (NRC) TASKCARD 212100-01-1-1 (01) / ITEM 1
FOUND AIR OUTLET AT F/O (FIN: 4501HM) DAMAGED (MISSING FLAP)
DESCRIPTION SIGN VAE03251
ACTION PERFORMED BY VAE03251 ON 03.AUG.2025, 09:15
C/O REPLACE INDIVIDUAL AIR-OUTLET AT F/O SIDE
REF AMM 21-21-44-000-001-A (AUG 01 2025)
PERFORMED SIGN VAE03251""",
            "expect_contains": [
                "FINDING (NRC) TASKCARD",
                "FOUND AIR OUTLET",
                "DAMAGED",
                "C/O REPLACE",
                "REF AMM 21-21-44",
            ],
        },
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases, 1):
        result = clean_wo_text(tc["input"])
        
        if "expect" in tc:
            if result == tc["expect"]:
                print(f"✅ Test {i}: {tc['name']}")
                passed += 1
            else:
                print(f"❌ Test {i}: {tc['name']}")
                print(f"   Expected: {tc['expect']}")
                print(f"   Got:      {result}")
                failed += 1
        
        elif "expect_contains" in tc:
            missing = [phrase for phrase in tc["expect_contains"] if phrase not in result]
            if not missing:
                print(f"✅ Test {i}: {tc['name']}")
                passed += 1
            else:
                print(f"❌ Test {i}: {tc['name']}")
                print(f"   Missing phrases: {missing}")
                print(f"   Result: {result[:200]}...")
                failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
