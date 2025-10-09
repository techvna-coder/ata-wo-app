# core/cleaning.py - OPTIMIZED VERSION
from __future__ import annotations
import regex as re

# ================================================================================
# NGUYÊN TẮC MỚI:
# 1. CHỈ loại bỏ header/footer/signature KHÔNG CÓ thông tin kỹ thuật
# 2. GIỮ LẠI toàn bộ dòng có: AMM/TSM/FIM, FOUND/DAMAGED/LEAK/REPLACE, ATA code
# 3. Loại bỏ metadata đơn thuần (BY xxx ON dd.MMM.yyyy, timestamp)
# ================================================================================

# Pattern cho dòng THUẦN TÚY là meta (không có keyword kỹ thuật)
_PURE_META_PATTERNS = [
    r"^\s*\d+\s+WORKSTEP\s+ADDED\s+BY\s+[A-Z]{3}\d+\s+ON\s+\d{2}\.[A-Z]{3}\.\d{4}[,\s]+\d{2}:\d{2}\s*$",
    r"^\s*(DESCRIPTION|ACTION\s+PERFORMED|PERFORMED)\s+SIGN\s+[A-Z]{3}\d+\s*$",
    r"^\s*BY\s+[A-Z]{3}\d{5}\s*$",
    r"^\s*ON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\s*$",
    r"^\s*\d{1,2}:\d{2}\s*$",
]

# Keyword kỹ thuật PHẢI GIỮ (mở rộng)
_TECHNICAL_KEYWORDS = re.compile(
    r"\b(AMM|TSM|FIM|ESPM|SB|AD|MEL|"  # Manuals & docs
    r"ECAM|EICAS|CAS|FWS|ACARS|"       # Systems
    r"FOUND|DAMAGE|CRACK|LEAK|FAIL|FAULT|INOP|OVERHEAT|SMOKE|VIBRAT|"  # Symptoms
    r"REPLACE|REPAIR|RECTIFY|TROUBLESHOOT|ADJUST|CALIBRAT|RIG|"  # Actions
    r"C/O|REF|TASKCARD|"  # Common abbreviations
    r"MISSING|BROKEN|WORN|ERODED|CORRODED|CONTAMINATED|"  # Conditions
    r"CHECK\s+SATIS|WITHIN\s+LIMITS|NO\s+ABNORMAL|"  # Findings
    r"\d{2}-\d{2}(-\d{2})?(-\d+)?)\b",  # ATA pattern
    re.I
)

_pure_meta_re = re.compile("|".join(_PURE_META_PATTERNS), re.I)
_whitespace_re = re.compile(r"[ \t]+")

def clean_wo_text(s: str) -> str:
    """
    Làm sạch THÔNG MINH:
    - Loại bỏ chỉ những dòng thuần túy là meta/signature (không có keyword kỹ thuật)
    - Giữ lại tất cả dòng có thông tin kỹ thuật (AMM ref, symptom, action...)
    - Xóa inline metadata trong dòng (BY xxx, ON date) nhưng GIỮ phần còn lại
    """
    if not s:
        return ""
    
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", str(s))]
    kept = []
    
    for ln in lines:
        if not ln:
            continue
        
        # Nếu dòng có keyword kỹ thuật → GIỮ NGUYÊN
        if _TECHNICAL_KEYWORDS.search(ln):
            kept.append(ln)
            continue
        
        # Nếu là pure meta (signature/timestamp đơn thuần) → BỎ
        if _pure_meta_re.match(ln):
            continue
        
        # Các dòng khác: xóa inline meta nhưng giữ phần còn lại
        cleaned_line = ln
        # Xóa "BY VAAxxxxx" ở giữa câu
        cleaned_line = re.sub(r"\s+BY\s+[A-Z]{3}\d{5}\b", "", cleaned_line, flags=re.I)
        # Xóa "ON dd.MMM.yyyy" ở giữa câu  
        cleaned_line = re.sub(r"\s+ON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b", "", cleaned_line, flags=re.I)
        # Xóa timestamp "hh:mm"
        cleaned_line = re.sub(r"\s+\d{1,2}:\d{2}\b", "", cleaned_line)
        
        cleaned_line = cleaned_line.strip()
        if cleaned_line and len(cleaned_line) > 5:  # Giữ nếu còn nội dung có nghĩa
            kept.append(cleaned_line)
    
    # Rút gọn khoảng trắng
    out = " ".join(kept)
    out = _whitespace_re.sub(" ", out)
    return out.strip()


# ============== TEST CASES ==============
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
    
    cleaned = clean_wo_text(test_wo)
    print("=== CLEANED OUTPUT ===")
    print(cleaned)
    print("\n=== LENGTH ===")
    print(f"Original: {len(test_wo)} chars")
    print(f"Cleaned: {len(cleaned)} chars")
    print("\n=== KEY CONTENT PRESERVED? ===")
    print(f"AMM refs: {'✓' if 'AMM 21-21-44' in cleaned else '✗'}")
    print(f"FOUND/DAMAGED: {'✓' if 'FOUND' in cleaned and 'DAMAGED' in cleaned else '✗'}")
    print(f"REPLACE: {'✓' if 'REPLACE' in cleaned else '✗'}")
