# core/refs.py - OPTIMIZED VERSION
import regex as re
from .constants import ATA_PATTERN

# Pattern mở rộng hỗ trợ:
# - Suffix revision: -A, -B, -001A, etc.
# - Flexible separators: space, dash, colon
# - Date parentheses: (AUG 01 2025)
REF_PATTERN = re.compile(
    r"\b(?P<manual>TSM|FIM|AMM|ESPM)\s*[-:\s]?\s*"
    r"(?P<seq>(\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?([- ]?[A-Z])?))"
    r"(\s*\([^)]{1,20}\))?",  # Optional date/revision in parentheses
    flags=re.I,
)

def _normalize_seq(seq: str) -> str:
    """
    Normalize sequence to AA-BB-CC-DDD format.
    Examples:
        21-21-44-000-001-A → 21-21-44-000-001
        212144000001A → 21-21-44-000-001
        21 21 44 000 001 → 21-21-44-000-001
    """
    # Remove revision suffix (single letter at end)
    seq_clean = re.sub(r"[- ]?[A-Z]$", "", seq, flags=re.I)
    
    # Extract only digits
    digits = re.sub(r"[^\d]", "", seq_clean)
    
    if len(digits) >= 4:
        parts = []
        parts.append(digits[:2])   # AA
        parts.append(digits[2:4])  # BB
        
        rest = digits[4:]
        # Split remaining into 2-digit or 3-digit chunks
        while rest:
            if len(rest) >= 3 and rest[0] != '0':  # Likely 3-digit task number
                parts.append(rest[:3])
                rest = rest[3:]
            elif len(rest) >= 2:
                parts.append(rest[:2])
                rest = rest[2:]
            else:
                parts.append(rest)
                break
        
        return "-".join(parts)
    
    return seq

def extract_citations(text: str):
    """
    Extract manual citations from text with improved pattern matching.
    Returns list of dicts with keys: manual, raw, normalized, ata04, task
    """
    out = []
    if not text:
        return out
    
    for m in REF_PATTERN.finditer(text):
        manual = m.group("manual").upper()
        raw = m.group("seq")
        normalized = _normalize_seq(raw)
        
        # Extract ATA04 (first 5 chars: AA-BB)
        ata04 = None
        task = normalized
        
        m2 = re.match(ATA_PATTERN, normalized)
        if m2:
            aa, bb = m2.group("aa"), m2.group("bb")
            ata04 = f"{aa}-{bb}"
        elif len(normalized) >= 5 and normalized[2] == '-':
            # Fallback: simple extraction
            ata04 = normalized[:5]
        
        out.append({
            "manual": manual,
            "raw": raw,
            "normalized": normalized,
            "ata04": ata04,
            "task": task,
        })
    
    return out


# ============== TEST CASES ==============
if __name__ == "__main__":
    test_cases = [
        "REF AMM 21-21-44-000-001-A (AUG 01 2025)",
        "AMM 21-21-44-400-001-A (AUG 01 2025)",
        "TSM 32-41-00-400-801",
        "FIM 24 11 00 001",
        "REF: AMM 212144000001B",
    ]
    
    print("=== CITATION EXTRACTION TEST ===\n")
    for tc in test_cases:
        cites = extract_citations(tc)
        print(f"Input:  {tc}")
        for c in cites:
            print(f"  → {c['manual']} {c['normalized']} (ATA: {c['ata04']})")
        print()
