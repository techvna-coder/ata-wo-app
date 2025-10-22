# core/cleaning_enhanced.py
"""
Enhanced text cleaning with:
1. Consistent reference normalization (AMM/TSM/FIM/ECAM)
2. Better meta-pattern removal
3. Preserves technical context
4. Handles multiple languages (EN/VN)
"""
from __future__ import annotations
import regex as re
from typing import List, Tuple, Optional

# ============================================================================
# META/AUDIT PATTERNS (Non-technical content to remove)
# ============================================================================
_META_PATTERNS = [
    # Workstep & personnel
    r"\b(WORKSTEP\s+ADDED\s+BY|ACTION\s+PERFORMED\s+BY|PERFORMED\s+SIGN|DESCRIPTION\s+SIGN)\b.*",
    r"\bFINDING\s*\(NRC\)\b.*",
    r"\bPART\s+REQUIREMENT\b.*",
    
    # Service orders & personnel codes
    r"\bS\.?O[-–][A-Z0-9\-\.]+\b.*",              # S.O-A321-3348-2019
    r"\bBY\s+[A-Z]{3}\d{5}\b.*",                  # BY VAE03251
    r"\bON\s+\d{1,2}\.[A-Z]{3}\.\d{4}\b.*",       # 05.AUG.2025
    r"\b\d{1,2}[:.]\d{2}\b.*",                    # 09:24, 11:22
    r"^\s*\d+\s*WORKSTEP\S*\b.*",                 # "1 WORKSTEP..."
    
    # Form fields & boilerplate
    r"\b(refer to|see also|as per|in accordance with|IAW)\s+(MEL|CDL|NEF|MMEL)\b.*",
    r"\b(ITEM|CATEGORY|REFERENCE|TASK CARD|JOB CARD)\s*\d+\b.*",
    
    # Vietnamese meta
    r"\b(thực hiện|ký bởi|người thực hiện|ngày thực hiện)\b.*",
]

_meta_re = re.compile("|".join(_META_PATTERNS), re.I)


# ============================================================================
# TECHNICAL KEYWORDS (Content to preserve)
# ============================================================================
_KEEP_HINT = re.compile(
    r"\b(AMM|TSM|FIM|IPC|WDM|SRM|"                  # Manuals
    r"ECAM|EICAS|CAS|CFDS|"                         # Systems
    r"FAULT|FAIL|LEAK|OVERHEAT|VIBRAT|SMOKE|"      # Symptoms
    r"WARNING|CAUTION|ADVISORY|"                    # Alerts
    r"INOP|INOPERATIVE|UNSERVICEABLE|"             # Status
    r"MEL|CDL|NEF|AOG|"                             # Operational
    r"LRU|LINE REPLACEABLE UNIT|"                   # Hardware
    r"BITE|BUILT.IN.TEST|"                          # Diagnostics
    r"FWC|FAULT WARNING|"                           # More systems
    r"ATA\s*\d{2})\b",
    re.I,
)


# ============================================================================
# REFERENCE NORMALIZATION
# ============================================================================
def normalize_manual_reference(text: str) -> str:
    """
    Normalize manual references to consistent format:
    - "AMM 21 51 00" → "AMM 21-51-00"
    - "TSM21-51-00-400-001" → "TSM 21-51-00-400-001"
    - "FIM  21  51" → "FIM 21-51"
    
    This ensures TF-IDF treats equivalent references as same token.
    """
    def _normalize_match(m):
        manual = m.group("manual").upper()
        seq = m.group("seq")
        
        # Extract all digits
        digits = re.findall(r"\d+", seq)
        
        if not digits:
            return m.group(0)  # Keep original if no digits
        
        # Format based on length
        if len(digits) == 1:
            # Single number (rare)
            return f"{manual} {digits[0]}"
        elif len(digits) == 2:
            # Two parts: "AMM 21 51" → "AMM 21-51"
            return f"{manual} {digits[0]}-{digits[1]}"
        elif len(digits) >= 3:
            # Three+ parts: "AMM 21 51 00 400 001" → "AMM 21-51-00-400-001"
            return f"{manual} {'-'.join(digits)}"
        else:
            return m.group(0)
    
    # Pattern matches manual references with flexible spacing/separators
    pattern = re.compile(
        r"\b(?P<manual>TSM|FIM|AMM|IPC|WDM|SRM)\s*[-:]?\s*(?P<seq>(?:\d+\s*[-\s]*)+\d+)\b",
        flags=re.I
    )
    
    return pattern.sub(_normalize_match, text)


def normalize_system_messages(text: str) -> str:
    """
    Normalize system messages:
    - "ECAM FAULT" → "ECAM-FAULT"
    - "BITE FAIL" → "BITE-FAIL"
    
    Hyphenated form improves TF-IDF token consistency.
    """
    # System + Status patterns
    patterns = [
        (r"\b(ECAM|EICAS|CAS|CFDS)\s+(FAULT|FAIL|WARN|CAUTION|ADVISORY)\b", r"\1-\2"),
        (r"\b(BITE|FWC)\s+(FAULT|FAIL|TEST)\b", r"\1-\2"),
        (r"\b(LRU|UNIT)\s+(FAIL|FAULT|INOP)\b", r"\1-\2"),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.I)
    
    return result


# ============================================================================
# MAIN CLEANING FUNCTION
# ============================================================================
def clean_wo_text(s: str, preserve_structure: bool = False) -> str:
    """
    Enhanced WO text cleaning:
    
    Args:
        s: Input text (defect or rectification)
        preserve_structure: If True, keep line breaks (useful for display)
    
    Returns:
        Cleaned text with:
        - Meta/audit content removed
        - References normalized
        - Technical context preserved
        - Consistent whitespace
    """
    if not s:
        return ""
    
    # Step 1: Normalize references BEFORE removing lines
    # This ensures "AMM 21 51 00" is recognized even in meta lines
    text = normalize_manual_reference(str(s))
    text = normalize_system_messages(text)
    
    # Step 2: Line-by-line filtering
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text)]
    kept = []
    
    for ln in lines:
        if not ln:
            continue
        
        # Always keep lines with technical hints
        if _KEEP_HINT.search(ln):
            kept.append(ln)
            continue
        
        # Remove meta/audit lines
        if _meta_re.search(ln):
            continue
        
        kept.append(ln)
    
    # Step 3: Compose output
    if preserve_structure:
        # Keep line breaks for readability
        result = "\n".join(kept)
    else:
        # Single line for TF-IDF
        result = " ".join(kept)
    
    # Step 4: Clean whitespace
    result = re.sub(r"\s+", " ", result)
    
    return result.strip()


# ============================================================================
# ADVANCED: EXTRACT STRUCTURED INFO
# ============================================================================
def extract_technical_keywords(text: str) -> List[str]:
    """
    Extract technical keywords for analysis/debugging.
    Returns list of: [manual refs, system msgs, ATA codes, etc.]
    """
    keywords = []
    
    # Manual references
    manual_pattern = re.compile(
        r"\b(TSM|FIM|AMM|IPC|WDM|SRM)\s+[\d\-]+\b",
        re.I
    )
    keywords.extend(manual_pattern.findall(text))
    
    # System messages
    system_pattern = re.compile(
        r"\b(ECAM|EICAS|CAS|BITE|FWC)[-\s](FAULT|FAIL|WARN|TEST)\b",
        re.I
    )
    keywords.extend([f"{m[0]}-{m[1]}".upper() for m in system_pattern.findall(text)])
    
    # ATA codes
    ata_pattern = re.compile(r"\bATA\s*\d{2}[-\s]?\d{2}\b", re.I)
    keywords.extend(ata_pattern.findall(text))
    
    return list(set(keywords))  # Deduplicate


def split_defect_action(text: str) -> Tuple[str, str]:
    """
    Smart split of combined defect+action text.
    Useful when text contains both in single field.
    
    Returns: (defect_part, action_part)
    """
    # Common separators
    separators = [
        r"\|\s*RECTIFICATION:",
        r"\|\s*ACTION:",
        r"\|\s*CORRECTIVE ACTION:",
        r"\bRECTIFICATION:\s*",
        r"\bACTION PERFORMED:\s*",
        r"\bCORRECTIVE ACTION:\s*",
    ]
    
    for sep in separators:
        parts = re.split(sep, text, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    
    # No clear separator found
    return text.strip(), ""


# ============================================================================
# UTILITY: COMPARISON HELPERS
# ============================================================================
def normalize_for_comparison(text: str) -> str:
    """
    Aggressive normalization for fuzzy matching.
    Used when comparing WO descriptions for deduplication.
    """
    # Apply standard cleaning
    cleaned = clean_wo_text(text)
    
    # Further normalization
    result = cleaned.lower()
    result = re.sub(r"[^\w\s-]", "", result)  # Remove punctuation
    result = re.sub(r"\s+", " ", result)      # Normalize spaces
    result = result.strip()
    
    return result


def text_similarity_score(text1: str, text2: str) -> float:
    """
    Quick similarity score (0.0 to 1.0) based on token overlap.
    Useful for detecting near-duplicate WOs.
    """
    tokens1 = set(normalize_for_comparison(text1).split())
    tokens2 = set(normalize_for_comparison(text2).split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
# If importing old cleaning module, this will work transparently
__all__ = [
    "clean_wo_text",
    "normalize_manual_reference",
    "normalize_system_messages",
    "extract_technical_keywords",
    "split_defect_action",
    "normalize_for_comparison",
    "text_similarity_score",
]
