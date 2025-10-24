"""
Enhanced Reference Extractor with Priority Logic

Business Rules:
1. Priority: TSM (#1) > IPC (#2) > AMM (#3) > FIM (#4)
2. CMM references are EXCLUDED (ATA codes are offset)
3. Task-only numbers (e.g., "21-26-40") are valid → Manual=UNKNOWN
4. Compressed formats (e.g., "292600") are parsed → "29-26-00"

Examples:
    "Refer to TSM 21-26-40 and AMM 21-26-50"
    → Returns: TSM 21-26-40 (highest priority)
    
    "See CMM 28-41-09"
    → Returns: None (CMM excluded)
    
    "Performed task 24-50-00"
    → Returns: Task 24-50-00, Manual=UNKNOWN
    
    "Refer to IPC 32-41-00-01"
    → Returns: IPC 32-41-00-01
"""
import regex as re
from typing import List, Dict, Optional, Tuple
from .constants import ATA_PATTERN


# ============================================================================
# PRIORITY CONFIGURATION
# ============================================================================
MANUAL_PRIORITY = {
    "TSM": 1,   # Troubleshooting Manual - Highest priority
    "IPC": 2,   # Illustrated Parts Catalog - High priority
    "AMM": 3,   # Aircraft Maintenance Manual - Medium priority
    "FIM": 4,   # Fault Isolation Manual - Low priority
    "CMM": 999, # Component Maintenance Manual - EXCLUDED (offset ATA)
    "UNKNOWN": 5, # Task-only (no prefix) - Lower than documented refs
}


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Pattern 1: With manual prefix (TSM, IPC, AMM, FIM, CMM)
REF_PATTERN_WITH_PREFIX = re.compile(
    r"\b(?P<manual>TSM|IPC|AMM|FIM|CMM|WDM|SRM)\s*[-:]?\s*"
    r"(?P<seq>(?:\d{2}[- ]?\d{2}(?:[- ]?\d{2,3}){0,3}|\d{6,8}))\b",
    flags=re.I,
)

# Pattern 2: Task-only (no prefix, strict format to avoid false positives)
# Matches: 21-26-40, 21-26-40-001, but NOT phone numbers or dates
TASK_ONLY_PATTERN = re.compile(
    r"(?<![A-Z\d])(?P<task>\d{2}[-]\d{2}(?:[-]\d{2,3}){0,3})(?![A-Z\d])",
    flags=re.I,
)

# Pattern 3: Compressed format (6-8 digits, no separators)
# Matches: 292600, 21264001
COMPRESSED_PATTERN = re.compile(
    r"(?<!\d)(?P<compressed>\d{6,8})(?!\d)",
)


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def _normalize_seq(seq: str) -> str:
    """
    Normalize sequence to standard format: AA-BB-CC-DD
    
    Examples:
        "21 26 40" → "21-26-40"
        "212640" → "21-26-40"
        "21-26-40-001" → "21-26-40-001"
    """
    # Extract all digits
    digits = re.sub(r"[^\d]", "", seq)
    
    if len(digits) < 4:
        return seq  # Too short, return as-is
    
    # Parse into pairs: AA-BB-CC-DD...
    aa, bb = digits[:2], digits[2:4]
    rest = digits[4:]
    
    parts = [aa, bb]
    
    # Split rest into 2-3 digit chunks
    while rest:
        if len(rest) >= 3:
            # Try 3-digit chunk first (for part numbers)
            parts.append(rest[:3])
            rest = rest[3:]
        elif len(rest) >= 2:
            parts.append(rest[:2])
            rest = rest[2:]
        else:
            parts.append(rest)
            rest = ""
    
    return "-".join(parts)


def _extract_ata04(normalized: str) -> Optional[str]:
    """
    Extract ATA04 (AA-BB) from normalized task.
    
    Examples:
        "21-26-40" → "21-26"
        "21-26-40-001" → "21-26"
        "21-26" → "21-26"
    """
    m = re.match(ATA_PATTERN, normalized)
    if m:
        aa, bb = m.group("aa"), m.group("bb")
        return f"{aa}-{bb}"
    
    # Fallback: extract first 2 pairs of digits
    parts = normalized.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    
    return None


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def _extract_with_prefix(text: str) -> List[Dict]:
    """Extract references with manual prefix."""
    results = []
    
    for m in REF_PATTERN_WITH_PREFIX.finditer(text):
        manual = m.group("manual").upper()
        raw = m.group("seq")
        normalized = _normalize_seq(raw)
        ata04 = _extract_ata04(normalized)
        
        results.append({
            "manual": manual,
            "raw": raw,
            "normalized": normalized,
            "ata04": ata04,
            "task": normalized,
            "priority": MANUAL_PRIORITY.get(manual, 999),
            "source": "prefix",
        })
    
    return results


def _extract_task_only(text: str) -> List[Dict]:
    """Extract task-only references (no prefix)."""
    results = []
    
    for m in TASK_ONLY_PATTERN.finditer(text):
        task = m.group("task")
        ata04 = _extract_ata04(task)
        
        results.append({
            "manual": "UNKNOWN",
            "raw": task,
            "normalized": task,
            "ata04": ata04,
            "task": task,
            "priority": MANUAL_PRIORITY["UNKNOWN"],
            "source": "task_only",
        })
    
    return results


def _extract_compressed(text: str) -> List[Dict]:
    """Extract compressed format (6-8 digits)."""
    results = []
    
    for m in COMPRESSED_PATTERN.finditer(text):
        compressed = m.group("compressed")
        
        # Skip if it looks like a date (YYYYMMDD pattern)
        if len(compressed) == 8 and compressed.startswith(('19', '20')):
            continue
        
        # Skip if it's part of a larger number
        if len(compressed) > 8:
            continue
        
        normalized = _normalize_seq(compressed)
        ata04 = _extract_ata04(normalized)
        
        results.append({
            "manual": "UNKNOWN",
            "raw": compressed,
            "normalized": normalized,
            "ata04": ata04,
            "task": normalized,
            "priority": MANUAL_PRIORITY["UNKNOWN"],
            "source": "compressed",
        })
    
    return results


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_citations(text: str, exclude_cmm: bool = True) -> List[Dict]:
    """
    Extract all citations from text.
    
    Args:
        text: Input text (WO Description or Action)
        exclude_cmm: If True, exclude CMM references (default: True)
    
    Returns:
        List of citation dicts sorted by priority
        
    Example:
        >>> citations = extract_citations("Refer to TSM 21-26-40 and AMM 21-26-50")
        >>> print(citations[0])
        {
            'manual': 'TSM',
            'normalized': '21-26-40',
            'ata04': '21-26',
            'task': '21-26-40',
            'priority': 1
        }
    """
    if not text:
        return []
    
    all_refs = []
    
    # Extract all types
    all_refs.extend(_extract_with_prefix(text))
    all_refs.extend(_extract_task_only(text))
    all_refs.extend(_extract_compressed(text))
    
    # Exclude CMM if requested
    if exclude_cmm:
        all_refs = [r for r in all_refs if r["manual"] != "CMM"]
    
    # Deduplicate by normalized task
    seen = {}
    unique_refs = []
    for ref in all_refs:
        key = ref["normalized"]
        if key not in seen or ref["priority"] < seen[key]["priority"]:
            seen[key] = ref
            if key in [r["normalized"] for r in unique_refs]:
                # Replace with higher priority
                unique_refs = [r for r in unique_refs if r["normalized"] != key]
            unique_refs.append(ref)
    
    # Sort by priority (lower number = higher priority)
    unique_refs.sort(key=lambda x: (x["priority"], x["normalized"]))
    
    return unique_refs


def get_best_citation(text: str, exclude_cmm: bool = True) -> Optional[Dict]:
    """
    Get single best citation based on priority.
    
    Priority order:
        1. TSM (Troubleshooting)
        2. IPC (Parts Catalog)
        3. AMM (Maintenance)
        4. FIM (Fault Isolation)
        5. UNKNOWN (task-only)
        ❌ CMM (excluded by default)
    
    Args:
        text: Input text
        exclude_cmm: Exclude CMM references (default: True)
    
    Returns:
        Best citation dict or None if no valid citations
        
    Example:
        >>> best = get_best_citation("TSM 21-26-40, AMM 21-26-50, CMM 21-26-60")
        >>> print(best['manual'], best['ata04'])
        TSM 21-26
    """
    citations = extract_citations(text, exclude_cmm=exclude_cmm)
    return citations[0] if citations else None


def extract_ata04_from_citation(text: str, exclude_cmm: bool = True) -> Optional[str]:
    """
    Extract ATA04 code from best citation.
    
    Args:
        text: Input text
        exclude_cmm: Exclude CMM references (default: True)
    
    Returns:
        ATA04 string (e.g., "21-26") or None
        
    Example:
        >>> ata04 = extract_ata04_from_citation("Refer to TSM 21-26-40")
        >>> print(ata04)
        21-26
    """
    best = get_best_citation(text, exclude_cmm=exclude_cmm)
    return best["ata04"] if best else None


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def extract_citations_legacy(text: str) -> List[Dict]:
    """
    Legacy function for backward compatibility.
    Behaves like old extract_citations but uses new logic.
    """
    citations = extract_citations(text, exclude_cmm=False)
    
    # Return in old format (remove new fields)
    return [{
        "manual": c["manual"],
        "raw": c["raw"],
        "normalized": c["normalized"],
        "ata04": c["ata04"],
        "task": c["task"],
    } for c in citations]


# ============================================================================
# VALIDATION & DEBUGGING
# ============================================================================

def validate_citation_extraction(test_cases: List[Tuple[str, Dict]]) -> Dict:
    """
    Validate extraction against test cases.
    
    Args:
        test_cases: List of (text, expected_result) tuples
    
    Returns:
        Validation report dict
    """
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    for text, expected in test_cases:
        actual = get_best_citation(text)
        
        # Compare key fields
        if actual and expected:
            if (actual["manual"] == expected.get("manual") and 
                actual["ata04"] == expected.get("ata04")):
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "text": text,
                    "expected": expected,
                    "actual": actual
                })
        elif not actual and not expected:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "text": text,
                "expected": expected,
                "actual": actual
            })
    
    return results


# ============================================================================
# EXAMPLE USAGE & TESTS
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Test 1: TSM priority
        ("Refer to TSM 21-26-40 and AMM 21-26-50", {"manual": "TSM", "ata04": "21-26"}),
        
        # Test 2: CMM exclusion
        ("See CMM 28-41-09 for overhaul", None),
        
        # Test 3: IPC priority over AMM
        ("IPC 32-41-00-01 and AMM 32-41-00-02", {"manual": "IPC", "ata04": "32-41"}),
        
        # Test 4: Task-only
        ("Performed task 24-50-00", {"manual": "UNKNOWN", "ata04": "24-50"}),
        
        # Test 5: Compressed format
        ("Refer to 292600", {"manual": "UNKNOWN", "ata04": "29-26"}),
        
        # Test 6: Multiple sources - TSM wins
        ("AMM 21-31-00, TSM 21-31-10, IPC 21-31-20", {"manual": "TSM", "ata04": "21-31"}),
        
        # Test 7: IPC only
        ("IPC 79-32-15-001", {"manual": "IPC", "ata04": "79-32"}),
    ]
    
    print("="*70)
    print("REFERENCE EXTRACTION - VALIDATION TESTS")
    print("="*70)
    
    report = validate_citation_extraction(test_cases)
    
    print(f"\nTotal tests: {report['total']}")
    print(f"✓ Passed: {report['passed']}")
    print(f"✗ Failed: {report['failed']}")
    
    if report['failures']:
        print("\nFailures:")
        for fail in report['failures']:
            print(f"\n  Text: {fail['text']}")
            print(f"  Expected: {fail['expected']}")
            print(f"  Actual: {fail['actual']}")
    
    print("\n" + "="*70)
    print("EXAMPLE EXTRACTIONS")
    print("="*70)
    
    examples = [
        "Refer to TSM 21-26-40 for troubleshooting ECAM FAULT",
        "IPC 32-41-00-01 shows brake assembly part number",
        "Performed task 24-50-00 as per maintenance schedule",
        "See CMM 28-41-09 for component overhaul procedures",
        "AMM 79-31-00, IPC 79-31-10, and TSM 79-31-20 referenced",
    ]
    
    for text in examples:
        best = get_best_citation(text)
        print(f"\nInput: {text}")
        if best:
            print(f"  → {best['manual']} {best['task']} (ATA04: {best['ata04']})")
        else:
            print("  → No valid citation found")
