# core/refs.py - WITH IPC SUPPORT
import regex as re
from .constants import ATA_PATTERN

# Supported manual types
SUPPORTED_MANUALS = ["AMM", "TSM", "FIM", "ESPM", "IPC", "CMM", "WDM", "SRM"]

# ================================================================================
# PATTERN 1: Standard manuals (AMM, TSM, FIM, ESPM)
# Format: AMM 21-21-44-000-001-A, TSM 32-41-00-400-801
# ================================================================================
MANUAL_PATTERN = re.compile(
    r"(?:\b(?:REF|PER|IAW)\s+)?(?P<manual>TSM|FIM|AMM|ESPM)\s*[:;\s-]*\s*"
    r"(?P<seq>(\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?([- ]?[A-Z])?))"
    r"(\s*\([^)]{1,20}\))?",  # Optional date/revision in parentheses
    flags=re.I,
)

# ================================================================================
# PATTERN 2: IPC (Illustrated Parts Catalog)
# Format: IPC 21-21-45 FIG 401 ITEM 21
#         IPC REF: 24-11-03 Figure 102 Item 5
#         IPC 27-21-31-901-801 Sheet 1
# ================================================================================
IPC_PATTERN = re.compile(
    r"(?:\b(?:REF|PER|IAW)\s+)?(?P<manual>IPC)\s*[:;\s-]*\s*"
    r"(?P<seq>\d{2}[- ]?\d{2}[- ]?\d{2}([- ]?\d{2,})?)"  # At least AA-BB-CC format
    r"(?:\s+(?:FIG(?:URE)?|SHEET|SHT)[\.\s]*(?P<fig>\d+))?"  # Optional FIG/SHEET number
    r"(?:\s+(?:ITEM|IT|ITM)[\.\s]*(?P<item>\d+))?",  # Optional ITEM number
    flags=re.I,
)


def _normalize_seq(seq: str) -> str:
    """
    Normalize sequence to AA-BB-CC-DDD format.
    Examples:
        21-21-44-000-001-A → 21-21-44-000-001
        212144000001A → 21-21-44-000-001
        21 21 44 000 001 → 21-21-44-000-001
        21-21-45 → 21-21-45
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
    Extract manual citations from text with support for AMM/TSM/FIM/ESPM/IPC.
    Returns list of dicts with keys: manual, raw, normalized, ata04, task, fig, item
    
    Examples:
        "REF AMM 21-21-44-000-001-A" → {manual: AMM, ata04: 21-21, task: 21-21-44-000-001}
        "IPC 21-21-45 FIG 401 ITEM 21" → {manual: IPC, ata04: 21-21, task: 21-21-45, fig: 401, item: 21}
    """
    out = []
    if not text:
        return out
    
    # ========== Extract Standard Manuals (AMM/TSM/FIM/ESPM) ==========
    for m in MANUAL_PATTERN.finditer(text):
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
            "fig": None,
            "item": None,
        })
    
    # ========== Extract IPC References ==========
    for m in IPC_PATTERN.finditer(text):
        manual = "IPC"
        raw = m.group("seq")
        normalized = _normalize_seq(raw)
        fig = m.group("fig") if m.group("fig") else None
        item = m.group("item") if m.group("item") else None
        
        # Extract ATA04 (first 5 chars: AA-BB)
        ata04 = None
        if len(normalized) >= 5 and normalized[2] == '-':
            ata04 = normalized[:5]
        
        # Build task identifier (include FIG/ITEM for uniqueness)
        task = normalized
        if fig:
            task += f" FIG{fig}"
        if item:
            task += f" IT{item}"
        
        out.append({
            "manual": manual,
            "raw": raw,
            "normalized": normalized,
            "ata04": ata04,
            "task": task,
            "fig": fig,
            "item": item,
        })
    
    return out


# ============== TEST CASES ==============
if __name__ == "__main__":
    test_cases = [
        # Standard manuals
        ("REF AMM 21-21-44-000-001-A (AUG 01 2025)", "AMM", "21-21"),
        ("AMM: 21-52-24-000-001-A", "AMM", "21-52"),
        ("TSM 32-41-00-400-801", "TSM", "32-41"),
        ("PER FIM 24-11-00-001", "FIM", "24-11"),
        
        # IPC references
        ("IPC 21-21-45 FIG 401 ITEM 21", "IPC", "21-21"),
        ("IPC REF: 24-11-03 Figure 102 Item 5", "IPC", "24-11"),
        ("IPC 27-21-31-901-801 Sheet 1", "IPC", "27-21"),
        ("IPC 32-41-00 FIG. 201 IT. 12", "IPC", "32-41"),
        
        # Mixed
        ("REF AMM 21-21-44-000-001 AND IPC 21-21-45 FIG 401", "AMM,IPC", "21-21"),
    ]
    
    print("="*80)
    print("CITATION EXTRACTION TEST (with IPC support)")
    print("="*80)
    
    for i, (text, expected_manual, expected_ata) in enumerate(test_cases, 1):
        cites = extract_citations(text)
        print(f"\n{i}. Input: {text}")
        
        if cites:
            for c in cites:
                status = "✓" if c['ata04'] == expected_ata else "✗"
                print(f"   {status} {c['manual']} {c['normalized']}")
                if c['fig']:
                    print(f"      FIG: {c['fig']}", end="")
                if c['item']:
                    print(f" ITEM: {c['item']}", end="")
                if c['fig'] or c['item']:
                    print()
                print(f"      → ATA04: {c['ata04']}")
        else:
            print(f"   ✗ No citations found!")
    
    print("\n" + "="*80)
    print("IPC-SPECIFIC TEST")
    print("="*80)
    
    ipc_text = """
    FOUND DAMAGED BRACKET AT ENGINE PYLON.
    REF IPC 54-12-03 FIG 201 ITEM 5
    C/O REPLACED BRACKET P/N 123456-001
    IAW IPC 54-12-03 Figure 201 Item 5
    """
    
    cites = extract_citations(ipc_text)
    print(f"\nText: {ipc_text.strip()}")
    print(f"\nFound {len(cites)} citation(s):")
    for c in cites:
        print(f"  • {c['manual']} {c['normalized']} FIG{c.get('fig','')} IT{c.get('item','')}")
        print(f"    ATA04: {c['ata04']}")
