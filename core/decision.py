# core/decision.py - OPTIMIZED VERSION
from .constants import CONF_MATRIX

def _is_valid_ata(ata):
    """
    Kiểm tra ATA có hợp lệ không (không phải placeholder/dummy).
    Invalid: 00-00, 99-99, None, empty, quá ngắn
    """
    if not ata or not isinstance(ata, str):
        return False
    ata = ata.strip().upper()
    if len(ata) < 4:
        return False
    # Placeholder codes
    if ata in ["00-00", "0000", "99-99", "9999", "XX-XX"]:
        return False
    return True

def decide(e0, e1_valid, e1_ata, e2_best, e2_all):
    """
    Decision logic với xử lý E0 invalid.
    
    Priority:
    1. E0 = E1 = E2 (all valid) → CONFIRM (0.97)
    2. E1 = E2 ≠ E0 (E1 valid) → CORRECT (0.95)
    3. E0 invalid + E1 valid → CORRECT (0.92)
    4. E0 invalid + E2 valid → CORRECT (0.88)
    5. E2 = E0 (no E1) → CONFIRM (0.86)
    6. E1 only → CONFIRM (0.92)
    7. E1 ≠ E2 → REVIEW (0.60)
    8. E0 only → REVIEW (0.50)
    9. No evidence → REVIEW (0.40)
    """
    e0_valid = _is_valid_ata(e0)
    e2_ata = e2_best["ata04"] if e2_best else None
    e2_valid = _is_valid_ata(e2_ata)
    
    # === CASE 1: All agree ===
    if e0_valid and e1_valid and e2_valid and (e0 == e1_ata == e2_ata):
        return "CONFIRM", CONF_MATRIX["E0_E1_E2_VALID"], "E0 = E1 = E2 (tất cả hợp lệ)."
    
    # === CASE 2: E1 = E2, both valid, differ from E0 ===
    if e1_valid and e2_valid and (e1_ata == e2_ata):
        if e0_valid and (e0 != e1_ata):
            return "CORRECT", CONF_MATRIX["E1_E2_NEQ_E0_VALID"], f"E1 = E2 = {e1_ata} ≠ E0 ({e0}) → sửa."
        elif not e0_valid:
            return "CORRECT", 0.92, f"E1 = E2 = {e1_ata}; E0 invalid → sửa."
    
    # === CASE 3: E0 invalid, but E1 valid ===
    if not e0_valid and e1_valid and e1_ata:
        return "CORRECT", 0.92, f"E0 invalid ('{e0}'), E1 citation hợp lệ: {e1_ata} → sửa."
    
    # === CASE 4: E0 invalid, but E2 valid (no E1) ===
    if not e0_valid and e2_valid and e2_ata:
        return "CORRECT", 0.88, f"E0 invalid ('{e0}'), E2 catalog hợp lệ: {e2_ata} → sửa."
    
    # === CASE 5: E2 matches E0, no E1 ===
    if e0_valid and e2_valid and (e2_ata == e0) and not e1_valid:
        return "CONFIRM", CONF_MATRIX["E2_EQ_E0_ONLY"], f"E2 = E0 = {e0}; không có E1."
    
    # === CASE 6: Only E1 valid ===
    if e1_valid and e1_ata and not e2_valid:
        if e0_valid and (e0 == e1_ata):
            return "CONFIRM", CONF_MATRIX["E1_ONLY_VALID"], f"E1 = E0 = {e1_ata}."
        elif e0_valid:
            return "CORRECT", 0.90, f"E1 hợp lệ ({e1_ata}) ≠ E0 ({e0}) → sửa."
        else:
            return "CORRECT", 0.88, f"E1 hợp lệ ({e1_ata}); E0 invalid → sửa."
    
    # === CASE 7: Conflict E1 ≠ E2 ===
    if e1_valid and e2_valid and (e1_ata != e2_ata):
        return "REVIEW", CONF_MATRIX["CONFLICT"], f"E1 ({e1_ata}) ≠ E2 ({e2_ata}) → cần xem xét."
    
    # === CASE 8: Only E0 valid ===
    if e0_valid and not (e1_valid or e2_valid):
        return "REVIEW", CONF_MATRIX["E0_ONLY"], f"Chỉ có E0 ({e0}), không có bằng chứng bổ sung."
    
    # === CASE 9: No valid evidence ===
    return "REVIEW", 0.40, f"Thiếu bằng chứng rõ ràng (E0: '{e0}', E1: {e1_valid}, E2: {e2_valid})."


# ============== TEST CASES ==============
if __name__ == "__main__":
    print("="*60)
    print("DECISION LOGIC TEST CASES")
    print("="*60)
    
    test_cases = [
        # (e0, e1_valid, e1_ata, e2_ata, expected_decision)
        ("21-21", True, "21-21", "21-21", "CONFIRM"),
        ("21-21", True, "21-52", "21-52", "CORRECT"),
        ("00-00", True, "21-52", None, "CORRECT"),  # Invalid E0
        ("00-00", False, None, "21-52", "CORRECT"),  # Invalid E0, catalog only
        ("21-21", False, None, "21-21", "CONFIRM"),
        ("21-21", True, "21-52", None, "CORRECT"),
        ("21-21", False, None, None, "REVIEW"),
        (None, False, None, None, "REVIEW"),
    ]
    
    for i, (e0, e1_v, e1_a, e2_a, expect) in enumerate(test_cases, 1):
        e2_best = {"ata04": e2_a, "score": 0.8} if e2_a else None
        dec, conf, reason = decide(e0, e1_v, e1_a, e2_best, None)
        status = "✓" if dec == expect else "✗"
        print(f"{status} Case {i}: E0={e0}, E1={e1_a}, E2={e2_a}")
        print(f"  → {dec} ({conf:.2f}): {reason}")
        if dec != expect:
            print(f"  ⚠ Expected: {expect}")
        print()
