# core/decision.py - V4 OPTIMIZED
"""
LOGIC MỚI - ƯU TIÊN E1/E2 THAY VÌ E0:

E0 (ATA Entered): Chỉ tham khảo, không tin tưởng tuyệt đối
E1 (Citation from AMM/TSM/FIM/IPC): Độ tin cậy CAO (từ tài liệu chính thức)
E2 (Catalog TF-IDF): Độ tin cậy TỐT (từ lịch sử WO tương tự)

PRIORITY:
1. E1 valid → Dùng E1 (confidence cao)
2. E1 invalid, E2 strong (score ≥ 0.4) → Dùng E2
3. E1 và E2 đều hợp lệ nhưng khác nhau → So sánh score, chọn tốt hơn
4. Không có E1/E2 → Dùng E0 (nhưng confidence thấp)
5. E0 invalid + không có E1/E2 → REVIEW

CONFIDENCE MATRIX (updated):
- E1 alone (strong): 0.92
- E2 strong (score ≥ 0.5): 0.88
- E1 = E2: 0.95
- E1 ≠ E2 (E1 win): 0.85
- E1 ≠ E2 (E2 win): 0.82
- E0 only (valid): 0.50
- No valid evidence: 0.30
"""

from typing import Optional, Dict, Any, Tuple

# ============================================================
# CONFIDENCE MATRIX
# ============================================================

CONFIDENCE = {
    # E1 scenarios (Citation from manual)
    "E1_STRONG": 0.92,              # E1 alone, strong evidence
    "E1_E2_AGREE": 0.95,            # E1 = E2 (high confidence)
    "E1_E2_DISAGREE_E1_WIN": 0.85,  # E1 ≠ E2, but E1 chosen
    
    # E2 scenarios (Catalog TF-IDF)
    "E2_STRONG": 0.88,              # E2 alone, score ≥ 0.5
    "E2_MEDIUM": 0.80,              # E2 alone, score 0.4-0.5
    "E2_WEAK": 0.70,                # E2 alone, score 0.3-0.4
    "E1_E2_DISAGREE_E2_WIN": 0.82,  # E1 ≠ E2, but E2 chosen
    
    # E0 scenarios (Manual entry - low trust)
    "E0_CONFIRMED": 0.75,           # E0 khớp với E1 hoặc E2
    "E0_ONLY": 0.50,                # Chỉ có E0 valid
    
    # Edge cases
    "NO_EVIDENCE": 0.30,            # Không có evidence nào
    "REVIEW": 0.60,                 # Cần review thủ công
}

# Score thresholds cho E2
E2_STRONG_THRESHOLD = 0.50
E2_MEDIUM_THRESHOLD = 0.40
E2_WEAK_THRESHOLD = 0.30


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _is_valid_ata(ata: Optional[str]) -> bool:
    """
    Kiểm tra ATA có hợp lệ không.
    Invalid: None, empty, placeholder (00-00, 99-99), quá ngắn
    """
    if not ata or not isinstance(ata, str):
        return False
    ata = ata.strip().upper()
    if len(ata) < 4:
        return False
    # Placeholder codes
    if ata in ["00-00", "0000", "99-99", "9999", "XX-XX", "00", "99"]:
        return False
    return True


def _normalize_ata(ata: Optional[str]) -> Optional[str]:
    """Chuẩn hóa format ATA về AA-BB"""
    if not _is_valid_ata(ata):
        return None
    ata = ata.strip().upper()
    # Remove non-digits except dash
    import re
    digits = re.sub(r"[^\d\-]", "", ata)
    # Ensure AA-BB format
    if "-" in digits and len(digits) >= 5:
        parts = digits.split("-")
        return f"{parts[0][:2]}-{parts[1][:2]}"
    elif digits.isdigit() and len(digits) >= 4:
        return f"{digits[:2]}-{digits[2:4]}"
    return ata if len(ata) >= 4 else None


def _e2_confidence_level(e2_best: Optional[Dict[str, Any]]) -> str:
    """Xác định mức độ tin cậy của E2 dựa trên score"""
    if not e2_best or "score" not in e2_best:
        return "NONE"
    
    score = float(e2_best.get("score", 0))
    
    if score >= E2_STRONG_THRESHOLD:
        return "STRONG"
    elif score >= E2_MEDIUM_THRESHOLD:
        return "MEDIUM"
    elif score >= E2_WEAK_THRESHOLD:
        return "WEAK"
    else:
        return "NONE"


# ============================================================
# MAIN DECISION FUNCTION
# ============================================================

def decide(
    e0: Optional[str],
    e1_valid: bool,
    e1_ata: Optional[str],
    e2_best: Optional[Dict[str, Any]],
    e2_all: Optional[Any] = None
) -> Tuple[str, float, str]:
    """
    Quyết định ATA04_Final với logic ưu tiên E1/E2.
    
    Args:
        e0: ATA Entered (manual entry) - CHỈ THAM KHẢO
        e1_valid: E1 có valid không (technical defect + có citation)
        e1_ata: ATA04 từ citation
        e2_best: Best prediction từ catalog {"ata04": str, "score": float, ...}
        e2_all: All predictions (không dùng trong V4)
    
    Returns:
        (decision: str, confidence: float, reason: str)
        decision: "CONFIRM" | "CORRECT" | "REVIEW"
    """
    
    # Normalize inputs
    e0_norm = _normalize_ata(e0)
    e1_norm = _normalize_ata(e1_ata) if e1_valid else None
    e2_norm = _normalize_ata(e2_best.get("ata04")) if e2_best else None
    e2_level = _e2_confidence_level(e2_best)
    e2_score = float(e2_best.get("score", 0)) if e2_best else 0
    
    e0_valid = _is_valid_ata(e0_norm)
    
    # ===== PRIORITY 1: E1 (Citation) - Highest Trust =====
    
    if e1_valid and e1_norm:
        # Case 1.1: E1 = E2 (cả hai đồng ý)
        if e2_norm and e1_norm == e2_norm:
            decision = "CONFIRM" if e0_valid and e0_norm == e1_norm else "CORRECT"
            return decision, CONFIDENCE["E1_E2_AGREE"], f"E1 = E2 = {e1_norm} (rất tin cậy)"
        
        # Case 1.2: E1 alone hoặc E2 weak/disagree
        if not e2_norm or e2_level in ["NONE", "WEAK"]:
            decision = "CONFIRM" if e0_valid and e0_norm == e1_norm else "CORRECT"
            return decision, CONFIDENCE["E1_STRONG"], f"E1 citation: {e1_norm} (từ manual)"
        
        # Case 1.3: E1 ≠ E2, cả hai strong → So sánh
        if e2_norm and e1_norm != e2_norm and e2_level in ["STRONG", "MEDIUM"]:
            # E1 thắng nếu E2 score không quá cao (< 0.6)
            if e2_score < 0.60:
                decision = "CORRECT" if not e0_valid or e0_norm != e1_norm else "CONFIRM"
                return decision, CONFIDENCE["E1_E2_DISAGREE_E1_WIN"], \
                       f"E1={e1_norm} vs E2={e2_norm}: Chọn E1 (citation > catalog)"
            
            # E2 thắng nếu score rất cao (≥ 0.6) và E2 khớp E0
            if e2_score >= 0.60 and e0_valid and e0_norm == e2_norm:
                decision = "CORRECT"
                return decision, CONFIDENCE["E1_E2_DISAGREE_E2_WIN"], \
                       f"E1={e1_norm} vs E2={e2_norm}: Chọn E2 (score cao + khớp E0)"
            
            # Mặc định: E1 thắng
            decision = "CORRECT"
            return decision, CONFIDENCE["E1_E2_DISAGREE_E1_WIN"], \
                   f"E1={e1_norm} vs E2={e2_norm}: Chọn E1 (citation ưu tiên)"
    
    # ===== PRIORITY 2: E2 (Catalog) - Good Trust =====
    
    if e2_norm and e2_level != "NONE":
        # Case 2.1: E2 strong
        if e2_level == "STRONG":
            decision = "CONFIRM" if e0_valid and e0_norm == e2_norm else "CORRECT"
            return decision, CONFIDENCE["E2_STRONG"], \
                   f"E2 catalog: {e2_norm} (score={e2_score:.3f}, strong)"
        
        # Case 2.2: E2 medium
        if e2_level == "MEDIUM":
            decision = "CONFIRM" if e0_valid and e0_norm == e2_norm else "CORRECT"
            return decision, CONFIDENCE["E2_MEDIUM"], \
                   f"E2 catalog: {e2_norm} (score={e2_score:.3f}, medium)"
        
        # Case 2.3: E2 weak
        if e2_level == "WEAK":
            # Nếu E0 valid và khớp E2 → CONFIRM
            if e0_valid and e0_norm == e2_norm:
                return "CONFIRM", CONFIDENCE["E0_CONFIRMED"], \
                       f"E2 weak ({e2_score:.3f}) nhưng khớp E0={e2_norm}"
            # Nếu E0 invalid hoặc khác E2 → REVIEW
            return "REVIEW", CONFIDENCE["E2_WEAK"], \
                   f"E2 weak ({e2_score:.3f}): {e2_norm}, cần review"
    
    # ===== PRIORITY 3: E0 Only (Low Trust) =====
    
    if e0_valid:
        return "REVIEW", CONFIDENCE["E0_ONLY"], \
               f"Chỉ có E0={e0_norm}, không có E1/E2 (confidence thấp)"
    
    # ===== PRIORITY 4: No Evidence =====
    
    return "REVIEW", CONFIDENCE["NO_EVIDENCE"], \
           "Không có evidence hợp lệ (E0/E1/E2)"


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("DECISION LOGIC V4 - TEST CASES")
    print("="*80)
    
    test_cases = [
        # E1 strong cases
        {
            "name": "E1 alone, strong",
            "e0": "00-00",
            "e1_valid": True,
            "e1_ata": "21-52",
            "e2_best": None,
            "expected_decision": "CORRECT",
            "expected_conf": 0.92,
        },
        {
            "name": "E1 = E2, both agree",
            "e0": "21-21",
            "e1_valid": True,
            "e1_ata": "21-52",
            "e2_best": {"ata04": "21-52", "score": 0.85},
            "expected_decision": "CORRECT",
            "expected_conf": 0.95,
        },
        {
            "name": "E1 = E0 = E2, all agree",
            "e0": "21-52",
            "e1_valid": True,
            "e1_ata": "21-52",
            "e2_best": {"ata04": "21-52", "score": 0.75},
            "expected_decision": "CONFIRM",
            "expected_conf": 0.95,
        },
        
        # E1 vs E2 conflict
        {
            "name": "E1 ≠ E2, E1 win (E2 weak)",
            "e0": "21-21",
            "e1_valid": True,
            "e1_ata": "21-52",
            "e2_best": {"ata04": "21-21", "score": 0.45},
            "expected_decision": "CORRECT",
            "expected_conf": 0.85,
        },
        {
            "name": "E1 ≠ E2, E2 win (E2 very strong + match E0)",
            "e0": "32-41",
            "e1_valid": True,
            "e1_ata": "32-40",
            "e2_best": {"ata04": "32-41", "score": 0.68},
            "expected_decision": "CORRECT",
            "expected_conf": 0.82,
        },
        
        # E2 only cases
        {
            "name": "E2 strong alone",
            "e0": "00-00",
            "e1_valid": False,
            "e1_ata": None,
            "e2_best": {"ata04": "24-11", "score": 0.72},
            "expected_decision": "CORRECT",
            "expected_conf": 0.88,
        },
        {
            "name": "E2 medium + E0 match",
            "e0": "27-21",
            "e1_valid": False,
            "e1_ata": None,
            "e2_best": {"ata04": "27-21", "score": 0.45},
            "expected_decision": "CONFIRM",
            "expected_conf": 0.80,
        },
        
        # E0 only (low confidence)
        {
            "name": "E0 only, valid",
            "e0": "21-21",
            "e1_valid": False,
            "e1_ata": None,
            "e2_best": None,
            "expected_decision": "REVIEW",
            "expected_conf": 0.50,
        },
        
        # No evidence
        {
            "name": "E0 invalid, no E1/E2",
            "e0": "00-00",
            "e1_valid": False,
            "e1_ata": None,
            "e2_best": None,
            "expected_decision": "REVIEW",
            "expected_conf": 0.30,
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases, 1):
        decision, conf, reason = decide(
            tc["e0"],
            tc["e1_valid"],
            tc["e1_ata"],
            tc["e2_best"]
        )
        
        dec_match = decision == tc["expected_decision"]
        conf_match = abs(conf - tc["expected_conf"]) < 0.05
        
        if dec_match and conf_match:
            print(f"\n✅ Test {i}: {tc['name']}")
            passed += 1
        else:
            print(f"\n❌ Test {i}: {tc['name']}")
            failed += 1
            if not dec_match:
                print(f"   Decision: Expected {tc['expected_decision']}, got {decision}")
            if not conf_match:
                print(f"   Confidence: Expected {tc['expected_conf']:.2f}, got {conf:.2f}")
        
        print(f"   Result: {decision} (conf={conf:.2f})")
        print(f"   Reason: {reason}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{len(test_cases)} passed")
    print("="*80)
