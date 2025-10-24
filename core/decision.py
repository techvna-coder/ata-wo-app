"""
Enhanced Decision Logic with Source-Based Confidence

Tri-validation: E0 (Entered) vs E1 (Citation) vs E2 (Catalog/LLM)

Confidence Matrix:
- E0=E1=E2 (E1 valid)     → CONFIRM (0.97)
- E1=E2≠E0 (E1 valid)     → CORRECT (0.95)
- E2=E0 (no E1)           → CONFIRM (0.83-0.88)
- E1 only (valid)         → CONFIRM (0.92)
- E1≠E2 (conflict)        → REVIEW (0.70-0.90, depends on E1 source)
- E0 only                 → REVIEW (0.50)

Source Priority Impact on E1≠E2 Conflict:
- TSM → 0.85 (Troubleshooting has highest confidence)
- IPC → 0.80 (Parts catalog is reliable)
- AMM → 0.75 (Maintenance manual is good)
- UNKNOWN → 0.70 (Task-only, no documentation context)
"""
from typing import Optional, Tuple
from .constants import CONF_MATRIX


# ============================================================================
# SOURCE-BASED CONFIDENCE FOR CONFLICTS
# ============================================================================

SOURCE_CONFIDENCE_ON_CONFLICT = {
    "TSM": 0.85,      # Troubleshooting Manual - Highest trust
    "IPC": 0.80,      # Parts Catalog - High trust
    "AMM": 0.75,      # Maintenance Manual - Medium trust
    "FIM": 0.72,      # Fault Isolation - Medium-low trust
    "UNKNOWN": 0.70,  # Task-only - Lower trust (no doc context)
}


# ============================================================================
# MAIN DECISION FUNCTION
# ============================================================================

def decide(
    e0: Optional[str],
    e1_valid: bool,
    e1_ata: Optional[str],
    e2_best: Optional[dict],
    e2_all: Optional[list],
    e1_source: Optional[str] = None
) -> Tuple[str, float, str]:
    """
    Make ATA04 decision based on tri-validation.
    
    Args:
        e0: ATA04_Entered (user input)
        e1_valid: Whether E1 citation is valid
        e1_ata: ATA04 from citation (E1)
        e2_best: Best result from catalog/LLM (E2)
        e2_all: All E2 candidates (for context)
        e1_source: Source of E1 citation (TSM/IPC/AMM/UNKNOWN)
    
    Returns:
        (decision, confidence, reason) tuple
        
    Decisions:
        - CONFIRM: E0 is correct
        - CORRECT: Should use different ATA04
        - REVIEW: Manual review needed
    """
    e2_ata = e2_best["ata04"] if e2_best else None
    
    # ========================================================================
    # Case 1: All three agree (E1 valid)
    # ========================================================================
    if e1_valid and e1_ata and e2_ata and (e0 == e1_ata == e2_ata):
        return "CONFIRM", CONF_MATRIX["E0_E1_E2_VALID"], "E0=E1=E2 (E1 valid)."
    
    # ========================================================================
    # Case 2: E1 and E2 agree, but differ from E0 (E1 valid)
    # ========================================================================
    if e1_valid and e1_ata and e2_ata and (e1_ata == e2_ata) and (e0 != e1_ata):
        return "CORRECT", CONF_MATRIX["E1_E2_NEQ_E0_VALID"], f"E1({e1_source or 'ref'})=E2≠E0 → sửa theo E1/E2."
    
    # ========================================================================
    # Case 3: E2 matches E0, no valid E1
    # ========================================================================
    if e2_ata and e0 and (e2_ata == e0) and not e1_valid:
        return "CONFIRM", CONF_MATRIX["E2_EQ_E0_ONLY"], "E2 khớp E0; không có E1."
    
    # ========================================================================
    # Case 4: Only E1 valid (no E2)
    # ========================================================================
    if e1_valid and e1_ata and not e2_ata:
        return "CONFIRM", CONF_MATRIX["E1_ONLY_VALID"], f"Chỉ E1({e1_source or 'ref'}) hợp lệ."
    
    # ========================================================================
    # Case 5: E1 ≠ E2 CONFLICT → Source-based confidence
    # ========================================================================
    if e1_valid and e1_ata and e2_ata and (e1_ata != e2_ata):
        # Get confidence based on E1 source
        confidence = SOURCE_CONFIDENCE_ON_CONFLICT.get(
            e1_source or "UNKNOWN",
            0.70
        )
        
        # Also consider E2 confidence
        e2_score = e2_best.get("score", 0) if e2_best else 0
        
        # If E2 has very high confidence, reduce E1 trust slightly
        if e2_score > 0.85:
            confidence = min(confidence, confidence * 0.95)
        
        # Build reason with source info
        source_label = e1_source or "citation"
        reason = f"E1({source_label}: {e1_ata}) ≠ E2({e2_ata})"
        
        if e1_source in ["TSM", "IPC"]:
            reason += " → Tin E1 nhiều hơn (high-priority source)"
        elif e2_score > 0.85:
            reason += f" → E2 có độ tin cậy cao ({e2_score:.2f})"
        else:
            reason += " → Cần xem xét cả hai"
        
        return "REVIEW", confidence, reason
    
    # ========================================================================
    # Case 6: Only E0 exists
    # ========================================================================
    if e0 and not (e1_valid or e2_ata):
        return "REVIEW", CONF_MATRIX["E0_ONLY"], "Chỉ có E0."
    
    # ========================================================================
    # Default: Insufficient evidence
    # ========================================================================
    return "REVIEW", 0.55, "Thiếu bằng chứng rõ ràng."


# ============================================================================
# ENHANCED DECISION WITH DETAILED CONTEXT
# ============================================================================

def decide_with_context(
    e0: Optional[str],
    e1_data: Optional[dict],  # Full citation dict with source, task, etc.
    e2_best: Optional[dict],
    e2_all: Optional[list] = None
) -> dict:
    """
    Enhanced decision with full context.
    
    Args:
        e0: ATA04_Entered
        e1_data: Full citation dict from refs.py with:
            - ata04: str
            - manual: str (TSM/IPC/AMM/UNKNOWN)
            - task: str
            - priority: int
        e2_best: Best E2 result
        e2_all: All E2 candidates
    
    Returns:
        Decision dict with:
            - decision: CONFIRM/CORRECT/REVIEW
            - confidence: float
            - reason: str
            - e1_source: str (if applicable)
            - e1_task: str (if applicable)
            - recommended_ata: str
    """
    # Extract E1 info
    e1_valid = bool(e1_data and e1_data.get("ata04"))
    e1_ata = e1_data.get("ata04") if e1_data else None
    e1_source = e1_data.get("manual") if e1_data else None
    e1_task = e1_data.get("task") if e1_data else None
    
    # Make decision
    decision, confidence, reason = decide(
        e0, e1_valid, e1_ata, e2_best, e2_all, e1_source
    )
    
    # Determine recommended ATA
    if decision == "CONFIRM":
        recommended_ata = e0
    elif decision == "CORRECT":
        recommended_ata = e1_ata or (e2_best["ata04"] if e2_best else e0)
    else:  # REVIEW
        # Prefer higher priority source
        if e1_valid and e1_source in ["TSM", "IPC"]:
            recommended_ata = e1_ata
        elif e2_best and e2_best.get("score", 0) > 0.8:
            recommended_ata = e2_best["ata04"]
        elif e1_valid:
            recommended_ata = e1_ata
        elif e2_best:
            recommended_ata = e2_best["ata04"]
        else:
            recommended_ata = e0
    
    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "e1_source": e1_source,
        "e1_task": e1_task,
        "e1_ata": e1_ata,
        "e2_ata": e2_best["ata04"] if e2_best else None,
        "e2_score": e2_best.get("score") if e2_best else None,
        "recommended_ata": recommended_ata,
    }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def decide_legacy(e0, e1_valid, e1_ata, e2_best, e2_all):
    """
    Legacy function signature for backward compatibility.
    Calls new decide() without source parameter.
    """
    return decide(e0, e1_valid, e1_ata, e2_best, e2_all, e1_source=None)


# ============================================================================
# BATCH DECISION PROCESSING
# ============================================================================

def decide_batch(records: list[dict]) -> list[dict]:
    """
    Process batch decisions efficiently.
    
    Args:
        records: List of dicts with keys:
            - e0, e1_data, e2_best, e2_all
    
    Returns:
        List of decision dicts
    """
    results = []
    
    for record in records:
        decision = decide_with_context(
            e0=record.get("e0"),
            e1_data=record.get("e1_data"),
            e2_best=record.get("e2_best"),
            e2_all=record.get("e2_all")
        )
        results.append(decision)
    
    return results


# ============================================================================
# STATISTICS & ANALYSIS
# ============================================================================

def analyze_decisions(decisions: list[dict]) -> dict:
    """
    Analyze batch decision results.
    
    Returns:
        Statistics dict with counts and distributions
    """
    total = len(decisions)
    
    decision_counts = {
        "CONFIRM": 0,
        "CORRECT": 0,
        "REVIEW": 0
    }
    
    source_impact = {
        "TSM": [],
        "IPC": [],
        "AMM": [],
        "UNKNOWN": []
    }
    
    confidence_ranges = {
        "high (>0.90)": 0,
        "medium (0.70-0.90)": 0,
        "low (<0.70)": 0
    }
    
    for d in decisions:
        # Count decisions
        decision_counts[d["decision"]] += 1
        
        # Track confidence by source
        if d.get("e1_source"):
            source_impact[d["e1_source"]].append(d["confidence"])
        
        # Confidence ranges
        conf = d["confidence"]
        if conf > 0.90:
            confidence_ranges["high (>0.90)"] += 1
        elif conf >= 0.70:
            confidence_ranges["medium (0.70-0.90)"] += 1
        else:
            confidence_ranges["low (<0.70)"] += 1
    
    # Calculate averages
    avg_conf_by_source = {
        src: sum(confs) / len(confs) if confs else 0
        for src, confs in source_impact.items()
    }
    
    return {
        "total": total,
        "decision_counts": decision_counts,
        "decision_percentages": {
            k: f"{v/total*100:.1f}%" for k, v in decision_counts.items()
        },
        "confidence_ranges": confidence_ranges,
        "avg_confidence_by_source": avg_conf_by_source,
        "overall_confidence": sum(d["confidence"] for d in decisions) / total if total else 0
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DECISION LOGIC - TEST CASES")
    print("="*70)
    
    test_cases = [
        {
            "name": "All agree (TSM)",
            "e0": "21-26",
            "e1_data": {"ata04": "21-26", "manual": "TSM", "task": "21-26-40"},
            "e2_best": {"ata04": "21-26", "score": 0.92},
            "expected": {"decision": "CONFIRM", "confidence": 0.97}
        },
        {
            "name": "E1(TSM) ≠ E2 conflict",
            "e0": "21-26",
            "e1_data": {"ata04": "21-31", "manual": "TSM", "task": "21-31-10"},
            "e2_best": {"ata04": "21-26", "score": 0.88},
            "expected": {"decision": "REVIEW", "confidence": 0.85}
        },
        {
            "name": "E1(IPC) ≠ E2 conflict",
            "e0": "32-41",
            "e1_data": {"ata04": "32-42", "manual": "IPC", "task": "32-42-00"},
            "e2_best": {"ata04": "32-41", "score": 0.75},
            "expected": {"decision": "REVIEW", "confidence": 0.80}
        },
        {
            "name": "E1(AMM) ≠ E2 conflict",
            "e0": "79-31",
            "e1_data": {"ata04": "79-32", "manual": "AMM", "task": "79-32-00"},
            "e2_best": {"ata04": "79-31", "score": 0.82},
            "expected": {"decision": "REVIEW", "confidence": 0.75}
        },
        {
            "name": "E1(UNKNOWN) ≠ E2 conflict",
            "e0": "24-50",
            "e1_data": {"ata04": "24-51", "manual": "UNKNOWN", "task": "24-51-00"},
            "e2_best": {"ata04": "24-50", "score": 0.85},
            "expected": {"decision": "REVIEW", "confidence": 0.70}
        },
        {
            "name": "CMM ignored, only E2",
            "e0": "28-41",
            "e1_data": None,  # CMM was excluded
            "e2_best": {"ata04": "28-41", "score": 0.86},
            "expected": {"decision": "CONFIRM", "confidence": 0.86}
        },
    ]
    
    passed = 0
    failed = 0
    
    for tc in test_cases:
        result = decide_with_context(
            tc["e0"],
            tc["e1_data"],
            tc["e2_best"]
        )
        
        expected = tc["expected"]
        
        # Check decision
        decision_match = result["decision"] == expected["decision"]
        
        # Check confidence (allow ±0.05 tolerance)
        conf_match = abs(result["confidence"] - expected["confidence"]) < 0.05
        
        status = "✓ PASS" if (decision_match and conf_match) else "✗ FAIL"
        
        if decision_match and conf_match:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} - {tc['name']}")
        print(f"  Decision: {result['decision']} (expected: {expected['decision']})")
        print(f"  Confidence: {result['confidence']:.2f} (expected: {expected['confidence']:.2f})")
        print(f"  Reason: {result['reason']}")
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
