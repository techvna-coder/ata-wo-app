# core/decision.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# =============================================================================
# Decision policy for ATA04 finalization
# Inputs:
#   e0: str|None                 # ATA entered by user (E0)
#   e1_valid: bool               # whether a valid manual citation (E1) exists
#   e1_ata: str|None             # ATA04 derived from citation (E1)
#   e2_best: dict|None           # best candidate from TF-IDF (E2), expects {"ata04": str, "score": float}
#   e2_all: list|None            # full TF-IDF list, e.g. [{"ata04": str, "score": float}, ...] sorted desc
#
# Returns:
#   (decision, confidence, reason)  # strings + float in [0,1]
#
# Policy priority (highest → lowest):
#   1) E1 (manual citation) dominates.
#   2) E2 (TF-IDF) with strong score + margin.
#   3) Else REVIEW.
# =============================================================================


def _norm_ata(ata: Optional[str]) -> Optional[str]:
    if not ata:
        return None
    s = str(ata).strip().replace(" ", "").upper()
    # accept formats like "02-05", "0205", "02.05", "02_05"
    import re
    m = re.match(r"^(\d{2})[-._]?(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s if "-" in s and len(s) >= 5 else None


def _extract_margin(e2_all: Optional[List[Dict]]) -> float:
    """Return difference between top-1 and top-2 scores (0 if insufficient)."""
    try:
        if not e2_all or len(e2_all) < 2:
            return 0.0
        s1 = float(e2_all[0].get("score", 0.0) or 0.0)
        s2 = float(e2_all[1].get("score", 0.0) or 0.0)
        return max(0.0, s1 - s2)
    except Exception:
        return 0.0


def decide(
    e0: Optional[str],
    e1_valid: bool,
    e1_ata: Optional[str],
    e2_best: Optional[Dict] = None,
    e2_all: Optional[List[Dict]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    return_explain: bool = False,
) -> Tuple[str, float, str] | Tuple[str, float, str, Dict]:
    """
    Make a robust decision using E0/E1/E2.
    thresholds (optional):
        - t_e2_strong: float  (default 0.42)  minimum score to accept E2 as strong
        - t_e2_ok:     float  (default 0.32)  minimum score to accept E2 as acceptable
        - t_margin:    float  (default 0.08)  minimum margin between top1 and top2
    """
    cfg = {
        "t_e2_strong": 0.42,
        "t_e2_ok": 0.32,
        "t_margin": 0.08,
    }
    if thresholds:
        cfg.update(thresholds)

    E0 = _norm_ata(e0)
    E1 = _norm_ata(e1_ata)
    e2_top_ata = _norm_ata(e2_best.get("ata04") if e2_best else None)
    e2_top_score = float((e2_best or {}).get("score", 0.0) or 0.0)
    e2_margin = _extract_margin(e2_all)

    # ---- 1) E1 dominates when valid ----------------------------------------
    if e1_valid and E1:
        if E0 and E0 == E1:
            decision, conf, reason = "CONFIRM", 0.92, "E1 (manual) trùng E0"
        else:
            # If E2 agrees with E1, boost confidence
            if e2_top_ata and e2_top_ata == E1 and e2_top_score >= cfg["t_e2_ok"]:
                decision, conf, reason = "CORRECT", 0.93, "E1 (manual) khác E0; E2 đồng thuận"
            else:
                decision, conf, reason = "CORRECT", 0.88, "E1 (manual) khác E0"
        explain = {
            "E0": E0, "E1": E1,
            "E2_top": e2_top_ata, "E2_score": round(e2_top_score, 4), "E2_margin": round(e2_margin, 4),
            "policy": "E1_priority"
        }
        return (decision, conf, reason, explain) if return_explain else (decision, conf, reason)

    # ---- 2) No E1: rely on E2 when strong enough ---------------------------
    if e2_top_ata:
        if e2_top_score >= cfg["t_e2_strong"] and e2_margin >= cfg["t_margin"]:
            # Strong TF-IDF signal
            if not E0:
                decision, conf, reason = "CORRECT", 0.82, "E2 mạnh; không có E0"
            elif E0 == e2_top_ata:
                decision, conf, reason = "CONFIRM", 0.80, "E2 mạnh; trùng E0"
            else:
                decision, conf, reason = "CORRECT", 0.78, "E2 mạnh; khác E0"
        elif e2_top_score >= cfg["t_e2_ok"]:
            # Acceptable TF-IDF signal (lower confidence)
            if not E0:
                decision, conf, reason = "CORRECT", 0.72, "E2 đủ; không có E0"
            elif E0 == e2_top_ata:
                decision, conf, reason = "CONFIRM", 0.70, "E2 đủ; trùng E0"
            else:
                decision, conf, reason = "CORRECT", 0.68, "E2 đủ; khác E0"
        else:
            # Weak TF-IDF -> REVIEW
            decision, conf, reason = "REVIEW", 0.55, "E2 yếu; thiếu E1"
        explain = {
            "E0": E0, "E1": None,
            "E2_top": e2_top_ata, "E2_score": round(e2_top_score, 4), "E2_margin": round(e2_margin, 4),
            "policy": "E2_fallback"
        }
        return (decision, conf, reason, explain) if return_explain else (decision, conf, reason)

    # ---- 3) Nothing but E0 --------------------------------------------------
    decision, conf, reason = "REVIEW", 0.5, "Chỉ có E0"
    explain = {"E0": E0, "E1": None, "E2_top": None, "E2_score": 0.0, "E2_margin": 0.0, "policy": "E0_only"}
    return (decision, conf, reason, explain) if return_explain else (decision, conf, reason)


# ====================== Basic self-test ======================
if __name__ == "__main__":
    # 1) E1 present & equals E0
    print(decide("21-21", True, "21-21", {"ata04": "21-21", "score": 0.55}, [{"ata04":"21-21","score":0.55},{"ata04":"21-52","score":0.40}], return_explain=True))
    # 2) E1 present & differs from E0; E2 agrees with E1
    print(decide("00-00", True, "21-52", {"ata04": "21-52", "score": 0.51}, [{"ata04":"21-52","score":0.51},{"ata04":"21-21","score":0.35}], return_explain=True))
    # 3) No E1, strong E2
    print(decide(None, False, None, {"ata04": "52-35", "score": 0.48}, [{"ata04":"52-35","score":0.48},{"ata04":"21-21","score":0.33}], return_explain=True))
    # 4) No E1, weak E2
    print(decide("00-00", False, None, {"ata04": "24-21", "score": 0.25}, [{"ata04":"24-21","score":0.25}], return_explain=True))
