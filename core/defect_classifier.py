# core/defect_classifier.py
"""
Technical Defect Classifier for Reliability Analysis

Business Rules:
1. SCHEDULED W/O: Only actual failures found during inspection count
2. Non-SCHEDULED W/O: Exclude routine maintenance & normal wear
3. Keywords-based classification with configurable patterns
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


# ============================================================================
# CLASSIFICATION PATTERNS
# ============================================================================

# Routine/Preventive maintenance keywords (NOT technical defect)
ROUTINE_KEYWORDS = [
    # Routine inspection/service
    r'\broutine\b',
    r'\bpreventive\b',
    r'\bpreventative\b',
    r'\bscheduled\s+inspection\b',
    r'\bscheduled\s+maintenance\b',
    r'\bscheduled\s+service\b',
    r'\bscheduled\s+check\b',
    r'\bpm\s+check\b',
    r'\bc-check\b',
    r'\ba-check\b',
    r'\bd-check\b',
    r'\boverhaul\b',
    r'\blife\s+limit\b',
    r'\btime\s+limit\b',
    r'\bcycle\s+limit\b',
    r'\bservice\s+bulletin\b',
    r'\bsb\s+\d+',
    r'\bairworthiness\s+directive\b',
    r'\bad\s+\d+',
    
    # Normal wear (NOT defect)
    r'\bnormal\s+wear\b',
    r'\bexpected\s+wear\b',
    r'\bwithin\s+limits\b',
    r'\bserviceable\b',
    r'\bwithin\s+tolerance\b',
    
    # Tire wear (normal)
    r'\btyre?\s+(wear|worn|tread)\b',
    r'\btire?\s+(wear|worn|tread)\b',
    r'\btread\s+depth\b',
    
    # Brake wear (normal)
    r'\bbrake\s+(wear|worn|pad)\b',
    r'\bbrake\s+lining\b',
    
    # Paint (cosmetic, not reliability issue)
    r'\bpaint\s+(chip|scratch|peel)\b',
    r'\bsơn\s+(bong|trầy|xước)\b',
    
    # Lead bonding (scheduled maintenance)
    r'\blead\s+bonding\b',
    r'\bbond\s+strip\b',
    
    # Oil/fluid service (scheduled)
    r'\boil\s+change\b',
    r'\bfluid\s+service\b',
    r'\bfilter\s+change\b',
    r'\btopping\s+up\b',
    
    # Lubrication (routine)
    r'\blubrication\b',
    r'\bgrease\b',
    
    # Life-limited parts replacement (scheduled)
    r'\bllp\s+replacement\b',
    r'\blife\s+limited\s+part\b',
]

# Actual failure/defect keywords (IS technical defect)
DEFECT_KEYWORDS = [
    # Failures
    r'\bfail(ed|ure)?\b',
    r'\bfault(y)?\b',
    r'\bdefect(ive)?\b',
    r'\bmalfunction\b',
    r'\binoperative\b',
    r'\binop\b',
    r'\bu/s\b',  # unserviceable
    r'\bunserviceable\b',
    r'\bnot\s+working\b',
    r'\bdoes\s+not\s+work\b',
    
    # Damage
    r'\bdamage(d)?\b',
    r'\bcrack(ed|ing)?\b',
    r'\bbroken\b',
    r'\bfracture(d)?\b',
    r'\bcorroded\b',
    r'\bcorrosion\b',
    r'\beroded\b',
    r'\berosion\b',
    
    # Leaks
    r'\bleak(age|ing)?\b',
    r'\bseep(age|ing)?\b',
    r'\bdrip(ping)?\b',
    
    # Abnormal conditions
    r'\babnormal\b',
    r'\bexcessive\b',
    r'\bover\s+limit\b',
    r'\bexceeds?\s+limit\b',
    r'\bout\s+of\s+(limit|tolerance)\b',
    r'\bhigh\s+temperature\b',
    r'\bhigh\s+vibration\b',
    r'\bfluctuating\b',
    
    # Alerts/Warnings
    r'\becam\s+(caution|warning|fault)\b',
    r'\bwarning\b',
    r'\balert\b',
    r'\bcaution\b',
    
    # Structural issues
    r'\bmissing\b',
    r'\bloose\b',
    r'\bdetached\b',
    r'\bdisconnected\b',
    r'\bdislodged\b',
    
    # System issues
    r'\boverheating\b',
    r'\bshort\s+circuit\b',
    r'\bintermittent\b',
    r'\bnoise\b',
    r'\bvibration\b',
    r'\bsmoke\b',
    r'\bburn(ed|ing)?\b',
]

# SCHEDULED W/O inspection findings (found defects)
INSPECTION_FINDING_KEYWORDS = [
    r'\bfound\b',
    r'\bdetected\b',
    r'\bdiscovered\b',
    r'\bidentified\b',
    r'\bobserved\b',
    r'\bnoted\b',
    r'\brectify\b',  # Rectify CMR means found issue
    r'\bcmr\b',      # Cabin Maintenance Report - found issue
]


# ============================================================================
# CLASSIFIER
# ============================================================================
@dataclass
class DefectClassification:
    """Classification result."""
    is_technical_defect: bool
    confidence: float  # 0-1
    reason: str
    matched_patterns: List[str]
    wo_type: str


class DefectClassifier:
    """
    Classify Work Orders as technical defects for reliability analysis.
    
    Rules:
    1. SCHEDULED W/O: Only if actual defect found during inspection
    2. Non-SCHEDULED W/O: Exclude routine/wear patterns
    3. Configurable keyword patterns
    """
    
    def __init__(
        self,
        routine_keywords: Optional[List[str]] = None,
        defect_keywords: Optional[List[str]] = None,
        inspection_keywords: Optional[List[str]] = None,
        case_sensitive: bool = False
    ):
        """
        Initialize classifier.
        
        Args:
            routine_keywords: Custom routine/preventive patterns
            defect_keywords: Custom defect patterns
            inspection_keywords: Custom inspection finding patterns
            case_sensitive: Case-sensitive matching
        """
        self.routine_patterns = self._compile_patterns(
            routine_keywords or ROUTINE_KEYWORDS,
            case_sensitive
        )
        self.defect_patterns = self._compile_patterns(
            defect_keywords or DEFECT_KEYWORDS,
            case_sensitive
        )
        self.inspection_patterns = self._compile_patterns(
            inspection_keywords or INSPECTION_FINDING_KEYWORDS,
            case_sensitive
        )
        self.case_sensitive = case_sensitive
    
    def _compile_patterns(self, patterns: List[str], case_sensitive: bool) -> List[re.Pattern]:
        """Compile regex patterns."""
        flags = 0 if case_sensitive else re.IGNORECASE
        return [re.compile(p, flags) for p in patterns]
    
    def _find_matches(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Find all matching patterns in text."""
        matches = []
        for pattern in patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches
    
    def classify(
        self,
        wo_type: str,
        defect_text: str,
        action_text: str = ""
    ) -> DefectClassification:
        """
        Classify a Work Order.
        
        Args:
            wo_type: WO Type (e.g., "SCHEDULED W/O", "UNSCHEDULED W/O")
            defect_text: Defect description
            action_text: Action taken (optional)
        
        Returns:
            DefectClassification with result and reasoning
        """
        # Combine texts for analysis
        full_text = f"{defect_text} {action_text}"
        
        # Normalize
        wo_type_norm = str(wo_type).strip().upper()
        
        # Check if SCHEDULED W/O
        is_scheduled = "SCHEDULED" in wo_type_norm and "UNSCHEDULED" not in wo_type_norm
        
        # Find pattern matches
        routine_matches = self._find_matches(full_text, self.routine_patterns)
        defect_matches = self._find_matches(full_text, self.defect_patterns)
        inspection_matches = self._find_matches(full_text, self.inspection_patterns)
        
        # === DECISION LOGIC ===
        
        # Rule 1: SCHEDULED W/O
        if is_scheduled:
            # Must have both:
            # a) Inspection finding indicator (found, detected, rectify, CMR)
            # b) Actual defect keyword (damage, fault, leak, etc.)
            # c) NOT routine maintenance pattern
            
            has_finding = bool(inspection_matches)
            has_defect = bool(defect_matches)
            has_routine = bool(routine_matches)
            
            if has_routine:
                return DefectClassification(
                    is_technical_defect=False,
                    confidence=0.9,
                    reason=f"SCHEDULED W/O: Routine maintenance detected: {routine_matches[:2]}",
                    matched_patterns=routine_matches,
                    wo_type=wo_type
                )
            
            if has_finding and has_defect:
                return DefectClassification(
                    is_technical_defect=True,
                    confidence=0.95,
                    reason=f"SCHEDULED W/O: Defect found during inspection: {defect_matches[:2]}",
                    matched_patterns=defect_matches + inspection_matches,
                    wo_type=wo_type
                )
            
            # Has defect keywords but no finding indicators - medium confidence
            if has_defect:
                return DefectClassification(
                    is_technical_defect=True,
                    confidence=0.7,
                    reason=f"SCHEDULED W/O: Defect indicators present: {defect_matches[:2]}",
                    matched_patterns=defect_matches,
                    wo_type=wo_type
                )
            
            # No clear defect found
            return DefectClassification(
                is_technical_defect=False,
                confidence=0.8,
                reason="SCHEDULED W/O: No defect found during inspection",
                matched_patterns=[],
                wo_type=wo_type
            )
        
        # Rule 2: Non-SCHEDULED W/O
        else:
            # Exclude routine/wear patterns
            if routine_matches:
                return DefectClassification(
                    is_technical_defect=False,
                    confidence=0.9,
                    reason=f"Non-scheduled: Routine/wear pattern: {routine_matches[:2]}",
                    matched_patterns=routine_matches,
                    wo_type=wo_type
                )
            
            # Has defect keywords = likely technical defect
            if defect_matches:
                return DefectClassification(
                    is_technical_defect=True,
                    confidence=0.9,
                    reason=f"Non-scheduled: Technical defect: {defect_matches[:2]}",
                    matched_patterns=defect_matches,
                    wo_type=wo_type
                )
            
            # No clear pattern - default to False for non-scheduled
            return DefectClassification(
                is_technical_defect=False,
                confidence=0.6,
                reason="Non-scheduled: No clear defect pattern",
                matched_patterns=[],
                wo_type=wo_type
            )
    
    def classify_batch(
        self,
        df: pd.DataFrame,
        wo_type_col: str = "WO_Type",
        defect_col: str = "WO_Description",
        action_col: str = "WO_Action"
    ) -> pd.DataFrame:
        """
        Classify entire DataFrame.
        
        Args:
            df: DataFrame with WO data
            wo_type_col: Column name for WO type
            defect_col: Column name for defect description
            action_col: Column name for action (optional)
        
        Returns:
            DataFrame with added columns:
            - Is_Technical_Defect (bool)
            - Defect_Confidence (float)
            - Defect_Reason (str)
        """
        results = []
        
        for idx, row in df.iterrows():
            wo_type = row.get(wo_type_col, "")
            defect = row.get(defect_col, "")
            action = row.get(action_col, "") if action_col in df.columns else ""
            
            result = self.classify(wo_type, str(defect), str(action))
            
            results.append({
                'Is_Technical_Defect': result.is_technical_defect,
                'Defect_Confidence': result.confidence,
                'Defect_Reason': result.reason,
                'Matched_Patterns': ", ".join(result.matched_patterns[:3])
            })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# ============================================================================
# EXAMPLE ANALYSIS
# ============================================================================
def analyze_example():
    """Analyze your specific example."""
    classifier = DefectClassifier()
    
    # Your example
    wo_type = "SCHEDULED W/O"
    defect_text = """
    WORKSTEP ADDED BY VAE02125 ON 06.SEP.2025, 15:23 
    102-2025 RECTIFY CMR T9-2025 
    30. ENG 2:HEAT SHIELD DAMAGED AT MANY POSITION.
    """
    action_text = """
    ACTION PERFORMED BY VAE03950 ON 13.SEP.2025, 21:16 
    102-2025 ENG 2 HEAT SHIELD REPAIRED COMPLETED 
    (IAW CMM 78-30-00 R.8/25)
    """
    
    result = classifier.classify(wo_type, defect_text, action_text)
    
    print("="*70)
    print("EXAMPLE CLASSIFICATION")
    print("="*70)
    print(f"\nWO Type: {wo_type}")
    print(f"\nDefect: {defect_text.strip()}")
    print(f"\nAction: {action_text.strip()}")
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Is Technical Defect: {result.is_technical_defect}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Reason: {result.reason}")
    print(f"Matched Patterns: {result.matched_patterns}")
    print("\n" + "="*70)
    print("EXPLANATION:")
    print("="*70)
    print("""
This is classified as TECHNICAL DEFECT because:
1. ✓ WO Type is "SCHEDULED W/O"
2. ✓ Contains inspection finding: "RECTIFY CMR" (CMR = Cabin Maintenance Report)
3. ✓ Contains actual defect: "DAMAGED AT MANY POSITION"
4. ✗ No routine maintenance keywords found

Therefore: Counts toward reliability metrics.
    """)


# ============================================================================
# CLI TOOL
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        analyze_example()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch processing
        if len(sys.argv) < 3:
            print("Usage: python defect_classifier.py --batch <wo_file.xlsx>")
            sys.exit(1)
        
        filepath = sys.argv[2]
        df = pd.read_excel(filepath)
        
        classifier = DefectClassifier()
        result_df = classifier.classify_batch(df)
        
        # Save
        output = filepath.replace('.xlsx', '_classified.xlsx')
        result_df.to_excel(output, index=False)
        
        # Stats
        technical_count = result_df['Is_Technical_Defect'].sum()
        total = len(result_df)
        
        print(f"✓ Processed {total} WO records")
        print(f"✓ Technical defects: {technical_count} ({technical_count/total*100:.1f}%)")
        print(f"✓ Saved to: {output}")
    
    else:
        print("Usage:")
        print("  python defect_classifier.py --example")
        print("  python defect_classifier.py --batch <wo_file.xlsx>")
