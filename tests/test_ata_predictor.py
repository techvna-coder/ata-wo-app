# tests/test_ata_predictor.py
"""
Unit tests for ata_predictor module.

Run:
    pytest tests/test_ata_predictor.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from core.ata_predictor import (
    ATAPredictor,
    normalize_ata,
    validate_ata,
    preprocess_batch,
    collect_e0_batch,
    collect_e1_batch,
    fuse_evidences,
    Evidence,
    WOInput,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    """Sample WO DataFrame for testing"""
    return pd.DataFrame({
        "Defect_Text": [
            "ECAM CAUTION: PACK 2 FAULT",
            "BABY BASSINET MISSING",
            "FOUND HYDRAULIC LEAK AT ACTUATOR REF AMM 29-51-00-400-001",
            "TYRE WEAR CHECK",
        ],
        "Rectification_Text": [
            "C/O REPLACED PACK 2 COMPRESSOR",
            "INSTALL NEW BASSINET",
            "REPLACE ACTUATOR SEAL PER AMM 29-51-00-400-001",
            "TREAD DEPTH WITHIN LIMITS",
        ],
        "WO_Type": ["", "", "SCHEDULED W/O", "SCHEDULED W/O"],
        "ATA04_Entered": ["21-21", None, "29-51", "32-41"]
    })


# ============================================================
# TEST: ATA NORMALIZATION
# ============================================================

def test_normalize_ata():
    """Test ATA code normalization"""
    # Valid formats
    assert normalize_ata("21-52") == "21-52"
    assert normalize_ata("2152") == "21-52"
    assert normalize_ata("21 52") == "21-52"
    assert normalize_ata("  21-52  ") == "21-52"
    
    # Invalid codes
    assert normalize_ata("00-00") is None
    assert normalize_ata("99-99") is None
    assert normalize_ata("XX-XX") is None
    assert normalize_ata("invalid") is None
    assert normalize_ata(None) is None
    assert normalize_ata("") is None


def test_validate_ata():
    """Test ATA code validation"""
    assert validate_ata("21-52") == True
    assert validate_ata("32-41") == True
    assert validate_ata("00-00") == False
    assert validate_ata("99-99") == False
    assert validate_ata(None) == False
    assert validate_ata("") == False


# ============================================================
# TEST: PREPROCESSING
# ============================================================

def test_preprocess_batch(sample_df):
    """Test batch preprocessing"""
    wos = preprocess_batch(sample_df)
    
    assert len(wos) == 4
    assert all(isinstance(wo, WOInput) for wo in wos)
    
    # Check technical classification
    assert wos[0].is_technical == True  # ECAM fault
    assert wos[1].is_technical == False  # Baby bassinet
    assert wos[2].is_technical == True  # Hydraulic leak
    assert wos[3].is_technical == False  # Tyre wear check (routine)
    
    # Check cleaning
    assert len(wos[0].defect_clean) > 0
    assert len(wos[0].combined_text) > 0
    assert wos[0].text_hash  # Hash generated


# ============================================================
# TEST: E0 COLLECTION
# ============================================================

def test_collect_e0_batch(sample_df):
    """Test E0 (Entered) evidence collection"""
    wos = preprocess_batch(sample_df)
    e0_list = collect_e0_batch(wos)
    
    assert len(e0_list) == 4
    
    # Row 0: valid E0
    assert e0_list[0] is not None
    assert e0_list[0].ata04 == "21-21"
    assert e0_list[0].confidence == 0.50
    
    # Row 1: no E0
    assert e0_list[1] is None
    
    # Row 2: valid E0
    assert e0_list[2].ata04 == "29-51"


# ============================================================
# TEST: E1 COLLECTION
# ============================================================

def test_collect_e1_batch(sample_df):
    """Test E1 (Citation) evidence collection"""
    wos = preprocess_batch(sample_df)
    e1_list = collect_e1_batch(wos, use_cache=False)
    
    assert len(e1_list) == 4
    
    # Row 0: no citation
    assert e1_list[0] is None
    
    # Row 1: non-technical, skipped
    assert e1_list[1] is None
    
    # Row 2: AMM citation should be extracted
    assert e1_list[2] is not None
    assert e1_list[2].ata04 == "29-51"
    assert e1_list[2].metadata["manual"] == "AMM"
    assert e1_list[2].confidence == 0.92


# ============================================================
# TEST: EVIDENCE FUSION
# ============================================================

def test_fuse_evidences_e1_e2_agree():
    """Test evidence fusion when E1 = E2"""
    e0 = Evidence("E0_entered", "21-21", 0.50)
    e1 = Evidence("E1_citation", "21-52", 0.92)
    e2 = Evidence("E2_catalog", "21-52", 0.85, {"score": 0.75})
    
    decision, conf, reason, ata = fuse_evidences(e0, e1, e2, [])
    
    assert ata == "21-52"
    assert decision in ["CONFIRM", "CORRECT"]
    assert conf >= 0.92


def test_fuse_evidences_e1_only():
    """Test evidence fusion with E1 only"""
    e1 = Evidence("E1_citation", "29-51", 0.92)
    
    decision, conf, reason, ata = fuse_evidences(None, e1, None, [])
    
    assert ata == "29-51"
    assert conf == 0.92


def test_fuse_evidences_no_evidence():
    """Test evidence fusion with no valid evidence"""
    decision, conf, reason, ata = fuse_evidences(None, None, None, [])
    
    assert decision == "REVIEW"
    assert conf == 0.30
    assert ata is None


# ============================================================
# TEST: FULL PIPELINE (MOCK CATALOG)
# ============================================================

@pytest.mark.skipif(not Path("catalog").exists(), reason="Catalog not built")
def test_predictor_full_pipeline(sample_df):
    """Test full prediction pipeline with real catalog"""
    from core.ata_catalog import ATACatalog
    
    catalog = ATACatalog("catalog")
    
    predictor = ATAPredictor(
        catalog=catalog,
        use_llm=False,
        verbose=False
    )
    
    results = predictor.predict_batch(sample_df)
    
    # Basic assertions
    assert len(results) == 4
    assert all(hasattr(r, 'ata04_final') for r in results)
    assert all(hasattr(r, 'confidence') for r in results)
    assert all(hasattr(r, 'decision') for r in results)
    
    # Check technical classification
    assert results[0].is_technical == True
    assert results[1].is_technical == False
    
    # Check decisions are valid
    valid_decisions = {"CONFIRM", "CORRECT", "REVIEW"}
    assert all(r.decision in valid_decisions for r in results)
    
    # Check confidence bounds
    assert all(0 <= r.confidence <= 1.0 for r in results)


def test_predictor_to_dataframe(sample_df):
    """Test conversion to DataFrame"""
    from core.ata_catalog import ATACatalog
    
    if not Path("catalog").exists():
        pytest.skip("Catalog not built")
    
    catalog = ATACatalog("catalog")
    predictor = ATAPredictor(catalog, use_llm=False, verbose=False)
    
    results = predictor.predict_batch(sample_df)
    output_df = predictor.to_dataframe(results)
    
    # Check DataFrame structure
    expected_cols = [
        "Is_Technical_Defect",
        "ATA04_Entered",
        "ATA04_From_Cited",
        "ATA04_Final",
        "Confidence",
        "Decision",
    ]
    
    for col in expected_cols:
        assert col in output_df.columns
    
    assert len(output_df) == len(sample_df)


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
