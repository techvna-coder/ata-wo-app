#!/usr/bin/env python3
# core/ata_map_loader.py
"""
Specialized loader for A321 Data ATA map.xlsx with UI markers.

Handles:
- UI markers: expand_more, chevron_right, \xa0 (non-breaking space)
- Unnamed columns
- Hierarchical structure
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import pandas as pd


def clean_ata_text(text: str) -> str:
    """
    Clean ATA text from UI markers and special characters.
    
    Examples:
        "expand_more\xa079 - (IAE) - OIL" → "79 - (IAE) - OIL"
        "chevron_right 79-32-15 - (IAE)" → "79-32-15 - (IAE)"
    """
    text = str(text).strip()
    
    # Remove UI markers
    text = re.sub(r'^(expand_more|chevron_right|xpand_more)\s*', '', text)
    
    # Remove non-breaking spaces and extra whitespace
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_ata_line(text: str) -> bool:
    """
    Check if line contains ATA code.
    
    Returns True for lines like:
        "79 - (IAE) - OIL"
        "79-32 - (IAE) - TEMP"
    """
    text = clean_ata_text(text)
    # Must start with 2 digits
    return bool(re.match(r'^\d{2}', text))


def load_a321_ata_map(
    filepath: str,
    output_parquet: str = "data_store/ata_map.parquet"
) -> pd.DataFrame:
    """
    Load A321 Data ATA map.xlsx and convert to standard format.
    
    Args:
        filepath: Path to A321 Data ATA map.xlsx
        output_parquet: Output path for parquet file
    
    Returns:
        DataFrame with columns: [ATA04, Title, RawText]
    
    Example:
        >>> df = load_a321_ata_map("A321 Data ATA map.xlsx")
        >>> print(df.head())
           ATA04                                              Title
        0  21-31  (CFM) - ENGINE - INDICATING - PRESSURE SYSTEM...
        1  79-32  (IAE) - OIL - GENERAL - TEMPERATURE INDICATING...
    """
    print(f"Loading {filepath}...")
    
    # Load Excel - read all sheets if multiple
    try:
        xls = pd.ExcelFile(filepath)
        print(f"Found {len(xls.sheet_names)} sheet(s): {xls.sheet_names}")
        
        # Use first sheet or sheet named 'ATA' or 'Data'
        sheet_name = None
        for name in ['ATA', 'Data', 'Sheet1', xls.sheet_names[0]]:
            if name in xls.sheet_names:
                sheet_name = name
                break
        
        print(f"Using sheet: {sheet_name}")
        df = pd.read_excel(filepath, sheet_name=sheet_name, dtype=str)
        
    except Exception as e:
        print(f"Error loading Excel: {e}")
        # Try simpler load
        df = pd.read_excel(filepath, dtype=str)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find column with ATA codes
    ata_col = None
    for col in df.columns:
        # Check first few non-null values
        sample_values = df[col].dropna().head(10)
        if any(is_ata_line(str(val)) for val in sample_values):
            ata_col = col
            break
    
    if ata_col is None:
        # Fallback: use first column
        ata_col = df.columns[0]
        print(f"Warning: Could not auto-detect ATA column, using: {ata_col}")
    else:
        print(f"Detected ATA column: {ata_col}")
    
    # Extract and clean ATA lines
    records = []
    
    for idx, row in df.iterrows():
        raw_text = str(row[ata_col]).strip()
        
        # Skip empty or invalid rows
        if not raw_text or raw_text.lower() in ['nan', 'none', '']:
            continue
        
        # Clean text
        clean_text = clean_ata_text(raw_text)
        
        # Check if valid ATA line
        if not is_ata_line(clean_text):
            continue
        
        records.append({
            'raw_text': raw_text,
            'clean_text': clean_text,
        })
    
    print(f"Extracted {len(records)} valid ATA lines")
    
    if not records:
        raise ValueError("No valid ATA codes found in file!")
    
    # Create DataFrame
    result_df = pd.DataFrame(records)
    
    # Now parse hierarchically using ata_map_processor
    from .ata_map_processor import ATAHierarchy, parse_ata_code
    
    hierarchy = ATAHierarchy()
    
    for _, row in result_df.iterrows():
        parsed = parse_ata_code(row['clean_text'])
        if parsed:
            ata_code, description, level = parsed
            hierarchy.add_node(ata_code, description, level)
    
    # Extract ATA04 with full descriptions
    ata04_codes = hierarchy.get_all_ata04()
    
    final_records = []
    for ata04 in ata04_codes:
        full_desc = hierarchy.get_full_description(ata04, include_children=True)
        final_records.append({
            "ATA04": ata04,
            "Title": full_desc
        })
    
    final_df = pd.DataFrame(final_records)
    
    # Save to parquet
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    
    print(f"\n✓ Exported {len(final_df)} ATA04 codes to {output_parquet}")
    print(f"\nSample entries:")
    print(final_df.head(10).to_string(index=False, max_colwidth=80))
    
    return final_df


def validate_a321_ata_map(filepath: str) -> dict:
    """
    Validate A321 ATA map file without processing.
    
    Returns validation report.
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        # Load file
        df = pd.read_excel(filepath, dtype=str)
        report["stats"]["total_rows"] = len(df)
        report["stats"]["columns"] = df.columns.tolist()
        
        # Check for ATA codes
        ata_count = 0
        for col in df.columns:
            for idx, val in enumerate(df[col].head(100)):
                if is_ata_line(str(val)):
                    ata_count += 1
                    if ata_count == 1:
                        report["stats"]["first_ata_line"] = clean_ata_text(str(val))
                        report["stats"]["first_ata_row"] = idx
        
        report["stats"]["ata_lines_found"] = ata_count
        
        if ata_count == 0:
            report["valid"] = False
            report["errors"].append("No ATA codes found in file")
        elif ata_count < 10:
            report["warnings"].append(f"Only {ata_count} ATA lines found (expected 100+)")
        
        # Check for UI markers
        ui_markers = 0
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ""
            if any(marker in sample for marker in ['expand_more', 'chevron_right', '\xa0']):
                ui_markers += 1
        
        report["stats"]["columns_with_ui_markers"] = ui_markers
        
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Could not load file: {str(e)}")
    
    return report


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ata_map_loader.py <excel_file> [--validate]")
        print()
        print("Examples:")
        print("  # Process and export")
        print("  python ata_map_loader.py 'A321 Data ATA map.xlsx'")
        print()
        print("  # Validate only")
        print("  python ata_map_loader.py 'A321 Data ATA map.xlsx' --validate")
        sys.exit(1)
    
    filepath = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--process"
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    if mode == "--validate":
        print("Validating file...")
        report = validate_a321_ata_map(filepath)
        
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        print(f"\nValid: {'✓ Yes' if report['valid'] else '✗ No'}")
        
        if report['errors']:
            print("\nErrors:")
            for err in report['errors']:
                print(f"  ✗ {err}")
        
        if report['warnings']:
            print("\nWarnings:")
            for warn in report['warnings']:
                print(f"  ⚠ {warn}")
        
        print("\nStatistics:")
        for key, val in report['stats'].items():
            print(f"  {key}: {val}")
        
        sys.exit(0 if report['valid'] else 1)
    
    else:  # --process
        try:
            df = load_a321_ata_map(filepath)
            print("\n✓ Processing completed successfully!")
            print(f"\nYou can now build catalog with:")
            print("  python build_ata_catalog_optimized.py --from-memory")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
