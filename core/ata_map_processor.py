# core/ata_map_processor.py
"""
Process hierarchical ATA map from Excel file with:
- Parent-child inheritance (79 → 79-30 → 79-32)
- Child aggregation (79-32 includes all 79-32-XX)
- ATA04 extraction (only XX-YY format)
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd


# ============================================================================
# ATA CODE PATTERNS
# ============================================================================
# Match different ATA code formats
ATA_PATTERNS = {
    "ata2": re.compile(r"^(\d{2})\s*-\s*(.+)$"),           # 79 - (IAE) - OIL
    "ata4": re.compile(r"^(\d{2})-(\d{2})\s*-\s*(.+)$"),   # 79-32 - (IAE) - ...
    "ata6": re.compile(r"^(\d{2})-(\d{2})-(\d{2})\s*-\s*(.+)$"),  # 79-32-15 - ...
    "ata8": re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2,})\s*-\s*(.+)$"),  # 79-32-15-01 - ...
}


def parse_ata_code(text: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse ATA code from text.
    
    Returns: (ata_code, description, level)
        - ata_code: normalized format (79, 79-32, 79-32-15, etc.)
        - description: full text after code
        - level: hierarchy level (2, 4, 6, 8)
    
    Examples:
        "79 - (IAE) - OIL - GENERAL" → ("79", "(IAE) - OIL - GENERAL", 2)
        "79-32 - (IAE) - TEMP" → ("79-32", "(IAE) - TEMP", 4)
    """
    text = str(text).strip()
    
    # Try patterns in order of specificity
    for level, pattern in [
        (8, ATA_PATTERNS["ata8"]),
        (6, ATA_PATTERNS["ata6"]),
        (4, ATA_PATTERNS["ata4"]),
        (2, ATA_PATTERNS["ata2"]),
    ]:
        m = pattern.match(text)
        if m:
            groups = m.groups()
            # Reconstruct ATA code
            if level == 2:
                ata_code = groups[0]
                desc = groups[1]
            elif level == 4:
                ata_code = f"{groups[0]}-{groups[1]}"
                desc = groups[2]
            elif level == 6:
                ata_code = f"{groups[0]}-{groups[1]}-{groups[2]}"
                desc = groups[3]
            else:  # level == 8
                ata_code = f"{groups[0]}-{groups[1]}-{groups[2]}-{groups[3]}"
                desc = groups[4]
            
            return ata_code, desc.strip(), level
    
    return None


def normalize_ata04(ata_code: str) -> Optional[str]:
    """
    Normalize to ATA04 format (XX-YY).
    
    Examples:
        "79-32" → "79-32"
        "79-32-15" → "79-32"
        "79" → None (not ATA04)
    """
    parts = ata_code.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return None


# ============================================================================
# HIERARCHICAL PROCESSING
# ============================================================================
class ATAHierarchy:
    """
    Build and query hierarchical ATA map.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # ata_code → node info
        self.children: Dict[str, List[str]] = {}  # ata_code → list of child codes
        self.parent: Dict[str, str] = {}  # ata_code → parent code
    
    def add_node(self, ata_code: str, description: str, level: int):
        """Add node to hierarchy."""
        self.nodes[ata_code] = {
            "code": ata_code,
            "description": description,
            "level": level,
        }
        
        # Determine parent
        parent_code = self._get_parent_code(ata_code, level)
        if parent_code:
            self.parent[ata_code] = parent_code
            if parent_code not in self.children:
                self.children[parent_code] = []
            self.children[parent_code].append(ata_code)
    
    def _get_parent_code(self, ata_code: str, level: int) -> Optional[str]:
        """Get parent code based on hierarchy."""
        parts = ata_code.split("-")
        
        if level == 4:
            # Parent is level 2
            return parts[0]
        elif level == 6:
            # Parent is level 4
            return f"{parts[0]}-{parts[1]}"
        elif level == 8:
            # Parent is level 6
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        
        return None
    
    def get_full_description(self, ata_code: str, include_children: bool = True) -> str:
        """
        Get full description with inheritance and aggregation.
        
        Args:
            ata_code: ATA code (e.g., "79-32")
            include_children: If True, append children descriptions
        
        Returns:
            Full description combining parent and children info
        
        Example:
            "79-32" → "(IAE) - OIL - GENERAL - TEMPERATURE INDICATING SYSTEM 
                       [Includes: THERMOCOUPLE - ACOC, THERMOCOUPLE - SCAVENGE]"
        """
        if ata_code not in self.nodes:
            return ""
        
        parts = []
        
        # 1. Collect parent descriptions (inheritance)
        current = ata_code
        parent_descs = []
        while current in self.parent:
            parent_code = self.parent[current]
            if parent_code in self.nodes:
                parent_descs.append(self.nodes[parent_code]["description"])
            current = parent_code
        
        # Reverse to get root → leaf order
        parent_descs.reverse()
        
        # 2. Add own description
        own_desc = self.nodes[ata_code]["description"]
        
        # Combine parent + own (remove duplicates)
        combined = self._merge_descriptions(parent_descs + [own_desc])
        parts.append(combined)
        
        # 3. Add children descriptions (aggregation)
        if include_children and ata_code in self.children:
            child_descs = []
            for child_code in self.children[ata_code]:
                if child_code in self.nodes:
                    child_desc = self.nodes[child_code]["description"]
                    # Extract equipment name (after last dash)
                    equipment = self._extract_equipment(child_desc)
                    if equipment:
                        child_descs.append(equipment)
            
            if child_descs:
                # Limit to first 10 to avoid too long
                child_descs = child_descs[:10]
                parts.append(f"[Includes: {', '.join(child_descs)}]")
        
        return " ".join(parts)
    
    def _merge_descriptions(self, descs: List[str]) -> str:
        """
        Merge descriptions by removing redundant parts.
        
        Example:
            ["(IAE) - OIL - GENERAL", "(IAE) - INDICATING", "(IAE) - TEMP"]
            → "(IAE) - OIL - GENERAL - INDICATING - TEMP"
        """
        if not descs:
            return ""
        
        # Extract unique meaningful parts
        seen = set()
        parts = []
        
        for desc in descs:
            # Split by dash and clean
            tokens = [t.strip() for t in desc.split("-")]
            for token in tokens:
                # Skip common prefixes
                if token in ["(IAE)", "(PW11)", "(CFM)", "(GE)"]:
                    if not seen:  # Only add engine type once
                        parts.append(token)
                        seen.add(token)
                elif token and token not in seen:
                    parts.append(token)
                    seen.add(token)
        
        return " - ".join(parts)
    
    def _extract_equipment(self, desc: str) -> str:
        """
        Extract equipment name from description.
        
        Example:
            "(IAE) - THERMOCOUPLE - ACOC OIL TEMPERATURE" → "THERMOCOUPLE - ACOC"
        """
        # Remove engine type prefix
        desc = re.sub(r"^\([A-Z0-9]+\)\s*-\s*", "", desc)
        
        # Take first 2-3 words
        words = desc.split()[:3]
        return " ".join(words)
    
    def get_all_ata04(self) -> List[str]:
        """Get all ATA04 codes (XX-YY format) in hierarchy."""
        ata04_codes = []
        for code in self.nodes.keys():
            if normalize_ata04(code) == code:  # Is already ATA04 format
                ata04_codes.append(code)
        return sorted(ata04_codes)


# ============================================================================
# EXCEL LOADER
# ============================================================================
def load_ata_map_from_excel(
    filepath: str,
    column_name: Optional[str] = None
) -> ATAHierarchy:
    """
    Load hierarchical ATA map from Excel file.
    
    Args:
        filepath: Path to Excel file
        column_name: Column containing ATA codes. If None, auto-detect.
    
    Returns:
        ATAHierarchy object with parsed data
    
    Example:
        >>> hierarchy = load_ata_map_from_excel("A321 Data ATA map.xlsx")
        >>> desc = hierarchy.get_full_description("79-32")
        >>> print(desc)
    """
    # Load Excel
    df = pd.read_excel(filepath, dtype=str)
    
    # Auto-detect column if not specified
    if column_name is None:
        # Look for column with ATA codes
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ""
            if re.match(r"^\d{2}", sample):
                column_name = col
                break
        
        if column_name is None:
            # Fallback to first column
            column_name = df.columns[0]
    
    # Build hierarchy
    hierarchy = ATAHierarchy()
    
    for idx, row in df.iterrows():
        text = str(row[column_name]).strip()
        
        # Skip empty or header rows
        if not text or text.lower() in ["nan", "none", ""]:
            continue
        
        # Parse ATA code
        parsed = parse_ata_code(text)
        if parsed:
            ata_code, description, level = parsed
            hierarchy.add_node(ata_code, description, level)
    
    return hierarchy


# ============================================================================
# CATALOG INTEGRATION
# ============================================================================
def enrich_catalog_with_ata_map(
    catalog_json_path: str,
    ata_map_excel_path: str,
    output_json_path: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Enrich existing catalog with hierarchical ATA map descriptions.
    
    Args:
        catalog_json_path: Path to existing ata_catalog.json
        ata_map_excel_path: Path to ATA map Excel file
        output_json_path: Output path (default: overwrite input)
    
    Returns:
        Enriched catalog dict
    """
    import json
    
    # Load existing catalog
    with open(catalog_json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    # Load ATA map
    hierarchy = load_ata_map_from_excel(ata_map_excel_path)
    
    # Enrich each entry
    enriched_count = 0
    for ata04 in catalog.keys():
        if ata04 in hierarchy.nodes:
            # Get full description with inheritance and children
            full_desc = hierarchy.get_full_description(ata04, include_children=True)
            
            # Update title
            catalog[ata04]["title"] = full_desc
            enriched_count += 1
    
    # Save
    output_path = output_json_path or catalog_json_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    
    print(f"Enriched {enriched_count}/{len(catalog)} catalog entries")
    return catalog


# ============================================================================
# EXPORT TO PARQUET (for catalog_builder)
# ============================================================================
def export_ata_map_to_parquet(
    excel_path: str,
    output_parquet: str = "data_store/ata_map.parquet"
) -> pd.DataFrame:
    """
    Export ATA04 codes with enriched descriptions to parquet.
    
    This creates the ata_map.parquet file that catalog_builder can use.
    
    Args:
        excel_path: Path to ATA map Excel
        output_parquet: Output parquet path
    
    Returns:
        DataFrame with [ATA04, Title] columns
    """
    # Load hierarchy
    hierarchy = load_ata_map_from_excel(excel_path)
    
    # Extract ATA04 codes
    ata04_codes = hierarchy.get_all_ata04()
    
    # Build DataFrame
    records = []
    for ata04 in ata04_codes:
        full_desc = hierarchy.get_full_description(ata04, include_children=True)
        records.append({
            "ATA04": ata04,
            "Title": full_desc
        })
    
    df = pd.DataFrame(records)
    
    # Save to parquet
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    
    print(f"Exported {len(df)} ATA04 codes to {output_parquet}")
    return df


# ============================================================================
# CLI TOOL
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ata_map_processor.py <excel_file> [--export|--enrich]")
        print()
        print("Examples:")
        print("  # Export to parquet for catalog builder")
        print("  python ata_map_processor.py 'A321 Data ATA map.xlsx' --export")
        print()
        print("  # Enrich existing catalog")
        print("  python ata_map_processor.py 'A321 Data ATA map.xlsx' --enrich")
        print()
        print("  # Test hierarchy")
        print("  python ata_map_processor.py 'A321 Data ATA map.xlsx'")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--test"
    
    if mode == "--export":
        df = export_ata_map_to_parquet(excel_path)
        print("\nSample entries:")
        print(df.head(10).to_string(index=False))
    
    elif mode == "--enrich":
        catalog_path = "catalog/ata_catalog.json"
        if not Path(catalog_path).exists():
            print(f"Error: {catalog_path} not found. Build catalog first.")
            sys.exit(1)
        
        enrich_catalog_with_ata_map(catalog_path, excel_path)
        print("Catalog enriched successfully!")
    
    else:  # --test
        hierarchy = load_ata_map_from_excel(excel_path)
        
        print(f"Loaded {len(hierarchy.nodes)} ATA codes")
        print(f"Found {len(hierarchy.get_all_ata04())} ATA04 codes")
        print()
        
        # Test examples
        test_codes = ["79-32", "79-31", "79-33", "21-51"]
        for ata04 in test_codes:
            if ata04 in hierarchy.nodes:
                desc = hierarchy.get_full_description(ata04)
                print(f"{ata04}:")
                print(f"  {desc}")
                print()
