#!/usr/bin/env python3
"""
Quick fix for A321 Data ATA map.xlsx issue.

Usage:
    python fix_a321_ata_map.py "A321 Data ATA map.xlsx"
"""
import sys
import os
from pathlib import Path


def main():
    print("="*70)
    print("A321 ATA MAP QUICK FIX")
    print("="*70)
    print()
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python fix_a321_ata_map.py <path_to_excel_file>")
        print()
        print("Example:")
        print('  python fix_a321_ata_map.py "A321 Data ATA map.xlsx"')
        print()
        sys.exit(1)
    
    excel_path = sys.argv[1]
    
    # Check file exists
    if not Path(excel_path).exists():
        print(f"❌ Error: File not found: {excel_path}")
        print()
        print("Check the path and try again.")
        sys.exit(1)
    
    print(f"📁 Input file: {excel_path}")
    print()
    
    # Step 1: Validate
    print("Step 1: Validating file format...")
    print("-" * 70)
    
    try:
        # Import specialized loader
        try:
            from core.ata_map_loader import load_a321_ata_map, validate_a321_ata_map
        except ImportError:
            print("❌ Error: ata_map_loader.py not found in core/")
            print()
            print("Please copy the file first:")
            print("  cp optimized_core_modules/ata_map_loader.py core/")
            print("  cp optimized_core_modules/ata_map_processor.py core/")
            sys.exit(1)
        
        # Validate
        report = validate_a321_ata_map(excel_path)
        
        if not report['valid']:
            print("❌ Validation failed:")
            for err in report['errors']:
                print(f"   {err}")
            sys.exit(1)
        
        print(f"✓ File format valid")
        print(f"✓ Found {report['stats']['ata_lines_found']} ATA lines")
        print()
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Process
    print("Step 2: Processing file...")
    print("-" * 70)
    
    try:
        df = load_a321_ata_map(excel_path)
        print()
        print(f"✓ Successfully processed {len(df)} ATA04 codes")
        print()
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Build catalog
    print("Step 3: Would you like to build the catalog now? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        print()
        print("Building catalog...")
        print("-" * 70)
        
        import subprocess
        result = subprocess.run([
            sys.executable,
            "build_ata_catalog_optimized.py",
            "--from-memory"
        ])
        
        if result.returncode == 0:
            print()
            print("✓ Catalog built successfully!")
        else:
            print()
            print("⚠ Catalog build had issues. Check output above.")
    
    # Summary
    print()
    print("="*70)
    print("✅ FIX COMPLETED!")
    print("="*70)
    print()
    print("What was done:")
    print("  ✓ Validated A321 ATA map format")
    print("  ✓ Processed hierarchical structure")
    print("  ✓ Exported to: data_store/ata_map.parquet")
    if response == 'y':
        print("  ✓ Built catalog with enriched descriptions")
    print()
    print("Next steps:")
    if response != 'y':
        print("  1. Build catalog:")
        print("     python build_ata_catalog_optimized.py --from-memory")
    print("  2. Test predictions:")
    print("     python -c \"from core.ata_catalog_optimized import ATACatalog; c=ATACatalog(); print(c.predict('OIL TEMP', 'REPLACED'))\"")
    print("  3. Use in your app!")
    print()


if __name__ == "__main__":
    main()
