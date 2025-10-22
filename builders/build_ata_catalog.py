#!/usr/bin/env python3
"""
Optimized ATA Catalog Builder with:
- Progress tracking
- Incremental build support
- Validation checks
- Performance metrics
- Error recovery

Usage:
    python build_ata_catalog_optimized.py --from-memory
    python build_ata_catalog_optimized.py --validate
    python build_ata_catalog_optimized.py --stats
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from core.catalog_builder import build_catalog_from_memory
    from core.store_optimized import get_training_stats
except ImportError:
    print("ERROR: Could not import core modules. Make sure they are in PYTHONPATH.")
    sys.exit(1)


# ============================================================================
# UTILITIES
# ============================================================================
class Colors:
    """ANSI colors for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(msg: str, status: str = "INFO"):
    """Print colored status message."""
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
    }
    color = color_map.get(status, "")
    print(f"{color}[{status}]{Colors.END} {msg}")


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print("=" * 70 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================
def validate_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print_status("Checking prerequisites...", "INFO")
    
    issues = []
    
    # Check training data
    wo_parquet = Path("data_store/wo_training.parquet")
    if not wo_parquet.exists():
        issues.append("Missing data_store/wo_training.parquet")
    else:
        try:
            stats = get_training_stats()
            rows = stats.get("total_rows", 0)
            ata_count = stats.get("distinct_ata04", 0)
            
            if rows < 100:
                issues.append(f"Too few training samples: {rows} (need at least 100)")
            elif rows < 1000:
                print_status(
                    f"Warning: Only {rows} training samples. Recommend at least 1000 for good accuracy.",
                    "WARNING"
                )
            
            if ata_count < 10:
                issues.append(f"Too few ATA classes: {ata_count} (need at least 10)")
            
            print(f"  ✓ Training data: {rows} WO across {ata_count} ATA classes")
        except Exception as e:
            issues.append(f"Could not read training data: {e}")
    
    # Check ATA map (optional but recommended)
    ata_parquet = Path("data_store/ata_map.parquet")
    if ata_parquet.exists():
        print("  ✓ ATA map available")
    else:
        print_status("  ℹ ATA map not found (optional, but recommended for better titles)", "INFO")
    
    # Check output directory
    catalog_dir = Path("catalog")
    if not catalog_dir.exists():
        print("  ℹ Creating catalog directory...")
        catalog_dir.mkdir(parents=True, exist_ok=True)
    print("  ✓ Output directory ready")
    
    if issues:
        print_status("Prerequisites check FAILED:", "ERROR")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    
    print_status("All prerequisites satisfied", "SUCCESS")
    return True


def validate_catalog_output() -> bool:
    """Validate generated catalog files."""
    print_status("Validating catalog output...", "INFO")
    
    required_files = [
        "catalog/ata_catalog.json",
        "catalog/model/tfidf_vectorizer.joblib",
        "catalog/model/tfidf_matrix.npz",
    ]
    
    issues = []
    
    for filepath in required_files:
        path = Path(filepath)
        if not path.exists():
            issues.append(f"Missing file: {filepath}")
        else:
            size = path.stat().st_size
            print(f"  ✓ {filepath} ({size:,} bytes)")
    
    # Try loading catalog
    try:
        from core.ata_catalog_optimized import ATACatalog
        catalog = ATACatalog()
        
        # Test prediction
        best, _ = catalog.predict("ECAM FAULT", "REPLACED LRU")
        if best:
            print(f"  ✓ Catalog functional (test prediction: {best['ata04']})")
        else:
            issues.append("Catalog loaded but test prediction returned None")
    except Exception as e:
        issues.append(f"Could not load catalog: {e}")
    
    if issues:
        print_status("Catalog validation FAILED:", "ERROR")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    
    print_status("Catalog validation passed", "SUCCESS")
    return True


# ============================================================================
# CATALOG BUILDING
# ============================================================================
def build_catalog(
    min_docs: int = 3,
    top_k: int = 15,
    sample_k: int = 3,
    max_docs: int = 2000,
    use_llm: bool = False,
    random_state: int = 42,
) -> Optional[dict]:
    """
    Build catalog with progress tracking.
    
    Returns dict with stats or None on failure.
    """
    print_header("Building TF-IDF Catalog")
    
    print_status(f"Configuration:", "INFO")
    print(f"  Min docs per class: {min_docs}")
    print(f"  Top keywords: {top_k}")
    print(f"  Sample sentences: {sample_k}")
    print(f"  Max docs per class: {max_docs}")
    print(f"  LLM enrichment: {'Yes' if use_llm else 'No'}")
    print()
    
    start_time = time.time()
    
    try:
        stat_df = build_catalog_from_memory(
            min_docs_per_class=min_docs,
            top_k=top_k,
            sample_k=sample_k,
            max_docs_per_class=max_docs,
            random_state=random_state,
            use_llm_enrich=use_llm,
        )
        
        elapsed = time.time() - start_time
        
        print_status(f"Catalog built successfully in {elapsed:.1f}s", "SUCCESS")
        print()
        
        # Print statistics
        print_status("Catalog Statistics:", "INFO")
        print(f"  Total ATA classes: {len(stat_df)}")
        print(f"  Total documents used: {stat_df['Docs'].sum()}")
        print(f"  Avg docs per class: {stat_df['Docs'].mean():.1f}")
        print(f"  Min docs: {stat_df['Docs'].min()}")
        print(f"  Max docs: {stat_df['Docs'].max()}")
        print()
        
        # Show top classes
        print("Top 10 ATA classes by training samples:")
        for idx, row in stat_df.head(10).iterrows():
            print(f"  {row['ATA04']}: {row['Docs']} samples")
        
        # Show classes with few samples (warning)
        few_samples = stat_df[stat_df['Docs'] < min_docs]
        if not few_samples.empty:
            print()
            print_status(
                f"Warning: {len(few_samples)} classes have fewer than {min_docs} samples:",
                "WARNING"
            )
            for idx, row in few_samples.head(5).iterrows():
                print(f"  {row['ATA04']}: {row['Docs']} samples")
            if len(few_samples) > 5:
                print(f"  ... and {len(few_samples) - 5} more")
        
        return {
            "success": True,
            "classes": len(stat_df),
            "total_docs": int(stat_df['Docs'].sum()),
            "elapsed_seconds": elapsed,
            "stats": stat_df,
        }
        
    except FileNotFoundError as e:
        print_status(f"Build failed: {e}", "ERROR")
        print("Make sure you have run data ingestion first.")
        return None
    
    except Exception as e:
        print_status(f"Build failed with error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STATISTICS
# ============================================================================
def show_catalog_stats():
    """Show statistics of existing catalog."""
    print_header("Catalog Statistics")
    
    catalog_json = Path("catalog/ata_catalog.json")
    if not catalog_json.exists():
        print_status("No catalog found. Run build first.", "ERROR")
        return False
    
    try:
        import json
        with open(catalog_json) as f:
            catalog = json.load(f)
        
        print_status("Catalog Information:", "INFO")
        print(f"  Total ATA classes: {len(catalog)}")
        
        # Analyze keywords
        keyword_counts = [len(v.get("keywords", [])) for v in catalog.values()]
        print(f"  Avg keywords per class: {sum(keyword_counts) / len(keyword_counts):.1f}")
        
        # Analyze samples
        sample_counts = [len(v.get("samples", [])) for v in catalog.values()]
        print(f"  Avg samples per class: {sum(sample_counts) / len(sample_counts):.1f}")
        
        # Show sample entries
        print()
        print("Sample catalog entries:")
        for ata, info in list(catalog.items())[:3]:
            title = info.get("title", "N/A")
            kws = ", ".join(info.get("keywords", [])[:5])
            print(f"\n  {ata}: {title}")
            print(f"    Keywords: {kws}")
        
        # Model files
        print()
        vec_path = Path("catalog/model/tfidf_vectorizer.joblib")
        mat_path = Path("catalog/model/tfidf_matrix.npz")
        
        if vec_path.exists() and mat_path.exists():
            vec_size = vec_path.stat().st_size / 1024
            mat_size = mat_path.stat().st_size / 1024
            print_status("Model Files:", "INFO")
            print(f"  Vectorizer: {vec_size:.1f} KB")
            print(f"  TF-IDF Matrix: {mat_size:.1f} KB")
            print(f"  Total: {vec_size + mat_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print_status(f"Could not read catalog: {e}", "ERROR")
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build TF-IDF catalog from training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build catalog from memory (standard)
  python build_ata_catalog_optimized.py --from-memory
  
  # Build with LLM enrichment (requires OPENAI_API_KEY)
  python build_ata_catalog_optimized.py --from-memory --use-llm
  
  # Show existing catalog stats
  python build_ata_catalog_optimized.py --stats
  
  # Validate catalog
  python build_ata_catalog_optimized.py --validate
  
  # Build with custom parameters
  python build_ata_catalog_optimized.py --from-memory --max-docs 1000 --top-k 10
        """
    )
    
    # Actions
    parser.add_argument(
        "--from-memory",
        action="store_true",
        help="Build catalog from data_store/wo_training.parquet"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing catalog without rebuilding"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics of existing catalog"
    )
    
    # Build parameters
    parser.add_argument(
        "--min-docs",
        type=int,
        default=3,
        help="Minimum documents per class (default: 3)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top keywords to extract (default: 15)"
    )
    parser.add_argument(
        "--sample-k",
        type=int,
        default=3,
        help="Number of sample sentences per class (default: 3)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2000,
        help="Maximum documents per class for balancing (default: 2000)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to enrich catalog entries (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks (not recommended)"
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not (args.from_memory or args.validate or args.stats):
        parser.print_help()
        return 0
    
    # Stats only
    if args.stats:
        success = show_catalog_stats()
        return 0 if success else 1
    
    # Validate only
    if args.validate:
        success = validate_catalog_output()
        return 0 if success else 1
    
    # Build catalog
    if args.from_memory:
        # Check prerequisites
        if not args.skip_validation:
            if not validate_prerequisites():
                print()
                print_status("Fix prerequisites and try again.", "ERROR")
                return 1
            print()
        
        # Build
        result = build_catalog(
            min_docs=args.min_docs,
            top_k=args.top_k,
            sample_k=args.sample_k,
            max_docs=args.max_docs,
            use_llm=args.use_llm,
        )
        
        if not result or not result["success"]:
            return 1
        
        print()
        
        # Validate output
        if not args.skip_validation:
            if not validate_catalog_output():
                print()
                print_status("Catalog built but validation failed. Please check output.", "WARNING")
                return 1
        
        print()
        print_status("=" * 70, "SUCCESS")
        print_status("Catalog build completed successfully!", "SUCCESS")
        print_status("=" * 70, "SUCCESS")
        print()
        print(f"Classes: {result['classes']}")
        print(f"Documents: {result['total_docs']}")
        print(f"Build time: {result['elapsed_seconds']:.1f}s")
        print()
        print("Next steps:")
        print("  1. Test predictions with: python -c \"from core.ata_catalog_optimized import ATACatalog; c=ATACatalog(); print(c.predict('ECAM FAULT', 'REPLACED LRU'))\"")
        print("  2. View stats with: python build_ata_catalog_optimized.py --stats")
        print("  3. Use in your application")
        print()
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
