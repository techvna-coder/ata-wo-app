#!/usr/bin/env python3
"""
Automated migration script for core module optimizations.

Usage:
    python migrate_to_optimized.py [--dry-run] [--backup-dir PATH]

Features:
- Automatic backup
- Data migration (parquet → DuckDB)
- Dependency check
- Rollback support
- Progress tracking
"""
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


# ============================================================================
# CONFIGURATION
# ============================================================================
REQUIRED_MODULES = ["duckdb", "scipy", "pandas", "numpy"]
BACKUP_PREFIX = "backup"
DATA_STORE = Path("data_store")
CATALOG_DIR = Path("catalog")


# ============================================================================
# UTILITIES
# ============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(msg: str, status: str = "INFO"):
    """Print formatted status message."""
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
    }
    color = color_map.get(status, "")
    print(f"{color}[{status}]{Colors.END} {msg}")


def check_dependencies() -> bool:
    """Check if all required modules are installed."""
    print_status("Checking dependencies...", "INFO")
    missing = []
    
    for module in REQUIRED_MODULES:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (MISSING)")
            missing.append(module)
    
    if missing:
        print_status(
            f"Missing dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            "ERROR"
        )
        return False
    
    print_status("All dependencies satisfied", "SUCCESS")
    return True


def create_backup(backup_dir: Path) -> bool:
    """Create backup of data_store and catalog."""
    print_status(f"Creating backup in {backup_dir}...", "INFO")
    
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup data_store
        if DATA_STORE.exists():
            shutil.copytree(
                DATA_STORE,
                backup_dir / "data_store",
                dirs_exist_ok=True
            )
            print("  ✓ Backed up data_store/")
        
        # Backup catalog
        if CATALOG_DIR.exists():
            shutil.copytree(
                CATALOG_DIR,
                backup_dir / "catalog",
                dirs_exist_ok=True
            )
            print("  ✓ Backed up catalog/")
        
        # Create manifest
        manifest = {
            "backup_time": datetime.now().isoformat(),
            "data_store_exists": DATA_STORE.exists(),
            "catalog_exists": CATALOG_DIR.exists(),
        }
        
        with open(backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        print_status("Backup completed successfully", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Backup failed: {e}", "ERROR")
        return False


# ============================================================================
# MIGRATION TASKS
# ============================================================================
def migrate_llm_cache(dry_run: bool = False) -> bool:
    """Migrate LLM cache from parquet to DuckDB."""
    print_status("Migrating LLM cache...", "INFO")
    
    old_cache = DATA_STORE / "llm_cache.parquet"
    new_cache = DATA_STORE / "llm_cache.duckdb"
    
    if not old_cache.exists():
        print("  ℹ No old cache found, skipping")
        return True
    
    try:
        import pandas as pd
        import duckdb
        from core.llm_cache_optimized import LLMCache
        
        # Load old cache
        df = pd.read_parquet(old_cache)
        print(f"  Found {len(df)} entries in old cache")
        
        if dry_run:
            print("  [DRY RUN] Would migrate cache entries")
            return True
        
        # Create new cache
        cache = LLMCache(str(new_cache))
        
        # Batch insert
        items = []
        for _, row in df.iterrows():
            try:
                import json
                value = json.loads(row["value"])
                items.append({
                    "key": row["key"],
                    "value": value,
                    "ttl_sec": int(row.get("ttl", 0))
                })
            except Exception as e:
                print(f"  ⚠ Skipped invalid entry: {e}")
        
        cache.put_batch(items)
        print(f"  ✓ Migrated {len(items)} cache entries")
        
        # Rename old file
        old_cache.rename(DATA_STORE / "llm_cache.parquet.old")
        print("  ✓ Renamed old cache file")
        
        print_status("LLM cache migration completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Cache migration failed: {e}", "ERROR")
        return False


def initialize_optimized_db(dry_run: bool = False) -> bool:
    """Initialize DuckDB schema for optimized store."""
    print_status("Initializing optimized database...", "INFO")
    
    if dry_run:
        print("  [DRY RUN] Would initialize DuckDB schema")
        return True
    
    try:
        from core.store_optimized import init_db
        
        init_db()
        print("  ✓ Created DuckDB schema")
        
        print_status("Database initialization completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Database initialization failed: {e}", "ERROR")
        return False


def verify_data_integrity() -> bool:
    """Verify that migrated data is intact."""
    print_status("Verifying data integrity...", "INFO")
    
    checks = []
    
    # Check WO training data
    wo_parquet = DATA_STORE / "wo_training.parquet"
    if wo_parquet.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(wo_parquet)
            required_cols = {"text", "ata04", "hash"}
            if required_cols.issubset(df.columns):
                print(f"  ✓ WO training data: {len(df)} rows")
                checks.append(True)
            else:
                print(f"  ✗ WO training data missing columns: {required_cols - set(df.columns)}")
                checks.append(False)
        except Exception as e:
            print(f"  ✗ Error reading WO data: {e}")
            checks.append(False)
    
    # Check ATA map
    ata_parquet = DATA_STORE / "ata_map.parquet"
    if ata_parquet.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(ata_parquet)
            print(f"  ✓ ATA map: {len(df)} entries")
            checks.append(True)
        except Exception as e:
            print(f"  ✗ Error reading ATA map: {e}")
            checks.append(False)
    
    # Check catalog
    catalog_json = CATALOG_DIR / "ata_catalog.json"
    if catalog_json.exists():
        try:
            with open(catalog_json) as f:
                catalog = json.load(f)
            print(f"  ✓ Catalog: {len(catalog)} ATA classes")
            checks.append(True)
        except Exception as e:
            print(f"  ✗ Error reading catalog: {e}")
            checks.append(False)
    
    # Check new cache
    cache_db = DATA_STORE / "llm_cache.duckdb"
    if cache_db.exists():
        try:
            from core.llm_cache_optimized import cache_stats
            stats = cache_stats()
            print(f"  ✓ LLM cache: {stats['active_entries']} entries")
            checks.append(True)
        except Exception as e:
            print(f"  ✗ Error checking cache: {e}")
            checks.append(False)
    
    if all(checks):
        print_status("Data integrity check passed", "SUCCESS")
        return True
    else:
        print_status("Data integrity check FAILED", "ERROR")
        return False


# ============================================================================
# ROLLBACK
# ============================================================================
def rollback(backup_dir: Path) -> bool:
    """Rollback to backup."""
    print_status(f"Rolling back from {backup_dir}...", "WARNING")
    
    if not backup_dir.exists():
        print_status("Backup directory not found", "ERROR")
        return False
    
    try:
        # Restore data_store
        backup_data = backup_dir / "data_store"
        if backup_data.exists():
            if DATA_STORE.exists():
                shutil.rmtree(DATA_STORE)
            shutil.copytree(backup_data, DATA_STORE)
            print("  ✓ Restored data_store/")
        
        # Restore catalog
        backup_catalog = backup_dir / "catalog"
        if backup_catalog.exists():
            if CATALOG_DIR.exists():
                shutil.rmtree(CATALOG_DIR)
            shutil.copytree(backup_catalog, CATALOG_DIR)
            print("  ✓ Restored catalog/")
        
        print_status("Rollback completed successfully", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Rollback failed: {e}", "ERROR")
        return False


# ============================================================================
# MAIN MIGRATION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Migrate core modules to optimized versions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(f"{BACKUP_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Backup directory path"
    )
    parser.add_argument(
        "--rollback",
        type=Path,
        help="Rollback to specified backup directory"
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup creation (NOT RECOMMENDED)"
    )
    
    args = parser.parse_args()
    
    # Handle rollback
    if args.rollback:
        return 0 if rollback(args.rollback) else 1
    
    # Print header
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}Core Module Migration to Optimized Versions{Colors.END}")
    print("=" * 70 + "\n")
    
    if args.dry_run:
        print_status("DRY RUN MODE - No changes will be made\n", "WARNING")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return 1
    
    print()
    
    # Step 2: Create backup
    if not args.skip_backup:
        if not create_backup(args.backup_dir):
            print_status("Migration aborted due to backup failure", "ERROR")
            return 1
        print()
    else:
        print_status("SKIPPING BACKUP (you've been warned!)", "WARNING")
        print()
    
    # Step 3: Run migrations
    success = True
    
    success &= initialize_optimized_db(args.dry_run)
    print()
    
    success &= migrate_llm_cache(args.dry_run)
    print()
    
    # Step 4: Verify (skip in dry-run)
    if not args.dry_run:
        success &= verify_data_integrity()
        print()
    
    # Step 5: Summary
    print("=" * 70)
    if success:
        print_status("Migration completed successfully! 🎉", "SUCCESS")
        print("\nNext steps:")
        print("1. Update imports in your code to use optimized modules")
        print("2. Run your application and verify functionality")
        print("3. Monitor performance improvements")
        print(f"\nBackup location: {args.backup_dir}")
        print(f"Rollback command: python {sys.argv[0]} --rollback {args.backup_dir}")
    else:
        print_status("Migration encountered errors", "ERROR")
        print(f"\nYou can rollback with:")
        print(f"  python {sys.argv[0]} --rollback {args.backup_dir}")
    print("=" * 70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
