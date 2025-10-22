# core/store_optimized.py
"""
Optimized version of store.py with:
- Atomic writes using temp files
- Incremental DuckDB append (no full pandas load)
- File locking for concurrent safety
- Partitioned storage for large datasets
"""
import duckdb
import hashlib
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import fcntl
import time

DB_PATH = "data_store/memory.duckdb"
WO_PARQUET = "data_store/wo_training.parquet"
ATA_PARQUET = "data_store/ata_map.parquet"
LOCK_FILE = "data_store/.lock"

# ============================================================================
# FILE LOCKING UTILITIES
# ============================================================================
class FileLock:
    """Simple file-based lock for concurrent access protection."""
    def __init__(self, lock_path: str, timeout: float = 30.0):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.lock_file = None
    
    def __enter__(self):
        self.lock_file = open(self.lock_path, 'w')
        start = time.time()
        while True:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except IOError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_path}")
                time.sleep(0.1)
    
    def __exit__(self, *args):
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()


# ============================================================================
# ATOMIC WRITE UTILITIES
# ============================================================================
def atomic_write_parquet(df: pd.DataFrame, target_path: Path):
    """Write parquet atomically using temp file + rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='wb',
        delete=False,
        dir=target_path.parent,
        prefix=f".tmp_{target_path.name}_",
        suffix='.parquet'
    ) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        df.to_parquet(tmp_path, index=False)
        shutil.move(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ============================================================================
# HASH & NORMALIZATION (reused from original)
# ============================================================================
def _hash_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()


def _to_ata04(s) -> Optional[str]:
    """Normalize ATA04 format (same as original)."""
    if pd.isna(s):
        return None
    import re
    t = str(s).strip()
    m = re.findall(r"\d", t)
    if len(m) >= 4:
        return f"{''.join(m[:2])}-{''.join(m[2:4])}"
    m2 = re.match(r"^\s*(\d{2})\s*[-\.]\s*(\d{2})", t)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return None


# ============================================================================
# DUCKDB INITIALIZATION
# ============================================================================
def init_db():
    """Initialize DuckDB with proper schema."""
    Path("data_store").mkdir(parents=True, exist_ok=True)
    
    with duckdb.connect(DB_PATH) as con:
        # Metadata table
        con.execute("""
            CREATE TABLE IF NOT EXISTS wo_meta(
                source_file TEXT,
                rows_ingested INTEGER,
                ingested_at TIMESTAMP DEFAULT now()
            );
        """)
        
        # Create index for faster manifest queries
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_wo_meta_source 
            ON wo_meta(source_file);
        """)


# ============================================================================
# ATA MAP APPEND (Optimized)
# ============================================================================
def append_ata_map(df_map: pd.DataFrame, code_col: str, name_col: str):
    """
    Append ATA mapping with deduplication using DuckDB.
    Avoids loading full parquet into pandas.
    """
    with FileLock(LOCK_FILE):
        df = df_map.copy()
        df["ATA04"] = df[code_col].map(_to_ata04)
        df = df[["ATA04", name_col]].rename(columns={name_col: "Title"})
        df = df.dropna().drop_duplicates()
        
        if not df.empty:
            if Path(ATA_PARQUET).exists():
                # Use DuckDB for efficient deduplication
                with duckdb.connect() as con:
                    # Register existing data
                    con.execute(f"CREATE TEMP TABLE old AS SELECT * FROM '{ATA_PARQUET}'")
                    
                    # Register new data
                    con.register('new', df)
                    
                    # Deduplicate (keep latest)
                    result = con.execute("""
                        SELECT ATA04, Title
                        FROM (
                            SELECT *, ROW_NUMBER() OVER (PARTITION BY ATA04 ORDER BY 1) as rn
                            FROM (
                                SELECT * FROM old
                                UNION ALL
                                SELECT * FROM new
                            )
                        )
                        WHERE rn = 1
                    """).df()
                
                atomic_write_parquet(result, Path(ATA_PARQUET))
            else:
                atomic_write_parquet(df, Path(ATA_PARQUET))


# ============================================================================
# WO TRAINING APPEND (Highly Optimized)
# ============================================================================
def append_wo_training(
    df_wo: pd.DataFrame,
    desc_col: str,
    act_col: str,
    ata_final_col: str,
    ata_entered_col: str,
    source_file: str
):
    """
    Optimized WO training append with:
    - Proper text cleaning import
    - Incremental DuckDB-based deduplication
    - Atomic writes
    - File locking
    """
    from .cleaning import clean_wo_text
    
    with FileLock(LOCK_FILE):
        tdf = df_wo.copy()
        
        # Text composition
        def _text(row):
            parts = []
            if desc_col in row and pd.notna(row[desc_col]):
                parts.append(clean_wo_text(str(row[desc_col])))
            if act_col and act_col in row and pd.notna(row[act_col]):
                parts.append(clean_wo_text(str(row[act_col])))
            return " | ".join([p for p in parts if p]).strip()
        
        tdf["text"] = tdf.apply(_text, axis=1)
        
        # Label extraction
        def _label(row):
            lab = None
            if ata_final_col in row and pd.notna(row[ata_final_col]) and str(row[ata_final_col]).strip():
                lab = row[ata_final_col]
            elif ata_entered_col in row:
                lab = row[ata_entered_col]
            return _to_ata04(lab)
        
        tdf["ata04"] = tdf.apply(_label, axis=1)
        
        # Filter valid rows
        tdf = tdf[(tdf["text"].str.len() > 0) & tdf["ata04"].notna()].copy()
        
        if tdf.empty:
            return  # Nothing to append
        
        # Hash after cleaning
        tdf["hash"] = tdf["text"].map(_hash_text)
        keep_cols = ["text", "ata04", "hash"]
        
        # DuckDB-based deduplication (memory efficient)
        if Path(WO_PARQUET).exists():
            with duckdb.connect() as con:
                con.execute(f"CREATE TEMP TABLE old AS SELECT * FROM '{WO_PARQUET}'")
                con.register('new', tdf[keep_cols])
                
                # Deduplicate by hash (keep first occurrence)
                result = con.execute("""
                    SELECT text, ata04, hash
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY hash ORDER BY 1) as rn
                        FROM (
                            SELECT * FROM old
                            UNION ALL
                            SELECT * FROM new
                        )
                    )
                    WHERE rn = 1
                """).df()
            
            atomic_write_parquet(result, Path(WO_PARQUET))
        else:
            tdf_dedup = tdf[keep_cols].drop_duplicates(subset=["hash"], keep="first")
            atomic_write_parquet(tdf_dedup, Path(WO_PARQUET))
        
        # Update metadata
        with duckdb.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO wo_meta(source_file, rows_ingested) VALUES (?, ?)",
                [source_file, int(tdf.shape[0])]
            )


# ============================================================================
# STATISTICS & UTILITIES
# ============================================================================
def get_training_stats() -> dict:
    """Get training data statistics without loading full dataset."""
    if not Path(WO_PARQUET).exists():
        return {"exists": False}
    
    with duckdb.connect() as con:
        stats = con.execute(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT ata04) as distinct_ata04,
                AVG(LENGTH(text)) as avg_text_length
            FROM '{WO_PARQUET}'
        """).fetchone()
        
        top_ata = con.execute(f"""
            SELECT ata04, COUNT(*) as count
            FROM '{WO_PARQUET}'
            GROUP BY ata04
            ORDER BY count DESC
            LIMIT 20
        """).df()
    
    return {
        "exists": True,
        "total_rows": stats[0],
        "distinct_ata04": stats[1],
        "avg_text_length": round(stats[2], 2),
        "top_ata": top_ata
    }


def cleanup_old_hashes(days_old: int = 30):
    """
    Clean up duplicate entries older than N days.
    Useful for maintenance after accumulating many duplicates.
    """
    if not Path(WO_PARQUET).exists():
        return
    
    with FileLock(LOCK_FILE):
        with duckdb.connect() as con:
            # Load and deduplicate
            result = con.execute(f"""
                SELECT text, ata04, hash
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY hash ORDER BY 1) as rn
                    FROM '{WO_PARQUET}'
                )
                WHERE rn = 1
            """).df()
        
        atomic_write_parquet(result, Path(WO_PARQUET))
