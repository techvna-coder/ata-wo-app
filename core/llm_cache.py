# core/llm_cache_optimized.py
"""
Optimized LLM cache with:
- DuckDB backend (indexed, efficient queries)
- Automatic TTL cleanup
- Batch operations support
- Memory-efficient storage
"""
from __future__ import annotations

import os
import time
import json
import duckdb
from typing import Any, Optional, List, Dict
from pathlib import Path

CACHE_DB = "data_store/llm_cache.duckdb"
os.makedirs("data_store", exist_ok=True)


class LLMCache:
    """
    DuckDB-based LLM response cache with:
    - O(1) lookups via indexed key
    - Automatic TTL expiration
    - Batch get/put operations
    - Periodic cleanup
    """
    
    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Initialize cache table with indexes."""
        with duckdb.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    ts BIGINT,
                    ttl BIGINT,
                    model VARCHAR,
                    tokens INTEGER
                );
            """)
            
            # Composite index for TTL cleanup
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_ttl 
                ON llm_cache(ttl, ts) 
                WHERE ttl > 0;
            """)
    
    def get(self, key: str) -> Optional[dict]:
        """
        Get cached value with TTL check.
        Returns None if expired or not found.
        """
        with duckdb.connect(self.db_path) as con:
            result = con.execute("""
                SELECT value, ts, ttl
                FROM llm_cache
                WHERE key = ?
            """, [key]).fetchone()
            
            if not result:
                return None
            
            value_json, ts, ttl = result
            
            # Check TTL
            if ttl > 0:
                age = int(time.time()) - ts
                if age > ttl:
                    # Expired - delete and return None
                    con.execute("DELETE FROM llm_cache WHERE key = ?", [key])
                    return None
            
            try:
                return json.loads(value_json)
            except:
                return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_sec: int = 0,
        model: Optional[str] = None,
        tokens: Optional[int] = None
    ):
        """
        Store value in cache with optional TTL and metadata.
        Uses INSERT OR REPLACE for upsert behavior.
        """
        value_json = json.dumps(value, ensure_ascii=False)
        
        with duckdb.connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO llm_cache (key, value, ts, ttl, model, tokens)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [key, value_json, int(time.time()), int(ttl_sec or 0), model, tokens])
    
    def get_batch(self, keys: List[str]) -> Dict[str, Optional[dict]]:
        """
        Batch get operation.
        Returns dict mapping key -> value (None if not found/expired).
        """
        if not keys:
            return {}
        
        result_map = {k: None for k in keys}
        current_time = int(time.time())
        
        with duckdb.connect(self.db_path) as con:
            # Use parameter list for IN clause
            placeholders = ','.join(['?'] * len(keys))
            results = con.execute(f"""
                SELECT key, value, ts, ttl
                FROM llm_cache
                WHERE key IN ({placeholders})
            """, keys).fetchall()
            
            expired_keys = []
            
            for key, value_json, ts, ttl in results:
                # Check TTL
                if ttl > 0 and (current_time - ts) > ttl:
                    expired_keys.append(key)
                    continue
                
                try:
                    result_map[key] = json.loads(value_json)
                except:
                    result_map[key] = None
            
            # Clean up expired entries
            if expired_keys:
                exp_placeholders = ','.join(['?'] * len(expired_keys))
                con.execute(f"""
                    DELETE FROM llm_cache 
                    WHERE key IN ({exp_placeholders})
                """, expired_keys)
        
        return result_map
    
    def put_batch(self, items: List[Dict[str, Any]]):
        """
        Batch insert/update operation.
        items: [{"key": ..., "value": ..., "ttl_sec": ..., ...}, ...]
        """
        if not items:
            return
        
        records = []
        for item in items:
            records.append((
                item["key"],
                json.dumps(item["value"], ensure_ascii=False),
                int(time.time()),
                int(item.get("ttl_sec", 0)),
                item.get("model"),
                item.get("tokens")
            ))
        
        with duckdb.connect(self.db_path) as con:
            con.executemany("""
                INSERT OR REPLACE INTO llm_cache (key, value, ts, ttl, model, tokens)
                VALUES (?, ?, ?, ?, ?, ?)
            """, records)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        Returns number of deleted rows.
        """
        current_time = int(time.time())
        
        with duckdb.connect(self.db_path) as con:
            result = con.execute("""
                DELETE FROM llm_cache
                WHERE ttl > 0 AND (? - ts) > ttl
                RETURNING count(*)
            """, [current_time]).fetchone()
            
            return result[0] if result else 0
    
    def clear(self):
        """Clear entire cache."""
        with duckdb.connect(self.db_path) as con:
            con.execute("DELETE FROM llm_cache")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with duckdb.connect(self.db_path) as con:
            total = con.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            
            # Count expired entries
            current_time = int(time.time())
            expired = con.execute("""
                SELECT COUNT(*) FROM llm_cache
                WHERE ttl > 0 AND (? - ts) > ttl
            """, [current_time]).fetchone()[0]
            
            # Get size estimate
            size = con.execute("""
                SELECT SUM(LENGTH(value)) FROM llm_cache
            """).fetchone()[0] or 0
            
            # Model distribution
            models = con.execute("""
                SELECT model, COUNT(*) as cnt
                FROM llm_cache
                WHERE model IS NOT NULL
                GROUP BY model
                ORDER BY cnt DESC
            """).df()
            
            # Token usage
            total_tokens = con.execute("""
                SELECT SUM(tokens) FROM llm_cache WHERE tokens IS NOT NULL
            """).fetchone()[0] or 0
            
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
            "total_tokens": total_tokens,
            "models": models.to_dict('records') if not models.empty else []
        }
    
    def vacuum(self):
        """Optimize database file size after deletions."""
        with duckdb.connect(self.db_path) as con:
            con.execute("VACUUM")


# ============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================================
_global_cache = LLMCache()


def cache_get(key: str) -> Optional[dict]:
    """Convenience function for backward compatibility."""
    return _global_cache.get(key)


def cache_put(key: str, obj: Any, ttl_sec: int = 0, **kwargs):
    """Convenience function for backward compatibility."""
    _global_cache.put(key, obj, ttl_sec, **kwargs)


def cache_cleanup():
    """Run cleanup of expired entries."""
    return _global_cache.cleanup_expired()


def cache_stats() -> dict:
    """Get cache statistics."""
    return _global_cache.stats()
