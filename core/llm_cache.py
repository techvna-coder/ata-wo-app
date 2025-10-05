# core/llm_cache.py
from __future__ import annotations
import os, time, json, hashlib
from typing import Any, Optional
import pandas as pd

CACHE_PATH = os.path.join("data_store", "llm_cache.parquet")
os.makedirs("data_store", exist_ok=True)

def _load_df() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        try:
            return pd.read_parquet(CACHE_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["key","value","ts","ttl"])

def _save_df(df: pd.DataFrame):
    df.to_parquet(CACHE_PATH, index=False)

def cache_get(key: str) -> Optional[dict]:
    df = _load_df()
    if df.empty: return None
    hit = df[df["key"] == key]
    if hit.empty: return None
    row = hit.iloc[-1]
    ts = int(row["ts"] or 0)
    ttl = int(row["ttl"] or 0)
    if ttl > 0 and (int(time.time()) - ts) > ttl:
        return None
    try:
        return json.loads(row["value"])
    except Exception:
        return None

def cache_put(key: str, obj: Any, ttl_sec: int = 0):
    df = _load_df()
    rec = {
        "key": key,
        "value": json.dumps(obj, ensure_ascii=False),
        "ts": int(time.time()),
        "ttl": int(ttl_sec or 0),
    }
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    _save_df(df)
