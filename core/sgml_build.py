# core/sgml_build.py
import os, tarfile, re, json, shutil, uuid, time
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from tqdm import tqdm
import duckdb

# Fallback Document
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # fallback

# Fallback Embeddings
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain.embeddings import OpenAIEmbeddings  # fallback

from langchain.text_splitter import RecursiveCharacterTextSplitter

TASK_PATTERN = re.compile(r'\b(\d{2})-(\d{2})-(\d{2})(?:-(\d{3})(?:-(\d{3}))*)?\b')
TASK_COMPACT = re.compile(r'\b(\d{2})(\d{2})(\d{2})(?:(\d{3}))?(?:(\d{3}))?\b')

def _norm_task_code(raw: str) -> Optional[str]:
    if not raw: 
        return None
    raw = raw.strip()
    m = TASK_PATTERN.search(raw)
    if m:
        aa, bb, cc = m.group(1), m.group(2), m.group(3)
        tail = "-".join([x for x in [m.group(4), m.group(5)] if x])
        return f"{aa}-{bb}-{cc}{('-'+tail) if tail else ''}"
    m2 = TASK_COMPACT.search(raw)
    if m2:
        aa, bb, cc = m2.group(1), m2.group(2), m2.group(3)
        tail = "-".join([x for x in [m2.group(4), m2.group(5)] if x])
        return f"{aa}-{bb}-{cc}{('-'+tail) if tail else ''}"
    return None

def _ata04_from_task(task_full: str) -> Optional[str]:
    if not task_full: return None
    m = TASK_PATTERN.search(task_full)
    if not m: return None
    return f"{m.group(1)}-{m.group(2)}"

def extract_tar(tar_path: str, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out)
    return str(out)

def iter_sgml_files(root: str) -> Iterator[str]:
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in [".sgml", ".sgm", ".xml", ".xsg", ".xmg"]:
            yield str(p)

def _get_text(node) -> str:
    return " ".join(node.stripped_strings) if node else ""

def parse_one_sgml(path: str, manual_type: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    # Parsers: thử lxml trước, nếu “khó” có thể đổi thành html.parser
    soup = BeautifulSoup(data, "lxml")

    records: List[Dict] = []
    candidates = []
    for tag in ["task", "dm", "procedure", "topic", "chapter", "section"]:
        candidates.extend(soup.find_all(tag))

    if candidates:
        for blk in candidates:
            title = None
            for ttag in ["title", "tasktitle", "name", "caption", "heading"]:
                tt = blk.find(ttag)
                if tt: 
                    title = _get_text(tt)
                    break

            task_full = None
            for key in ["tasknumber", "task", "dmcode", "id", "code", "ident"]:
                v = blk.get(key) if hasattr(blk, "get") else None
                task_full = _norm_task_code(v) or task_full
            if not task_full:
                for t in ["tasknumber", "dmcode", "ref", "id", "code"]:
                    el = blk.find(t)
                    if el:
                        task_full = _norm_task_code(el.get_text())
                        if task_full: break
            if not task_full:
                task_full = _norm_task_code(_get_text(blk))
            if not task_full:
                continue

            content = _get_text(blk)
            records.append({
                "task_full": task_full,
                "ata04": _ata04_from_task(task_full),
                "title": title or "",
                "doc_type": manual_type,
                "source_file": path,
                "content": content
            })
    else:
        # Fallback: quét bằng regex toàn file
        for m in TASK_PATTERN.finditer(data):
            task_full = _norm_task_code(m.group(0))
            if not task_full: 
                continue
            start = max(0, m.start() - 1500)
            end = min(len(data), m.end() + 2500)
            snippet = data[start:end]
            title = ""
            records.append({
                "task_full": task_full,
                "ata04": _ata04_from_task(task_full),
                "title": title,
                "doc_type": manual_type,
                "source_file": path,
                "content": BeautifulSoup(snippet, "lxml").get_text(" ", strip=True)
            })
    return records

def parse_files_parallel(file_paths: List[str], manual_type: str, max_workers: int = 4) -> Iterable[Dict]:
    def _worker(fp):
        try:
            return parse_one_sgml(fp, manual_type)
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, fp) for fp in file_paths]
        for fut in as_completed(futures):
            recs = fut.result()
            for r in recs:
                yield r

def build_registry_and_chunks(
    extracted_dir: str,
    manual_type: str,
    duckdb_path: str,
    vector_dir: str,
    openai_api_key: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "text-embedding-3-small",
    embed_batch_size: int = 64,
    shard_size: int = 3000,
    max_workers_parse: int = 4,
    no_embed: bool = False,
    resume: bool = False,
) -> Tuple[int, int]:

    from math import inf
    # Fallback FAISS import
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        from langchain.vectorstores import FAISS  # fallback

    files = list(iter_sgml_files(extracted_dir))
    if not files:
        raise RuntimeError("Không tìm thấy file SGML/XML trong thư mục.")

    # DuckDB
    con = duckdb.connect(duckdb_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS refs(
            task_full TEXT,
            task_short TEXT,
            ata04 TEXT,
            manual_type TEXT,
            title TEXT,
            page INTEGER,
            source_file TEXT
        )
    """)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Shard & resume
    os.makedirs(vector_dir, exist_ok=True)
    existing_shards = sorted([d for d in os.listdir(vector_dir) if d.startswith("shard_")])
    shard_idx = (int(existing_shards[-1].split("_")[1]) + 1) if (resume and existing_shards) else 1

    # Embeddings
    if not no_embed:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=embedding_model)

    total_tasks = 0
    total_chunks = 0
    duckdb_rows: List[Tuple] = []
    shard_docs: List[Document] = []

    def flush_duckdb_rows(rows):
        if not rows:
            return
        con.executemany("INSERT INTO refs VALUES (?,?,?,?,?,?,?)", rows)
        rows.clear()

    def save_faiss_shard(shard_docs: List[Document], shard_idx: int):
        if not shard_docs:
            return 0
        store = FAISS.from_documents(shard_docs, embeddings)
        shard_dir = os.path.join(vector_dir, f"shard_{shard_idx:04d}")
        os.makedirs(shard_dir, exist_ok=True)
        store.save_local(shard_dir)
        return len(shard_docs)

    # Parse song song → ghi DuckDB theo lô → chunk → nhúng thành shard
    for rec in parse_files_parallel(files, manual_type, max_workers=max_workers_parse):
        m = TASK_PATTERN.search(rec["task_full"])
        aa_bb_cc = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        duckdb_rows.append((rec["task_full"], aa_bb_cc, rec["ata04"], rec["doc_type"], rec["title"], None, rec["source_file"]))
        total_tasks += 1
        if len(duckdb_rows) >= 2000:
            flush_duckdb_rows(duckdb_rows)

        for i, ch in enumerate(splitter.split_text(rec["content"])):
            meta = {
                "task_full": rec["task_full"], "ata04": rec["ata04"], "doc_type": rec["doc_type"],
                "title": rec["title"], "source_file": rec["source_file"], "chunk_id": f"{rec['task_full']}#{i}"
            }
            shard_docs.append(Document(page_content=ch, metadata=meta))
            total_chunks += 1

            if (not no_embed) and len(shard_docs) >= shard_size:
                save_faiss_shard(shard_docs, shard_idx)
                shard_docs.clear()
                shard_idx += 1

    flush_duckdb_rows(duckdb_rows)
    con.close()

    if not no_embed and shard_docs:
        save_faiss_shard(shard_docs, shard_idx)
        shard_docs.clear()

    return (total_tasks, total_chunks)
