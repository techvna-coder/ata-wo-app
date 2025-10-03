# core/sgml_build.py
# ------------------------------------------------------------------------------------
# SGML → DuckDB (refs) + FAISS (sharded)
# - Parse SGML/XML (TSM/FIM/AMM) an toàn (BeautifulSoup + lxml, fallback regex)
# - Ghi DuckDB theo lô
# - Chunk → Embed (OpenAIEmbeddings) với retry + token clipping
# - Sharding + Resume cho FAISS
# - Tương thích nhiều phiên bản langchain_* khi tạo FAISS từ embeddings
# ------------------------------------------------------------------------------------

import os
import re
import tarfile
import time
import shutil
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
from bs4 import BeautifulSoup
from tqdm import tqdm
import tiktoken

# Fallback Document
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # fallback rất cũ

# Fallback Embeddings
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain.embeddings import OpenAIEmbeddings  # fallback rất cũ

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Fallback FAISS Vectorstore (class wrapper)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # fallback rất cũ


# -------------------------
# Regex nhận diện task/ATA
# -------------------------
TASK_PATTERN = re.compile(
    r'\b(\d{2})-(\d{2})-(\d{2})(?:-(\d{3})(?:-(\d{3}))*)?\b'
)
TASK_COMPACT = re.compile(
    r'\b(\d{2})(\d{2})(\d{2})(?:(\d{3}))?(?:(\d{3}))?\b', re.I
)

def _norm_task_code(raw: str) -> Optional[str]:
    """Chuẩn hóa chuỗi về định dạng task: AA-BB-CC(-DDD-EEE…)."""
    if not raw:
        return None
    raw = str(raw).strip()
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
    if not task_full:
        return None
    m = TASK_PATTERN.search(task_full)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


# -------------------------
# I/O tiện ích
# -------------------------
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


# -------------------------
# Parse SGML/XML → records
# -------------------------
def parse_one_sgml(path: str, manual_type: str) -> List[Dict]:
    """Trả về danh sách record: task_full, ata04, title, doc_type, source_file, content."""
    try:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

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
            # Thử thuộc tính
            for key in ["tasknumber", "task", "dmcode", "id", "code", "ident"]:
                v = blk.get(key) if hasattr(blk, "get") else None
                task_full = _norm_task_code(v) or task_full
            # Thử nội dung các thẻ con
            if not task_full:
                for t in ["tasknumber", "dmcode", "ref", "id", "code"]:
                    el = blk.find(t)
                    if el:
                        task_full = _norm_task_code(el.get_text())
                        if task_full:
                            break
            # Fallback: regex toàn block
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
        # Fallback regex toàn file
        for m in TASK_PATTERN.finditer(data):
            task_full = _norm_task_code(m.group(0))
            if not task_full:
                continue
            start = max(0, m.start() - 1500)
            end = min(len(data), m.end() + 2500)
            snippet = data[start:end]
            records.append({
                "task_full": task_full,
                "ata04": _ata04_from_task(task_full),
                "title": "",
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


# -------------------------
# Token utils (tiktoken)
# -------------------------
def _tiktoken_len(text: str, model: str = "text-embedding-3-small") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

def _clip_to_tokens(text: str, max_tokens: int, model: str = "text-embedding-3-small") -> str:
    if not text:
        return ""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


# -------------------------
# Build pipeline chính
# -------------------------
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
    # giới hạn an toàn cho mỗi item embeddings (gần ngưỡng 8k tokens của dòng t-e-3-*)
    per_item_token_max: int = 7900,
) -> Tuple[int, int]:
    """
    Trả về: (số task ghi DuckDB, số chunk đã xử lý)
    """

    files = list(iter_sgml_files(extracted_dir))
    if not files:
        raise RuntimeError("Không tìm thấy file SGML/XML trong thư mục.")

    # 1) DuckDB: tạo bảng nếu chưa có
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

    # 2) Shard & resume
    os.makedirs(vector_dir, exist_ok=True)
    existing_shards = sorted([d for d in os.listdir(vector_dir) if d.startswith("shard_")])
    shard_idx = (int(existing_shards[-1].split("_")[1]) + 1) if (resume and existing_shards) else 1

    # 3) Embeddings
    if not no_embed:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=embedding_model)
    else:
        embeddings = None  # type: ignore

    total_tasks = 0
    total_chunks = 0
    duckdb_rows: List[Tuple] = []
    shard_docs: List[Document] = []

    def flush_duckdb_rows(rows):
        if not rows:
            return
        con.executemany("INSERT INTO refs VALUES (?,?,?,?,?,?,?)", rows)
        rows.clear()

    def _embed_texts_with_retry(texts: List[str]) -> List[Optional[List[float]]]:
        """Chuẩn hoá + cắt token + embed theo batch, có retry/backoff. Trả về list vector/None."""
        if embeddings is None:
            # nếu no_embed=True, không embed
            return [None] * len(texts)

        clean_texts = []
        idx_map = []
        skipped = 0
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                skipped += 1
                continue
            if _tiktoken_len(t, embedding_model) > per_item_token_max:
                t = _clip_to_tokens(t, per_item_token_max, embedding_model)
            if not t:
                skipped += 1
                continue
            clean_texts.append(t)
            idx_map.append(i)

        if not clean_texts:
            return [None] * len(texts)

        vecs_all: List[Optional[List[float]]] = [None] * len(texts)
        start = 0
        while start < len(clean_texts):
            batch = clean_texts[start:start+embed_batch_size]
            for attempt in range(6):
                try:
                    vecs = embeddings.embed_documents(batch)  # type: ignore
                    break
                except Exception:
                    # backoff 1,2,4,8,16,32s
                    time.sleep(1.0 * (2 ** attempt))
                    if attempt == 5:
                        raise
            for j, v in enumerate(vecs):
                vecs_all[idx_map[start+j]] = v
            start += embed_batch_size

        return vecs_all

    def save_faiss_shard(shard_docs: List[Document], shard_idx: int) -> int:
        """
        Tạo & lưu một shard FAISS từ danh sách Document:
        - Tách vectors / texts / metadatas / ids
        - Thử nhiều chữ ký from_embeddings(...) để tương thích phiên bản
        - Fallback: tự dựng faiss.IndexFlatL2 + InMemoryDocstore
        """
        if not shard_docs:
            return 0

        # 1) Chuẩn bị dữ liệu
        texts = [d.page_content or "" for d in shard_docs]
        metas = [d.metadata or {} for d in shard_docs]

        # 2) Tính embedding có kiểm soát
        vecs = _embed_texts_with_retry(texts)

        # 3) Lọc lỗi
        vectors, texts_ok, metas_ok, ids_ok = [], [], [], []
        dropped = 0
        for i, (v, t, m) in enumerate(zip(vecs, texts, metas)):
            if v is None:
                dropped += 1
                continue
            vectors.append(v)
            texts_ok.append(t)
            metas_ok.append(m)
            cid = m.get("chunk_id") if isinstance(m, dict) else None
            ids_ok.append(str(cid) if cid else f"{shard_idx:04d}_{i:06d}")

        if not vectors:
            return 0

        # 4) Xây Vectorstore
        try:
            # New-style: keyword params đầy đủ
            store = FAISS.from_embeddings(
                embeddings=vectors,
                texts=texts_ok,
                metadatas=metas_ok,
                ids=ids_ok,
                embedding=embeddings  # type: ignore
            )
        except TypeError:
            try:
                # Old-style: positional / hỗn hợp (tuỳ version)
                store = FAISS.from_embeddings(
                    vectors,
                    metadatas=metas_ok,
                    ids=ids_ok,
                    embedding=embeddings,  # type: ignore
                    texts=texts_ok
                )
            except Exception:
                # Fallback thủ công
                try:
                    import numpy as np
                except Exception as e:
                    raise RuntimeError("Thiếu numpy – cần 'numpy' để dựng FAISS thủ công.") from e
                try:
                    import faiss as faiss_lib
                except Exception as e:
                    raise RuntimeError("Thiếu faiss – cần 'faiss-cpu' để dựng FAISS thủ công.") from e
                try:
                    from langchain_community.docstore.in_memory import InMemoryDocstore
                except Exception:
                    from langchain.docstore.in_memory import InMemoryDocstore  # fallback rất cũ

                dim = len(vectors[0])
                index = faiss_lib.IndexFlatL2(dim)
                index.add((__import__("numpy")).array(vectors, dtype="float32"))

                # map id -> Document
                docstore_dict = {ids_ok[i]: Document(page_content=texts_ok[i], metadata=metas_ok[i])
                                 for i in range(len(ids_ok))}
                docstore = InMemoryDocstore(docstore_dict)
                index_to_docstore_id = {i: ids_ok[i] for i in range(len(ids_ok))}

                store = FAISS(
                    embedding_function=embeddings,  # type: ignore
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                )

        # 5) Lưu shard
        shard_dir = os.path.join(vector_dir, f"shard_{shard_idx:04d}")
        os.makedirs(shard_dir, exist_ok=True)
        store.save_local(shard_dir)
        if dropped:
            print(f"[WARN] Shard #{shard_idx}: dropped {dropped} empty/oversized chunks")
        return len(texts_ok)

    # -------------------------
    # Vòng lặp xử lý
    # -------------------------
    for rec in parse_files_parallel(files, manual_type, max_workers=max_workers_parse):
        # Ghi DuckDB theo lô
        m = TASK_PATTERN.search(rec["task_full"])
        aa_bb_cc = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        duckdb_rows.append((
            rec["task_full"], aa_bb_cc, rec["ata04"], rec["doc_type"], rec["title"], None, rec["source_file"]
        ))
        total_tasks += 1
        if len(duckdb_rows) >= 2000:
            flush_duckdb_rows(duckdb_rows)

        # Chunk
        for i, ch in enumerate(splitter.split_text(rec["content"] or "")):
            meta = {
                "task_full": rec["task_full"],
                "ata04": rec["ata04"],
                "doc_type": rec["doc_type"],
                "title": rec["title"],
                "source_file": rec["source_file"],
                "chunk_id": f"{rec['task_full']}#{i}",
            }
            shard_docs.append(Document(page_content=ch, metadata=meta))
            total_chunks += 1

            # Đủ 1 shard → embed & lưu
            if (not no_embed) and len(shard_docs) >= shard_size:
                save_faiss_shard(shard_docs, shard_idx)
                shard_docs.clear()
                shard_idx += 1

    # Flush phần còn lại
    flush_duckdb_rows(duckdb_rows)
    con.close()

    if not no_embed and shard_docs:
        save_faiss_shard(shard_docs, shard_idx)
        shard_docs.clear()

    return (total_tasks, total_chunks)
