# core/sgml_build.py
import os, tarfile, re, json, shutil, uuid
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import duckdb

TASK_PATTERN = re.compile(r'\b(\d{2})-(\d{2})-(\d{2})(?:-(\d{3})(?:-(\d{3}))*)?\b')
# Một số OEM viết 212600 → chuẩn hóa về 21-26-00
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
    """
    Trả về danh sách task records:
    {task_full, ata04, title, doc_type, source_file, content}
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    # Parse “mềm” để chịu được SGML/DTD
    soup = BeautifulSoup(data, "lxml")  # hoặc "html.parser" nếu lỗi lxml

    records: List[Dict] = []

    # Heuristic theo iSpec 2200/S1000D: task, dm, proc, topic, chap…
    # Ưu tiên thẻ có số hiệu riêng; fallback: regex quét toàn file theo block.
    candidates = []
    for tag in ["task", "dm", "procedure", "topic", "chapter", "section"]:
        candidates.extend(soup.find_all(tag))

    if candidates:
        for blk in candidates:
            # tìm tiêu đề
            title = None
            for ttag in ["title", "tasktitle", "name", "caption", "heading"]:
                tt = blk.find(ttag)
                if tt: 
                    title = _get_text(tt)
                    break
            # tìm task number
            task_full = None
            # 1) thuộc tính/element chuyên biệt
            for key in ["tasknumber", "task", "dmcode", "id", "code", "ident"]:
                v = blk.get(key) if hasattr(blk, "get") else None
                task_full = _norm_task_code(v) or task_full
            if not task_full:
                for t in ["tasknumber", "dmcode", "ref", "id", "code"]:
                    el = blk.find(t)
                    if el:
                        task_full = _norm_task_code(el.get_text())
                        if task_full: break
            # 2) fallback: quét text khối
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
        # Không có thẻ gợi ý → fallback: cắt theo regex toàn file
        for m in TASK_PATTERN.finditer(data):
            task_full = _norm_task_code(m.group(0))
            if not task_full: 
                continue
            # lấy vùng lân cận làm content
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

def build_registry_and_chunks(
    extracted_dir: str,
    manual_type: str,
    duckdb_path: str,
    vector_dir: str,
    openai_api_key: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Tuple[int, int]:
    """
    Duyệt toàn bộ SGML → DuckDB 'refs' + FAISS vectorstore.
    Trả về (số_task, số_chunk).
    """
    # 1) Parse → records
    all_records: List[Dict] = []
    files = list(iter_sgml_files(extracted_dir))
    for fp in tqdm(files, desc=f"Parsing {manual_type} SGML"):
        try:
            recs = parse_one_sgml(fp, manual_type)
            all_records.extend(recs)
        except Exception as e:
            # Bỏ qua file lỗi (ghi log nếu cần)
            pass

    if not all_records:
        raise RuntimeError("Không tìm thấy task hợp lệ trong SGML.")

    # 2) Ghi registry vào DuckDB
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
    # page: SGML không có page theo PDF → để NULL; nếu SGML có số trang, có thể map sau.
    # task_short = AA-BB-CC
    rows = []
    for r in all_records:
        aa_bb_cc = None
        m = TASK_PATTERN.search(r["task_full"])
        if m:
            aa_bb_cc = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        rows.append((
            r["task_full"], aa_bb_cc, r["ata04"], r["doc_type"], r["title"], None, r["source_file"]
        ))
    con.execute("DELETE FROM refs WHERE manual_type = ?", [manual_type])
    con.executemany("INSERT INTO refs VALUES (?,?,?,?,?,?,?)", rows)
    con.close()

    # 3) Chunking → Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for r in all_records:
        chunks = splitter.split_text(r["content"])
        for i, ch in enumerate(chunks):
            meta = {
                "task_full": r["task_full"],
                "ata04": r["ata04"],
                "doc_type": r["doc_type"],
                "title": r["title"],
                "source_file": r["source_file"],
                "chunk_id": f"{r['task_full']}#{i}"
            }
            docs.append(Document(page_content=ch, metadata=meta))

    # 4) Embedding + FAISS
    from langchain_community.vectorstores import FAISS
    os.makedirs(vector_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(vector_dir)

    return (len(all_records), len(docs))
