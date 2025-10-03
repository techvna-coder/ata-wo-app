# app.py
# -------------------------------------------
# WO ATA04 Verifier – Streamlit App (Full, Catalog + RAG)
# -------------------------------------------
# Chức năng:
# 1) Xử lý tài liệu SGML_*.tar (TSM/FIM/AMM) -> parse/chunk/embed (FAISS) [tuỳ chọn]
# 2) "Catalog mode": build/lấy Catalog ATA (JSON + TF-IDF) -> dự đoán ATA rất nhanh (không cần OpenAI)
# 3) Upload Excel WO -> Lọc Non-Defect -> Tam-đối-soát:
#     E0: ATA_Entered, E1: ATA từ TSM/FIM cited (đã xác thực qua DuckDB nếu có),
#     E2: ATA từ Catalog (ưu tiên) hoặc RAG (FAISS) -> Decision + Confidence + Evidence
# 4) Export kết quả ra Excel
#
# Yêu cầu:
# - requirements.txt cần có: streamlit, pandas, openpyxl, duckdb, beautifulsoup4, lxml, regex,
#   tqdm, rapidfuzz, faiss-cpu, langchain, langchain-core, langchain-openai, langchain-community,
#   tiktoken, scikit-learn, joblib
#
# Thư mục/Files:
#   catalog/ata_catalog.json
#   catalog/model/ (tfidf_vectorizer.joblib, ata_tfidf_matrix.joblib)
#   reference_index.duckdb
#   vectorstore/tsm/shard_0001 ... (tuỳ chọn)
# -------------------------------------------

import os
import io
import re
import shutil
import tempfile
from hashlib import sha1

import pandas as pd
import streamlit as st

# Core modules
from core.rag_store import load_faiss
from core.refregistry import RefRegistry
from core.sgml_build import extract_tar, build_registry_and_chunks

# Catalog (TF-IDF, không cần embedding OpenAI)
from core.ata_catalog import (
    load_catalog, load_vectorizer_and_matrix, predict_ata
)

# -----------------------
# Cấu hình & hằng số
# -----------------------

st.set_page_config(page_title="WO ATA04 Verifier", layout="wide")

# Regex trích mã reference trong Rectification
TSM_FIM_REF_RE = re.compile(
    r'\b(?:TSM|FIM)\s*([0-9]{1,2})[-\s]?([0-9]{1,2})(?:[-\s]?([0-9]{1,2}))?(?:-?([0-9]{3}))?(?:-?([0-9]{3}))?\b',
    re.IGNORECASE
)
AMM_REF_RE = re.compile(
    r'\bAMM\s*([0-9]{1,2})[-\s]?([0-9]{1,2})(?:[-\s]?([0-9]{1,2}))?(?:-?([0-9]{3}))?(?:-?([0-9]{3}))?\b',
    re.IGNORECASE
)
TASK_STD_RE = re.compile(r'\b(\d{2})-(\d{2})-(\d{2})(?:-(\d{3})(?:-(\d{3}))*)?\b')

# Từ khóa Non-Defect (có thể tinh chỉnh thêm)
EXCLUDE_PATTERNS = [
    r'\bclean(ing|ed)?\b', r'\blub(e|rication|ricated)\b', r'\bservicing\b',
    r'\bfirst aid kit\b', r'\bty(re|re) wear\b', r'\broutine\b',
    r'\bno fault found\b', r'\bscheduled (task|maintenance)\b', r'\bwater service\b',
    r'\boxygen service\b', r'\bIFE (content|update)\b', r'\bsoftware load\b'
]
EXCLUDE_RE = re.compile('|'.join(EXCLUDE_PATTERNS), re.IGNORECASE)
FAIL_KEYWORDS_RE = re.compile(r'\b(failure|leak|burn|overheat|vibration|ecam|fault|abnormal|smoke|noise|warning)\b', re.IGNORECASE)

# Ngưỡng chọn ứng viên
DERIVED_SCORE_GOOD = 0.55
DERIVED_SCORE_STRONG = 0.70

# -----------------------
# Tiện ích xử lý text/ATA
# -----------------------

def normalize_ata4(x):
    if not x or not isinstance(x, str):
        return None
    m = TASK_STD_RE.search(x)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Nếu nhập dạng "AA-BB"
    m2 = re.match(r'^\s*(\d{2})[-](\d{2})\s*$', x)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return None

def _norm_task_code_from_match(m):
    aa = f"{int(m.group(1)):02d}"
    bb = f"{int(m.group(2)):02d}"
    cc = f"{int(m.group(3) or 0):02d}"
    ddd = m.group(4)
    eee = m.group(5)
    parts = [aa, bb, cc]
    tail = []
    if ddd: tail.append(ddd)
    if eee: tail.append(eee)
    return f"{'-'.join(parts + tail) if tail else '-'.join(parts)}", f"{aa}-{bb}"

def parse_cited_refs(rect_text: str):
    """Trả về list các dict: {manual, task_full, ata04} trích từ Rectification"""
    out = []
    if not rect_text:
        return out
    for m in TSM_FIM_REF_RE.finditer(rect_text):
        task_full, ata4 = _norm_task_code_from_match(m)
        manual = "TSM" if "TSM" in rect_text[m.start()-10:m.start()+3].upper() else "FIM"
        out.append({"manual": manual, "task_full": task_full, "ata04": ata4})
    for m in AMM_REF_RE.finditer(rect_text):
        task_full, ata4 = _norm_task_code_from_match(m)
        out.append({"manual": "AMM", "task_full": task_full, "ata04": ata4})
    # Loại trùng
    uniq = {(d["manual"], d["task_full"]): d for d in out}
    return list(uniq.values())

def is_non_defect(defect_text: str, rect_text: str) -> bool:
    text = f"{defect_text or ''} {rect_text or ''}"
    if EXCLUDE_RE.search(text) and not (defect_text and FAIL_KEYWORDS_RE.search(defect_text)):
        return True
    return False

def normalize_query(defect, rect):
    q = " ".join([str(defect or ""), str(rect or "")]).strip()
    q = re.sub(r'\s+', ' ', q)
    return q[:5000]

# -----------------------
# RAG: chọn ứng viên từ manuals (FAISS, hỗ trợ nhiều shard)
# -----------------------

def get_openai_key():
    return st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

def ensure_stores_loaded(api_key):
    """Nạp FAISS stores (hỗ trợ nhiều shard) vào session_state: ['TSM','FIM','AMM']"""
    if "stores" not in st.session_state:
        st.session_state.stores = {}
    for mt in ["tsm", "fim", "amm"]:
        base_dir = os.path.join("vectorstore", mt)
        if not os.path.isdir(base_dir):
            continue
        shard_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("shard_")])
        if shard_dirs:
            stores = []
            for sd in shard_dirs:
                try:
                    stores.append(load_faiss(sd, api_key))
                except Exception:
                    pass
            if stores:
                st.session_state.stores[mt.upper()] = stores
        else:
            # fallback: 1 index duy nhất
            try:
                st.session_state.stores[mt.upper()] = load_faiss(base_dir, api_key)
            except Exception:
                pass

def _search_store_any(store_or_list, query, k):
    """Tìm best hit trong 1 store hoặc danh sách shard; trả về (doc, score) tốt nhất (score = 1/(1+dist))."""
    best_doc, best_score = None, -1.0
    stores = store_or_list if isinstance(store_or_list, list) else [store_or_list]
    for s in stores:
        try:
            hits = s.similarity_search_with_score(query, k=k)
        except Exception:
            hits = []
        for doc, dist in hits:
            sim = 1.0 / (1.0 + float(dist))
            if sim > best_score:
                best_doc, best_score = doc, sim
    return best_doc, best_score

def derive_from_manuals(stores: dict, query: str, prefer_order=("TSM", "FIM", "AMM"), topk=4):
    best = None
    for mt in prefer_order:
        stobj = stores.get(mt)
        if not stobj:
            continue
        doc, score = _search_store_any(stobj, query, topk)
        if doc:
            cand = {
                "ata04_derived": normalize_ata4(doc.metadata.get("ata04")),
                "task_full": doc.metadata.get("task_full"),
                "doc_type": mt,
                "score": score,
                "title": doc.metadata.get("title", ""),
                "source_file": doc.metadata.get("source_file", ""),
                "snippet": doc.page_content[:800],
            }
            if cand["ata04_derived"] and (not best or cand["score"] > best["score"]):
                best = cand
        if best and best["score"] >= DERIVED_SCORE_STRONG:
            break
    return best if (best and best["score"] >= DERIVED_SCORE_GOOD and best["ata04_derived"]) else None

# -----------------------
# Catalog mode (TF-IDF)
# -----------------------

def ensure_catalog_loaded():
    """Nạp Catalog & TF-IDF vào session_state nếu sẵn có."""
    catalog_ready = os.path.exists("catalog/ata_catalog.json") and os.path.isdir("catalog/model")
    if not catalog_ready:
        st.session_state.catalog_ready = False
        return False
    if "catalog_df" not in st.session_state:
        try:
            st.session_state.catalog_df = load_catalog("catalog/ata_catalog.json")
            vec, X, labels = load_vectorizer_and_matrix("catalog/model")
            st.session_state.tfidf_vec = vec
            st.session_state.tfidf_X = X
            st.session_state.tfidf_labels = labels
            st.session_state.catalog_ready = True
        except Exception:
            st.session_state.catalog_ready = False
    return st.session_state.get("catalog_ready", False)

def derive_from_catalog(defect_text, rect_text=None, topk=3):
    if not st.session_state.get("catalog_ready"):
        return None
    vec = st.session_state.tfidf_vec
    X = st.session_state.tfidf_X
    labels = st.session_state.tfidf_labels
    if vec is None or X is None or labels is None:
        return None
    query = " ".join(filter(None, [str(defect_text or ""), str(rect_text or "")]))
    preds = predict_ata(query, vec, X, labels, topk=topk)
    if not preds:
        return None
    ata, score = preds[0]
    return {
        "ata04_derived": ata,
        "task_full": None,
        "doc_type": "CATALOG",
        "score": float(score),  # 0..1
        "title": None,
        "source_file": "catalog/ata_catalog.json",
        "snippet": None
    }

# -----------------------
# Triangulation (Tam-đối-soát)
# -----------------------

def decision(status, ata_final, conf, reason, evidence=None):
    evidence = evidence or {}
    return {
        "Decision": status,
        "ATA04_Final": ata_final,
        "Confidence": round(conf, 3),
        "Reason": reason,
        **{f"Ev_{k}": v for k, v in evidence.items()}
    }

def reconcile_triplet(ata_entered, cited_best, derived_best):
    """
    cited_best: dict hoặc None:
        {"ata04":..., "task_full":..., "manual": "TSM|FIM|AMM", "exists": True/False}
    derived_best: dict hoặc None:
        {"ata04_derived":..., "task_full":..., "doc_type":..., "score": float, ...}
    """
    e0 = normalize_ata4(ata_entered)
    e1 = normalize_ata4(cited_best["ata04"]) if cited_best else None
    e2 = normalize_ata4(derived_best["ata04_derived"]) if derived_best else None

    # 1) Tất cả trùng
    if e0 and e1 and e2 and (e0 == e1 == e2) and cited_best.get("exists", False):
        return decision("CONFIRM", e2, 0.97, "All sources agree",
                        evidence={"TSM_task": cited_best.get("task_full"),
                                  "Derived_task": derived_best.get("task_full")})

    # 2) E1 = E2 != E0
    if e1 and e2 and (e1 == e2) and e0 != e1 and cited_best.get("exists", False):
        return decision("CORRECT", e1, 0.95, "TSM-cited & Derived agree vs Entered",
                        evidence={"TSM_task": cited_best.get("task_full"),
                                  "Derived_task": derived_best.get("task_full")})

    # 3) Chỉ E2 khớp E0, không có E1
    if (not e1) and e0 and e2 and (e0 == e2):
        base = 0.88 if (derived_best and derived_best.get("score", 0) >= DERIVED_SCORE_STRONG) else 0.83
        return decision("CONFIRM", e2, base, "Entered matches AI-derived (no cited)",
                        evidence={"Derived_task": derived_best.get("task_full")})

    # 4) Chỉ E1 hợp lệ, E2 yếu/không có
    if e1 and (not e2) and cited_best.get("exists", False):
        return decision("CONFIRM", e1, 0.92, "Valid TSM-cited only",
                        evidence={"TSM_task": cited_best.get("task_full")})

    # 5) E1 vs E2 khác nhau → chọn theo chất lượng
    if e1 and e2 and (e1 != e2):
        if cited_best.get("exists", False) and (not derived_best or derived_best.get("score", 0) < DERIVED_SCORE_GOOD):
            return decision("CONFIRM", e1, 0.90, "Prefer valid cited; derived weak",
                            evidence={"TSM_task": cited_best.get("task_full")})
        if derived_best and derived_best.get("score", 0) >= DERIVED_SCORE_STRONG:
            return decision("CORRECT", e2, 0.90, "Derived strong vs cited",
                            evidence={"Derived_task": derived_best.get("task_full")})
        return decision("REVIEW", None, 0.0, "Cited vs Derived conflict")

    # 6) Chỉ có Entered
    if e0 and not e1 and not e2:
        return decision("REVIEW", None, 0.0, "No reliable evidence")

    # 7) Trường hợp còn lại
    return decision("REVIEW", e2 or e1 or None, 0.0, "Ambiguous")

# -----------------------
# UI – Sidebar: SGML build + Catalog info
# -----------------------

with st.sidebar:
    st.header("1) Xử lý SGML (TSM/FIM/AMM) → Index (tuỳ chọn)")
    api_key_sb = st.text_input("OPENAI_API_KEY (embedding, nếu dùng RAG)", type="password", value=get_openai_key() or "")
    if api_key_sb:
        st.session_state["OPENAI_API_KEY"] = api_key_sb

    sgml_tar = st.file_uploader("Upload SGML_*.tar", type=["tar"])
    manual_type = st.selectbox("Manual type", ["TSM", "FIM", "AMM"], index=0)
    build_clicked = st.button("Build Index từ SGML (FAISS)")

    if build_clicked:
        if not sgml_tar:
            st.warning("Vui lòng upload file SGML_*.tar trước.")
        elif not get_openai_key():
            st.warning("Cần OPENAI_API_KEY để tạo embedding (FAISS). Hoặc dùng Catalog mode bên dưới.")
        else:
            with st.spinner("Đang giải nén, parse SGML, chunking & embedding (shard)..."):
                tmpdir = tempfile.mkdtemp(prefix="sgml_")
                tar_path = os.path.join(tmpdir, sgml_tar.name)
                with open(tar_path, "wb") as f:
                    f.write(sgml_tar.getbuffer())
                extracted = extract_tar(tar_path, tmpdir)

                # copy vào reference_docs/<manual_type>
                final_dir = os.path.join("reference_docs", manual_type.lower())
                if os.path.exists(final_dir):
                    shutil.rmtree(final_dir)
                shutil.copytree(extracted, final_dir)

                # build index (sử dụng default an toàn trong core/sgml_build.py đã vá)
                n_task, n_chunk = build_registry_and_chunks(
                    extracted_dir=final_dir,
                    manual_type=manual_type,
                    duckdb_path="reference_index.duckdb",
                    vector_dir=os.path.join("vectorstore", manual_type.lower()),
                    openai_api_key=get_openai_key(),
                )
                st.success(f"Đã build {manual_type}. Tasks={n_task}, Chunks={n_chunk}")

            # sau khi build xong, tự nạp stores/registry
            ensure_stores_loaded(get_openai_key())
            if os.path.exists("reference_index.duckdb"):
                st.session_state.registry = RefRegistry("reference_index.duckdb")

    st.markdown("---")
    st.header("2) Catalog mode (không cần OpenAI)")
    if ensure_catalog_loaded():
        st.success("Catalog sẵn sàng: catalog/ata_catalog.json & catalog/model/*")
    else:
        st.info("Để dùng Catalog mode, chạy script:\n\n"
                "`python build_ata_catalog.py --tar SGML_*.tar --manual-type TSM`\n\n"
                "Script sẽ tạo `catalog/ata_catalog.json` và `catalog/model/*`.")

# -----------------------
# UI – Main: Excel WO
# -----------------------

st.title("WO ATA04 – Xác nhận & Hiệu chỉnh dựa trên Manuals (TSM/FIM/AMM) & Catalog")

# Tự nạp RAG stores (nếu có) và DuckDB
ensure_stores_loaded(get_openai_key() or "")
if "registry" not in st.session_state and os.path.exists("reference_index.duckdb"):
    st.session_state.registry = RefRegistry("reference_index.duckdb")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload Excel WO", type=["xlsx"])
with col2:
    run_btn = st.button("Chạy xử lý Excel")

if uploaded:
    df = pd.read_excel(uploaded)
    # Chuẩn hoá tên cột tối thiểu
    for col in ["ATA04_Entered", "Defect_Text", "Rectification_Text"]:
        if col not in df.columns:
            df[col] = None
    st.write("Xem nhanh 20 dòng đầu:")
    st.dataframe(df.head(20), use_container_width=True)

if uploaded and run_btn:
    stores = st.session_state.get("stores", {})          # RAG (tuỳ có)
    registry = st.session_state.get("registry", None)    # DuckDB (tuỳ có)
    catalog_ready = st.session_state.get("catalog_ready", False)

    # Pha 1: Vectorized – Non-Defect + TSM/FIM/AMM cited
    work = df.copy()

    def vec_is_non_defect(row):
        return is_non_defect(row.get("Defect_Text"), row.get("Rectification_Text"))
    work["Is_Technical_Defect"] = work.apply(vec_is_non_defect, axis=1)

    def vec_cited(row):
        refs = parse_cited_refs(row.get("Rectification_Text"))
        return refs
    work["Cited_Refs"] = work.apply(vec_cited, axis=1)

    def pick_valid_cited(refs):
        if not refs:
            return None
        preferred = sorted(refs, key=lambda x: {"TSM":0, "FIM":1, "AMM":2}.get(x["manual"], 9))
        for r in preferred:
            exists = False
            if registry is not None:
                try:
                    dfq = registry.exact_lookup(r["task_full"])
                    exists = not dfq.empty
                except Exception:
                    exists = False
            r2 = {**r, "exists": bool(exists)}
            if exists:
                return r2
        r = preferred[0]
        return {**r, "exists": False}

    work["Cited_Best"] = work["Cited_Refs"].apply(pick_valid_cited)

    # Pha 2: Derive (ưu tiên Catalog nhanh; nếu không tự tin, mới dùng RAG)
    rag_cache = {}

    def derive_row(row):
        q = normalize_query(row.get("Defect_Text"), row.get("Rectification_Text"))
        # Catalog trước
        if catalog_ready:
            d = derive_from_catalog(row.get("Defect_Text"), row.get("Rectification_Text"), topk=3)
            # ngưỡng Catalog có thể điều chỉnh theo thực nghiệm
            if d and d.get("score", 0) >= 0.25:
                return d
        # RAG nếu có store và là defect thực sự
        if stores and row.get("Is_Technical_Defect", False):
            key = sha1(q.encode("utf-8")).hexdigest()
            if key in rag_cache:
                return rag_cache[key]
            d = derive_from_manuals(stores, q, prefer_order=("TSM","FIM","AMM"), topk=4)
            rag_cache[key] = d
            return d
        return None

    progress = st.progress(0, text="Đang suy luận ATA (Catalog/RAG)…")
    derived_list = []
    total = len(work)
    for idx, row in work.iterrows():
        d = derive_row(row)
        derived_list.append(d)
        if (len(derived_list) % max(1, total//50)) == 0:
            progress.progress(min(1.0, len(derived_list)/total), text=f"Suy luận {len(derived_list)}/{total}")
    work["Derived_Best"] = derived_list
    progress.progress(1.0, text="Hoàn tất suy luận.")

    # Tam-đối-soát
    def reconcile_row(row):
        e0 = row.get("ATA04_Entered")
        e1 = row.get("Cited_Best")
        e2 = row.get("Derived_Best")
        dec = reconcile_triplet(e0, e1, e2)
        ev = {
            "ATA04_Entered": normalize_ata4(e0),
            "ATA04_From_Cited": normalize_ata4(e1.get("ata04")) if e1 else None,
            "Cited_Manual": e1.get("manual") if e1 else None,
            "Cited_Task": e1.get("task_full") if e1 else None,
            "Cited_Exists": e1.get("exists") if e1 else None,
            "ATA04_Derived": normalize_ata4(e2.get("ata04_derived")) if e2 else None,
            "Derived_Task": e2.get("task_full") if e2 else None,
            "Derived_DocType": e2.get("doc_type") if e2 else None,
            "Derived_Score": round(e2.get("score", 0), 3) if e2 else None,
            "Derived_Title": e2.get("title") if e2 else None,
            "Derived_Source": e2.get("source_file") if e2 else None,
            "Evidence_Snippet": (e2.get("snippet")[:600] if e2 and e2.get("snippet") else None),
        }
        return {**dec, **ev}

    result_records = []
    progress2 = st.progress(0, text="Đang đối chiếu và kết luận…")
    for i, row in work.iterrows():
        out = reconcile_row(row)
        result_records.append(out)
        if (len(result_records) % max(1, total//50)) == 0:
            progress2.progress(min(1.0, len(result_records)/total), text=f"Reconcile {len(result_records)}/{total}")
    progress2.progress(1.0, text="Hoàn tất kết luận.")

    res_df = pd.concat([work, pd.DataFrame(result_records)], axis=1)

    st.success("Đã xử lý xong.")
    st.write("Bộ lọc nhanh:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Chỉ hiện REVIEW"):
            st.dataframe(res_df[res_df["Decision"]=="REVIEW"], use_container_width=True)
    with c2:
        if st.button("Chỉ hiện CORRECT"):
            st.dataframe(res_df[res_df["Decision"]=="CORRECT"], use_container_width=True)
    with c3:
        if st.button("Chỉ hiện CONFIRM"):
            st.dataframe(res_df[res_df["Decision"]=="CONFIRM"], use_container_width=True)

    st.write("Bảng kết quả (rút gọn cột chính):")
    show_cols = [
        "Is_Technical_Defect",
        "ATA04_Entered", "ATA04_From_Cited", "Cited_Manual", "Cited_Task", "Cited_Exists",
        "ATA04_Derived", "Derived_Task", "Derived_DocType", "Derived_Score",
        "Decision", "ATA04_Final", "Confidence", "Reason"
    ]
    for c in show_cols:
        if c not in res_df.columns:
            res_df[c] = None
    st.dataframe(res_df[show_cols].head(200), use_container_width=True)

    # Export Excel
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="openpyxl") as xw:
        res_df.to_excel(xw, index=False, sheet_name="WO_ATA_checked")
    st.download_button("Tải kết quả (xlsx)", data=out_buf.getvalue(),
                       file_name="WO_ATA_checked.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------
# Footer
# -----------------------
st.caption("© WO ATA04 Verifier – Tam-đối-soát: Entered vs TSM/FIM-cited vs Catalog/RAG từ manuals (TSM/FIM/AMM).")
