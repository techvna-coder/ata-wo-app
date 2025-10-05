# app.py
import os
import re
import io
import json
import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.gdrive_sync import sync_drive_folder
from core.store import init_db, append_wo_training, append_ata_map
from core.catalog_builder import build_catalog_from_memory
from core.io_excel import load_wo_excel, write_result
from core.nondefect import is_technical_defect
from core.refs import extract_citations
from core.ata_catalog import ATACatalog
from core.decision import decide

# (tùy chọn) Fallback LLM khi mơ hồ
try:
    from core.openai_helpers import llm_suggest_ata
    HAS_LLM = True
except Exception:
    HAS_LLM = False

# (tùy chọn) Audit
try:
    from core.audit import audit_store, list_ingested_files, classify_all_ingested
    HAS_AUDIT = True
except Exception:
    HAS_AUDIT = False

# ----------------------------
# Cấu hình trang
# ----------------------------
st.set_page_config(page_title="WO → ATA04 Checker (Drive + Memory + Optional LLM)", layout="wide")
st.title("WO → ATA04 Checker (Drive + Incremental Memory)")

# Khởi tạo kho dữ liệu
init_db()

# ----------------------------
# Tiện ích: kiểm tra/tạo catalog
# ----------------------------
CAT_JSON = Path("catalog/ata_catalog.json")
CAT_VEC  = Path("catalog/model/tfidf_vectorizer.joblib")
CAT_MAT  = Path("catalog/model/tfidf_matrix.npz")

def catalog_exists() -> bool:
    return CAT_JSON.exists() and CAT_VEC.exists() and CAT_MAT.exists()

def ensure_dirs():
    Path("catalog/model").mkdir(parents=True, exist_ok=True)
    Path("data_store").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Đọc Secrets (nếu có)
# ----------------------------
def _read_sa_json_from_secrets() -> bytes | None:
    candidates = [
        "GDRIVE_SERVICE_ACCOUNT_JSON",
        "SERVICE_ACCOUNT_JSON",
        "gdrive_service_account",
        "service_account",
    ]
    if hasattr(st, "secrets"):
        for k in candidates:
            if k in st.secrets:
                val = st.secrets[k]
                if isinstance(val, str):
                    return val.encode("utf-8")
                elif isinstance(val, dict):
                    import json as _json
                    return _json.dumps(val).encode("utf-8")
    return None

def _default_folder_from_secrets() -> str | None:
    if hasattr(st, "secrets"):
        for k in ("GDRIVE_FOLDER_ID", "DRIVE_FOLDER_ID"):
            if k in st.secrets:
                return st.secrets[k]
    return None

# Xuất OPENAI_API_KEY vào env (nếu có trong secrets)
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

default_folder_id = _default_folder_from_secrets()
default_sa_json_bytes = _read_sa_json_from_secrets()

# ----------------------------
# Sidebar: Kết nối Google Drive và tuỳ chọn
# ----------------------------
with st.sidebar:
    st.header("Kết nối Google Drive")

    folder_id = st.text_input(
        "Drive Folder ID",
        value=default_folder_id or "",
        help="URL dạng drive.google.com/drive/folders/<FOLDER_ID> (chỉ cần phần <FOLDER_ID>)"
    )

    sa_json_upload = st.file_uploader(
        "Service Account JSON (bỏ qua nếu đã cài trong Secrets)",
        type=["json"]
    )
    sa_json_bytes = default_sa_json_bytes
    if not sa_json_bytes and sa_json_upload:
        sa_json_bytes = sa_json_upload.read()

    do_sync = st.button("Đồng bộ từ Drive (incremental)")

    st.markdown("---")
    do_rebuild = st.button("Rebuild Catalog từ bộ nhớ")

    use_catalog = st.checkbox("Dùng Catalog (TF-IDF) offline", value=True)

    
    # Tùy chọn LLM
    st.markdown("---")
    use_llm_fallback = st.checkbox("Dùng OpenAI khi mơ hồ/không xác định", value=True and HAS_LLM, disabled=(not HAS_LLM))
    use_llm_enrich_build = st.checkbox("Dùng OpenAI làm giàu Catalog khi build", value=False, disabled=(not HAS_LLM))
    low_conf_thresh = st.slider("Ngưỡng confidence kích hoạt LLM", 0.0, 1.0, 0.82, 0.01)

    st.markdown("---")
    show_debug = st.checkbox("Hiển thị debug", value=False)

    st.markdown("---")
    do_audit = st.button("Kiểm tra dữ liệu đã đồng bộ (Audit)", disabled=(not HAS_AUDIT))

# ----------------------------
# Đồng bộ & ingest dữ liệu
# ----------------------------
if do_sync:
    if not folder_id or not sa_json_bytes:
        st.error("Thiếu Folder ID hoặc Service Account JSON (secrets hoặc upload).")
        st.stop()

    ensure_dirs()
    sa_path = "data_store/sa.json"
    with open(sa_path, "wb") as f:
        f.write(sa_json_bytes)

    try:
        changed = sync_drive_folder(folder_id, sa_path)
    except Exception as e:
        st.error(f"Lỗi xác thực hoặc truy cập Drive: {e}")
        st.stop()

    st.success(f"Đồng bộ xong. {len(changed)} file mới/cập nhật.")

    # Ingest → memory (phân loại file ATA map / WO)
    inj_dir = Path("data_store/ingest")
    files = sorted([p for p in inj_dir.glob("*.xls*")])
    n_map, n_wo = 0, 0

    for p in files:
        try:
            df = pd.read_excel(p, dtype=str)
            cols = [c.lower() for c in df.columns]

            # Heuristic nhận diện file: ATA map vs WO
            is_ata_map = (
                any(re.search(r"ata.*0?4|^ata$|code", c) for c in cols)
                and any(re.search(r"name|title|system|mô tả|mo ta|description", c) for c in cols)
            )

            # Mở rộng pattern tiếng Anh + tiếng Việt cho WO
            desc_patterns = [
                r"^W/?O\s*Description$", r"\b(description|defect|symptom)\b",
                r"\bmô\s*tả\b", r"\bmo\s*ta\b",
            ]
            action_patterns = [
                r"^W/?O\s*Action$", r"\b(rectification|action|repair|corrective|rectify)\b",
                r"\bhành\s*động\b", r"\bhanh\s*dong\b", r"\bkhắc\s*phục\b", r"\bkhac\s*phuc\b",
            ]
            ata_final_patterns = [
                r"\bATA\s*0?4\s*(Corrected|Final)\b", r"\bATA\s*final\b", r"\bATA04_Final\b",
                r"\bATA\s*Corrected\b", r"\bATA\s*04\s*Corrected\b",
            ]
            ata_entered_patterns = [
                r"^ATA$", r"\bATA\s*0?4\b", r"\bATA\s*04\b", r"\bATA04_Entered\b",
                r"\bATA\s*Code\b", r"\bATA_Code\b",
            ]

            def find_col(pats):
                for pat in pats:
                    for c in df.columns:
                        if re.search(pat, c, flags=re.I):
                            return c
                return None

            is_wo = False
            desc_col = find_col(desc_patterns)
            act_col  = find_col(action_patterns)
            ata_final_col   = find_col(ata_final_patterns)
            ata_entered_col = find_col(ata_entered_patterns)

            # Thư giãn điều kiện: coi là WO nếu có mô tả + (ATA_final hoặc ATA_entered).
            if desc_col and (ata_final_col or ata_entered_col):
                is_wo = True

            if is_ata_map and not is_wo:
                code_col = next((c for c in df.columns if re.search(r"ata.*0?4|^ata$|code", c, flags=re.I)), None)
                name_col = next((c for c in df.columns if re.search(r"name|title|system|mô tả|mo ta|description", c, flags=re.I)), None)
                if code_col and name_col:
                    append_ata_map(df, code_col, name_col)
                    n_map += 1
            elif is_wo:
                append_wo_training(df, desc_col, act_col or "", ata_final_col or "", ata_entered_col or "", p.name)
                n_wo += 1
            else:
                st.info(f"Không nhận diện được loại file: {p.name}. Một số cột đầu: {df.columns.tolist()[:12]}")
        except Exception as e:
            st.warning(f"Lỗi đọc {p.name}: {e}")

    st.info(f"Đã cập nhật bộ nhớ: {n_map} file ATA map, {n_wo} file WO.")

    # Chẩn đoán nhanh sau đồng bộ
    if Path("data_store/wo_training.parquet").exists():
        st.success("Đã có data_store/wo_training.parquet (WO lịch sử).")
    else:
        st.warning("Chưa thấy wo_training.parquet. Kiểm tra lại định dạng cột WO để nhận diện đúng (Description/Action/ATA...).")

    if Path("data_store/ata_map.parquet").exists():
        st.success("Đã có data_store/ata_map.parquet (bảng ATA).")
    else:
        st.warning("Chưa thấy ata_map.parquet. Kiểm tra lại file định nghĩa ATA (phải có cột mã & tên).")

    # Tự build catalog nếu chưa có và đã có WO training
    try:
        ensure_dirs()
        if not catalog_exists():
            if Path("data_store/wo_training.parquet").exists():
                stat = build_catalog_from_memory(use_llm_enrich=bool(use_llm_enrich_build))
                st.success(f"Catalog chưa có → đã build mới từ bộ nhớ ({len(stat)} lớp).")
            else:
                st.warning("Không thể build catalog do chưa có data_store/wo_training.parquet.")
        else:
            st.info("Catalog đã tồn tại. Bạn có thể bấm 'Rebuild Catalog' nếu muốn cập nhật.")
    except Exception as e:
        st.warning(f"Không build được catalog sau đồng bộ: {e}")

# ----------------------------
# Build lại Catalog TF-IDF từ bộ nhớ (thủ công)
# ----------------------------
if do_rebuild:
    try:
        ensure_dirs()
        stat = build_catalog_from_memory(use_llm_enrich=bool(use_llm_enrich_build))
        st.success(f"Đã build Catalog từ bộ nhớ ({len(stat)} lớp).")
        st.dataframe(stat.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi build catalog: {e}")

# ----------------------------
# KIỂM TRA DỮ LIỆU ĐÃ ĐỒNG BỘ (AUDIT)
# ----------------------------
if do_audit and HAS_AUDIT:
    st.header("Kiểm tra dữ liệu đã đồng bộ (Audit)")

    files = list_ingested_files()
    st.subheader("Danh sách file đã đồng bộ (manifest)")
    if files:
        df_files = pd.DataFrame(files)
        show_cols = [c for c in ["name", "path", "modified"] if c in df_files.columns]
        st.dataframe(df_files[show_cols], use_container_width=True)
    else:
        st.info("Chưa có manifest hoặc chưa sync file nào.")

    st.subheader("Phân loại file trong data_store/ingest/")
    classified = classify_all_ingested(limit_preview_rows=5)
    for item in classified:
        st.write(f"**{item.get('path')}** → kind: `{item.get('kind')}`")
        if item.get("kind") == "WO":
            st.write(f"- desc_col: `{item.get('desc_col')}`, action_col: `{item.get('action_col')}`, "
                     f"ata_final_col: `{item.get('ata_final_col')}`, ata_entered_col: `{item.get('ata_entered_col')}`")
            if "preview" in item and isinstance(item["preview"], pd.DataFrame):
                st.dataframe(item["preview"], use_container_width=True)
        elif item.get("kind") == "ATA_MAP":
            st.write("- Phát hiện là bảng ATA (mã + tên).")
        elif item.get("kind") == "error":
            st.warning(f"- Lỗi đọc: {item.get('error')}")
        else:
            st.info("- Không nhận diện được loại file. Hãy xem lại tên cột.")

    st.subheader("Tổng hợp bộ nhớ (Parquet)")
    rep = audit_store()

    wo_rep = rep.get("wo_training", {})
    if wo_rep.get("exists"):
        st.success(f"wo_training.parquet: {wo_rep['rows']} dòng, {wo_rep['distinct_ata04']} ATA04 khác nhau.")
        st.write("Top 20 ATA04 theo số mẫu:")
        st.dataframe(wo_rep["top_ata"], use_container_width=True)
        st.write("Mẫu dữ liệu:")
        st.dataframe(wo_rep["sample"], use_container_width=True)
    else:
        st.warning("Chưa có data_store/wo_training.parquet.")

    ata_rep = rep.get("ata_map", {})
    if ata_rep.get("exists"):
        st.success(f"ata_map.parquet: {ata_rep['rows']} dòng.")
        if "coverage_on_training" in ata_rep:
            st.info(f"Độ phủ tên gọi ATA trên training: {ata_rep['coverage_on_training']}%")
        st.write("Mẫu dữ liệu:")
        st.dataframe(ata_rep["sample"], use_container_width=True)
    else:
        st.warning("Chưa có data_store/ata_map.parquet.")

# ----------------------------
# Xử lý WO mới (upload thủ công)
# ----------------------------
st.header("Xử lý WO mới")
uploaded = st.file_uploader("Upload Excel WO cần suy luận ATA corrected", type=["xlsx", "xls"])

if uploaded is not None:
    # df: khung nội bộ (đã map các cột chuẩn), dùng để suy luận
    df = load_wo_excel(uploaded)
    st.success(f"Đã nạp {len(df)} dòng.")

    catalog = None
    if use_catalog:
        try:
            if not catalog_exists():
                ensure_dirs()
                if Path("data_store/wo_training.parquet").exists():
                    stat = build_catalog_from_memory(use_llm_enrich=bool(use_llm_enrich_build))
                    st.info(f"Catalog chưa có → vừa build từ bộ nhớ ({len(stat)} lớp).")
                else:
                    st.error("Chưa có data_store/wo_training.parquet để build catalog. Hãy đồng bộ Drive chứa WO lịch sử.")
                    st.stop()
            catalog = ATACatalog("catalog")
        except Exception as e:
            st.error(f"Không load được Catalog: {e}")
            st.stop()

    results = []
    citations_example = None

    from core.openai_helpers import llm_arbitrate_when_review  # dùng arbitration mới
    
    # ...
    
    results = []
    citations_example = None
    
    for _, row in df.iterrows():
        defect = row.get("Defect_Text")
        action = row.get("Rectification_Text")
        e0 = row.get("ATA04_Entered")
    
        # 1) Non-Defect filter
        wo_type = row.get("WO_Type")
        is_tech = is_technical_defect(defect, action, wo_type=wo_type)
    
        # 2) E1: trích dẫn (AMM/TSM/FIM)
        citations = extract_citations(f"{defect or ''} {action or ''}")
        if citations and citations_example is None:
            citations_example = citations[:]
        e1_valid, e1_ata, cited_manual, cited_task = False, None, None, None
        for c in citations:
            if c.get("ata04"):
                e1_valid = True
                e1_ata = c["ata04"]
                cited_manual = c.get("manual")
                cited_task = c.get("task")
                break
    
        # 3) E2: Catalog (Top-K cho LLM)
        e2_best = None
        e2_all = []
        derived_task = derived_doc = derived_score = evidence_snip = evidence_src = None
        ata_name_map = {}
        if use_catalog:
            if catalog and is_tech:
                e2_best, e2_all = catalog.predict(defect, action)   # best, list
                if e2_best:
                    derived_task = e2_best.get("ata04")
                    derived_doc  = e2_best.get("doc")
                    derived_score= e2_best.get("score")
                    evidence_snip= e2_best.get("snippet")
                    evidence_src = e2_best.get("source")
                ata_name_map = catalog.name_map()
    
        # 4) Tam-đối-soát
        decision, conf, reason = decide(
            e0=(e0 if isinstance(e0, str) and len(e0) >= 5 else None),
            e1_valid=(is_tech and e1_valid),
            e1_ata=e1_ata,
            e2_best=e2_best,
            e2_all=e2_all
        )
        ata_final = (e1_ata or derived_task or e0) if decision in ("CONFIRM", "CORRECT") else (derived_task or e1_ata or e0)
    
        # 5) LLM arbitration khi REVIEW & Technical Defect
        llm_used = False
        if use_llm_fallback and HAS_LLM and is_tech and decision == "REVIEW":
            # Chuẩn bị tập ứng viên cho LLM: E1 + Top-K + E0
            uniq = {}
            # E1 (nếu có)
            if e1_ata:
                uniq[e1_ata.upper()] = {"ata04": e1_ata.upper(), "why": "citation", "score": 1.0, "snippet": "", "source": "citation"}
            # E2 all
            for c in (e2_all or []):
                a = (c.get("ata04") or "").upper()
                if a and a not in uniq:
                    uniq[a] = {"ata04": a, "why": "tfidf", "score": float(c.get("score") or 0.0),
                               "snippet": c.get("snippet"), "source": c.get("source")}
            # E0 (nếu hợp lệ)
            if isinstance(e0, str) and len(e0) >= 4:
                a0 = e0.strip().upper()
                if a0 not in uniq:
                    uniq[a0] = {"ata04": a0, "why": "entered", "score": 0.5, "snippet": "", "source": "entered"}
    
            cand_list = sorted(uniq.values(), key=lambda x: (-float(x.get("score") or 0.0), x.get("why","")))
            # citations compact (đưa cho LLM)
            cits = []
            for c in (citations or [])[:5]:
                cits.append({
                    "manual": c.get("manual"),
                    "task": c.get("task"),
                    "ata04": c.get("ata04")
                })
    
            arb = llm_arbitrate_when_review(
                desc=defect or "", action=action or "",
                candidates=cand_list[:5], citations=cits, ata_name_map=ata_name_map,
                force_from_candidates=True
            )
            a_llm = (arb.get("ata04") or "").upper()
            if a_llm:
                llm_used = True
                ata_final = a_llm
                chosen_from = arb.get("chosen_from") or ""
                # Hiệu chỉnh confidence
                conf_llm = float(arb.get("confidence") or 0.0)
                # đặt sàn cao hơn khi LLM chọn đúng E1
                if a_llm == (e1_ata or "").upper():
                    conf = max(conf or 0.0, 0.92, conf_llm)
                    decision = "CONFIRM"
                else:
                    conf = max(conf or 0.0, 0.86, conf_llm)
                    decision = "CORRECT" if ata_final and ata_final != (e0 or "") else "CONFIRM"
                reason = (reason or "")
                reason = (reason + f" | LLM arbitration: {arb.get('rationale','')[:300]}").strip()
    
        results.append({
            "Is_Technical_Defect": bool(is_tech),
            "ATA04_Entered": e0,
            "ATA04_From_Cited": e1_ata,
            "Cited_Manual": cited_manual,
            "Cited_Task": cited_task,
            "Cited_Exists": False,
            "ATA04_Derived": derived_task,
            "Derived_Task": derived_task,
            "Derived_DocType": derived_doc,
            "Derived_Score": derived_score,
            "Evidence_Snippet": evidence_snip,
            "Evidence_Source": evidence_src,
            "Decision": decision,
            "ATA04_Final": ata_final,
            "Confidence": conf,
            "Reason": reason,
            "LLM_Used": llm_used,
        })

    res_df = pd.DataFrame(results)

    # === Xuất Excel: giữ NGUYÊN cột gốc, chỉ thêm các cột yêu cầu ===
    raw_df = pd.read_excel(uploaded, dtype=str)  # giữ đúng tiêu đề/thứ tự cột gốc
    out_df = raw_df.copy()

    # Bổ sung cột theo yêu cầu (không đụng vào "ATA 04 Corrected")
    out_df["Is_Technical_Defect"] = res_df["Is_Technical_Defect"].astype(object)
    out_df["ATA04_Final"]         = res_df["ATA04_Final"].astype(object)
    out_df["Confidence"]           = res_df["Confidence"].astype(object)
    out_df["Decision"]             = res_df["Decision"].astype(object)
    out_df["Reason"]               = res_df["Reason"].astype(object)

    # Nếu có cột WO_Number nhưng trống toàn bộ → loại bỏ
    if "WO_Number" in out_df.columns:
        if out_df["WO_Number"].replace(r"^\s*$", np.nan, regex=True).isna().all():
            out_df = out_df.drop(columns=["WO_Number"])

    # Hiển thị tóm tắt kết quả
    view_cols = ["Is_Technical_Defect", "ATA04_Final", "Confidence", "Decision", "Reason"]
    st.subheader("Kết quả")
    st.dataframe(out_df[[c for c in view_cols if c in out_df.columns]].head(200), use_container_width=True)

    # Xuất file kết quả
    from pathlib import Path as _Path
    out_name = f"{_Path(uploaded.name).stem}_ATA_checked.xlsx"
    path = write_result(out_df, path=out_name)
    st.download_button("Tải kết quả Excel", data=open(path, "rb"), file_name=out_name)

    if show_debug:
        st.write("Ví dụ citations:", citations_example[:3] if citations_example else None)

# ----------------------------
# BỔ SUNG: Kiểm tra & tải Catalog
# ----------------------------

def _flatten_catalog_to_df(catalog_path: str) -> pd.DataFrame:
    """
    Đọc catalog/ata_catalog.json và làm phẳng để soát xét.
    Tương thích nhiều schema:
    - {"entries": { "21-52": {...}, "27-10": {...}, ...}}
    - {"entries": [ {"ATA04":"21-52", ...}, {"ata04":"27-10", ...}, ... ]}
    - [ {"ATA04":"21-52", ...}, ... ]
    - {"catalog":[...]}  (ít gặp)
    Mỗi item có thể có: titles, keywords, samples, source, notes.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Không tìm thấy {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        cat = json.load(f)

    # Helper an toàn
    def _to_str_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if x is not None]
        return [str(v)]

    def _pick_ata04(d, fallback_key=None):
        # Lấy mã ATA04 từ nhiều key khả dĩ
        for k in ("ATA04", "ata04", "ATA_04", "ata_04", "code", "Code", "ata", "ATA"):
            if isinstance(d, dict) and k in d and d[k]:
                return str(d[k]).strip().upper()
        if fallback_key:
            return str(fallback_key).strip().upper()
        return None

    # Chuẩn hoá về list các item
    items = []
    if isinstance(cat, dict) and "entries" in cat:
        entries = cat["entries"]
        if isinstance(entries, dict):
            # Dạng map: {"21-52": {...}, ...}
            for key, val in entries.items():
                ata = _pick_ata04(val, fallback_key=key)
                items.append((ata, val or {}))
        elif isinstance(entries, list):
            for val in entries:
                ata = _pick_ata04(val)
                items.append((ata, val or {}))
        else:
            raise ValueError("Trường 'entries' có kiểu không hỗ trợ.")
    elif isinstance(cat, list):
        for val in cat:
            ata = _pick_ata04(val)
            items.append((ata, val or {}))
    elif isinstance(cat, dict) and "catalog" in cat and isinstance(cat["catalog"], list):
        for val in cat["catalog"]:
            ata = _pick_ata04(val)
            items.append((ata, val or {}))
    else:
        # Thử coi cả object là 1 item
        ata = _pick_ata04(cat)
        items.append((ata, cat if isinstance(cat, dict) else {}))

    rows = []
    for ata, it in items:
        titles   = _to_str_list(it.get("titles") or it.get("Titles"))
        keywords = _to_str_list(it.get("keywords") or it.get("Keywords"))
        samples  = _to_str_list(it.get("samples") or it.get("Samples") or it.get("sample_phrases"))
        source   = it.get("source") or it.get("Source") or ""
        notes    = it.get("notes")  or it.get("Notes")  or ""

        rows.append({
            "ATA04": ata,
            "Titles": "; ".join(titles),
            "Keywords": "; ".join(keywords),
            "Sample_Phrases": "; ".join(samples),
            "Source": source,
            "Notes": notes,
        })

    df = pd.DataFrame(rows)
    # Dọn dẹp: chỉ giữ các dòng có ATA04 hợp lệ dạng AA-BB (nếu có)
    if "ATA04" in df.columns:
        df["ATA04"] = df["ATA04"].astype(str).str.upper().str.strip()
        # Chuẩn hoá về AA-BB nếu có dạng "21-5200" or "2152"
        df["ATA04"] = df["ATA04"].str.replace(r"[^0-9\-]", "", regex=True)
        def _fix(ata):
            if not isinstance(ata, str) or not ata:
                return ata
            if "-" in ata:
                parts = ata.split("-")
                if len(parts[0]) >= 2 and len(parts[1]) >= 2:
                    return f"{parts[0][:2]}-{parts[1][:2]}"
            # Nếu là 4 số liền: 2152 → 21-52
            if ata.isdigit() and len(ata) >= 4:
                return f"{ata[:2]}-{ata[2:4]}"
            return ata
        df["ATA04"] = df["ATA04"].map(_fix)
        # loại dòng thiếu mã
        df = df[df["ATA04"].notna() & (df["ATA04"].str.len() >= 5)]
        if not df.empty and "ATA04" in df.columns:
            df = df.sort_values(by=["ATA04"]).reset_index(drop=True)

    return df


def _make_catalog_zip(zip_name: str = None):
    """
    Đóng gói:
    - catalog/ata_catalog.json
    - catalog/model/* (nếu có)
    - data_store/ata_map.parquet, data_store/wo_training.parquet (nếu có)
    - catalog/catalog_flat.csv (nếu làm phẳng thành công)
    """
    cat_dir = "catalog"
    if not os.path.isdir(cat_dir):
        raise FileNotFoundError("Chưa có thư mục 'catalog/'. Hãy build catalog trước.")
    json_path = os.path.join(cat_dir, "ata_catalog.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError("Không thấy 'catalog/ata_catalog.json'.")

    # Cố gắng làm phẳng; nếu lỗi thì vẫn tạo zip nhưng bỏ qua CSV
    csv_bytes = None
    try:
        df_flat = _flatten_catalog_to_df(json_path)
        if not df_flat.empty:
            csv_bytes = df_flat.to_csv(index=False).encode("utf-8")
    except Exception as ex:
        # Không raise để không chặn việc tải gói — chỉ cảnh báo ở UI
        csv_bytes = None

    buf = io.BytesIO()
    if not zip_name:
        zip_name = "catalog_bundle.zip"

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # JSON chính
        zf.write(json_path, arcname="catalog/ata_catalog.json")

        # Model TF-IDF (nếu có)
        model_dir = os.path.join(cat_dir, "model")
        if os.path.isdir(model_dir):
            for path in glob.glob(os.path.join(model_dir, "**"), recursive=True):
                if os.path.isfile(path):
                    arc = os.path.relpath(path, ".")
                    zf.write(path, arcname=arc)

        # Metadata trong data_store nếu có
        for fname in ("ata_map.parquet", "wo_training.parquet"):
            p = os.path.join("data_store", fname)
            if os.path.exists(p):
                zf.write(p, arcname=f"data_store/{fname}")

        # CSV làm phẳng (nếu có)
        if csv_bytes:
            zf.writestr("catalog/catalog_flat.csv", csv_bytes)

    buf.seek(0)
    return buf, zip_name


st.markdown("---")
st.subheader("Kiểm tra & tải Catalog")

col1, col2 = st.columns(2)
with col1:
    if st.button("Xem nhanh catalog (bảng phẳng)"):
        try:
            if not catalog_exists():
                st.warning("Catalog chưa sẵn sàng. Hãy build catalog trước.")
            else:
                df_flat = _flatten_catalog_to_df(str(CAT_JSON))
                st.success(f"Đã đọc catalog: {len(df_flat)} dòng.")
                st.dataframe(df_flat, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Tải catalog_flat.csv",
                    data=df_flat.to_csv(index=False).encode("utf-8"),
                    file_name="catalog_flat.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Không đọc được catalog. Lỗi: {e}")

with col2:
    if st.button("Tạo & tải gói Catalog (.zip)"):
        try:
            buf, zip_name = _make_catalog_zip()
            st.success("Đã tạo gói Catalog.")
            st.download_button(
                label="Tải catalog_bundle.zip",
                data=buf.getvalue(),
                file_name=zip_name,
                mime="application/zip",
            )
            # Thử preview bảng phẳng để báo rõ nếu CSV bị bỏ qua
            try:
                df_preview = _flatten_catalog_to_df(str(CAT_JSON))
                st.info(f"Bảng phẳng có {len(df_preview)} dòng.")
            except Exception as ex2:
                st.warning(f"Đã tạo gói nhưng không thể tạo catalog_flat.csv: {ex2}")
        except Exception as e:
            st.error(f"Không thể tạo gói Catalog: {e}")
