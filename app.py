import os
import re
from pathlib import Path

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
from core.mapping import ORDER_OUTCOLS

# ----------------------------
# Cấu hình trang
# ----------------------------
st.set_page_config(page_title="WO → ATA04 Checker (Drive + Memory)", layout="wide")
st.title("WO → ATA04 Checker (Drive + Incremental Memory)")
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
    """
    Tìm Service Account JSON trong st.secrets.
    Hỗ trợ cả dạng string (JSON text) lẫn dict (object).
    """
    candidates = [
        "GDRIVE_SERVICE_ACCOUNT_JSON",
        "SERVICE_ACCOUNT_JSON",
        "gdrive_service_account",
        "service_account",
    ]
    for k in candidates:
        if k in st.secrets:
            val = st.secrets[k]
            if isinstance(val, str):
                return val.encode("utf-8")
            elif isinstance(val, dict):
                import json
                return json.dumps(val).encode("utf-8")
    return None

def _default_folder_from_secrets() -> str | None:
    for k in ("GDRIVE_FOLDER_ID", "DRIVE_FOLDER_ID"):
        if k in st.secrets:
            return st.secrets[k]
    return None

# Xuất OPENAI_API_KEY vào env (nếu có trong secrets)
if "OPENAI_API_KEY" in st.secrets and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

default_folder_id = _default_folder_from_secrets()
default_sa_json_bytes = _read_sa_json_from_secrets()

# ----------------------------
# Sidebar: Kết nối Google Drive
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
    use_catalog = st.checkbox("Dùng Catalog (TF-IDF)", value=True)
    show_debug = st.checkbox("Hiển thị debug", value=False)

# ----------------------------
# Đồng bộ & ingest dữ liệu
# ----------------------------
if do_sync:
    if not folder_id or not sa_json_bytes:
        st.error("Thiếu Folder ID hoặc Service Account JSON (secrets hoặc upload).")
        st.stop()

    ensure_dirs()
    # Lưu SA JSON tạm
    sa_path = "data_store/sa.json"
    with open(sa_path, "wb") as f:
        f.write(sa_json_bytes)

    # Gọi đồng bộ (bọc lỗi để hiển thị rõ ràng)
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
                and any(re.search(r"name|title|system|mô tả|description", c) for c in cols)
            )
            is_wo = (
                any(re.search(r"w/?o.*desc|description|defect|symptom", c) for c in cols)
                and any(re.search(r"w/?o.*action|rectification|action|repair|corrective", c) for c in cols)
            )

            if is_ata_map:
                code_col = next((c for c in df.columns if re.search(r"ata.*0?4|^ata$|code", c, flags=re.I)), None)
                name_col = next((c for c in df.columns if re.search(r"name|title|system|mô tả|description", c, flags=re.I)), None)
                if code_col and name_col:
                    append_ata_map(df, code_col, name_col)
                    n_map += 1
            elif is_wo:
                def find_col(pats):
                    for pat in pats:
                        for c in df.columns:
                            if re.search(pat, c, flags=re.I):
                                return c
                    return None
                desc_col = find_col([r"^W/?O\s*Description$", r"\b(description|defect|symptom)\b"])
                act_col  = find_col([r"^W/?O\s*Action$", r"\b(rectification|action|repair|corrective)\b"])
                ata_final_col   = find_col([r"\bATA\s*0?4\s*Corrected\b", r"\bATA\s*final\b", r"\bATA04_Final\b"])
                ata_entered_col = find_col([r"^ATA$", r"\bATA\s*0?4\b", r"\bATA04_Entered\b"])

                if desc_col and act_col and (ata_final_col or ata_entered_col):
                    append_wo_training(df, desc_col, act_col, ata_final_col or "", ata_entered_col or "", p.name)
                    n_wo += 1
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

    # Tự build catalog nếu chưa có
    try:
        ensure_dirs()
        if not catalog_exists():
            stat = build_catalog_from_memory()
            st.success(f"Catalog chưa có → đã build mới từ bộ nhớ ({len(stat)} lớp).")
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
        stat = build_catalog_from_memory()
        st.success(f"Đã build Catalog từ bộ nhớ ({len(stat)} lớp).")
        st.dataframe(stat.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi build catalog: {e}")

# ----------------------------
# Xử lý WO mới (upload thủ công)
# ----------------------------
st.header("Xử lý WO mới")
uploaded = st.file_uploader("Upload Excel WO cần suy luận ATA corrected", type=["xlsx", "xls"])

if uploaded is not None:
    df = load_wo_excel(uploaded)
    st.success(f"Đã nạp {len(df)} dòng.")

    catalog = None
    if use_catalog:
        try:
            # Nếu chưa có catalog, cố build từ bộ nhớ trước khi nạp
            if not catalog_exists():
                ensure_dirs()
                stat = build_catalog_from_memory()
                st.info(f"Catalog chưa có → vừa build từ bộ nhớ ({len(stat)} lớp).")
            catalog = ATACatalog("catalog")
        except Exception as e:
            st.error(f"Không load được Catalog: {e}")
            st.stop()

    results = []
    citations_example = None

    for _, row in df.iterrows():
        defect = row.get("Defect_Text")
        action = row.get("Rectification_Text")
        e0 = row.get("ATA04_Entered")

        # 1) Non-Defect filter
        is_tech = is_technical_defect(defect, action)

        # 2) E1: citations
        citations = extract_citations(f"{defect or ''} {action or ''}")
        if citations and citations_example is None:
            citations_example = citations[:]
        e1_valid, e1_ata, cited_manual, cited_task = False, None, None, None
        for c in citations:
            if c.get("ata04"):
                e1_valid = True
                e1_ata = c["ata04"]
                cited_manual = c["manual"]
                cited_task = c["task"]
                break

        # 3) E2: Catalog
        e2_best = None
        derived_task = derived_doc = derived_score = evidence_snip = evidence_src = None
        if use_catalog and catalog and is_tech:
            e2_best, _ = catalog.predict(defect, action)
            if e2_best:
                derived_task = e2_best.get("ata04")
                derived_doc  = e2_best.get("doc")
                derived_score= e2_best.get("score")
                evidence_snip= e2_best.get("snippet")
                evidence_src = e2_best.get("source")

        # 4) Decision
        decision, conf, reason = decide(
            e0=(e0 if isinstance(e0, str) and len(e0) >= 5 else None),
            e1_valid=(is_tech and e1_valid),
            e1_ata=e1_ata,
            e2_best=e2_best,
            e2_all=None
        )
        ata_final = (e1_ata or derived_task or e0) if decision in ("CONFIRM", "CORRECT") else (derived_task or e1_ata or e0)

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
        })

    res_df = pd.DataFrame(results)
    df["ATA04_Final"] = res_df["ATA04_Final"]
    view_cols = [c for c in ORDER_OUTCOLS if c in res_df.columns]
    st.subheader("Kết quả")
    st.dataframe(res_df[view_cols].head(200), use_container_width=True)

    path = write_result(df, path="WO_ATA_checked.xlsx")
    st.download_button("Tải kết quả Excel", data=open(path, "rb"), file_name="WO_ATA_checked.xlsx")

    if show_debug:
        st.write("Ví dụ citations:", citations_example[:3] if citations_example else None)
