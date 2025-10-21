# app.py - OPTIMIZED VERSION
from __future__ import annotations

import os
import re
import io
import json
import glob
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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

# (tùy chọn) LLM
HAS_LLM = False
try:
    from core.openai_helpers import llm_arbitrate_when_review
    HAS_LLM = True
except Exception:
    HAS_LLM = False

# (tùy chọn) Audit
HAS_AUDIT = False
try:
    from core.audit import audit_store, list_ingested_files, classify_all_ingested
    HAS_AUDIT = True
except Exception:
    HAS_AUDIT = False

# --------------------------------
# Cấu hình trang
# --------------------------------
st.set_page_config(page_title="WO → ATA04 Checker (Optimized)", layout="wide")
st.title("WO → ATA04 Checker (Drive + Incremental Memory) - OPTIMIZED")

# Khởi tạo kho dữ liệu
init_db()

# --------------------------------
# Tiện ích: kiểm tra/tạo catalog
# --------------------------------
CAT_JSON = Path("catalog/ata_catalog.json")
CAT_VEC  = Path("catalog/model/tfidf_vectorizer.joblib")
CAT_MAT  = Path("catalog/model/tfidf_matrix.npz")

def catalog_exists() -> bool:
    return CAT_JSON.exists() and CAT_VEC.exists() and CAT_MAT.exists()

def ensure_dirs():
    Path("catalog/model").mkdir(parents=True, exist_ok=True)
    Path("data_store").mkdir(parents=True, exist_ok=True)

# --------------------------------
# Đọc Secrets (nếu có)
# --------------------------------
def _read_sa_json_from_secrets() -> Optional[bytes]:
    candidates = [
        "GDRIVE_SERVICE_ACCOUNT_JSON",
        "SERVICE_ACCOUNT_JSON",
        "gdrive_service_account",
        "service_account",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_CREDENTIALS_JSON",
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

def _default_folder_from_secrets() -> Optional[str]:
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

# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:
    st.header("Kết nối Google Drive")

    folder_id = st.text_input(
        "Drive Folder ID",
        value=default_folder_id or "",
        help="URL dạng drive.google.com/drive/folders/<FOLDER_ID> (chỉ cần phần <FOLDER_ID>)",
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

    st.markdown("---")
    use_llm_fallback = st.checkbox("Dùng OpenAI khi mơ hồ/không xác định", value=HAS_LLM, disabled=(not HAS_LLM))
    use_llm_enrich_build = st.checkbox("Dùng OpenAI làm giàu Catalog khi build", value=False, disabled=(not HAS_LLM))
    low_conf_thresh = st.slider("Ngưỡng confidence kích hoạt LLM", 0.0, 1.0, 0.82, 0.01)

    st.markdown("---")
    max_llm_calls = st.number_input("Giới hạn số LLM calls", min_value=0, max_value=5000, value=300, step=50, help="0 để tắt fallback")
    show_debug = st.checkbox("Hiển thị debug", value=False)

    st.markdown("---")
    do_audit = st.button("Kiểm tra dữ liệu đã đồng bộ (Audit)", disabled=(not HAS_AUDIT))

# --------------------------------
# Đồng bộ & ingest dữ liệu
# --------------------------------
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

# --------------------------------
# Build lại Catalog thủ công
# --------------------------------
if do_rebuild:
    try:
        ensure_dirs()
        stat = build_catalog_from_memory(use_llm_enrich=bool(use_llm_enrich_build))
        st.success(f"Đã build Catalog từ bộ nhớ ({len(stat)} lớp).")
        st.dataframe(stat.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi build catalog: {e}")

# --------------------------------
# Xử lý WO mới (upload thủ công) – TỐI ƯU TỐC ĐỘ
# --------------------------------
st.header("Xử lý WO mới")
uploaded = st.file_uploader("Upload Excel WO cần suy luận ATA corrected", type=["xlsx", "xls"])

@st.cache_resource(show_spinner=False)
def _load_catalog_resource() -> Optional[ATACatalog]:
    if not catalog_exists():
        return None
    return ATACatalog("catalog")

if uploaded is not None:
    import time
    from datetime import datetime
    
    # Giữ nguyên tiêu đề/thứ tự cột gốc để xuất
    raw_df = pd.read_excel(uploaded, dtype=str)
    # Khung nội bộ chuẩn hoá
    df = load_wo_excel(uploaded)
    st.success(f"Đã nạp {len(df)} dòng.")
    
    # ===== INITIALIZE CATALOG =====
    catalog = None
    if use_catalog:
        try:
            if not catalog_exists():
                ensure_dirs()
                if Path("data_store/wo_training.parquet").exists():
                    _ = build_catalog_from_memory(use_llm_enrich=bool(use_llm_enrich_build))
                else:
                    st.error("Chưa có data_store/wo_training.parquet để build catalog. Hãy đồng bộ Drive chứa WO lịch sử.")
                    st.stop()
            catalog = _load_catalog_resource()
            if catalog is None:
                catalog = ATACatalog("catalog")
        except Exception as e:
            st.error(f"Không load được Catalog: {e}")
            st.stop()
    else:
        st.error("Cần enable 'Dùng Catalog (TF-IDF) offline' để predict ATA!")
        st.stop()
    
    # ===== GET ATA NAME MAP =====
    ata_name_map = None
    try:
        if catalog and hasattr(catalog, "name_map"):
            ata_name_map = catalog.name_map()
    except Exception:
        pass
    
    # ===== CREATE PREDICTOR =====
    from core.ata_predictor import ATAPredictor
    
    predictor = ATAPredictor(
        catalog=catalog,
        ata_name_map=ata_name_map,
        use_llm=use_llm_fallback,
        llm_conf_threshold=float(low_conf_thresh),
        max_llm_calls=int(max_llm_calls),
        verbose=show_debug
    )
    
    # ===== PREDICT (WITH TIMING) =====
    start_time = time.time()
    start_dt = datetime.now()
    
    with st.spinner("🔄 Đang xử lý..."):
        results = predictor.predict_batch(df)
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = len(df) / duration if duration > 0 else 0
    
    # Convert to DataFrame
    res_df = predictor.to_dataframe(results)
    
    # ===== MERGE WITH ORIGINAL =====
    out_df = raw_df.copy()
    for col in res_df.columns:
        out_df[col] = res_df[col].astype(object)
    
    # Remove empty WO_Number column if exists
    if "WO_Number" in out_df.columns:
        if out_df["WO_Number"].replace(r"^\s*$", np.nan, regex=True).isna().all():
            out_df = out_df.drop(columns=["WO_Number"])
    
    # ===== DISPLAY SUMMARY =====
    st.success(f"✅ Processed {len(df)} WOs in {duration:.1f}s ({throughput:.1f} WO/s)")
    
    st.subheader("📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_tech = sum(r.is_technical for r in results)
        st.metric("Technical Defects", f"{n_tech}/{len(results)}", 
                  f"{n_tech/len(results)*100:.1f}%")
    
    with col2:
        n_confirm = sum(1 for r in results if r.decision == "CONFIRM")
        st.metric("CONFIRM", n_confirm, 
                  f"{n_confirm/len(results)*100:.1f}%")
    
    with col3:
        n_correct = sum(1 for r in results if r.decision == "CORRECT")
        st.metric("CORRECT", n_correct,
                  f"{n_correct/len(results)*100:.1f}%")
    
    with col4:
        n_review = sum(1 for r in results if r.decision == "REVIEW")
        st.metric("REVIEW", n_review,
                  f"{n_review/len(results)*100:.1f}%", 
                  delta="⚠️ Need attention" if n_review > 0 else None)
    
    # ===== EVIDENCE STATISTICS =====
    with st.expander("🔍 Evidence Statistics"):
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            n_e0 = sum(1 for r in results if r.e0 and r.e0.is_valid())
            st.metric("E0 (Entered)", f"{n_e0}/{len(results)}")
        
        with col_e2:
            n_e1 = sum(1 for r in results if r.e1 and r.e1.is_valid())
            n_tech = sum(1 for r in results if r.is_technical)
            st.metric("E1 (Citation)", f"{n_e1}/{n_tech}" if n_tech > 0 else "0/0")
        
        with col_e3:
            n_e2 = sum(1 for r in results if r.e2 and r.e2.is_valid())
            st.metric("E2 (Catalog)", f"{n_e2}/{n_tech}" if n_tech > 0 else "0/0")
    
    # ===== CONFIDENCE DISTRIBUTION =====
    with st.expander("📈 Confidence Distribution"):
        try:
            import plotly.express as px
            
            conf_data = pd.DataFrame({
                "Confidence": [r.confidence for r in results],
                "Decision": [r.decision for r in results]
            })
            
            fig = px.histogram(
                conf_data, 
                x="Confidence", 
                color="Decision",
                nbins=20,
                title="Confidence Score Distribution",
                color_discrete_map={
                    "CONFIRM": "#28a745",
                    "CORRECT": "#ffc107", 
                    "REVIEW": "#dc3545"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for visualization: pip install plotly")
    
    # ===== MAIN RESULTS TABLE =====
    st.subheader("📋 Results Table")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        filter_decision = st.multiselect(
            "Filter by Decision",
            ["CONFIRM", "CORRECT", "REVIEW"],
            default=["CONFIRM", "CORRECT", "REVIEW"]
        )
    
    with col_f2:
        filter_tech = st.radio(
            "Filter by Type",
            ["All", "Technical Only", "Non-Technical Only"],
            index=0
        )
    
    with col_f3:
        min_conf = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.0, 0.05
        )
    
    # Apply filters
    filtered_df = out_df.copy()
    
    if filter_decision:
        filtered_df = filtered_df[filtered_df["Decision"].isin(filter_decision)]
    
    if filter_tech == "Technical Only":
        filtered_df = filtered_df[filtered_df["Is_Technical_Defect"] == True]
    elif filter_tech == "Non-Technical Only":
        filtered_df = filtered_df[filtered_df["Is_Technical_Defect"] == False]
    
    if "Confidence" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Confidence"].isna()) | 
            (filtered_df["Confidence"] >= min_conf)
        ]
    
    st.info(f"Showing {len(filtered_df)}/{len(out_df)} rows")
    
    # Display columns
    view_cols = [
        "Is_Technical_Defect", 
        "ATA04_Entered",
        "ATA04_From_Cited",
        "ATA04_Derived",
        "ATA04_Final", 
        "Confidence", 
        "Decision", 
        "Reason"
    ]
    
    # Color coding for decisions
    def highlight_decision(row):
        colors = {
            "CONFIRM": "background-color: #d4edda; color: #155724",
            "CORRECT": "background-color: #fff3cd; color: #856404", 
            "REVIEW": "background-color: #f8d7da; color: #721c24"
        }
        decision = row.get("Decision", "")
        style = colors.get(decision, "")
        return [style] * len(row)
    
    display_df = filtered_df[[c for c in view_cols if c in filtered_df.columns]].head(200)
    
    try:
        styled_df = display_df.style.apply(highlight_decision, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
    except Exception:
        # Fallback without styling
        st.dataframe(display_df, use_container_width=True, height=600)
    
    # ===== DEBUG INFO =====
    if show_debug:
        st.subheader("🐛 Debug Information")
        
        # Show detailed evidence for first 5 rows
        debug_count = min(5, len(results))
        st.write(f"Showing details for first {debug_count} rows:")
        
        for i in range(debug_count):
            r = results[i]
            
            with st.expander(f"Row {i}: {r.ata04_final or 'N/A'} ({r.decision}, conf={r.confidence:.2f})"):
                st.write("**WO Info:**")
                wo = predictor  # Access from closure - we can improve this
                st.code(f"Technical: {r.is_technical}")
                
                st.write("**Evidences:**")
                
                if r.e0:
                    st.write(f"- **E0 (Entered):** `{r.e0.ata04}` (conf: {r.e0.confidence:.2f})")
                    st.json(r.e0.metadata)
                else:
                    st.write("- E0: None")
                
                if r.e1:
                    st.write(f"- **E1 (Citation):** `{r.e1.ata04}` (conf: {r.e1.confidence:.2f})")
                    st.json(r.e1.metadata)
                else:
                    st.write("- E1: None")
                
                if r.e2:
                    st.write(f"- **E2 (Catalog):** `{r.e2.ata04}` (conf: {r.e2.confidence:.2f})")
                    st.json(r.e2.metadata)
                else:
                    st.write("- E2: None")
                
                st.write(f"**Final Decision:** {r.decision} (confidence: {r.confidence:.2f})")
                st.write(f"**Reason:** {r.reason}")
                
                if r.llm_used and r.llm_result:
                    st.write("**LLM Result:**")
                    st.json(r.llm_result)
        
        # Cache statistics
        from core.ata_predictor import get_cache_stats
        cache_stats = get_cache_stats()
        st.write("**Citation Cache:**")
        st.json(cache_stats)
    
    # ===== DOWNLOAD RESULTS =====
    st.subheader("💾 Download Results")
    
    from pathlib import Path as _Path
    
    out_name = f"{_Path(uploaded.name).stem}_ATA_checked.xlsx"
    path = write_result(out_df, path=out_name)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.download_button(
            "📥 Download Full Results (Excel)",
            data=open(path, "rb"),
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    
    with col_d2:
        # Download filtered results
        if len(filtered_df) < len(out_df):
            filtered_name = f"{_Path(uploaded.name).stem}_filtered.xlsx"
            filtered_path = write_result(filtered_df, path=filtered_name)
            st.download_button(
                "📥 Download Filtered Results",
                data=open(filtered_path, "rb"),
                file_name=filtered_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # ===== PERFORMANCE LOG =====
    if use_llm_fallback:
        llm_used_count = sum(1 for r in results if r.llm_result)
        llm_flagged = sum(1 for r in results if r.llm_used)
        st.info(f"🤖 LLM: {llm_used_count}/{llm_flagged} flagged calls made (quota: {max_llm_calls})")
    
    # Save performance log
    perf_log = {
        "timestamp": start_dt.isoformat(),
        "filename": uploaded.name,
        "n_wos": len(df),
        "n_technical": sum(1 for r in results if r.is_technical),
        "duration_sec": round(duration, 2),
        "throughput_per_sec": round(throughput, 2),
        "decisions": {
            "CONFIRM": sum(1 for r in results if r.decision == "CONFIRM"),
            "CORRECT": sum(1 for r in results if r.decision == "CORRECT"),
            "REVIEW": sum(1 for r in results if r.decision == "REVIEW"),
        },
        "evidences": {
            "E0": sum(1 for r in results if r.e0 and r.e0.is_valid()),
            "E1": sum(1 for r in results if r.e1 and r.e1.is_valid()),
            "E2": sum(1 for r in results if r.e2 and r.e2.is_valid()),
        },
        "llm_calls": llm_used_count if use_llm_fallback else 0,
        "avg_confidence": float(np.mean([r.confidence for r in results])),
    }
    
    # Optionally save log
    try:
        log_dir = Path("data_store/performance_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"perf_{start_dt.strftime('%Y%m%d_%H%M%S')}.json"
        log_file.write_text(json.dumps(perf_log, indent=2))
    except Exception:
        pass
# --------------------------------
# BỔ SUNG: Kiểm tra & tải Catalog
# --------------------------------
def _flatten_catalog_to_df(catalog_path: str) -> pd.DataFrame:
    """
    Đọc catalog/ata_catalog.json và làm phẳng để soát xét.
    Hỗ trợ nhiều schema và chuẩn hoá mã ATA04 về AA-BB.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Không tìm thấy {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        cat = json.load(f)

    def _to_str_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if x is not None]
        return [str(v)]

    def _pick_ata04(d, fallback_key=None):
        for k in ("ATA04", "ata04", "ATA_04", "ata_04", "code", "Code", "ata", "ATA"):
            if isinstance(d, dict) and k in d and d[k]:
                return str(d[k]).strip().upper()
        if fallback_key:
            return str(fallback_key).strip().upper()
        return None

    items = []
    if isinstance(cat, dict) and "entries" in cat:
        entries = cat["entries"]
        if isinstance(entries, dict):
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
        ata = _pick_ata04(cat)
        items.append((ata, cat if isinstance(cat, dict) else {}))

    rows = []
    for ata, it in items:
        titles   = _to_str_list(it.get("titles") or it.get("Titles") or it.get("title"))
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
    if "ATA04" in df.columns:
        df["ATA04"] = df["ATA04"].astype(str).str.upper().str.strip()
        df["ATA04"] = df["ATA04"].str.replace(r"[^0-9\-]", "", regex=True)

        def _fix(ata):
            if not isinstance(ata, str) or not ata:
                return ata
            if "-" in ata:
                parts = ata.split("-")
                if len(parts[0]) >= 2 and len(parts[1]) >= 2:
                    return f"{parts[0][:2]}-{parts[1][:2]}"
            if ata.isdigit() and len(ata) >= 4:
                return f"{ata[:2]}-{ata[2:4]}"
            return ata

        df["ATA04"] = df["ATA04"].map(_fix)
        df = df[df["ATA04"].notna() & (df["ATA04"].str.len() >= 5)]
        if not df.empty:
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

    csv_bytes = None
    try:
        df_flat = _flatten_catalog_to_df(json_path)
        if not df_flat.empty:
            csv_bytes = df_flat.to_csv(index=False).encode("utf-8")
    except Exception:
        csv_bytes = None

    buf = io.BytesIO()
    if not zip_name:
        zip_name = "catalog_bundle.zip"

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="catalog/ata_catalog.json")
        model_dir = os.path.join(cat_dir, "model")
        if os.path.isdir(model_dir):
            for path in glob.glob(os.path.join(model_dir, "**"), recursive=True):
                if os.path.isfile(path):
                    arc = os.path.relpath(path, ".")
                    zf.write(path, arcname=arc)
        for fname in ("ata_map.parquet", "wo_training.parquet"):
            p = os.path.join("data_store", fname)
            if os.path.exists(p):
                zf.write(p, arcname=f"data_store/{fname}")
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
            try:
                df_preview = _flatten_catalog_to_df(str(CAT_JSON))
                st.info(f"Bảng phẳng có {len(df_preview)} dòng.")
            except Exception as ex2:
                st.warning(f"Đã tạo gói nhưng không thể tạo catalog_flat.csv: {ex2}")
        except Exception as e:
            st.error(f"Không thể tạo gói Catalog: {e}")

# --------------------------------
# KIỂM TRA DỮ LIỆU ĐÃ ĐỒNG BỘ (AUDIT)
# --------------------------------
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
