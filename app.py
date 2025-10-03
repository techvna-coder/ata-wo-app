import os
import streamlit as st
import pandas as pd
from pathlib import Path
import re

from core.gdrive_sync import sync_drive_folder
from core.store import init_db, append_wo_training, append_ata_map
from core.catalog_builder import build_catalog_from_memory
from core.io_excel import load_wo_excel, write_result
from core.nondefect import is_technical_defect
from core.refs import extract_citations
from core.ata_catalog import ATACatalog
from core.decision import decide
from core.mapping import ORDER_OUTCOLS

st.set_page_config(page_title="WO → ATA04 Checker (Drive + Memory)", layout="wide")
st.title("WO → ATA04 Checker (Drive + Incremental Memory)")
init_db()

with st.sidebar:
    st.header("Kết nối Google Drive")
    folder_id = st.text_input("Drive Folder ID", value="", help="URL dạng drive.google.com/drive/folders/<FOLDER_ID>")
    sa_json = st.file_uploader("Service Account JSON", type=["json"], help="Upload khoá service account")
    do_sync = st.button("Đồng bộ từ Drive (incremental)")
    st.markdown("---")
    do_rebuild = st.button("Rebuild Catalog từ bộ nhớ")
    use_catalog = st.checkbox("Dùng Catalog (TF-IDF)", value=True)
    show_debug = st.checkbox("Hiển thị debug", value=False)

if do_sync:
    if not folder_id or not sa_json:
        st.error("Thiếu Folder ID hoặc Service Account JSON.")
    else:
        sa_path = "data_store/sa.json"
        Path("data_store").mkdir(parents=True, exist_ok=True)
        with open(sa_path, "wb") as f:
            f.write(sa_json.read())
        changed = sync_drive_folder(folder_id, sa_path)
        st.success(f"Đồng bộ xong. {len(changed)} file mới/cập nhật.")

        # Ingest về bộ nhớ
        inj_dir = Path("data_store/ingest")
        files = sorted([p for p in inj_dir.glob("*.xls*")])
        n_map, n_wo = 0, 0

        for p in files:
            try:
                df = pd.read_excel(p, dtype=str)
                cols = [c.lower() for c in df.columns]
                is_ata_map = any(re.search(r"ata.*0?4|^ata$|code", c) for c in cols) and any(re.search(r"name|title|system|mô tả|description", c) for c in cols)
                is_wo = any(re.search(r"w/?o.*desc|description|defect", c) for c in cols) and any(re.search(r"w/?o.*action|rectification|action", c) for c in cols)

                if is_ata_map:
                    code_col = [c for c in df.columns if re.search(r"ata.*0?4|^ata$|code", c, flags=re.I)]
                    name_col = [c for c in df.columns if re.search(r"name|title|system|mô tả|description", c, flags=re.I)]
                    if code_col and name_col:
                        append_ata_map(df, code_col[0], name_col[0])
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
                    ata_final_col = find_col([r"\bATA\s*0?4\s*Corrected\b", r"\bATA\s*final\b", r"\bATA04_Final\b"])
                    ata_entered_col = find_col([r"^ATA$", r"\bATA\s*0?4\b", r"\bATA04_Entered\b"])

                    if desc_col and act_col and (ata_final_col or ata_entered_col):
                        append_wo_training(df, desc_col, act_col, ata_final_col or "", ata_entered_col or "", p.name)
                        n_wo += 1
            except Exception as e:
                st.warning(f"Lỗi đọc {p.name}: {e}")

        st.info(f"Đã cập nhật bộ nhớ: {n_map} file ATA map, {n_wo} file WO.")

if do_rebuild:
    try:
        stat = build_catalog_from_memory()
        st.success(f"Đã build Catalog từ bộ nhớ ({len(stat)} lớp).")
        st.dataframe(stat.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi build catalog: {e}")

st.header("Xử lý WO mới")
uploaded = st.file_uploader("Upload Excel WO cần suy luận ATA corrected", type=["xlsx","xls"])
if uploaded is not None:
    df = load_wo_excel(uploaded)
    st.success(f"Đã nạp {len(df)} dòng.")
    if use_catalog:
        try:
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

        is_tech = is_technical_defect(defect, action)

        citations = extract_citations(f"{defect or ''} {action or ''}")
        if citations and citations_example is None:
            citations_example = citations
        e1_valid, e1_ata, cited_manual, cited_task = False, None, None, None
        for c in citations:
            if c.get("ata04"):
                e1_valid = True
                e1_ata = c["ata04"]
                cited_manual = c["manual"]
                cited_task = c["task"]
                break

        e2_best = None
        derived_task = derived_doc = derived_score = evidence_snip = evidence_src = None
        if use_catalog and is_tech:
            e2_best, _ = catalog.predict(defect, action)
            if e2_best:
                derived_task = e2_best.get("ata04")
                derived_doc  = e2_best.get("doc")
                derived_score= e2_best.get("score")
                evidence_snip= e2_best.get("snippet")
                evidence_src = e2_best.get("source")

        decision, conf, reason = decide(
            e0=(e0 if isinstance(e0,str) and len(e0)>=5 else None),
            e1_valid=(is_tech and e1_valid),
            e1_ata=e1_ata,
            e2_best=e2_best,
            e2_all=None
        )
        ata_final = (e1_ata or derived_task or e0) if decision in ("CONFIRM","CORRECT") else (derived_task or e1_ata or e0)

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
