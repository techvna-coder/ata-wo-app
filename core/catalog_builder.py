# core/catalog_builder.py
from __future__ import annotations
"""
Xây Catalog TF-IDF (offline) từ bộ nhớ:
- Input:
    data_store/wo_training.parquet  (các cột: text, ata04, hash)
    data_store/ata_map.parquet      (tuỳ chọn: ánh xạ ATA04 -> Title/Name/System)
- Output:
    catalog/ata_catalog.json
    catalog/model/tfidf_vectorizer.joblib
    catalog/model/tfidf_matrix.npz

Tính năng:
- Làm sạch văn bản (dùng core.cleaning.clean_wo_text nếu có).
- Cân bằng số mẫu mỗi lớp bằng max_docs_per_class.
- Trích top từ khoá đại diện theo TF-IDF trung bình.
- Lấy samples đại diện theo tổng trọng số TF-IDF theo hàng.
- Tuỳ chọn AI-enrich (chuẩn hoá title/keywords/samples) qua OpenAI.
"""

from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Đường dẫn kho dữ liệu/artefact
WO_PARQUET = Path("data_store/wo_training.parquet")
ATA_PARQUET = Path("data_store/ata_map.parquet")
OUT_JSON = Path("catalog/ata_catalog.json")
OUT_VEC = Path("catalog/model/tfidf_vectorizer.joblib")
OUT_MAT = Path("catalog/model/tfidf_matrix.npz")

# --- Tiện ích văn bản --------------------------------------------------------
def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    return " ".join(s.split())

def _ensure_out_dirs():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAT.parent.mkdir(parents=True, exist_ok=True)

# --- TF-IDF đại diện lớp -----------------------------------------------------
def _top_terms(matrix, inv_vocab: Dict[int, str], row_ids: List[int], top_k: int) -> List[str]:
    """Lấy top-k token theo TF-IDF trung bình trên một lớp."""
    if not row_ids:
        return []
    sub = matrix[row_ids]  # (n_i, V)
    mean_vec = np.asarray(sub.mean(axis=0)).ravel()  # (V,)
    top_idx = mean_vec.argsort()[::-1][:top_k]
    return [inv_vocab.get(i, "") for i in top_idx if i in inv_vocab]

def _rep_samples(df: pd.DataFrame, matrix, row_ids: List[int], text_col: str, sample_k: int) -> List[str]:
    """Chọn sample câu điển hình theo tổng TF-IDF mỗi hàng."""
    if not row_ids:
        return []
    sub = matrix[row_ids]  # (n_i, V)
    row_scores = np.asarray(sub.sum(axis=1)).ravel()
    ranked = [row_ids[i] for i in np.argsort(row_scores)[::-1]]
    out: List[str] = []
    for ridx in ranked:
        t = str(df.loc[ridx, text_col]).strip()
        if len(t) >= 40:
            out.append(t[:240])
        if len(out) >= sample_k:
            break
    i = 0
    while len(out) < sample_k and i < len(ranked):
        t = str(df.loc[ranked[i], text_col]).strip()[:240]
        if t and t not in out:
            out.append(t)
        i += 1
    return out[:sample_k]

# --- AI enrich (tuỳ chọn) ----------------------------------------------------
def _llm_enrich(ata04: str, samples: List[str], title_hint: str, top_k: int):
    """Gọi OpenAI nếu khả dụng; trả dict hoặc None."""
    try:
        from .openai_helpers import llm_enrich_catalog_entry
    except Exception:
        return None
    try:
        return llm_enrich_catalog_entry(ata04=ata04, samples=samples, title_hint=title_hint, top_k=top_k)
    except Exception:
        return None

# --- Hàm chính ----------------------------------------------------------------
def build_catalog_from_memory(
    min_docs_per_class: int = 3,
    top_k: int = 15,
    sample_k: int = 3,
    max_docs_per_class: int = 2000,
    random_state: int = 42,
    use_llm_enrich: bool = False,
) -> pd.DataFrame:
    """
    Xây Catalog TF-IDF từ dữ liệu training.
    Trả về DataFrame thống kê: [ATA04, Docs].
    """
    _ensure_out_dirs()

    if not WO_PARQUET.exists():
        raise FileNotFoundError(f"Không tìm thấy {WO_PARQUET}. Hãy đồng bộ & ingest WO trước.")

    df = pd.read_parquet(WO_PARQUET).copy()
    if not {"text", "ata04"}.issubset(df.columns):
        raise ValueError("wo_training.parquet thiếu cột 'text' hoặc 'ata04'.")

    # Làm sạch nhất quán với ingest/predict
    try:
        from .cleaning import clean_wo_text
        df["text_norm"] = df["text"].astype(str).map(clean_wo_text).map(_normalize_text)
    except Exception:
        df["text_norm"] = df["text"].astype(str).map(_normalize_text)

    # Cân bằng số mẫu mỗi lớp để giảm lệch lớp
    if isinstance(max_docs_per_class, int) and max_docs_per_class > 0:
        dfs = []
        for ata, grp in df.groupby("ata04", sort=False):
            if len(grp) > max_docs_per_class:
                dfs.append(grp.sample(n=max_docs_per_class, random_state=random_state))
            else:
                dfs.append(grp)
        df = pd.concat(dfs, ignore_index=True)

    # Danh sách lớp
    classes = sorted(df["ata04"].dropna().unique().tolist())

    # TF-IDF toàn bộ câu để trích đặc trưng cho từng lớp
    vec_all = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000)
    X_all = vec_all.fit_transform(df["text_norm"].tolist())
    inv_vocab = {v: k for k, v in vec_all.vocabulary_.items()}

    # Ánh xạ tiêu đề từ ata_map (nếu có)
    title_map: Dict[str, str] = {}
    if ATA_PARQUET.exists():
        try:
            amap = pd.read_parquet(ATA_PARQUET)
            # ưu tiên cột tên hệ thống
            cand = [c for c in amap.columns if str(c).lower() in {"title", "name", "system", "description"}]
            tcol = cand[0] if cand else None
            if "ATA04" in amap.columns and tcol:
                title_map = {str(r["ATA04"]): str(r[tcol] or "") for _, r in amap.iterrows()}
        except Exception:
            title_map = {}

    # Gom chỉ số hàng theo lớp
    idx_by_cls = {ata: list(df.index[df["ata04"] == ata].values) for ata in classes}

    # Xây catalog từng lớp
    catalog: Dict[str, Dict[str, List[str]]] = {}
    ata_list: List[str] = []
    for ata in classes:
        row_ids = idx_by_cls[ata]

        # Keywords & Samples từ TF-IDF
        if len(row_ids) >= 1:
            kws = _top_terms(X_all, inv_vocab, row_ids, top_k)
            sps = _rep_samples(df, X_all, row_ids, "text_norm", sample_k)
        else:
            kws, sps = [], []

        # Title từ ata_map (nếu có)
        title = title_map.get(ata, "")

        # AI enrich (tuỳ chọn)
        if use_llm_enrich:
            enriched = _llm_enrich(ata04=ata, samples=sps, title_hint=title, top_k=top_k)
            if enriched:
                # Ưu tiên nội dung do LLM chuẩn hoá; fallback TF-IDF nếu thiếu
                title = enriched.get("title") or title
                llm_kws = enriched.get("keywords") or []
                llm_sps = enriched.get("samples") or []
                if llm_kws:
                    kws = llm_kws[:top_k]
                if llm_sps:
                    sps = llm_sps[:sample_k]

        # Nếu số mẫu quá ít dưới min_docs_per_class, vẫn ghi lớp (để không mất lớp),
        # nhưng hiểu rằng độ tin cậy thấp – downstream sẽ xử lý.
        catalog[ata] = {"title": title, "keywords": kws, "samples": sps}
        ata_list.append(ata)

    # Biên soạn tài liệu đại diện cho từng lớp để huấn luyện vectorizer catalog
    docs: List[str] = []
    for ata in ata_list:
        info = catalog[ata]
        doc = " ".join([
            info.get("title", "") or "",
            " ".join(info.get("keywords", []) or []),
            " ".join(info.get("samples", []) or []),
        ]).strip()
        docs.append(doc if doc else ata)

    # Vectorizer cho catalog (nhẹ, phục vụ truy vấn)
    vec_cat = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X_cat = vec_cat.fit_transform(docs)  # shape: (n_classes, V_cat)

    # Ghi artefacts
    OUT_JSON.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(vec_cat, OUT_VEC)
    save_npz(OUT_MAT, X_cat)

    # Thống kê trả về
    stat = pd.DataFrame({"ATA04": ata_list, "Docs": [len(idx_by_cls[a]) for a in ata_list]})
    return stat.sort_values(["Docs", "ATA04"], ascending=[False, True]).reset_index(drop=True)
