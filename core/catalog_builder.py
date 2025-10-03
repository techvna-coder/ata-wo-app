# core/catalog_builder.py
from __future__ import annotations

"""
Xây Catalog TF-IDF (offline) từ bộ nhớ huấn luyện:
- Input:
    data_store/wo_training.parquet  (các cột: text, ata04, hash)
    data_store/ata_map.parquet      (tuỳ chọn: ánh xạ ATA04 -> Title)
- Output:
    catalog/ata_catalog.json
    catalog/model/tfidf_vectorizer.joblib
    catalog/model/tfidf_matrix.npz

Gợi ý quy trình:
1) Đồng bộ Drive (ingest) -> tạo/ghi data_store/*.parquet
2) Gọi build_catalog_from_memory() để sinh catalog + mô hình
3) ATACatalog (core/ata_catalog.py) nạp và dự đoán
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Đường dẫn kho mặc định
WO_PARQUET = Path("data_store/wo_training.parquet")
ATA_PARQUET = Path("data_store/ata_map.parquet")

OUT_JSON = Path("catalog/ata_catalog.json")
OUT_VEC = Path("catalog/model/tfidf_vectorizer.joblib")
OUT_MAT = Path("catalog/model/tfidf_matrix.npz")


# -----------------------------
# Tiện ích xử lý văn bản
# -----------------------------
def _normalize_text(s: str) -> str:
    """Chuẩn hoá nhẹ (lower + rút gọn khoảng trắng). ĐÃ có clean_wo_text trước đó."""
    s = (s or "").lower()
    return " ".join(s.split())


def _ensure_out_dirs():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAT.parent.mkdir(parents=True, exist_ok=True)


def _top_terms(matrix, inv_vocab: Dict[int, str], row_ids: List[int], top_k: int) -> List[str]:
    """
    Lấy top-k cụm từ có trọng số TF-IDF trung bình cao nhất trong một lớp.
    - matrix: ma trận TF-IDF (toàn bộ tập), shape (N_samples, V)
    - inv_vocab: map chỉ số -> token
    - row_ids: các chỉ số hàng thuộc lớp
    """
    if not row_ids:
        return []
    sub = matrix[row_ids]  # (n_i, V)
    mean_vec = np.asarray(sub.mean(axis=0)).ravel()  # (V,)
    top_idx = mean_vec.argsort()[::-1][:top_k]
    return [inv_vocab[i] for i in top_idx if i in inv_vocab]


def _rep_samples(df: pd.DataFrame, matrix, row_ids: List[int], text_col: str, sample_k: int) -> List[str]:
    """
    Lấy một vài câu đại diện (samples) cho lớp dựa trên tổng trọng số TF-IDF theo hàng.
    """
    if not row_ids:
        return []
    sub = matrix[row_ids]                   # (n_i, V)
    row_scores = np.asarray(sub.sum(axis=1)).ravel()  # (n_i,)
    ranked = [row_ids[i] for i in np.argsort(row_scores)[::-1]]
    out: List[str] = []
    for ridx in ranked:
        t = str(df.loc[ridx, text_col]).strip()
        if len(t) >= 40:  # ưu tiên câu đủ dài để có ngữ cảnh
            out.append(t[:240])
        if len(out) >= sample_k:
            break
    # nếu chưa đủ, bổ sung từ đầu danh sách
    i = 0
    while len(out) < sample_k and i < len(ranked):
        t = str(df.loc[ranked[i], text_col]).strip()[:240]
        if t not in out:
            out.append(t)
        i += 1
    return out[:sample_k]


# -----------------------------
# Hàm chính: Build Catalog
# -----------------------------
def build_catalog_from_memory(
    min_docs_per_class: int = 3,
    top_k: int = 15,
    sample_k: int = 3,
    max_docs_per_class: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Xây Catalog TF-IDF từ data_store/wo_training.parquet và (tuỳ chọn) ata_map.parquet.

    Tham số:
    - min_docs_per_class: số mẫu tối thiểu để một lớp được xem là “đủ dữ liệu”
    - top_k: số từ khoá đặc trưng lấy cho mỗi lớp
    - sample_k: số câu đại diện minh hoạ mỗi lớp
    - max_docs_per_class: giới hạn mẫu tối đa mỗi lớp để tránh lệch dữ liệu
    - random_state: seed cho lấy mẫu ngẫu nhiên

    Trả về: DataFrame thống kê (ATA04, Docs)
    """
    _ensure_out_dirs()

    if not WO_PARQUET.exists():
        raise FileNotFoundError(f"Không tìm thấy {WO_PARQUET}. Hãy đồng bộ & ingest WO trước.")

    # Đọc training
    df = pd.read_parquet(WO_PARQUET).copy()
    if not {"text", "ata04"}.issubset(df.columns):
        raise ValueError("wo_training.parquet thiếu cột 'text' hoặc 'ata04'.")

    # Làm sạch nhất quán với ingest/predict
    try:
        from .cleaning import clean_wo_text
        df["text_norm"] = df["text"].astype(str).map(clean_wo_text).map(_normalize_text)
    except Exception:
        # Dự phòng: nếu module cleaning chưa có
        df["text_norm"] = df["text"].astype(str).map(_normalize_text)

    # Cân bằng số mẫu mỗi lớp để tránh overfit lệch lớp
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

    # Vector hoá toàn bộ câu (để tính top terms/samples nội bộ)
    vec_all = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000)
    X_all = vec_all.fit_transform(df["text_norm"].tolist())  # shape: (N, V)
    inv_vocab = {v: k for k, v in vec_all.vocabulary_.items()}

    # Chuẩn bị map tiêu đề ATA (nếu có)
    title_map: Dict[str, str] = {}
    if ATA_PARQUET.exists():
        try:
            amap = pd.read_parquet(ATA_PARQUET)
            # kỳ vọng có cột ATA04 và Title (hoặc tương đương)
            tcol = None
            for c in amap.columns:
                if str(c).lower() in {"title", "name", "system", "description"}:
                    tcol = c
                    break
            if "ATA04" in amap.columns and tcol:
                title_map = {str(r["ATA04"]): str(r[tcol] or "") for _, r in amap.iterrows()}
        except Exception:
            # nếu lỗi đọc/thiếu cột, để title_map rỗng
            title_map = {}

    # Gom chỉ số hàng theo lớp
    idx_by_cls = {ata: list(df.index[df["ata04"] == ata].values) for ata in classes}

    # Tạo catalog JSON (mỗi ATA04 gồm: title, keywords, samples)
    catalog: Dict[str, Dict[str, List[str]]] = {}
    ata_list: List[str] = []
    for ata in classes:
        row_ids = idx_by_cls[ata]
        if len(row_ids) < min_docs_per_class:
            # vẫn sinh thông tin ở mức tối thiểu nếu có dữ liệu
            kws = _top_terms(X_all, inv_vocab, row_ids, top_k) if row_ids else []
            sps = _rep_samples(df, X_all, row_ids, "text_norm", sample_k) if row_ids else []
        else:
            kws = _top_terms(X_all, inv_vocab, row_ids, top_k)
            sps = _rep_samples(df, X_all, row_ids, "text_norm", sample_k)

        title = title_map.get(ata, "")
        catalog[ata] = {"title": title, "keywords": kws, "samples": sps}
        ata_list.append(ata)

    # Biên soạn “tài liệu đại diện” cho từng lớp để train vectorizer catalog
    docs: List[str] = []
    for ata in ata_list:
        info = catalog[ata]
        doc = " ".join(
            [
                info.get("title", "") or "",
                " ".join(info.get("keywords", []) or []),
                " ".join(info.get("samples", []) or []),
            ]
        ).strip()
        docs.append(doc if doc else ata)  # dự phòng: nếu rỗng thì dùng nhãn

    # Vectorizer cho catalog (nhẹ hơn so với vec_all)
    vec_cat = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X_cat = vec_cat.fit_transform(docs)  # shape: (n_classes, V_cat)

    # Ghi artefacts
    OUT_JSON.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(vec_cat, OUT_VEC)
    save_npz(OUT_MAT, X_cat)

    # Thống kê trả về
    stat = pd.DataFrame({"ATA04": ata_list, "Docs": [len(idx_by_cls[a]) for a in ata_list]})
    return stat.sort_values(["Docs", "ATA04"], ascending=[False, True]).reset_index(drop=True)
