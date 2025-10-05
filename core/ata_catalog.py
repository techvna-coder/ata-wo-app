# core/ata_catalog.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import load
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import TOP_K_TFIDF, MIN_SCORE_CONFIRM
from .cleaning import clean_wo_text


class ATACatalog:
    """
    Bộ phân loại ATA04 dựa trên TF-IDF Catalog (offline).

    Cấu trúc file:
      - catalog/ata_catalog.json:
          {
            "AA-BB": {"title": "...", "keywords": [...], "samples": [...]},
            "27-10": {...},
            ...
          }
      - catalog/model/tfidf_vectorizer.joblib: vectorizer cho tài liệu catalog
      - catalog/model/tfidf_matrix.npz: ma trận TF-IDF (shape: n_classes x vocab)

    APIs chính:
      - predict(defect_text, rect_text, top_k=None) -> (best_result, top_results)
      - topk_candidates(defect_text, rect_text, k=5) -> List[result]
      - name_map() -> Dict[ata04, system_name] (đọc từ data_store/ata_map.parquet nếu có)
    """

    def __init__(self, catalog_dir: str = "catalog") -> None:
        self.catalog_dir = catalog_dir
        self.catalog_path = os.path.join(catalog_dir, "ata_catalog.json")
        self.vec_path = os.path.join(catalog_dir, "model", "tfidf_vectorizer.joblib")
        self.mat_path = os.path.join(catalog_dir, "model", "tfidf_matrix.npz")

        # Nạp catalog JSON (dict: ata04 -> info)
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog: Dict[str, Dict[str, Any]] = json.load(f)

        # Danh sách lớp theo cùng thứ tự khi build ma trận TF-IDF
        # (phải khớp thứ tự khi sinh ma trận trong catalog_builder)
        self.ata_list: List[str] = list(self.catalog.keys())

        # Nạp vectorizer + ma trận TF-IDF (n_classes x vocab)
        self.vectorizer: TfidfVectorizer = load(self.vec_path)
        self.tfidf = load_npz(self.mat_path)  # shape: (n_classes, vocab)

        # Kiểm tra đồng nhất kích thước
        if self.tfidf.shape[0] != len(self.ata_list):
            raise ValueError(
                f"TF-IDF matrix rows ({self.tfidf.shape[0]}) "
                f"không khớp số lớp ATA ({len(self.ata_list)})."
            )

        # Bảng tên hệ thống (có thể trống nếu chưa có ata_map.parquet)
        self._name_map: Dict[str, str] = self._load_name_map()

    # ------------------------------
    # Tiện ích nội bộ
    # ------------------------------
    @staticmethod
    def _compose_doc(text: str) -> str:
        """Chuẩn hoá nhẹ chuỗi đầu vào (trim)."""
        return (text or "").strip()

    def _format_result(self, idx: int, score: float) -> Dict[str, Any]:
        """
        Định dạng một kết quả dự đoán theo ATA04.
        - snippet ưu tiên: title -> keywords[0] -> samples[0]
        """
        ata = self.ata_list[idx]
        info = self.catalog.get(ata, {}) or {}
        title = info.get("title") or info.get("Titles")  # phòng khi có schema khác
        keywords = info.get("keywords") or info.get("Keywords") or []
        samples = info.get("samples") or info.get("Samples") or []

        snippet = title
        if not snippet and isinstance(keywords, list) and keywords:
            snippet = keywords[0]
        if not snippet and isinstance(samples, list) and samples:
            snippet = samples[0]

        return {
            "ata04": ata,
            "score": float(score),
            "doc": "CATALOG",
            "snippet": snippet or "",
            "source": "catalog/ata_catalog.json",
        }

    def _embed_query(self, defect_text: Optional[str], rect_text: Optional[str]) -> Optional[np.ndarray]:
        """
        Làm sạch + ghép Defect/Rectification rồi vector hoá bằng TF-IDF.
        Trả về vector shape (1, vocab) hoặc None nếu rỗng.
        """
        q_def = clean_wo_text(defect_text or "")
        q_rect = clean_wo_text(rect_text or "")
        q = self._compose_doc(f"{q_def}\n{q_rect}")
        if not q:
            return None
        return self.vectorizer.transform([q])  # (1, vocab)

    def _load_name_map(self) -> Dict[str, str]:
        """
        Đọc data_store/ata_map.parquet nếu có để lấy tên hệ thống.
        Tìm cột mã ~ ('ATA04'|'ATA'|'Code') và cột tên ~ ('Name'|'Title'|'System'|'Description'|'Mô tả').
        """
        path = os.path.join("data_store", "ata_map.parquet")
        if not os.path.exists(path):
            return {}
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            # Tìm code col
            code_col = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("ata04", "ata", "code") or "ata 04" in cl or "ata_04" in cl:
                    code_col = c
                    break
            # Tìm name col
            name_col = None
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in ("name", "title", "system", "desc", "mô tả", "mo ta")):
                    name_col = c
                    break
            if code_col and name_col:
                tmp = df[[code_col, name_col]].dropna()
                return {
                    str(row[code_col]).strip().upper(): str(row[name_col]).strip()
                    for _, row in tmp.iterrows()
                }
        except Exception:
            pass
        return {}

    # ------------------------------
    # Dự đoán / Ứng viên
    # ------------------------------
    def topk_candidates(
        self,
        defect_text: Optional[str],
        rect_text: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Trả về danh sách ứng viên Top-K (mặc định TOP_K_TFIDF).
        Mỗi item: {"ata04","score","doc","snippet","source"}
        """
        kk = k or TOP_K_TFIDF
        qv = self._embed_query(defect_text, rect_text)
        if qv is None:
            return []

        # Cosine approx bằng tích ma trận (qv: 1 x V) @ (tfidf: n x V).T = 1 x n
        scores = (qv @ self.tfidf.T).toarray()[0]  # (n_classes,)
        if scores.size == 0:
            return []

        top_idx = np.argsort(scores)[::-1][:kk]
        return [self._format_result(int(i), float(scores[int(i)])) for i in top_idx]

    def predict(
        self,
        defect_text: Optional[str],
        rect_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Suy luận ATA04 từ mô tả/hành động.
        - Làm sạch văn bản để loại bỏ meta/audit (WORKSTEP..., NRC...) trước khi vector hoá.
        - Trả về (best_result, top_results).
        - Không áp ngưỡng cứng ở đây; để lớp quyết định/tam-đối-soát xử lý.
        """
        kk = top_k or TOP_K_TFIDF
        candidates = self.topk_candidates(defect_text, rect_text, k=kk)
        best = candidates[0] if candidates else None

        # Nếu muốn áp ngưỡng niềm tin tối thiểu cho best:
        if best and best["score"] < float(MIN_SCORE_CONFIRM):
            # vẫn trả best cho downstream quyết định; ngưỡng dùng ở lớp quyết định
            pass

        return best, candidates

    # ------------------------------
    # Bảng tên hệ thống (cho LLM arbitration / UI)
    # ------------------------------
    def name_map(self) -> Dict[str, str]:
        """
        Trả về ánh xạ ATA04 -> system name (nếu có).
        """
        return dict(self._name_map)
