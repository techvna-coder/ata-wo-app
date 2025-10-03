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
    - catalog/ata_catalog.json: { "AA-BB": {"title": "...", "keywords": [...], "samples": [...]}, ... }
    - catalog/model/tfidf_vectorizer.joblib: vectorizer cho tài liệu catalog
    - catalog/model/tfidf_matrix.npz: ma trận TF-IDF (docs của từng ATA04)

    Gợi ý pipeline xây:
      1) Ingest WO lịch sử -> data_store/wo_training.parquet
      2) build_catalog_from_memory() -> sinh file JSON + model TF-IDF
      3) Nạp ATACatalog và gọi predict(defect, rectification)
    """

    def __init__(self, catalog_dir: str = "catalog") -> None:
        self.catalog_dir = catalog_dir
        self.catalog_path = os.path.join(catalog_dir, "ata_catalog.json")
        self.vec_path = os.path.join(catalog_dir, "model", "tfidf_vectorizer.joblib")
        self.mat_path = os.path.join(catalog_dir, "model", "tfidf_matrix.npz")

        # Nạp catalog JSON
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog: Dict[str, Dict[str, Any]] = json.load(f)

        # Danh sách lớp theo cùng thứ tự khi build ma trận TF-IDF
        self.ata_list: List[str] = list(self.catalog.keys())

        # Nạp vectorizer + ma trận TF-IDF
        self.vectorizer: TfidfVectorizer = load(self.vec_path)
        self.tfidf = load_npz(self.mat_path)

        # Kiểm tra đồng nhất kích thước
        if self.tfidf.shape[0] != len(self.ata_list):
            raise ValueError(
                f"TF-IDF matrix rows ({self.tfidf.shape[0]}) "
                f"không khớp số lớp ATA ({len(self.ata_list)})."
            )

    # ------------------------------
    # Tiện ích nội bộ
    # ------------------------------
    @staticmethod
    def _compose_doc(text: str) -> str:
        """Chuẩn hoá nhẹ chuỗi đầu vào (trim)."""
        return (text or "").strip()

    def _format_result(self, idx: int, score: float) -> Dict[str, Any]:
        """Định dạng một kết quả dự đoán theo ATA04."""
        ata = self.ata_list[idx]
        info = self.catalog.get(ata, {})
        snippet = (
            info.get("title")
            or (info.get("keywords") or [""])[0]
            or (info.get("samples") or [""])[0]
        )
        return {
            "ata04": ata,
            "score": float(score),
            "doc": "CATALOG",
            "snippet": snippet,
            "source": "catalog/ata_catalog.json",
        }

    # ------------------------------
    # Dự đoán
    # ------------------------------
    def predict(
        self,
        defect_text: Optional[str],
        rect_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Suy luận ATA04 từ mô tả/rectification.
        - Làm sạch văn bản để loại bỏ meta/audit (WORKSTEP..., NRC...) trước khi vector hoá.
        - Trả về (best_result, top_results).
        """
        k = top_k or TOP_K_TFIDF

        # Làm sạch đầu vào
        q_def = clean_wo_text(defect_text or "")
        q_rect = clean_wo_text(rect_text or "")
        q = self._compose_doc(f"{q_def}\n{q_rect}")

        if not q:
            return None, None

        # Vector hoá query
        qv = self.vectorizer.transform([q])  # shape: (1, vocab)
        # Tính điểm cosine xấp xỉ bằng tích ma trận (qv * tfidf.T)
        # tfidf shape: (n_classes, vocab) -> cần transpose
        scores = (qv @ self.tfidf.T).toarray()[0]  # shape: (n_classes,)

        # Lấy top-k chỉ mục theo điểm giảm dần
        top_idx = np.argsort(scores)[::-1][:k]

        # Biên dịch danh sách top-k kết quả
        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            results.append(self._format_result(idx, scores[idx]))

        best = results[0] if results else None

        # Nếu muốn áp ngưỡng niềm tin tối thiểu cho best:
        if best and best["score"] < float(MIN_SCORE_CONFIRM):
            # vẫn trả best cho downstream quyết định; ngưỡng sẽ dùng ở lớp quyết định
            pass

        return best, results
