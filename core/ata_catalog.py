# core/ata_catalog.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import load
from scipy.sparse import load_npz, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import TOP_K_TFIDF, MIN_SCORE_CONFIRM
from .cleaning import clean_wo_text


class ATACatalog:
    def __init__(self, catalog_dir: str = "catalog") -> None:
        self.catalog_dir = catalog_dir
        self.catalog_path = os.path.join(catalog_dir, "ata_catalog.json")
        self.vec_path = os.path.join(catalog_dir, "model", "tfidf_vectorizer.joblib")
        self.mat_path = os.path.join(catalog_dir, "model", "tfidf_matrix.npz")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog: Dict[str, Dict[str, Any]] = json.load(f)

        self.ata_list: List[str] = list(self.catalog.keys())

        self.vectorizer: TfidfVectorizer = load(self.vec_path)
        self.tfidf: csr_matrix = load_npz(self.mat_path)  # shape: (n_classes, vocab)

        if self.tfidf.shape[0] != len(self.ata_list):
            raise ValueError(
                f"TF-IDF matrix rows ({self.tfidf.shape[0]}) "
                f"không khớp số lớp ATA ({len(self.ata_list)})."
            )

    @staticmethod
    def _compose_doc(text: str) -> str:
        return (text or "").strip()

    def _format_result(self, idx: int, score: float) -> Dict[str, Any]:
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

    # ======== API đơn lẻ (giữ tương thích) ========
    def predict(
        self,
        defect_text: Optional[str],
        rect_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        k = top_k or TOP_K_TFIDF
        q_def = clean_wo_text(defect_text or "")
        q_rect = clean_wo_text(rect_text or "")
        q = self._compose_doc(f"{q_def}\n{q_rect}")
        if not q:
            return None, None
        qv = self.vectorizer.transform([q])          # (1, vocab)
        scores = (qv @ self.tfidf.T).toarray()[0]    # (n_classes,)
        top_idx = np.argsort(scores)[::-1][:k]
        results = [self._format_result(i, scores[i]) for i in top_idx]
        best = results[0] if results else None
        return best, results

    # ======== API theo lô (cực nhanh) ========
    def predict_batch(
        self,
        pairs: List[Tuple[Optional[str], Optional[str]]],
        top_k: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[List[Optional[Dict[str, Any]]], Optional[List[List[Dict[str, Any]]]]]:
        """
        pairs: [(defect, action), ...]
        Trả về:
          - best_list: list[Optional[dict]] độ dài = len(pairs)
          - all_list : nếu return_all=True → list[list[dict]] top-k cho từng dòng
        """
        k = top_k or TOP_K_TFIDF
        # 1) Làm sạch & ghép query (vector hoá 1 lần)
        texts: List[str] = []
        for d, a in pairs:
            q_def = clean_wo_text(d or "")
            q_act = clean_wo_text(a or "")
            texts.append(self._compose_doc(f"{q_def}\n{q_act}"))
        # Nếu tất cả rỗng
        if not any(texts):
            return [None] * len(texts), ([] if return_all else None)

        Q = self.vectorizer.transform(texts)      # (N, vocab) – 1 lần
        S = (Q @ self.tfidf.T).toarray()          # (N, n_classes)

        # 2) Lấy top-k bằng argpartition (nhanh hơn sort toàn bộ)
        if k >= S.shape[1]:
            topk_idx = np.argsort(S, axis=1)[:, ::-1]
        else:
            part = np.argpartition(S, -k, axis=1)[:, -k:]  # (N, k) chỉ số không sắp xếp
            # sắp xếp k phần tử theo điểm giảm dần
            row_arange = np.arange(S.shape[0])[:, None]
            part_sorted = part[row_arange, np.argsort(S[row_arange, part], axis=1)[:, ::-1]]
            topk_idx = part_sorted  # (N, k)

        best_list: List[Optional[Dict[str, Any]]] = []
        all_list: List[List[Dict[str, Any]]] = []

        for r, idxs in enumerate(topk_idx):
            row = []
            for j in idxs:
                row.append(self._format_result(int(j), float(S[r, j])))
            all_list.append(row)
            best_list.append(row[0] if row else None)

        return best_list, (all_list if return_all else None)
