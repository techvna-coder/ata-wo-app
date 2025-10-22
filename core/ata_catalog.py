# core/ata_catalog_optimized.py
"""
Optimized ATA Catalog with:
- Sparse matrix operations (no unnecessary toarray())
- Vectorized batch processing
- Memory-efficient top-K retrieval
- Caching for repeated queries
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
from joblib import load
from scipy.sparse import load_npz, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import TOP_K_TFIDF, MIN_SCORE_CONFIRM
from .cleaning import clean_wo_text


class ATACatalogOptimized:
    """
    Optimized version with:
    1. Sparse operations throughout
    2. Efficient top-K using argpartition
    3. LRU cache for repeated queries
    4. Batch processing with minimal memory footprint
    """
    
    def __init__(self, catalog_dir: str = "catalog") -> None:
        self.catalog_dir = catalog_dir
        self.catalog_path = os.path.join(catalog_dir, "ata_catalog.json")
        self.vec_path = os.path.join(catalog_dir, "model", "tfidf_vectorizer.joblib")
        self.mat_path = os.path.join(catalog_dir, "model", "tfidf_matrix.npz")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog: Dict[str, Dict[str, Any]] = json.load(f)

        self.ata_list: List[str] = list(self.catalog.keys())
        self.vectorizer: TfidfVectorizer = load(self.vec_path)
        self.tfidf: csr_matrix = load_npz(self.mat_path)  # Keep sparse!

        if self.tfidf.shape[0] != len(self.ata_list):
            raise ValueError(
                f"TF-IDF matrix rows ({self.tfidf.shape[0]}) "
                f"không khớp số lớp ATA ({len(self.ata_list)})."
            )
        
        # Precompute norms for normalization (if needed)
        self._class_norms = None
    
    @staticmethod
    def _compose_doc(text: str) -> str:
        return (text or "").strip()
    
    def _format_result(self, idx: int, score: float) -> Dict[str, Any]:
        """Format single result."""
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
    
    @lru_cache(maxsize=1000)
    def _vectorize_single(self, text: str) -> tuple:
        """
        Cache vectorized queries (LRU for repeated queries).
        Returns tuple of (data, indices, indptr) for sparse reconstruction.
        """
        q_clean = clean_wo_text(text)
        q_vec = self.vectorizer.transform([q_clean])
        
        # Convert to tuple for caching (csr_matrix not hashable)
        return (
            tuple(q_vec.data),
            tuple(q_vec.indices),
            tuple(q_vec.indptr)
        )
    
    def _reconstruct_sparse(self, cached_data: tuple) -> csr_matrix:
        """Reconstruct sparse matrix from cached tuple."""
        data, indices, indptr = cached_data
        return csr_matrix(
            (np.array(data), np.array(indices), np.array(indptr)),
            shape=(1, len(self.vectorizer.vocabulary_))
        )

    # ======== OPTIMIZED SINGLE PREDICTION ========
    def predict(
        self,
        defect_text: Optional[str],
        rect_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Single prediction with caching support.
        """
        k = top_k or TOP_K_TFIDF
        q_def = clean_wo_text(defect_text or "")
        q_rect = clean_wo_text(rect_text or "")
        q = self._compose_doc(f"{q_def}\n{q_rect}")
        
        if not q:
            return None, None
        
        # Try cache first
        try:
            cached = self._vectorize_single(q)
            qv = self._reconstruct_sparse(cached)
        except:
            # Fallback if caching fails
            qv = self.vectorizer.transform([q])
        
        # ✅ KEEP SPARSE: dot product with sparse matrices
        scores_sparse = qv @ self.tfidf.T  # (1, n_classes) sparse
        scores = np.asarray(scores_sparse.todense()).ravel()  # Only convert final result
        
        # Efficient top-K
        if k >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n) vs argsort O(n log n)
            part_idx = np.argpartition(scores, -k)[-k:]
            top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
        
        results = [self._format_result(int(i), float(scores[i])) for i in top_idx]
        best = results[0] if results else None
        return best, results

    # ======== HIGHLY OPTIMIZED BATCH PREDICTION ========
    def predict_batch(
        self,
        pairs: List[Tuple[Optional[str], Optional[str]]],
        top_k: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[List[Optional[Dict[str, Any]]], Optional[List[List[Dict[str, Any]]]]]:
        """
        Batch prediction with:
        1. Single vectorization pass
        2. Sparse matrix operations throughout
        3. Efficient top-K extraction
        4. Minimal memory footprint
        """
        k = top_k or TOP_K_TFIDF
        
        # 1) Prepare texts (clean & compose)
        texts: List[str] = []
        for d, a in pairs:
            q_def = clean_wo_text(d or "")
            q_act = clean_wo_text(a or "")
            texts.append(self._compose_doc(f"{q_def}\n{q_act}"))
        
        # Handle empty case
        if not any(texts):
            return [None] * len(texts), ([] if return_all else None)
        
        # 2) Vectorize once
        Q = self.vectorizer.transform(texts)  # (N, vocab) sparse
        
        # 3) ✅ KEEP SPARSE: Compute similarities without converting to dense
        S_sparse = Q @ self.tfidf.T  # (N, n_classes) sparse
        
        # 4) Extract top-K efficiently PER ROW using sparse operations
        best_list: List[Optional[Dict[str, Any]]] = []
        all_list: List[List[Dict[str, Any]]] = []
        
        for i in range(S_sparse.shape[0]):
            # Get single row (still sparse)
            row_sparse = S_sparse.getrow(i)
            
            # Convert only this row to dense (much cheaper than full matrix)
            row_scores = row_sparse.toarray().ravel()
            
            # Efficient top-K for this row
            if k >= len(row_scores):
                top_idx = np.argsort(row_scores)[::-1]
            else:
                part_idx = np.argpartition(row_scores, -k)[-k:]
                top_idx = part_idx[np.argsort(row_scores[part_idx])[::-1]]
            
            # Format results
            row_results = [
                self._format_result(int(idx), float(row_scores[idx]))
                for idx in top_idx
            ]
            
            all_list.append(row_results)
            best_list.append(row_results[0] if row_results else None)
        
        return best_list, (all_list if return_all else None)
    
    # ======== ADVANCED: BATCH WITH EARLY STOPPING ========
    def predict_batch_with_threshold(
        self,
        pairs: List[Tuple[Optional[str], Optional[str]]],
        confidence_threshold: float = MIN_SCORE_CONFIRM,
        top_k: Optional[int] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Batch prediction with early stopping for high-confidence predictions.
        Returns only best result per query if score >= threshold, else None.
        """
        k = top_k or TOP_K_TFIDF
        texts: List[str] = []
        
        for d, a in pairs:
            q_def = clean_wo_text(d or "")
            q_act = clean_wo_text(a or "")
            texts.append(self._compose_doc(f"{q_def}\n{q_act}"))
        
        if not any(texts):
            return [None] * len(texts)
        
        Q = self.vectorizer.transform(texts)
        S_sparse = Q @ self.tfidf.T
        
        results: List[Optional[Dict[str, Any]]] = []
        
        for i in range(S_sparse.shape[0]):
            row_scores = S_sparse.getrow(i).toarray().ravel()
            best_idx = np.argmax(row_scores)
            best_score = row_scores[best_idx]
            
            if best_score >= confidence_threshold:
                results.append(self._format_result(int(best_idx), float(best_score)))
            else:
                results.append(None)
        
        return results
    
    # ======== UTILITY: CLEAR CACHE ========
    def clear_cache(self):
        """Clear LRU cache (useful after model updates)."""
        self._vectorize_single.cache_clear()
    
    # ======== UTILITY: GET CACHE STATS ========
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        info = self._vectorize_single.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================
class ATACatalog(ATACatalogOptimized):
    """
    Backward compatible wrapper.
    Existing code using ATACatalog will automatically use optimized version.
    """
    pass
