import json, os
import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from .constants import TOP_K_TFIDF, MIN_SCORE_CONFIRM

class ATACatalog:
    def __init__(self, catalog_dir="catalog"):
        self.catalog_path = os.path.join(catalog_dir, "ata_catalog.json")
        self.vec_path = os.path.join(catalog_dir, "model", "tfidf_vectorizer.joblib")
        self.mat_path = os.path.join(catalog_dir, "model", "tfidf_matrix.npz")
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)  # {"AA-BB": {...}}
        self.ata_list = list(self.catalog.keys())
        self.vectorizer: TfidfVectorizer = load(self.vec_path)
        self.tfidf = load_npz(self.mat_path)

    def _compose_doc(self, text: str):
        return (text or "").strip()

    def predict(self, defect_text: str, rect_text: str = None):
        q = self._compose_doc(f"{defect_text or ''}\n{rect_text or ''}")
        if not q:
            return None, None
        qv = self.vectorizer.transform([q])
        scores = (qv @ self.tfidf.T).toarray()[0]
        top_idx = np.argsort(scores)[::-1][:TOP_K_TFIDF]
        results = []
        for idx in top_idx:
            ata = self.ata_list[idx]
            score = float(scores[idx])
            info = self.catalog.get(ata, {})
            snippet = info.get("title") or (info.get("keywords") or [""])[0]
            results.append({
                "ata04": ata,
                "score": score,
                "doc": "CATALOG",
                "snippet": snippet,
                "source": "catalog/ata_catalog.json"
            })
        best = results[0] if results else None
        if best and best["score"] < MIN_SCORE_CONFIRM:
            pass
        return best, results
