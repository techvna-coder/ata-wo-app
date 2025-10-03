# core/ata_catalog.py
# Xây "ATA Catalog" nhẹ từ SGML, rồi suy đoán ATA theo TF-IDF (cục bộ, không cần OpenAI)
import os, re, tarfile, json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
import joblib

ATA_RE_STD = re.compile(r'\b(\d{2})-(\d{2})-(\d{2})(?:-(\d{3})(?:-(\d{3}))*)?\b')
ATA_RE_COMPACT = re.compile(r'\b(\d{2})(\d{2})(\d{2})(?:(\d{3}))?(?:(\d{3}))?\b', re.I)

WARN_PATTERNS = [
    r'\bECAM\b', r'\bEICAS\b', r'\bCAS\b', r'\bMASTER CAUTION\b', r'\bMASTER WARNING\b',
    r'\bWARNING\b', r'\bCAUTION\b', r'\bALERT\b', r'\bMSG\b', r'\bFAULT\b', r'\bFAIL\b'
]
WARN_RE = re.compile("|".join(WARN_PATTERNS), re.I)

def _norm_ata(task: str) -> Optional[str]:
    if not task: return None
    m = ATA_RE_STD.search(task)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m2 = ATA_RE_COMPACT.search(task)
    if m2: return f"{m2.group(1)}-{m2.group(2)}"
    return None

def _extract_text(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)

def _sgml_files_from_tar(tar_path: str, tmp_dir: str) -> List[str]:
    outdir = Path(tmp_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(outdir)
    files = []
    for p in outdir.rglob("*"):
        if p.suffix.lower() in [".sgml", ".sgm", ".xml", ".xsg", ".xmg"]:
            files.append(str(p))
    return files

def build_catalog_from_sgml_files(files: List[str], manual_type: str = "TSM") -> pd.DataFrame:
    """
    Quét nhanh các file SGML/XML, gom theo ATA04:
      - ata04
      - system_titles: tiêu đề/chủ đề thường gặp
      - warnings: cụm cảnh báo/alert/ECAM/EICAS
      - keywords: từ/cụm từ hay xuất hiện (unigrams/bigrams)
      - sample_text: vài câu tiêu biểu (phục vụ TF-IDF)
    """
    buckets: Dict[str, Dict[str, Counter]] = defaultdict(lambda: {
        "titles": Counter(),
        "warnings": Counter(),
        "tokens": Counter(),
        "samples": Counter(),
    })

    for fp in files:
        try:
            raw = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        soup = BeautifulSoup(raw, "lxml")
        # lấy các block có khả năng chứa task/ATA (heuristic)
        blocks = soup.find_all(["task","dm","procedure","topic","chapter","section","para","p","title","name","caption","heading"])
        if not blocks:
            text = _extract_text(raw)
            ata04 = _norm_ata(text)
            if not ata04: 
                continue
            buckets[ata04]["samples"][text[:800]] += 1
            if WARN_RE.search(text):
                buckets[ata04]["warnings"][text[:120]] += 1
            continue

        for blk in blocks:
            blk_text = " ".join(blk.stripped_strings)
            if not blk_text: 
                continue
            ata04 = _norm_ata(blk_text) or _norm_ata(str(blk.get("id") or blk.get("code") or ""))
            if not ata04:
                continue

            # title
            title = None
            for ttag in ["title","tasktitle","name","caption","heading"]:
                t = blk.find(ttag)
                if t and t.get_text(strip=True):
                    title = t.get_text(" ", strip=True)[:150]
                    break
            if title: buckets[ata04]["titles"][title] += 1

            # warnings
            if WARN_RE.search(blk_text):
                # cắt câu chứa từ cảnh báo
                for sent in re.split(r'(?<=[\.\!\?])\s+', blk_text):
                    if WARN_RE.search(sent):
                        buckets[ata04]["warnings"][sent[:200]] += 1

            # tokens (thô, để chọn top keywords)
            tokens = re.findall(r"[A-Za-z0-9\-_/\.]+", blk_text)
            for tk in tokens:
                if len(tk) >= 3:
                    buckets[ata04]["tokens"][tk.lower()] += 1

            buckets[ata04]["samples"][blk_text[:800]] += 1

    rows = []
    for ata, d in buckets.items():
        titles = [t for t,_ in d["titles"].most_common(10)]
        warns  = [w for w,_ in d["warnings"].most_common(20)]
        keys   = [k for k,_ in d["tokens"].most_common(60)]
        sents  = [s for s,_ in d["samples"].most_common(40)]
        rows.append({
            "ata04": ata,
            "manual_type": manual_type,
            "system_titles": titles,
            "warnings": warns,
            "keywords": keys,
            "sample_text": " \n".join(sents)[:8000]
        })
    df = pd.DataFrame(rows).sort_values("ata04")
    return df.reset_index(drop=True)

def save_catalog(df: pd.DataFrame, out_json: str):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)

def load_catalog(json_path: str) -> pd.DataFrame:
    return pd.read_json(json_path)

# ====== TF-IDF suy đoán ATA theo mô tả hỏng hóc ======
def build_vectorizer_and_matrix(catalog_df: pd.DataFrame, model_dir: str):
    """
    Huấn luyện TF-IDF trên văn bản đại diện mỗi ATA: title + warnings + keywords + sample_text.
    Lưu vectorizer & ma trận centroid theo ATA.
    """
    texts, labels = [], []
    for _, r in catalog_df.iterrows():
        doc = " ".join([
            " ".join(r.get("system_titles", [])),
            " ".join(r.get("warnings", [])),
            " ".join(r.get("keywords", [])),
            r.get("sample_text", "") or ""
        ])
        texts.append(doc)
        labels.append(r["ata04"])

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1,2),
        min_df=1,
        max_features=100_000
    )
    X = vec.fit_transform(texts)
    X = normalize(X, norm="l2")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    # Lưu ma trận và nhãn
    joblib.dump({"X": X, "labels": labels}, os.path.join(model_dir, "ata_tfidf_matrix.joblib"))

def load_vectorizer_and_matrix(model_dir: str):
    vec = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    pack = joblib.load(os.path.join(model_dir, "ata_tfidf_matrix.joblib"))
    return vec, pack["X"], pack["labels"]

def predict_ata(defect_text: str, vec, X, labels, topk=3) -> List[Tuple[str, float]]:
    if not defect_text:
        return []
    q = vec.transform([defect_text])
    q = normalize(q, norm="l2")
    sims = linear_kernel(q, X).flatten()  # cosine vì đã l2-normalize
    idx = sims.argsort()[::-1][:topk]
    return [(labels[i], float(sims[i])) for i in idx]
