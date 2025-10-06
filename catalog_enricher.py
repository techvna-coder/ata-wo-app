#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Catalog Enricher for WO → ATA04 Checker
--------------------------------------

Mục tiêu
- Đọc file Excel "A321 Data ATA map.xlsx" (không theo cấu trúc cố định),
  trích xuất các mã ATA cấp 2/4/6 theo regex.
- Tổng hợp mô tả theo cấp và làm giàu nội dung cho cấp ATA4 (mức huấn luyện chính).
- Với chương ATA >= 70, nhân đôi nhánh theo Engine_Type (IAE, PW11).
- Hợp nhất với bộ nhớ hiện có (Parquet) và rebuild TF‑IDF model cho catalog.

Kết quả
- catalog/ata_catalog.json: danh mục enriched theo ATA4 (kèm Engine_Type khi áp dụng)
- catalog/model/tfidf_vectorizer.joblib
- catalog/model/tfidf_matrix.npz

Cách chạy
python catalog_enricher.py \
  --excel "./A321 Data ATA map.xlsx" \
  --catalog-dir ./catalog \
  --data-store ./data_store \
  --engine-types IAE,PW11 \
  --min-docs-per-ata4 1 \
  --dry-run False

Phụ thuộc: pandas, numpy, openpyxl, regex, scikit-learn, joblib, scipy
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import regex as re2
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# Regex & Chuẩn hoá mã ATA
# ------------------------------
RE_ATA2 = re2.compile(r"(?<!\d)(\d{2})(?!\d)")
RE_ATA4 = re2.compile(r"(?<!\d)(\d{2})[- ]?(\d{2})(?!\d)")
RE_ATA6 = re2.compile(r"(?<!\d)(\d{2})[- ]?(\d{2})[- ]?(\d{2})(?!\d)")


def norm_ata2(s: str) -> Optional[str]:
    m = RE_ATA2.search(s or "")
    if m:
        return f"{int(m.group(1)):02d}"
    return None


def norm_ata4(s: str) -> Optional[str]:
    m = RE_ATA4.search(s or "")
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def norm_ata6(s: str) -> Optional[str]:
    m = RE_ATA6.search(s or "")
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


# ------------------------------
# Dataclasses
# ------------------------------
@dataclass
class ATARecord:
    ata2: Optional[str]
    ata4: Optional[str]
    ata6: Optional[str]
    title: Optional[str]
    description: Optional[str]
    engine_type: Optional[str] = None  # IAE, PW11 hoặc None

    def text(self) -> str:
        parts = []
        if self.title:
            parts.append(str(self.title))
        if self.description:
            parts.append(str(self.description))
        # Gắn nhãn cấp để cải thiện TF‑IDF (hint n‑gram theo cấp)
        tags = []
        if self.ata2:
            tags.append(f"[ATA2:{self.ata2}]")
        if self.ata4:
            tags.append(f"[ATA4:{self.ata4}]")
        if self.ata6:
            tags.append(f"[ATA6:{self.ata6}]")
        if self.engine_type:
            tags.append(f"[ENG:{self.engine_type}]")
        if tags:
            parts.append(" ".join(tags))
        return "\n".join(parts)


# ------------------------------
# Đọc Excel không cấu trúc: quét toàn bộ cell để bắt mã ATA & lấy text lân cận
# ------------------------------

def read_unstructured_excel(path: Path) -> List[ATARecord]:
    """Đọc mọi sheet; với mỗi hàng, cố gắng trích mã ATA6/ATA4/ATA2
    và ghép tên/mô tả từ các cột còn lại cùng hàng.
    """
    xl = pd.ExcelFile(path)
    records: List[ATARecord] = []

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype=str)
        except Exception:
            warnings.warn(f"Bỏ qua sheet '{sheet}' (không đọc được)")
            continue
        if df.empty:
            continue
        df = df.fillna("")
        # Chuẩn hoá tên cột để tăng cơ hội bắt text mô tả/tiêu đề
        df.columns = [str(c).strip() for c in df.columns]
        for _, row in df.iterrows():
            row_texts = [str(v).strip() for v in row.values if str(v).strip()]
            if not row_texts:
                continue
            joined = " | ".join(row_texts)
            a6 = norm_ata6(joined)
            a4 = norm_ata4(joined)
            a2 = norm_ata2(joined)
            if not any([a6, a4, a2]):
                continue
            # Ưu tiên lấy "title" là cell chứa code nhất (a6>a4>a2); description là phần còn lại
            title = None
            description = None
            # tìm cell gần code nhất
            def nearest_cell_with(subpat: re2.Pattern) -> Optional[str]:
                for v in row.values:
                    sv = str(v).strip()
                    if sv and subpat.search(sv):
                        return sv
                return None

            if a6:
                title = nearest_cell_with(RE_ATA6)
            elif a4:
                title = nearest_cell_with(RE_ATA4)
            elif a2:
                title = nearest_cell_with(RE_ATA2)

            # description = tất cả cell trừ title, nối lại
            desc_parts = []
            for v in row.values:
                sv = str(v).strip()
                if not sv:
                    continue
                if title and sv == title:
                    continue
                desc_parts.append(sv)
            description = " | ".join(desc_parts) if desc_parts else None

            rec = ATARecord(ata2=a2, ata4=a4, ata6=a6, title=title, description=description)
            records.append(rec)
    return records


# ------------------------------
# Gộp về cấp ATA4 (mức chính cho TF‑IDF)
# ------------------------------

def group_to_ata4(records: List[ATARecord], engine_types: List[str]) -> pd.DataFrame:
    """Tạo DataFrame với mỗi dòng là một tài liệu đại diện cho một ATA4.
    - Văn bản = tổng hợp từ: các record cùng ATA4 + ngữ cảnh ATA2/ATA6
    - Với chương >= 70: nhân đôi theo Engine_Type (IAE, PW11)
    """
    # Chuẩn hoá: nếu có ATA6 mà thiếu ATA4 thì rút gọn
    for r in records:
        if not r.ata4 and r.ata6:
            parts = r.ata6.split("-")
            r.ata4 = "-".join(parts[:2])
        if not r.ata2 and r.ata4:
            r.ata2 = r.ata4.split("-")[0]

    # Gom theo ATA4
    buckets: Dict[str, List[ATARecord]] = {}
    for r in records:
        if not r.ata4:
            continue
        buckets.setdefault(r.ata4, []).append(r)

    rows = []
    for ata4, rs in buckets.items():
        ata2 = rs[0].ata2 if rs and rs[0].ata2 else (ata4.split("-")[0] if "-" in ata4 else None)
        # Tổng hợp văn bản
        titles = [x.title for x in rs if x.title]
        descs = [x.description for x in rs if x.description]
        a6_children = [x.ata6 for x in rs if x.ata6]
        text = "\n".join([*titles, *descs, *(f"[CHILD:{c}]" for c in a6_children) if a6_children else []])
        # Bổ sung nhãn cấp
        header = f"[ATA4:{ata4}] [ATA2:{ata2}]"
        base_text = f"{header}\n{text}" if text else header

        chap = int(ata4.split("-")[0])
        if chap >= 70 and engine_types:
            for eng in engine_types:
                rows.append({
                    "ata4": ata4,
                    "ata2": ata2,
                    "engine_type": eng,
                    "doc_text": f"{base_text}\n[ENG:{eng}]",
                })
        else:
            rows.append({
                "ata4": ata4,
                "ata2": ata2,
                "engine_type": None,
                "doc_text": base_text,
            })

    df = pd.DataFrame(rows)
    # Loại bỏ trùng lặp văn bản (nếu có)
    if not df.empty:
        df = df.drop_duplicates(subset=["ata4", "engine_type", "doc_text"]).reset_index(drop=True)
    return df


# ------------------------------
# Hợp nhất với bộ nhớ Parquet (WO training & ata_map)
# ------------------------------

def load_memory_frames(data_store_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (wo_training, ata_map) nếu có, ngược lại trả DataFrame rỗng.
    wo_training.parquet: nên có cột Defect_Text, Rectification_Text, ATA04_Final (hoặc ATA04_Entered)
    ata_map.parquet: mapping tên hệ thống theo ATA04 (tuỳ chọn)
    """
    wo_p = data_store_dir / "wo_training.parquet"
    map_p = data_store_dir / "ata_map.parquet"

    if wo_p.exists():
        wo = pd.read_parquet(wo_p)
    else:
        wo = pd.DataFrame()

    if map_p.exists():
        ata_map = pd.read_parquet(map_p)
    else:
        ata_map = pd.DataFrame()
    return wo, ata_map


def build_docs_from_wo(wo: pd.DataFrame, min_docs_per_ata4: int = 1) -> pd.DataFrame:
    """Tạo tài liệu lịch sử từ WO để bổ sung vào catalog (mức ATA4).
    Văn bản = Defect_Text + Rectification_Text (khi có). Sử dụng ATA04_Final ưu tiên; rơi về ATA04_Entered.
    """
    if wo is None or wo.empty:
        return pd.DataFrame(columns=["ata4", "engine_type", "doc_text"])  # engine_type None

    # Xác định cột ATA
    ata_col = None
    for c in ["ATA04_Final", "ATA 04 Corrected", "ATA04_Entered", "ATA", "ATA04"]:
        if c in wo.columns:
            ata_col = c
            break
    if ata_col is None:
        return pd.DataFrame(columns=["ata4", "engine_type", "doc_text"])  # không có cột ATA

    def to_ata4(x: str) -> Optional[str]:
        return norm_ata4(str(x)) if pd.notna(x) else None

    wo = wo.copy()
    wo["ata4"] = wo[ata_col].map(to_ata4)

    # Văn bản
    def build_text(row) -> str:
        parts = []
        for c in ["Defect_Text", "W/O Description", "Rectification_Text", "W/O Action"]:
            if c in wo.columns and pd.notna(row.get(c)):
                parts.append(str(row.get(c)))
        header = f"[WO] [ATA4:{row.get('ata4')}]"
        return header + "\n" + "\n".join(parts)

    wo = wo[wo["ata4"].notna()].copy()
    if wo.empty:
        return pd.DataFrame(columns=["ata4", "engine_type", "doc_text"])  # no rows

    wo["doc_text"] = wo.apply(build_text, axis=1)
    docs = wo[["ata4", "doc_text"]].dropna()

    # Bảo đảm tối thiểu số doc cho mỗi ATA4, có thể group + sample
    grp = docs.groupby("ata4")["doc_text"].apply(list).reset_index()
    rows = []
    for _, r in grp.iterrows():
        ata4 = r["ata4"]
        texts = r["doc_text"]
        # ghép thành 1 tài liệu dài để giảm độ phân mảnh
        merged = "\n\n".join(texts)
        rows.append({"ata4": ata4, "engine_type": None, "doc_text": merged})
    return pd.DataFrame(rows)


# ------------------------------
# TF‑IDF Builder & Saver
# ------------------------------

def build_tfidf(docs: pd.DataFrame, catalog_dir: Path) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    model_dir = catalog_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Vector hoá theo n‑gram, bỏ rất chung chung, giữ số & dấu '-'
    vectorizer = TfidfVectorizer(
        strip_accents=None,
        lowercase=True,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.98,
        token_pattern=r"(?u)\b[\w\-/\.]{2,}\b",
    )

    X = vectorizer.fit_transform(docs["doc_text"].tolist())

    # Lưu
    dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    sparse.save_npz(model_dir / "tfidf_matrix.npz", X)
    return vectorizer, X


# ------------------------------
# Catalog JSON Saver
# ------------------------------

def save_catalog_json(docs: pd.DataFrame, ata_map: pd.DataFrame, catalog_dir: Path) -> None:
    """Sinh catalog/ata_catalog.json theo định dạng phẳng.
    Trường bắt buộc: ata4, engine_type, name (nếu tìm thấy), keywords, sample
    """
    # Mapping tên hệ thống nếu có
    name_map: Dict[str, str] = {}
    for cand in ["ATA04", "ata4", "ATA_04", "ATA Code", "ATA code", "Code", "code"]:
        if cand in (ata_map.columns if isinstance(ata_map, pd.DataFrame) else []):
            key_col = cand
            # tìm cột tên
            name_col = None
            for nc in [
                "Name", "System Name", "ATA Name", "Description", "System",
                "Tên", "Mô tả", "Title",
            ]:
                if nc in ata_map.columns:
                    name_col = nc
                    break
            if name_col:
                tmp = ata_map[[key_col, name_col]].dropna()
                for _, r in tmp.iterrows():
                    k = norm_ata4(str(r[key_col]))
                    if k:
                        name_map[k] = str(r[name_col])
            break

    # Gợi ý keywords đơn giản từ doc_text (top frequent tokens)
    def extract_keywords(text: str, topk: int = 12) -> List[str]:
        text = text.lower()
        toks = re.findall(r"[a-z0-9\-/\.]{3,}", text)
        # loại bỏ tag
        toks = [t for t in toks if not t.startswith("[ata") and not t.startswith("[wo") and not t.startswith("[eng") and not t.startswith("[child")]
        if not toks:
            return []
        vc = pd.Series(toks).value_counts()
        return vc.head(topk).index.tolist()

    items = []
    for _, r in docs.iterrows():
        ata4 = r["ata4"]
        eng = r.get("engine_type")
        text = r.get("doc_text") or ""
        name = name_map.get(ata4)
        kw = extract_keywords(text)
        items.append({
            "ata4": ata4,
            "name": name,
            "engine_type": eng,
            "keywords": kw,
            "sample": text[:1200],  # cắt mẫu để tránh file quá lớn
        })

    out = {
        "version": 2,
        "generated_by": "catalog_enricher.py",
        "count": len(items),
        "items": items,
    }
    (catalog_dir / "ata_catalog.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------
# Pipeline chính
# ------------------------------

def run_pipeline(
    excel_path: Path,
    catalog_dir: Path,
    data_store_dir: Path,
    engine_types: List[str],
    min_docs_per_ata4: int = 1,
    dry_run: bool = False,
) -> None:
    print("[1/5] Đọc file Excel nhà sản xuất (unstructured)…", file=sys.stderr)
    records = read_unstructured_excel(excel_path)
    if not records:
        raise RuntimeError("Không trích xuất được bản ghi ATA nào từ file Excel.")

    print(f"  → Bắt được {len(records)} record thô.", file=sys.stderr)

    print("[2/5] Gộp về mức ATA4 và nhân nhánh theo Engine_Type (>=70)…", file=sys.stderr)
    df_man = group_to_ata4(records, engine_types)
    print(f"  → Sinh {len(df_man)} tài liệu từ dữ liệu nhà sản xuất.", file=sys.stderr)

    print("[3/5] Nạp bộ nhớ WO & ATA map (Parquet)…", file=sys.stderr)
    wo, ata_map = load_memory_frames(data_store_dir)
    df_wo = build_docs_from_wo(wo, min_docs_per_ata4=min_docs_per_ata4)
    print(f"  → Sinh {len(df_wo)} tài liệu lịch sử từ WO.", file=sys.stderr)

    print("[4/5] Hợp nhất & rebuild TF‑IDF…", file=sys.stderr)
    docs = pd.concat([df_man, df_wo], ignore_index=True)
    # Loại bỏ NA và trùng
    docs = docs.dropna(subset=["ata4", "doc_text"]).drop_duplicates()
    if docs.empty:
        raise RuntimeError("Không có tài liệu để huấn luyện TF‑IDF.")

    if dry_run:
        print("  [DRY-RUN] Bỏ qua huấn luyện TF‑IDF.", file=sys.stderr)
    else:
        build_tfidf(docs, catalog_dir)

    print("[5/5] Xuất catalog JSON (enriched)…", file=sys.stderr)
    save_catalog_json(docs, ata_map, catalog_dir)

    print("Hoàn tất. Catalog & model đã được cập nhật tại:")
    print(f"  - {catalog_dir / 'ata_catalog.json'}")
    if not dry_run:
        print(f"  - {catalog_dir / 'model' / 'tfidf_vectorizer.joblib'}")
        print(f"  - {catalog_dir / 'model' / 'tfidf_matrix.npz'}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalog Enricher for WO → ATA04 Checker")
    parser.add_argument("--excel", required=True, help="Đường dẫn file Excel ATA map của nhà sản xuất")
    parser.add_argument("--catalog-dir", default="./catalog", help="Thư mục output catalog")
    parser.add_argument("--data-store", default="./data_store", help="Thư mục data_store chứa .parquet")
    parser.add_argument("--engine-types", default="IAE,PW11", help="Danh sách engine types, ví dụ: IAE,PW11")
    parser.add_argument("--min-docs-per-ata4", type=int, default=1, help="Tối thiểu số doc/ATA4 từ WO khi gộp")
    parser.add_argument("--dry-run", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False, help="Chạy thử, không huấn luyện TF‑IDF")

    args = parser.parse_args()

    excel_path = Path(args.excel)
    catalog_dir = Path(args.catalog_dir)
    data_store_dir = Path(args.data_store)
    engine_types = [s.strip() for s in str(args.engine_types).split(",") if s.strip()]

    run_pipeline(
        excel_path=excel_path,
        catalog_dir=catalog_dir,
        data_store_dir=data_store_dir,
        engine_types=engine_types,
        min_docs_per_ata4=args.min_docs_per_ata4,
        dry_run=args.dry_run,
    )
