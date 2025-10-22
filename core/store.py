import duckdb, re, hashlib, pandas as pd
from pathlib import Path

DB_PATH = "data_store/memory.duckdb"
WO_PARQUET = "data_store/wo_training.parquet"
ATA_PARQUET = "data_store/ata_map.parquet"

def _hash_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8","ignore")).hexdigest()

def _to_ata04(s):
    if pd.isna(s): return None
    t = str(s).strip()
    m = re.findall(r"\d", t)
    if len(m) >= 4:
        return f"{''.join(m[:2])}-{''.join(m[2:4])}"
    m2 = re.match(r"^\s*(\d{2})\s*[-\.]\s*(\d{2})", t)
    if m2: return f"{m2.group(1)}-{m2.group(2)}"
    return None

def init_db():
    Path("data_store").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS wo_meta(
        source_file TEXT,
        rows_ingested INTEGER,
        ingested_at TIMESTAMP DEFAULT now()
    );
    """)
    con.close()

def append_ata_map(df_map: pd.DataFrame, code_col: str = None, name_col: str = None):
    """
    Tự động xử lý file ATA map có cấu trúc cây (A321 Data ATA map.xlsx).
    - Nhận dạng code và mô tả từ cột duy nhất.
    - Kế thừa mô tả từ hệ thống cha.
    - Gom nhóm về ATA04 và tạo mô tả đầy đủ.
    """
    import re

    df = df_map.copy()
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()

    # 1️⃣ Trích ATA_Code và mô tả (nếu file chỉ có 1 cột)
    if not code_col or code_col not in df.columns:
        df["ATA_Code"] = df[first_col].apply(
            lambda x: re.findall(r"\d{2}(?:-\d{2}){0,3}", x)[0] if re.search(r"\d{2}(?:-\d{2}){0,3}", x) else None
        )
    else:
        df["ATA_Code"] = df[code_col]

    if not name_col or name_col not in df.columns:
        df["Title"] = df[first_col].apply(
            lambda x: re.sub(r"^\s*\d{2}(?:-\d{2}){0,3}\s*-\s*", "", x).strip()
        )
    else:
        df["Title"] = df[name_col].astype(str)

    # 2️⃣ Xây map hierarchy cha–con
    def parent_code(code):
        if not code or not isinstance(code, str): return None
        parts = code.split("-")
        if len(parts) > 2:
            return "-".join(parts[:-1])
        elif len(parts) == 2:
            return parts[0]
        return None

    df["Parent"] = df["ATA_Code"].apply(parent_code)

    desc_map = {r["ATA_Code"]: r["Title"] for _, r in df.iterrows() if r["ATA_Code"]}

    enriched = []
    for code, desc in desc_map.items():
        full_desc = desc
        p = parent_code(code)
        # kế thừa mô tả từ cha, ghép theo chiều tăng dần độ chi tiết
        while p and p in desc_map:
            full_desc = desc_map[p] + " & " + full_desc
            p = parent_code(p)
        enriched.append({"ATA_Code": code, "Full_Description": full_desc})

    df_enriched = pd.DataFrame(enriched)

    # 3️⃣ Tính ATA04 (2 cấp)
    def to_ata04(code):
        if not code or not isinstance(code, str): return None
        parts = code.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return code

    df_enriched["ATA04"] = df_enriched["ATA_Code"].apply(to_ata04)

    # 4️⃣ Gộp các dòng cùng ATA04 thành 1 mô tả phong phú
    agg = (
        df_enriched.groupby("ATA04")["Full_Description"]
        .apply(lambda x: " | ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"Full_Description": "Title"})
    )
    agg["Source_Count"] = agg["Title"].apply(lambda t: t.count("|") + 1)

    # 5️⃣ Ghi vào parquet (merge nếu đã có dữ liệu cũ)
    if Path(ATA_PARQUET).exists():
        old = pd.read_parquet(ATA_PARQUET)
        out = pd.concat([old, agg], ignore_index=True).drop_duplicates(subset=["ATA04"], keep="last")
    else:
        out = agg

    out.to_parquet(ATA_PARQUET, index=False)
    print(f"✅ ATA map enriched: {len(out)} entries written to {ATA_PARQUET}")


# core/store.py (chỉ thay hàm append_wo_training)
def append_wo_training(df_wo: pd.DataFrame, desc_col: str, act_col: str,
                       ata_final_col: str, ata_entered_col: str, source_file: str):
    from .cleaning import clean_wo_text  # import nội bộ để dùng chung

    tdf = df_wo.copy()

    def _text(row):
        parts = []
        if desc_col in row and pd.notna(row[desc_col]):
            parts.append(clean_wo_text(str(row[desc_col])))
        if act_col and act_col in row and pd.notna(row[act_col]):
            parts.append(clean_wo_text(str(row[act_col])))
        # ghép bằng ' | ' để giữ ngữ cảnh nhưng ổn định khi hash
        return " | ".join([p for p in parts if p]).strip()

    tdf["text"] = tdf.apply(_text, axis=1)

    def _label(row):
        lab = None
        if ata_final_col in row and pd.notna(row[ata_final_col]) and str(row[ata_final_col]).strip():
            lab = row[ata_final_col]
        elif ata_entered_col in row:
            lab = row[ata_entered_col]
        return _to_ata04(lab)

    tdf["ata04"] = tdf.apply(_label, axis=1)

    # bỏ dòng trống hoặc không chuẩn hoá được ATA04
    tdf = tdf[(tdf["text"].str.len() > 0) & tdf["ata04"].notna()].copy()

    # Hash sau khi làm sạch → chống trùng theo nội dung kỹ thuật thực
    tdf["hash"] = tdf["text"].map(_hash_text)
    keep_cols = ["text", "ata04", "hash"]

    if Path(WO_PARQUET).exists():
        old = pd.read_parquet(WO_PARQUET)
        merged = pd.concat([old[keep_cols], tdf[keep_cols]], ignore_index=True)
        merged = merged.drop_duplicates(subset=["hash"], keep="first")
    else:
        merged = tdf[keep_cols].drop_duplicates(subset=["hash"], keep="first")

    merged.to_parquet(WO_PARQUET, index=False)
    con = duckdb.connect(DB_PATH)
    con.execute("INSERT INTO wo_meta(source_file, rows_ingested) VALUES (?, ?)", [source_file, int(tdf.shape[0])])
    con.close()
