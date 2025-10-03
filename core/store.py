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


def append_ata_map(df_map: pd.DataFrame, code_col: str, name_col: str):
df = df_map.copy()
df["ATA04"] = df[code_col].map(_to_ata04)
df = df[["ATA04", name_col]].rename(columns={name_col:"Title"}).dropna().drop_duplicates()
if Path(ATA_PARQUET).exists():
old = pd.read_parquet(ATA_PARQUET)
out = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["ATA04"], keep="last")
else:
out = df
out.to_parquet(ATA_PARQUET, index=False)


def append_wo_training(df_wo: pd.DataFrame, desc_col: str, act_col: str, ata_final_col: str, ata_entered_col: str, source_file: str):
tdf = df_wo.copy()
def _text(row):
parts = []
if desc_col in row and pd.notna(row[desc_col]): parts.append(str(row[desc_col]))
if act_col in row and pd.notna(row[act_col]): parts.append(str(row[act_col]))
return " | ".join(parts).strip()
tdf["text"] = tdf.apply(_text, axis=1)
def _label(row):
lab = None
if ata_final_col in row and pd.notna(row[ata_final_col]) and str(row[ata_final_col]).strip():
lab = row[ata_final_col]
elif ata_entered_col in row:
lab = row[ata_entered_col]
return _to_ata04(lab)
tdf["ata04"] = tdf.apply(_label, axis=1)
tdf = tdf[(tdf["text"].str.len()>0) & tdf["ata04"].notna()].copy()
tdf["hash"] = tdf["text"].map(_hash_text)
keep_cols = ["text","ata04","hash"]
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
