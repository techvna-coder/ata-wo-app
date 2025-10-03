import pandas as pd
from .mapping import MAP_IN2INTERNAL


def load_wo_excel(file):
df = pd.read_excel(file, dtype=str)
cols = {}
for k, v in MAP_IN2INTERNAL.items():
if k in df.columns:
cols[k] = v
df = df.rename(columns=cols)
for need in MAP_IN2INTERNAL.values():
if need not in df.columns:
df[need] = None
for c in ("Open_Date","Close_Date"):
df[c] = pd.to_datetime(df[c], errors="coerce")
return df


def write_result(df, path="WO_ATA_checked.xlsx"):
df.to_excel(path, index=False)
return path
