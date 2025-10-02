# core/refregistry.py
import duckdb
import pandas as pd

class RefRegistry:
    def __init__(self, db_path: str = "reference_index.duckdb", read_only: bool = True):
        self.con = duckdb.connect(db_path, read_only=read_only)

    def exact_lookup(self, task_full: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM refs WHERE task_full = ? LIMIT 1", [task_full]).fetchdf()

    def by_ata04(self, ata04: str, manual_type: str = None, limit: int = 50) -> pd.DataFrame:
        if manual_type:
            q = "SELECT * FROM refs WHERE ata04 = ? AND manual_type = ? LIMIT ?"
            return self.con.execute(q, [ata04, manual_type, limit]).fetchdf()
        q = "SELECT * FROM refs WHERE ata04 = ? LIMIT ?"
        return self.con.execute(q, [ata04, limit]).fetchdf()

    def close(self):
        self.con.close()
