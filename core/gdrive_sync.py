import json
import re
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

MANIFEST = "data_store/manifest.json"
INGEST_DIR = "data_store/ingest"

def _ensure_dirs():
    Path("data_store").mkdir(parents=True, exist_ok=True)
    Path(INGEST_DIR).mkdir(parents=True, exist_ok=True)

def _load_manifest() -> Dict:
    p = Path(MANIFEST)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"by_id": {}}

def _save_manifest(m: Dict):
    Path(MANIFEST).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def _auth_service_account(sa_json_path: str):
    """Xác thực Google Drive bằng service account JSON."""
    gauth = GoogleAuth(settings={
        "client_config_backend": "service",
        "service_config": {"client_json_file_path": sa_json_path}
    })
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột, loại bỏ ký tự ẩn và tiền tố UI."""
    clean_cols = []
    for c in df.columns:
        c2 = re.sub(r"[\xa0\s]+", " ", str(c))
        c2 = re.sub(r"(expand_more|chevron_right)\s*", "", c2, flags=re.I)
        clean_cols.append(c2.strip())
    df.columns = clean_cols
    return df

def list_folder_files(drive: GoogleDrive, folder_id: str):
    """Liệt kê toàn bộ file trong 1 thư mục Drive (không phân trang thủ công)."""
    q = f"'{folder_id}' in parents and trashed = false"
    fl = drive.ListFile({'q': q, 'maxResults': 1000}).GetList()
    return fl

def _detect_file_kind(file_path: str) -> str:
    """Nhận diện loại file (ATA_MAP hoặc WO) dựa theo cấu trúc dữ liệu."""
    try:
        df = pd.read_excel(file_path, dtype=str, nrows=100)
        df = _clean_colnames(df)

        if len(df.columns) == 1:
            sample = " ".join(df.iloc[:, 0].astype(str).tolist())
            if len(re.findall(r"\b\d{2}-\d{2}(?:-\d{2})?\b", sample)) >= 3:
                return "ATA_MAP"
        else:
            cols_lower = [c.lower() for c in df.columns]
            if any("ata" in c for c in cols_lower) and any(
                re.search(r"(title|name|system|desc|mô tả|mo ta)", c, re.I)
                for c in cols_lower
            ):
                return "ATA_MAP"
            elif any(re.search(r"(defect|action|rectification)", c, re.I) for c in cols_lower):
                return "WO"
    except Exception as e:
        print(f"⚠️ Lỗi đọc {file_path}: {e}")
    return "unknown"

def sync_drive_folder(folder_id: str, sa_json_path: str) -> List[str]:
    """
    Đồng bộ incremental thư mục Drive về data_store/ingest.
    - Tự export Google Sheets thành .xlsx
    - Chỉ tải file Excel (.xlsx/.xls)
    - Nhận diện tự động loại file (WO / ATA_MAP)
    Trả về danh sách file local đã tải/cập nhật.
    """
    _ensure_dirs()
    try:
        drive = _auth_service_account(sa_json_path)
    except Exception as e:
        raise RuntimeError(f"❌ Lỗi xác thực service account: {e}")

    manifest = _load_manifest()
    by_id = manifest.get("by_id", {})

    try:
        files = list_folder_files(drive, folder_id)
    except Exception as e:
        raise RuntimeError(f"❌ Không list được folder '{folder_id}': {e}")

    changed_paths = []

    for f in files:
        fid = f["id"]
        name = f.get("title") or f.get("name")
        mime = f.get("mimeType", "")
        modified = f.get("modifiedDate") or f.get("modifiedTime") or ""

        if mime == "application/vnd.google-apps.folder":
            continue

        # Google Sheets → export thành .xlsx
        if mime == "application/vnd.google-apps.spreadsheet":
            local_path = str(Path(INGEST_DIR) / (name if name.lower().endswith(".xlsx") else f"{name}.xlsx"))
            rec = by_id.get(fid, {})
            if rec.get("modified") == modified and rec.get("name") == name:
                continue
            try:
                gfile = drive.CreateFile({"id": fid})
                gfile.GetContentFile(
                    local_path,
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                print(f"⚠️ Export lỗi '{name}': {e}")
                continue

            file_kind = _detect_file_kind(local_path)
            by_id[fid] = {
                "name": name, "modified": modified,
                "path": local_path, "kind": file_kind,
                "time_synced": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            changed_paths.append(local_path)
            print(f"✅ Export: {name} ({file_kind})")
            continue

        # File Excel gốc
        if re.search(r"\.(xlsx|xls)$", name, flags=re.I):
            rec = by_id.get(fid, {})
            if rec.get("modified") == modified and rec.get("name") == name:
                continue
            local_path = str(Path(INGEST_DIR) / name)
            try:
                gfile = drive.CreateFile({"id": fid})
                gfile.GetContentFile(local_path)
            except Exception as e:
                print(f"⚠️ Tải file '{name}' lỗi: {e}")
                continue

            file_kind = _detect_file_kind(local_path)
            by_id[fid] = {
                "name": name, "modified": modified,
                "path": local_path, "kind": file_kind,
                "time_synced": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            changed_paths.append(local_path)
            print(f"✅ Đồng bộ: {name} ({file_kind})")

    manifest["by_id"] = by_id
    _save_manifest(manifest)

    print(f"\n📦 Hoàn tất: {len(changed_paths)} file được cập nhật.")
    return changed_paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đồng bộ thư mục Google Drive về local.")
    parser.add_argument("--folder", required=True, help="Folder ID trên Google Drive")
    parser.add_argument("--sa", required=True, help="Đường dẫn tới file service_account.json")
    args = parser.parse_args()

    paths = sync_drive_folder(args.folder, args.sa)
    print(paths)
