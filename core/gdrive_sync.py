# core/gdrive_sync.py
from __future__ import annotations
import re, json, io, time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

DATA_DIR = Path("data_store")
INGEST_DIR = DATA_DIR / "ingest"
MANIFEST_FILE = DATA_DIR / "manifest.json"

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột, loại bỏ ký tự ẩn (expand_more, chevron_right, \xa0...)."""
    clean_cols = []
    for c in df.columns:
        c2 = re.sub(r"[\xa0\s]+", " ", str(c))
        c2 = re.sub(r"(expand_more|chevron_right)\s*", "", c2, flags=re.I)
        clean_cols.append(c2.strip())
    df.columns = clean_cols
    return df

def _auth_drive() -> GoogleDrive:
    """Xác thực service account."""
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("service_account.json")
    if not gauth.credentials:
        raise RuntimeError("❌ Không tìm thấy file service_account.json.")
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {"by_id": {}}

def _save_manifest(manifest: Dict[str, Any]):
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

def _download_file(drive: GoogleDrive, file_id: str, name: str) -> Path:
    """Tải file từ Google Drive."""
    file_path = INGEST_DIR / name
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    f = drive.CreateFile({"id": file_id})
    f.GetContentFile(str(file_path))
    return file_path

def sync_drive_folder(*args, **kwargs) -> List[Dict[str, Any]]:
    """
    Đồng bộ toàn bộ file trong thư mục Google Drive.
    Cho phép gọi theo 2 dạng:
      - sync_drive_folder(folder_id)
      - sync_drive_folder(drive, folder_id)
    """
    # Xử lý linh hoạt số đối số
    if len(args) == 1:
        folder_id = args[0]
        drive = _auth_drive()
    elif len(args) == 2:
        drive, folder_id = args
    else:
        raise TypeError("sync_drive_folder() requires 1 or 2 arguments: [drive], folder_id")

    manifest = _load_manifest()

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    synced_files = []

    for f in file_list:
        name = f["title"]
        fid = f["id"]
        mime = f["mimeType"]
        if not (name.endswith(".xlsx") or name.endswith(".xls")):
            continue  # bỏ qua file không phải Excel

        print(f"📥 Phát hiện file: {name} ({fid})")
        file_path = _download_file(drive, fid, name)

        # Đọc thử để nhận diện nhanh loại file
        try:
            df = pd.read_excel(file_path, dtype=str, nrows=50)
            df = _clean_colnames(df)

            # Nhận diện loại file (ATA_MAP / WO)
            kind = "unknown"
            if len(df.columns) == 1:
                sample = " ".join(df.iloc[:, 0].astype(str).tolist())
                if len(re.findall(r"\b\d{2}-\d{2}(?:-\d{2})?\b", sample)) >= 3:
                    kind = "ATA_MAP"
            else:
                cols_lower = [c.lower() for c in df.columns]
                if any("ata" in c for c in cols_lower) and any(
                    re.search(r"(title|name|system|desc|mô tả|mo ta)", c, re.I) for c in cols_lower
                ):
                    kind = "ATA_MAP"
                elif any(re.search(r"(defect|action|rectification)", c, re.I) for c in cols_lower):
                    kind = "WO"

            manifest["by_id"][fid] = {
                "name": name,
                "kind": kind,
                "updated": f["modifiedDate"],
                "size": f["fileSize"],
                "path": str(file_path),
                "time_synced": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            synced_files.append({"id": fid, "name": name, "kind": kind, "path": str(file_path)})
            print(f"✅ Đã đồng bộ: {name} → loại: {kind}")

        except Exception as e:
            print(f"⚠️ Lỗi đọc {name}: {e}")
            manifest["by_id"][fid] = {"name": name, "kind": "error", "error": str(e)}

    _save_manifest(manifest)
    return synced_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đồng bộ file từ Google Drive folder.")
    parser.add_argument("--folder", required=True, help="Google Drive folder ID")
    args = parser.parse_args()
    files = sync_drive_folder(args.folder)
    print(f"\nHoàn tất đồng bộ: {len(files)} file được cập nhật.")
