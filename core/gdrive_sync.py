import json
import re
from pathlib import Path
from typing import List, Dict

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

MANIFEST = "data_store/manifest.json"
INGEST_DIR = "data_store/ingest"


def _ensure_dirs():
    Path(INGEST_DIR).mkdir(parents=True, exist_ok=True)
    Path("data_store").mkdir(parents=True, exist_ok=True)

def _load_manifest() -> Dict:
    p = Path(MANIFEST)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"by_id": {}}

def _save_manifest(m: Dict):
    Path(MANIFEST).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def _auth_service_account(sa_json_path: str):
    """
    Xác thực service account cho pydrive2 bằng config nội tuyến.
    Tránh dùng LoadServiceConfigSettings() để khỏi phụ thuộc file cấu hình mặc định.
    """
    gauth = GoogleAuth(settings={
        "client_config_backend": "service",
        "service_config": {
            "client_json_file_path": sa_json_path
        }
    })
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

def list_folder_files(drive: GoogleDrive, folder_id: str):
    q = f"'{folder_id}' in parents and trashed = false"
    file_list = []
    # pydrive2 đã handle phân trang nội bộ trong GetList()
    fl = drive.ListFile({'q': q, 'maxResults': 1000}).GetList()
    file_list.extend(fl)
    return file_list

def sync_drive_folder(folder_id: str, sa_json_path: str) -> List[str]:
    """
    Đồng bộ incremental 1 thư mục Drive về data_store/ingest.
    - Tự export Google Sheets thành .xlsx
    - Chỉ nhận file Excel (.xlsx/.xls)
    Trả về danh sách file path đã được tải/cập nhật.
    """
    _ensure_dirs()
    try:
        drive = _auth_service_account(sa_json_path)
    except Exception as e:
        raise RuntimeError(f"Auth lỗi (service account): {e}")

    manifest = _load_manifest()
    by_id = manifest.get("by_id", {})

    try:
        files = list_folder_files(drive, folder_id)
    except Exception as e:
        raise RuntimeError(f"Không list được folder '{folder_id}': {e}")

    changed_paths = []

    for f in files:
        fid = f["id"]
        name = f.get("title") or f.get("name")
        mime = f.get("mimeType", "")
        modified = f.get("modifiedDate") or f.get("modifiedTime") or ""

        # Bỏ qua thư mục con
        if mime == "application/vnd.google-apps.folder":
            continue

        # Google Sheets → export .xlsx
        if mime == "application/vnd.google-apps.spreadsheet":
            # tên file local đảm bảo có .xlsx
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
                raise RuntimeError(f"Export Google Sheet '{name}' lỗi: {e}")
            by_id[fid] = {"name": name, "modified": modified, "path": local_path}
            changed_paths.append(local_path)
            continue

        # File Excel chuẩn tải trực tiếp
        if re.search(r"\.(xlsx|xls)$", name, flags=re.I):
            rec = by_id.get(fid, {})
            if rec.get("modified") == modified and rec.get("name") == name:
                continue
            local_path = str(Path(INGEST_DIR) / name)
            try:
                gfile = drive.CreateFile({"id": fid})
                gfile.GetContentFile(local_path)
            except Exception as e:
                raise RuntimeError(f"Tải file '{name}' lỗi: {e}")
            by_id[fid] = {"name": name, "modified": modified, "path": local_path}
            changed_paths.append(local_path)
        # Bỏ qua các loại file khác

    manifest["by_id"] = by_id
    _save_manifest(manifest)
    return changed_paths
