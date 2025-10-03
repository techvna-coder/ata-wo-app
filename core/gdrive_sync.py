import json, re
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
    gauth = GoogleAuth()
    gauth.ServiceAuthCredentials = None
    gauth.LoadServiceConfigSettings()
    gauth.settings["service_config"]["client_json_file_path"] = sa_json_path
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

def list_folder_files(drive: GoogleDrive, folder_id: str):
    q = f"'{folder_id}' in parents and trashed = false"
    file_list = []
    fl = drive.ListFile({'q': q, 'maxResults': 1000}).GetList()
    file_list.extend(fl)
    return file_list

def sync_drive_folder(folder_id: str, sa_json_path: str) -> List[str]:
    _ensure_dirs()
    drive = _auth_service_account(sa_json_path)
    manifest = _load_manifest()
    by_id = manifest.get("by_id", {})

    files = list_folder_files(drive, folder_id)
    changed_paths = []

    for f in files:
        fid = f['id']
        name = f.get('title') or f.get('name')
        mime = f.get('mimeType', '')
        modified = f.get('modifiedDate') or f.get('modifiedTime') or ''
        if mime == "application/vnd.google-apps.folder":
            continue
        if not re.search(r"\.(xlsx|xls)$", name, flags=re.I):
            continue

        rec = by_id.get(fid, {})
        if rec.get("modified") == modified and rec.get("name") == name:
            continue

        local_path = str(Path(INGEST_DIR) / name)
        gfile = drive.CreateFile({"id": fid})
        gfile.GetContentFile(local_path)
        by_id[fid] = {"name": name, "modified": modified, "path": local_path}
        changed_paths.append(local_path)

    manifest["by_id"] = by_id
    _save_manifest(manifest)
    return changed_paths
