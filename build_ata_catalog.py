# build_ata_catalog.py
# Dùng khi muốn build Catalog từ file SGML_*.tar mà không cần embedding OpenAI
import argparse, tempfile, shutil, os
from core.ata_catalog import _sgml_files_from_tar, build_catalog_from_sgml_files, save_catalog, build_vectorizer_and_matrix

def main():
    ap = argparse.ArgumentParser(description="Build ATA Catalog (JSON + TF-IDF) from SGML TAR")
    ap.add_argument("--tar", required=True, help="Path to SGML_*.tar")
    ap.add_argument("--manual-type", default="TSM", choices=["TSM","FIM","AMM"])
    ap.add_argument("--out-json", default="catalog/ata_catalog.json")
    ap.add_argument("--model-dir", default="catalog/model")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="sgml_")
    try:
        files = _sgml_files_from_tar(args.tar, tmp)
        df = build_catalog_from_sgml_files(files, manual_type=args.manual_type)
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        save_catalog(df, args.out_json)
        build_vectorizer_and_matrix(df, args.model_dir)
        print(f"OK. Catalog: {args.out_json}")
        print(f"OK. Model dir: {args.model_dir}")
        print(f"ATA count: {df['ata04'].nunique()}, rows: {len(df)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
