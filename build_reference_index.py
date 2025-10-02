# build_reference_index.py
import argparse, os, shutil, sys, tempfile
from core.sgml_build import extract_tar, build_registry_and_chunks

def main():
    ap = argparse.ArgumentParser(description="Build DuckDB + FAISS from SGML tar")
    ap.add_argument("--tar", required=True, help="Đường dẫn file SGML_*.tar")
    ap.add_argument("--manual-type", required=True, choices=["TSM", "FIM", "AMM"], help="Loại tài liệu")
    ap.add_argument("--out-db", default="reference_index.duckdb", help="DuckDB registry output")
    ap.add_argument("--out-vectors", default="vectorstore", help="Thư mục FAISS output")
    ap.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OPENAI_API_KEY")
    ap.add_argument("--tmpdir", default=None, help="Thư mục tạm (mặc định: hệ thống)")
    args = ap.parse_args()

    if not args.openai_key:
        print("Thiếu OPENAI_API_KEY. Truyền --openai-key hoặc set biến môi trường.", file=sys.stderr)
        sys.exit(1)

    workdir = args.tmpdir or tempfile.mkdtemp(prefix="sgml_")
    try:
        extracted = extract_tar(args.tar, workdir)
        # Lưu vào reference_docs/<manual_type> để tái sử dụng
        final_dir = os.path.join("reference_docs", args.manual_type.lower())
        os.makedirs(final_dir, exist_ok=True)
        # Copy tree (nếu muốn giữ)
        if extracted != final_dir:
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.copytree(extracted, final_dir)

        # Vectorstore riêng theo manual-type
        vec_dir = os.path.join(args.out_vectors, args.manual_type.lower())
        os.makedirs(vec_dir, exist_ok=True)

        n_task, n_chunk = build_registry_and_chunks(
            extracted_dir=final_dir,
            manual_type=args.manual_type,
            duckdb_path=args.out_db,
            vector_dir=vec_dir,
            openai_api_key=args.openai_key
        )
        print(f"Hoàn tất. Tasks: {n_task} | Chunks: {n_chunk}")
        print(f"DuckDB: {args.out_db}")
        print(f"FAISS:   {vec_dir}")
    finally:
        if not args.tmpdir:
            shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    main()
