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

    # Tuỳ chọn hiệu năng/độ bền
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=3000, help="Số chunk/FAISS shard")
    ap.add_argument("--max-workers-parse", type=int, default=4)
    ap.add_argument("--no-embed", action="store_true", help="Chỉ parse & build DuckDB, chưa embed")
    ap.add_argument("--resume", action="store_true", help="Tiếp tục từ shard đã có")

    args = ap.parse_args()

    if not args.openai_key and not args.no_embed:
        print("Thiếu OPENAI_API_KEY. Truyền --openai-key hoặc set biến môi trường, hoặc dùng --no-embed.", file=sys.stderr)
        sys.exit(1)

    workdir = tempfile.mkdtemp(prefix="sgml_")
    try:
        extracted = extract_tar(args.tar, workdir)
        final_dir = os.path.join("reference_docs", args.manual_type.lower())
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(extracted, final_dir)

        vec_dir = os.path.join(args.out_vectors, args.manual_type.lower())
        os.makedirs(vec_dir, exist_ok=True)

        n_task, n_chunk = build_registry_and_chunks(
            extracted_dir=final_dir,
            manual_type=args.manual_type,
            duckdb_path=args.out_db,
            vector_dir=vec_dir,
            openai_api_key=args.openai_key or "",
            embedding_model=args.embedding_model,
            embed_batch_size=args.embed_batch_size,
            shard_size=args.shard_size,
            max_workers_parse=args.max_workers_parse,
            no_embed=args.no_embed,
            resume=args.resume,
        )
        print(f"Hoàn tất. Tasks: {n_task} | Chunks: {n_chunk}")
        print(f"DuckDB: {args.out_db}")
        print(f"FAISS:   {vec_dir} (shard mode)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    main()
