"""
Build TF-IDF catalog từ file catalog/ata_catalog.json (nếu có) hoặc từ bộ nhớ bằng core/catalog_builder.py.
"""
import argparse
from core.catalog_builder import build_catalog_from_memory


if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--from-memory", action="store_true", help="Build từ data_store parquet")
args = parser.parse_args()
if args.from_memory:
stat = build_catalog_from_memory()
print(f"Built from memory: {len(stat)} classes")
else:
# khuyến nghị dùng --from-memory
stat = build_catalog_from_memory()
print(f"Built from memory (default): {len(stat)} classes")
