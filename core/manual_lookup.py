# core/manual_lookup.py
def derive_ata_from_description(store, defect_text, rect_text=None, k=4):
    query = normalize_for_query(" ".join([defect_text or "", rect_text or ""]))
    # tìm trong TSM trước
    tsm_hits = store.search(query, k=k, doc_type="TSM/FIM")
    amm_hits = [] if is_confident(tsm_hits) else store.search(query, k=k, doc_type="AMM")

    candidates = rank_candidates(tsm_hits + amm_hits)  # xếp hạng theo: semantic + component-keyword + meta ata04
    best = candidates[0] if candidates else None
    if not best:
        return None
    return {
        "ata04_derived": best.meta.get("ata04"),
        "task_full": best.meta.get("task_full"),
        "doc_type": best.meta.get("doc_type"),
        "doc_page": best.meta.get("page"),
        "score": best.score,
        "snippet": best.text[:600]
    }
