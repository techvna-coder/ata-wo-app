# core/reconcile.py
def reconcile_triplet(ata_entered, cited, derived):
    """
    cited: {"ata04":..., "exists":True/False, "task_full":..., "page":...} or None
    derived: {"ata04_derived":..., "task_full":..., "doc_type":..., "score":...} or None
    """
    e0 = normalize_ata4(ata_entered)
    e1 = normalize_ata4(cited.get("ata04")) if cited else None
    e2 = normalize_ata4(derived.get("ata04_derived")) if derived else None

    if e0 and e1 and e2 and (e0==e1==e2):
        return decision("CONFIRM", e2, 0.97, reason="All sources agree")

    if e1 and e2 and (e1==e2) and e0!=e1 and cited.get("exists", False):
        return decision("CORRECT", e1, 0.95, reason="TSM-cited & RAG agree vs Entered",
                        evidence={"TSM_task":cited.get("task_full"), "page":cited.get("page"),
                                  "Derived_task":derived.get("task_full")})

    if (not e1) and e0 and e2 and (e0==e2):
        return decision("CONFIRM", e2, 0.88, reason="Entered matches AI-derived (no cited)")

    if e1 and not e2 and cited.get("exists", False):
        return decision("CONFIRM", e1, 0.92, reason="Valid TSM-cited only")

    if e1 and e2 and (e1!=e2):
        # chọn theo chất lượng bằng chứng
        if cited.get("exists", False) and derived.get("score",0) < THRESH_LOW:
            return decision("CONFIRM", e1, 0.90, reason="Prefer valid cited; derived weak")
        if derived.get("score",0) >= THRESH_HIGH:
            return decision("CORRECT", e2, 0.90, reason="Derived strong vs cited", 
                            evidence={"Derived_task":derived.get("task_full")})
        return decision("REVIEW", None, 0.0, reason="Cited vs Derived conflict")

    if e0 and not e1 and not e2:
        return decision("REVIEW", None, 0.0, reason="No reliable evidence")

    return decision("REVIEW", e2 or e1 or None, 0.0, reason="Ambiguous")
