from .constants import CONF_MATRIX

def decide(e0, e1_valid, e1_ata, e2_best, e2_all):
    e2_ata = e2_best["ata04"] if e2_best else None

    if e1_valid and e1_ata and e2_ata and (e0 == e1_ata == e2_ata):
        return "CONFIRM", CONF_MATRIX["E0_E1_E2_VALID"], "E0=E1=E2 (E1 valid)."

    if e1_valid and e1_ata and e2_ata and (e1_ata == e2_ata) and (e0 != e1_ata):
        return "CORRECT", CONF_MATRIX["E1_E2_NEQ_E0_VALID"], "E1=E2≠E0 (E1 valid) → sửa theo E1/E2."

    if e2_ata and e0 and (e2_ata == e0) and not e1_valid:
        return "CONFIRM", CONF_MATRIX["E2_EQ_E0_ONLY"], "E2 khớp E0; không có E1."

    if e1_valid and e1_ata and not e2_ata:
        return "CONFIRM", CONF_MATRIX["E1_ONLY_VALID"], "Chỉ E1 hợp lệ."

    if e1_valid and e1_ata and e2_ata and (e1_ata != e2_ata):
        return "REVIEW", CONF_MATRIX["CONFLICT"], "E1≠E2 → cần xem xét."

    if e0 and not (e1_valid or e2_ata):
        return "REVIEW", CONF_MATRIX["E0_ONLY"], "Chỉ có E0."

    return "REVIEW", 0.55, "Thiếu bằng chứng rõ ràng."
