# core/refs.py
# =============================================================================
# Chức năng:
#   - Trích các tham chiếu tài liệu kỹ thuật (AMM, TSM, FIM, AFI, IPC) từ chuỗi văn bản WO.
#   - Chuẩn hoá chuỗi nhiệm vụ (task) và suy ra ATA04 = "AA-BB".
#   - Ưu tiên thứ tự manual: TSM/FIM/AFI (cao nhất) > AMM > IPC (thấp nhất).
#
# API:
#   extract_citations(text: str) -> list[dict]
#     Mỗi dict có dạng: {"manual": "AMM|TSM|FIM|AFI|IPC", "task": "AA-BB[-CC[-DDD[-...]]]", "ata04": "AA-BB"}
#
# Ghi chú:
#   - Regex đủ “mềm” để bắt các biến thể: có/không dấu “:”, các ký tự phân tách (-, /, ., khoảng trắng, _),
#     có phần hậu tố chữ (ví dụ “-A”), có ngoặc đơn kèm mốc thời gian.
#   - Nếu xuất hiện nhiều tham chiếu, kết quả được sắp theo độ ưu tiên manual, rồi theo thứ tự xuất hiện.
# =============================================================================

from __future__ import annotations

# Ưu tiên dùng 'regex' (PCRE-like), fallback 're' nếu thiếu.
try:
    import regex as re
except Exception:  # pragma: no cover
    import re  # type: ignore

from typing import Iterable, Iterator, Optional, Tuple, List, Dict

# ----------------------------------------------------------------------------- 
# Thứ tự ưu tiên manual
# -----------------------------------------------------------------------------
_MANUAL_PRIORITY = {
    "TSM": 3,
    "FIM": 3,
    "AFI": 3,
    "AMM": 2,
    "IPC": 1,
}

# ----------------------------------------------------------------------------- 
# Regex lõi cho các manual cần bắt.
# Mẫu số: AA BB [CC] [DDD] [SUFFIX] với các phân tách linh hoạt.
# Ví dụ hợp lệ:
#   AMM 21-52-24-000-001-A
#   TSM: 28 41 00 710
#   FIM 27-51-00
#   AFI 21.21.44.000.001
#   IPC 29-11-00
#
# Chú ý:
# - Cho phép có/không dấu “:”.
# - Cho phép ký tự phân tách bất kỳ không phải số giữa các nhóm (\D{0,3} hoặc \W{0,3}).
# - CC (2 số) và DDD (3 số) là tuỳ chọn; có thể xuất hiện thêm “-001” hoặc hậu tố chữ "A".
# -----------------------------------------------------------------------------
_MANUAL_TAG = r"(?P<manual>AMM|TSM|FIM|AFI|IPC)\b\s*:?"
# AA, BB (bắt buộc), CC (tuỳ chọn), DDD (tuỳ chọn), EEE (tuỳ chọn, đôi khi số mở rộng), và SUFFIX chữ (A/B/…)
# Ta tách riêng các nhóm số chính, phần mở rộng và hậu tố chữ.
_NUM_SEQ = (
    r"(?P<AA>\d{2})\D{0,3}(?P<BB>\d{2})"
    r"(?:\D{0,3}(?P<CC>\d{2}))?"
    r"(?:\D{0,3}(?P<DDD>\d{3}))?"
    r"(?:\D{0,3}(?P<EEE>\d{3}))?"
    r"(?:\D{0,3}(?P<SUFFIX>[A-Z]))?"
)
# Tổng hợp pattern manual + chuỗi số
_MANUAL_RE = re.compile(_MANUAL_TAG + r"\s*" + _NUM_SEQ, re.IGNORECASE)

# Một số biến thể có thể viết liền “AA BB CC DDD” không prefix manual → BỎ QUA trong extract_citations (chỉ nhận khi có manual)
# Vì E1 cần dựa trên manual rõ ràng để nâng độ tin cậy.


def _norm_ata04(aa: Optional[str], bb: Optional[str]) -> Optional[str]:
    """Chuẩn hoá ATA04 = 'AA-BB' từ AA, BB (2 chữ số)."""
    if not aa or not bb:
        return None
    aa = aa.zfill(2)
    bb = bb.zfill(2)
    return f"{aa}-{bb}"


def _compose_task(aa: str, bb: str, cc: Optional[str], ddd: Optional[str],
                  eee: Optional[str], suffix: Optional[str]) -> str:
    """
    Chuẩn hoá chuỗi 'task' theo thứ tự các nhóm có mặt.
    Ưu tiên dạng: AA-BB[-CC[-DDD[-EEE]]][-SUFFIX]
    """
    parts: List[str] = [aa.zfill(2), bb.zfill(2)]
    if cc:
        parts.append(cc.zfill(2))
    if ddd:
        parts.append(ddd.zfill(3))
    if eee:
        parts.append(eee.zfill(3))

    task = "-".join(parts)
    if suffix:
        # dính hậu tố chữ ở cuối
        task = f"{task}-{suffix.upper()}"
    return task


def _iter_manual_matches(text: str) -> Iterator[Tuple[int, Dict[str, str]]]:
    """
    Duyệt tất cả match manual trong văn bản, yield (start_pos, info_dict).
    info_dict = {"manual","task","ata04"}
    """
    if not text:
        return
    for m in _MANUAL_RE.finditer(text):
        gd = m.groupdict()
        manual = (gd.get("manual") or "").upper()
        aa, bb = gd.get("AA"), gd.get("BB")
        cc, ddd = gd.get("CC"), gd.get("DDD")
        eee, sf = gd.get("EEE"), gd.get("SUFFIX")

        ata04 = _norm_ata04(aa, bb)
        if not ata04:
            continue

        task = _compose_task(aa, bb, cc, ddd, eee, sf)
        yield (m.start(), {"manual": manual, "task": task, "ata04": ata04})


def _priority(manual: str) -> int:
    """Trả về mức ưu tiên của manual."""
    return _MANUAL_PRIORITY.get(manual.upper(), 0)


def _dedup_preserve_order(items: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Khử trùng lặp theo (manual, task). Giữ nguyên thứ tự lần đầu xuất hiện.
    """
    seen = set()
    out: List[Dict[str, str]] = []
    for it in items:
        key = (it.get("manual", ""), it.get("task", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Trích các citation (E1) từ văn bản WO.
    - Chỉ nhận các mẫu có prefix manual (AMM/TSM/FIM/AFI/IPC).
    - Chuẩn hoá 'task' và 'ata04'.
    - Sắp xếp theo ưu tiên manual: TSM/FIM/AFI > AMM > IPC; nếu ngang nhau, theo vị trí xuất hiện.

    Returns:
        List[{"manual": str, "task": str, "ata04": str}]
    """
    text = text or ""
    matches: List[Tuple[int, Dict[str, str]]] = list(_iter_manual_matches(text))
    if not matches:
        return []

    # Sắp theo (priority desc, start_pos asc)
    matches.sort(key=lambda t: (-_priority(t[1]["manual"]), t[0]))

    # Lấy phần dict, khử trùng lặp theo (manual, task)
    ordered = [info for _, info in matches]
    return _dedup_preserve_order(ordered)


# ====================== Tự kiểm thử cơ bản ======================
if __name__ == "__main__":
    s = """
    REF AMM: 21-52-24-000-001-A (AUG 01 2025) AND AMM 21-52-24-400-001-A.
    Also found TSM 27 51 00 710 and FIM: 29-11-00. AFI 21.21.44.000.001 BLAH.
    IPC 32-41-00 may appear too.
    """
    cits = extract_citations(s)
    print("CITATIONS:")
    for c in cits:
        print(c)
    # Kỳ vọng: các bản ghi TSM/FIM/AFI nằm trước AMM, và cuối cùng là IPC
