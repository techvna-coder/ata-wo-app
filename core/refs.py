# core/refs.py
# =============================================================================
# Chức năng:
#   - Trích các tham chiếu tài liệu kỹ thuật (AMM, TSM, FIM, AFI, IPC) từ chuỗi văn bản WO.
#   - Chuẩn hoá chuỗi nhiệm vụ (task) và suy ra ATA04 = "AA-BB".
#   - Ưu tiên thứ tự manual: TSM/FIM/AFI (cao nhất) > AMM > IPC (thấp nhất).
#
# API:
#   extract_citations(text: str) -> list[dict]
#     Mỗi dict có dạng: {"manual": "AMM|TSM|FIM|AFI|IPC", "task": "AA-BB[-CC[-DDD[-EEE]]][-SUFFIX]", "ata04": "AA-BB"}
# =============================================================================

from __future__ import annotations

try:
    import regex as re
except Exception:  # pragma: no cover
    import re  # type: ignore

from typing import Iterable, Iterator, Optional, Tuple, List, Dict

# Thứ tự ưu tiên manual
_MANUAL_PRIORITY = {
    "TSM": 3,
    "FIM": 3,
    "AFI": 3,
    "AMM": 2,
    "IPC": 1,
}

# Regex lõi
_MANUAL_TAG = r"(?P<manual>AMM|TSM|FIM|AFI|IPC)\b\s*:?"
_NUM_SEQ = (
    r"(?P<AA>\d{2})\W{0,3}(?P<BB>\d{2})"
    r"(?:\W{0,3}(?P<CC>\d{2}))?"
    r"(?:\W{0,3}(?P<DDD>\d{3}))?"
    r"(?:\W{0,3}(?P<EEE>\d{3}))?"
    r"(?:\W{0,3}(?P<SUFFIX>[A-Z]))?"
)
_MANUAL_RE = re.compile(_MANUAL_TAG + r"\s*" + _NUM_SEQ, re.IGNORECASE)


def _norm_ata04(aa: Optional[str], bb: Optional[str]) -> Optional[str]:
    if not aa or not bb:
        return None
    return f"{aa.zfill(2)}-{bb.zfill(2)}"


def _compose_task(aa: str, bb: str, cc: Optional[str], ddd: Optional[str],
                  eee: Optional[str], suffix: Optional[str]) -> str:
    parts: List[str] = [aa.zfill(2), bb.zfill(2)]
    if cc:
        parts.append(cc.zfill(2))
    if ddd:
        parts.append(ddd.zfill(3))
    if eee:
        parts.append(eee.zfill(3))
    task = "-".join(parts)
    if suffix:
        task = f"{task}-{suffix.upper()}"
    return task


def _iter_manual_matches(text: str) -> Iterator[Tuple[int, Dict[str, str]]]:
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
    return {"TSM":3, "FIM":3, "AFI":3, "AMM":2, "IPC":1}.get(manual.upper(), 0)


def _dedup_preserve_order(items: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
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
    text = text or ""
    matches: List[Tuple[int, Dict[str, str]]] = list(_iter_manual_matches(text))
    if not matches:
        return []
    # sort by priority desc, then position asc
    matches.sort(key=lambda t: (-_priority(t[1]["manual"]), t[0]))
    ordered = [info for _, info in matches]
    return _dedup_preserve_order(ordered)


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
