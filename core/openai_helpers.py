# core/openai_helpers.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # sẽ kiểm tra ở _ensure_openai()

_CLIENT = None
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ----------------------------
# Tiện ích & an toàn kiểu dữ liệu
# ----------------------------
def _ensure_openai():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if OpenAI is None:
        raise RuntimeError("Thư viện openai chưa sẵn sàng.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong biến môi trường.")
    _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _is_nan(x: Any) -> bool:
    # Không nhập pandas; NaN float có tính chất x != x
    try:
        return isinstance(x, float) and x != x
    except Exception:
        return False


def _to_text(x: Any, maxlen: Optional[int] = None) -> str:
    """Ép bất kỳ giá trị nào thành chuỗi 'an toàn' (loại NaN/None, strip, cắt độ dài)."""
    if x is None or _is_nan(x):
        s = ""
    elif isinstance(x, str):
        s = x
    else:
        # Một số kiểu (numpy.*) có __str__ ổn; nếu lỗi thì rỗng
        try:
            s = str(x)
        except Exception:
            s = ""
    s = s.strip()
    if maxlen is not None and maxlen > 0 and len(s) > maxlen:
        s = s[:maxlen]
    return s


def _norm_candidates(cands: Any) -> List[str]:
    """
    Chuẩn hoá danh sách ứng viên:
      - nếu là list[str] => giữ
      - nếu là list[dict] có khóa 'ata04' => lấy giá trị
      - bỏ rỗng, bỏ None, loại trùng
    """
    out: List[str] = []
    if isinstance(cands, (list, tuple)):
        for c in cands:
            if isinstance(c, dict):
                v = c.get("ata04") or c.get("ATA04") or c.get("code")
                v = _to_text(v)
            else:
                v = _to_text(c)
            if v:
                out.append(v)
    vset = []
    for v in out:
        if v not in vset:
            vset.append(v)
    return vset


def _norm_citations(cits: Any) -> List[str]:
    """Rút gọn danh sách citations sang chuỗi dễ đọc (ví dụ 'AMM 21-51-00-400-001')."""
    out: List[str] = []
    if isinstance(cits, (list, tuple)):
        for c in cits:
            if isinstance(c, dict):
                manual = _to_text(c.get("manual"))
                task = _to_text(c.get("task"))
                ata04 = _to_text(c.get("ata04"))
                s = " ".join([p for p in [manual, task, ata04] if p])
                s = s.strip()
            else:
                s = _to_text(c)
            if s:
                out.append(s)
    return out[:6]


def _ata_name(ata: str, ata_name_map: Optional[Dict[str, str]]) -> str:
    if not ata_name_map:
        return ""
    return _to_text(ata_name_map.get(ata))


# ----------------------------
# 1) Gợi ý ATA khi mơ hồ (đơn giản)
# ----------------------------
def llm_suggest_ata(
    defect_text: str,
    action_text: str,
    top_candidates: Optional[List[str]] = None,
    cited_refs: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Dùng LLM để đề xuất 1 ATA04 khi pipeline TF-IDF mơ hồ.
    Trả về: {"ata04": "...", "reason": "...", "used_model": "..."} hoặc None.
    """
    client = _ensure_openai()
    model = model or _DEFAULT_MODEL

    desc = _to_text(defect_text, 2000)
    act  = _to_text(action_text, 2000)
    cands = _norm_candidates(top_candidates or [])
    refs  = _norm_citations(cited_refs or [])

    sys = (
        "Bạn là kỹ sư khai thác/độ tin cậy. Hãy chọn 1 mã ATA04 phù hợp nhất với mô tả WO "
        "(ưu tiên trong danh sách ứng viên nếu có). Trả về JSON {'ata04': 'AA-BB', 'reason': '...'}."
    )
    user_payload = {
        "wo": {"desc": desc, "action": act},
        "candidates": cands,
        "cited_refs": refs,
        "rule_of_thumb": [
            "Ưu tiên mã ATA xuất hiện trong trích dẫn AMM/TSM/FIM (nếu có).",
            "Nếu không có trích dẫn, chọn ứng viên gần nhất theo triệu chứng và LRU được nhắc.",
            "Không bịa mã, không trả về rỗng. Nếu bất khả kháng, trả về ứng viên đầu tiên và nêu lý do."
        ],
    }

    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        txt = rsp.choices[0].message.content.strip()
        # Tìm JSON trong câu trả lời
        try:
            if txt.startswith("{"):
                data = json.loads(txt)
            else:
                # Thử tìm khối JSON
                lb = txt.find("{")
                rb = txt.rfind("}")
                data = json.loads(txt[lb:rb+1]) if (lb >= 0 and rb >= 0 and rb > lb) else {}
        except Exception:
            data = {}
        ata = _to_text(data.get("ata04"))
        reason = _to_text(data.get("reason"))
        if ata:
            return {"ata04": ata, "reason": reason, "used_model": model}
        # fallback: nếu có candidates
        if cands:
            return {"ata04": cands[0], "reason": "Fallback: chọn ứng viên đầu.", "used_model": model}
        return None
    except Exception:
        # Nếu LLM lỗi, trả None để caller tự xử lý
        return None


# ----------------------------
# 2) Trọng tài khi REVIEW (ưu tiên candidates)
# ----------------------------
def llm_arbitrate_when_review(
    desc: Any,
    action: Any,
    candidates: Optional[List[Union[str, Dict[str, Any]]]],
    citations: Optional[List[Any]] = None,
    ata_name_map: Optional[Dict[str, str]] = None,
    force_from_candidates: bool = True,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Dùng LLM khi Decision=REVIEW. Trả về:
      {"ata04": "...", "reason": "...", "used_model": "...", "candidate_used": True/False}
    """
    client = _ensure_openai()
    model = model or _DEFAULT_MODEL

    # Ép an toàn mọi trường
    s_desc = _to_text(desc, 3000)
    s_act  = _to_text(action, 3000)
    cands  = _norm_candidates(candidates or [])
    cits   = _norm_citations(citations or [])

    # Kèm tên hệ thống nếu có
    cand_with_names = [{"ata04": c, "name": _ata_name(c, ata_name_map)} for c in cands]

    sys = (
        "Bạn là chuyên gia phân loại ATA cho WO. Nhiệm vụ: chọn đúng 1 mã ATA04 "
        "từ danh sách ứng viên (nếu danh sách không rỗng) dựa trên mô tả defect và action. "
        "Nếu danh sách rỗng, bạn có thể đề xuất mã hợp lý nhất, nhưng phải nêu lý do và không bịa đặt.\n"
        "Trả về JSON: {'ata04': 'AA-BB', 'reason': '...'}."
    )
    user_payload = {
        "wo": {"desc": s_desc, "action": s_act},
        "candidates": cand_with_names,
        "citations": cits,
        "policy": {
            "force_from_candidates": bool(force_from_candidates),
            "tie_break": "ưu tiên ứng viên trùng với ATA trong trích dẫn; nếu vẫn hoà, ưu tiên candidate có tên hệ thống phù hợp ngữ cảnh."
        }
    }

    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        txt = rsp.choices[0].message.content.strip()
        try:
            if txt.startswith("{"):
                data = json.loads(txt)
            else:
                lb = txt.find("{")
                rb = txt.rfind("}")
                data = json.loads(txt[lb:rb+1]) if (lb >= 0 and rb >= 0 and rb > lb) else {}
        except Exception:
            data = {}

        ata = _to_text(data.get("ata04"))
        reason = _to_text(data.get("reason"))
        if ata:
            used = ata in cands if cands else False
            return {
                "ata04": ata,
                "reason": reason,
                "used_model": model,
                "candidate_used": used
            }

        # fallback hợp lệ nếu ép phải chọn trong candidates
        if force_from_candidates and cands:
            return {
                "ata04": cands[0],
                "reason": "Fallback: chọn ứng viên đầu do không parse được đầu ra LLM.",
                "used_model": model,
                "candidate_used": True
            }
        return None
    except Exception:
        # Nếu call lỗi, fallback tối thiểu
        if force_from_candidates and cands:
            return {
                "ata04": cands[0],
                "reason": "Fallback offline: lỗi gọi LLM, dùng ứng viên đầu.",
                "used_model": model,
                "candidate_used": True
            }
        return None
