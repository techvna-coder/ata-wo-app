# core/openai_helpers.py
from __future__ import annotations
import os, json, hashlib, time
from typing import List, Dict, Any, Optional, Tuple

# Hỗ trợ cả SDK mới (v1) và SDK cũ (0.x)
_SDK_MODE = -1   # -1: không có, 0: legacy 0.x, 1: new 1.x
_client_v1 = None
_openai_legacy = None

def _detect_sdk() -> int:
    global _SDK_MODE, _client_v1, _openai_legacy
    if _SDK_MODE != -1:
        return _SDK_MODE
    # Thử SDK v1
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            _SDK_MODE = -1
            return _SDK_MODE
        _client_v1 = OpenAI(api_key=api_key)
        _SDK_MODE = 1
        return _SDK_MODE
    except Exception:
        pass
    # Thử SDK legacy 0.x
    try:
        import openai as _openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            _SDK_MODE = -1
            return _SDK_MODE
        _openai.api_key = api_key
        globals()["_openai_legacy"] = _openai
        _SDK_MODE = 0
        return _SDK_MODE
    except Exception:
        _SDK_MODE = -1
        return _SDK_MODE

def has_llm_available() -> bool:
    """Dùng trong app.py để bật/tắt checkbox LLM đúng thực tế."""
    return _detect_sdk() in (0, 1)

# ---------------- Cache đơn giản ----------------
from .llm_cache import cache_get, cache_put

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

def _hash_key(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _chat_completion(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """
    Gọi Chat Completions, tương thích cả SDK v1 và 0.x.
    Trả về text (assistant content). Ném Exception nếu SDK không có.
    """
    mode = _detect_sdk()
    if mode == 1:
        # SDK v1
        resp = _client_v1.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    elif mode == 0:
        # SDK legacy 0.x
        resp = _openai_legacy.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]
    else:
        raise RuntimeError("OpenAI SDK chưa sẵn sàng (chưa cài hoặc thiếu OPENAI_API_KEY).")

# ------------------------------------------------------------------
# 1) LLM arbitration khi REVIEW (chọn trong tập ứng viên hạn chế)
# ------------------------------------------------------------------
def llm_arbitrate_when_review(
    desc: str,
    action: str,
    candidates: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    ata_name_map: Dict[str, str],
    force_from_candidates: bool = True,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_ttl_sec: int = 30*24*3600,
) -> Dict[str, Any]:
    """
    Trả về:
      { "ata04": "21-52", "confidence": 0.90, "rationale": "...", "chosen_from": "citations|tfidf|entered", "evidence_span": "..." }
    Nếu SDK/OpenAI không sẵn sàng → không raise, trả object rỗng kèm rationale.
    """
    payload = {
        "wo": {"desc": (desc or "")[:1500], "action": (action or "")[:1500]},
        "candidates": (candidates or [])[:5],
        "citations": (citations or [])[:5],
        "ata_map": {k: ata_name_map.get(k, "") for k in [c.get("ata04") for c in (candidates or []) if c.get("ata04")]},
        "rules": {
            "format": "AA-BB only",
            "must_choose_from_candidates": bool(force_from_candidates),
            "tie_break": "citations > symptom match > tfidf score > E0"
        }
    }
    key = _hash_key(payload)
    hit = cache_get(key)
    if hit:
        return hit

    if not has_llm_available():
        return {"ata04": "", "confidence": 0.0, "rationale": "LLM not available", "chosen_from": "", "evidence_span": ""}

    system = (
        "Bạn là chuyên gia bảo dưỡng tàu bay. Nhiệm vụ: chọn đúng MỘT mã ATA 4 ký tự (AA-BB) phù hợp nhất, "
        "ưu tiên (1) trích dẫn AMM/TSM/FIM hợp lệ, (2) khớp triệu chứng, (3) điểm TF-IDF. "
        "Chỉ trả JSON: {ata04, confidence, rationale, chosen_from, evidence_span}."
    )
    user = json.dumps(payload, ensure_ascii=False)

    try:
        text = _chat_completion(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        return {"ata04": "", "confidence": 0.0, "rationale": f"LLM error: {e}", "chosen_from": "", "evidence_span": ""}

    # Parse JSON
    ata04 = ""
    conf = 0.0
    rationale = ""
    chosen_from = ""
    evidence_span = ""
    try:
        obj = json.loads(text)
        ata04 = (obj.get("ata04") or "").strip().upper()
        conf = float(obj.get("confidence") or 0.0)
        rationale = obj.get("rationale") or ""
        chosen_from = obj.get("chosen_from") or ""
        evidence_span = obj.get("evidence_span") or ""
    except Exception:
        rationale = f"Unparseable JSON: {text}"

    # Chuẩn hoá AA-BB
    def _fix(ata: str) -> str:
        if not ata: return ata
        ata = ata.replace(" ", "")
        if "-" in ata:
            a, b = ata.split("-", 1)
            return f"{a[:2]}-{b[:2]}"
        if ata.isdigit() and len(ata) >= 4:
            return f"{ata[:2]}-{ata[2:4]}"
        return ata
    ata04 = _fix(ata04)

    # Ép phải thuộc tập ứng viên nếu cờ bật
    set_cands = { (c.get("ata04") or "").upper() for c in (candidates or []) if c.get("ata04") }
    if force_from_candidates and ata04 not in set_cands and set_cands:
        ata04 = (candidates[0].get("ata04") or "").upper()
        chosen_from = chosen_from or "tfidf"

    out = {
        "ata04": ata04,
        "confidence": max(0.0, min(1.0, conf)),
        "rationale": (rationale or "")[:1200],
        "chosen_from": chosen_from,
        "evidence_span": (evidence_span or "")[:600],
        "ts": int(time.time()),
    }
    cache_put(key, out, ttl_sec=cache_ttl_sec)
    return out

# ------------------------------------------------------------------
# 2) API cũ để gợi ý đơn mã (giữ tương thích)
# ------------------------------------------------------------------
def llm_suggest_ata(
    defect_text: str,
    rect_text: str,
    top_candidates: Optional[List[str]] = None,
    cited_refs: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    if not has_llm_available():
        return {"ata04": "", "confidence": 0.0, "reason": "LLM not available"}
    system = "Bạn là chuyên gia bảo dưỡng tàu bay. Hãy đề xuất một mã ATA 4 ký tự (AA-BB) phù hợp nhất."
    user = {
        "desc": (defect_text or "")[:1500],
        "action": (rect_text or "")[:1500],
        "candidates": top_candidates or [],
        "cited_refs": cited_refs or [],
        "rules": {"format": "AA-BB recommended"}
    }
    try:
        text = _chat_completion(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(user, ensure_ascii=False)}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        obj = json.loads(text)
        return {
            "ata04": obj.get("ata04"),
            "confidence": obj.get("confidence", 0.0),
            "reason": obj.get("reason", "")
        }
    except Exception as e:
        return {"ata04": "", "confidence": 0.0, "reason": f"LLM error: {e}"}
