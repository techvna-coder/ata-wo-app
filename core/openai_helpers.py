# core/openai_helpers.py
from __future__ import annotations
import os, json, hashlib, time
from typing import List, Dict, Any, Optional, Tuple

try:
    import openai
except Exception:
    openai = None

from .llm_cache import cache_get, cache_put

# ----- Cấu hình mặc định -----
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

def _ensure_openai():
    if openai is None:
        raise RuntimeError("Thư viện openai chưa sẵn sàng.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong biến môi trường.")
    openai.api_key = api_key

def _hash_key(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

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
      {
        "ata04": "21-52",
        "confidence": 0.90,
        "rationale": "...",
        "chosen_from": "citations|tfidf|entered",
        "evidence_span": "..."
      }
    - Luôn ưu tiên chọn trong 'candidates' (force_from_candidates=True)
    - Có cache theo sha1(desc|action|candidates|citations|ata_name_map)
    """
    payload = {
        "wo": {"desc": (desc or "")[:1500], "action": (action or "")[:1500]},
        "candidates": candidates[:5] if candidates else [],
        "citations": citations[:5] if citations else [],
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

    _ensure_openai()

    system = (
        "Bạn là chuyên gia bảo dưỡng tàu bay. Nhiệm vụ: chọn đúng MỘT mã ATA 4 ký tự (AA-BB) phù hợp nhất, "
        "ưu tiên theo: (1) trích dẫn AMM/TSM/FIM hợp lệ, (2) khớp triệu chứng với tiêu đề/từ khóa ATA, (3) điểm TF-IDF. "
        "Trả về JSON theo schema: {ata04, confidence, rationale, chosen_from, evidence_span}. "
        "Không trả văn bản ngoài JSON."
    )
    user = json.dumps(payload, ensure_ascii=False)

    # Dùng Chat Completions (tương thích)
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp["choices"][0]["message"]["content"]
    except Exception as e:
        # Khi lỗi model, trả về rỗng
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
            a,b = ata.split("-",1)
            return f"{a[:2]}-{b[:2]}"
        if ata.isdigit() and len(ata) >= 4:
            return f"{ata[:2]}-{ata[2:4]}"
        return ata
    ata04 = _fix(ata04)

    # Nếu ép phải nằm trong candidates, kiểm tra và ràng buộc
    set_cands = { (c.get("ata04") or "").upper() for c in (candidates or []) if c.get("ata04") }
    if force_from_candidates and ata04 not in set_cands and set_cands:
        # Nếu LLM chọn ngoài tập → ràng buộc về top-1 TF-IDF
        # Hoặc chọn ứng viên có mention trong rationale/evidence
        top1 = candidates[0].get("ata04") if candidates else ""
        ata04 = top1 or ata04
        chosen_from = chosen_from or "tfidf"

    out = {
        "ata04": ata04,
        "confidence": max(0.0, min(1.0, conf)),
        "rationale": rationale[:1200],
        "chosen_from": chosen_from,
        "evidence_span": evidence_span[:600],
        "ts": int(time.time()),
    }
    cache_put(key, out, ttl_sec=cache_ttl_sec)
    return out

# ------------------------------------------------------------------
# 2) API cũ để gợi ý đơn mã (không bắt buộc candidates)
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
    """
    Giữ tương thích ngược: chỉ trả 1 mã đề xuất từ ngữ cảnh. Không ràng buộc candidates.
    """
    _ensure_openai()
    system = ("Bạn là chuyên gia bảo dưỡng tàu bay. Hãy đề xuất một mã ATA 4 ký tự (AA-BB) phù hợp nhất với mô tả/hành động.")
    user = {
        "desc": (defect_text or "")[:1500],
        "action": (rect_text or "")[:1500],
        "candidates": top_candidates or [],
        "cited_refs": cited_refs or [],
        "rules": {"format": "AA-BB recommended"}
    }
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(user, ensure_ascii=False)}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp["choices"][0]["message"]["content"]
        obj = json.loads(text)
        return {
            "ata04": obj.get("ata04"),
            "confidence": obj.get("confidence", 0.0),
            "reason": obj.get("reason", "")
        }
    except Exception as e:
        return {"ata04": "", "confidence": 0.0, "reason": f"LLM error: {e}"}

# ------------------------------------------------------------------
# 3) Làm giàu Catalog (tuỳ chọn khi build)
# ------------------------------------------------------------------
def enrich_catalog_entries(
    ata_samples: Dict[str, List[str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> Dict[str, Dict[str, Any]]:
    """
    ata_samples: {"21-52": ["desc1 ... action1", "desc2 ... action2", ...], ...}
    trả về: {"21-52": {"keywords": [...], "samples": [...]} }
    """
    _ensure_openai()
    out = {}
    for ata, texts in ata_samples.items():
        j = {"ata04": ata, "examples": [t[:800] for t in texts[:5]]}
        system = ("Chuẩn hoá từ khóa & câu mô tả mẫu cho mã ATA. "
                  "Trả JSON: {keywords:[...], samples:[...]} ngắn gọn, kỹ thuật, không riêng máy bay cụ thể.")
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(j, ensure_ascii=False)}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp["choices"][0]["message"]["content"]
            obj = json.loads(text)
            kws = obj.get("keywords", [])
            sents = obj.get("samples", [])
            if isinstance(kws, list) and isinstance(sents, list):
                out[ata] = {"keywords": [str(x) for x in kws][:20], "samples": [str(x) for x in sents][:10]}
        except Exception:
            continue
    return out
