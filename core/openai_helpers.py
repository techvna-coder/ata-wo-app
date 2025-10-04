# core/openai_helpers.py
from __future__ import annotations
import os, time, json
from typing import Dict, Any, List, Optional

# Hỗ trợ OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# Cấu hình & tiện ích
# =========================
def _client() -> Optional["OpenAI"]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def _safe_truncate(txt: str, limit: int) -> str:
    if not txt:
        return ""
    txt = str(txt)
    return txt[:limit]

def _retry(fn, retries=2, backoff=1.5):
    last = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(backoff ** (i + 1))
    raise last

# =========================
# Fallback gợi ý ATA04
# =========================
def llm_suggest_ata(
    defect_text: str,
    rect_text: str,
    top_candidates: Optional[List[str]] = None,
    cited_refs: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> Optional[Dict[str, Any]]:
    """
    Trả về gợi ý {"ata04": "AA-BB", "reason": "..."} hoặc None nếu không khả dụng.
    Gửi tối thiểu dữ liệu, đã rút gọn.
    """
    cli = _client()
    if cli is None:
        return None

    defect_text = _safe_truncate(defect_text or "", 1200)
    rect_text   = _safe_truncate(rect_text or "", 1200)
    cand_txt = ", ".join(top_candidates or [])[:200] if top_candidates else ""
    refs_txt = "; ".join(cited_refs or [])[:250] if cited_refs else ""

    system = (
        "Bạn là kỹ sư bảo dưỡng máy bay. Nhiệm vụ: xác định mã ATA 04 (dạng AA-BB) "
        "phù hợp nhất cho mô tả WO. Ưu tiên dựa vào triệu chứng/symptom, chỉ dấu ECAM/EICAS/CAS, "
        "và tham chiếu AMM/TSM/FIM (nếu có). Xuất JSON ngắn gọn."
    )
    user = {
        "wo_defect": defect_text,
        "wo_action": rect_text,
        "top_candidates": top_candidates or [],
        "cited": refs_txt,
        "output": "Trả về JSON dạng: {\"ata04\": \"AA-BB\", \"reason\": \"...\"}."
    }

    def _call():
        resp = cli.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        return resp

    try:
        resp = _retry(_call)
        txt = resp.choices[0].message.content or ""
        # tìm JSON trong câu trả lời
        start = txt.find("{")
        end   = txt.rfind("}")
        if start >= 0 and end > start:
            js = json.loads(txt[start:end+1])
            ata = js.get("ata04")
            reason = js.get("reason", "")
            if isinstance(ata, str) and len(ata) >= 5:
                return {"ata04": ata.strip(), "reason": str(reason)[:500]}
    except Exception:
        return None
    return None

# =========================
# Làm giàu Catalog (title/keywords)
# =========================
def llm_enrich_catalog_entry(
    ata04: str,
    samples: List[str],
    title_hint: str = "",
    model: str = "gpt-4o-mini",
    top_k: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Từ vài câu mẫu + title_hint (nếu có), sinh:
      {"title": "...", "keywords": ["..."], "samples": ["..."]}  (keywords <= top_k)
    """
    cli = _client()
    if cli is None:
        return None

    samples = [ _safe_truncate(s, 240) for s in (samples or []) ][:5]

    system = (
        "Bạn là chuyên gia hệ thống máy bay. Hãy đặt tiêu đề hệ thống ngắn gọn, "
        "chuẩn ATA và liệt kê từ khóa đặc trưng (symptom, ECAM/EICAS/CAS cues, LRUs) "
        "cho lớp ATA04 được cung cấp."
    )
    user = {
        "ata04": ata04,
        "title_hint": _safe_truncate(title_hint or "", 120),
        "samples": samples,
        "need": f"Trả JSON: {{\"title\": \"...\", \"keywords\": [<= {top_k} từ/cụm], \"samples\": [<=3 câu chọn lại]}}"
    }

    def _call():
        return cli.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=300,
        )

    try:
        resp = _retry(_call)
        txt = resp.choices[0].message.content or ""
        start = txt.find("{")
        end   = txt.rfind("}")
        if start >= 0 and end > start:
            js = json.loads(txt[start:end+1])
            title = str(js.get("title", "")).strip()
            kws   = [str(k).strip() for k in (js.get("keywords") or []) if str(k).strip()]
            sps   = [str(s).strip() for s in (js.get("samples") or []) if str(s).strip()]
            return {
                "title": title[:120],
                "keywords": kws[:top_k],
                "samples": sps[:3],
            }
    except Exception:
        return None
    return None
