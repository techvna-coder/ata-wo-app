import os
from typing import Optional
from openai import OpenAI

def summarize_evidence(defect_text: str, rect_text: str, ata04: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""You are an aviation maintenance reliability assistant.
Given:
- Defect: {defect_text}
- Rectification: {rect_text}
- Proposed ATA04: {ata04}
Write a concise, evidence-style justification (1-2 sentences) referencing symptoms/components typical of this ATA04."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
