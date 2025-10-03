import regex as re
from .constants import ATA_PATTERN


REF_PATTERN = re.compile(
r"\b(?P<manual>TSM|FIM|AMM)\s*[-:]?\s*(?P<seq>(\d{2}[- ]?\d{2}([- ]?\d{2})?([- ]?\d{2,})?|\d{6,}))\b",
flags=re.I,
)


def _normalize_seq(seq: str) -> str:
digits = re.sub(r"[^\d]", "", seq)
if len(digits) >= 4:
aa, bb = digits[:2], digits[2:4]
rest = digits[4:]
parts = [aa, bb]
while len(rest) >= 2:
parts.append(rest[:2])
rest = rest[2:]
return "-".join(parts)
return seq


def extract_citations(text: str):
out = []
if not text:
return out
for m in REF_PATTERN.finditer(text):
manual = m.group("manual").upper()
raw = m.group("seq")
normalized = _normalize_seq(raw)
ata04 = None
task = normalized
m2 = re.match(ATA_PATTERN, normalized)
if m2:
aa, bb = m2.group("aa"), m2.group("bb")
ata04 = f"{aa}-{bb}"
out.append({
"manual": manual,
"raw": raw,
"normalized": normalized,
"ata04": ata04,
"task": task,
})
return out
