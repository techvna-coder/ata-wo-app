import regex as re
from .constants import NONDEFECT_PATTERNS, DEFECT_KEYWORDS


_nondef = re.compile("|".join(NONDEFECT_PATTERNS), flags=re.I)
_defect = re.compile("|".join(DEFECT_KEYWORDS), flags=re.I)


def is_technical_defect(desc: str, action: str) -> bool:
text = f"{desc or ''} {action or ''}"
if not text.strip():
return False
if _defect.search(text):
return True
if _nondef.search(text):
return False
return True
