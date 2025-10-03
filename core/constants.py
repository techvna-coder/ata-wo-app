NONDEFECT_PATTERNS = [
r"\b(clean(ing)?|lubrication|servic(e|ing)|first aid kit|tyre wear|tire wear|scheduled (check|maintenance)|software (load|upgrade|update)|cabin tidy|interior clean|disinfection|water service|lavatory service)\b",
r"\b(gse|ground support|paint touch(-|\s)?up|refuel(l)?ing|defuel(l)?ing|galley restock)\b",
]
DEFECT_KEYWORDS = [
r"\b(fault|fail(ure)?|leak|smoke|burn|overheat|vibration|jam|stuck|noise|abnorm(al)?|warning|caution|advisory|ecam|eicas|cas|msg|cb tripped?)\b"
]
ATA_PATTERN = r"^(?P<aa>\d{2})-(?P<bb>\d{2})(?:-(?P<cc>\d{2}))?(?:-(?P<rest>\d{2,}))?$"


CONF_MATRIX = {
"E0_E1_E2_VALID": 0.97,
"E1_E2_NEQ_E0_VALID": 0.95,
"E2_EQ_E0_ONLY": 0.86,
"E1_ONLY_VALID": 0.92,
"CONFLICT": 0.60,
"E0_ONLY": 0.50,
}


TOP_K_TFIDF = 3
MIN_SCORE_CONFIRM = 0.35
