import re


CIT_RE = re.compile(r"\bDOC:[A-Za-z0-9_\-]+\.c\d+\b")

def extract_citations_from_text(text):
    return set(CIT_RE.findall(text or ""))

def validate_no_hallucinated_citations(answer_text, allowed):
    found = extract_citations_from_text(answer_text)
    invalid = sorted([c for c in found if c not in allowed])
    return invalid