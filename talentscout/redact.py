"""Blind screening: strip identity and protected attributes before any LLM sees a CV.

Pure code, no LLM, so it is predictable and testable. Removes:
- the candidate's name (heading line and every later mention of name parts)
- emails, phone numbers, URLs / profile links, photo references
- labelled personal fields: date of birth, age, gender, marital status,
  nationality, religion, address
- gendered pronouns (replaced with they/them/their)

Returns the redacted text plus the categories that were removed, so the
report can show what the scorer did *not* see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PERSONAL_FIELDS = r"date of birth|dob|age|gender|sex|marital status|nationality|religion|address|photo"
FIELD_RE = re.compile(rf"\b(?:{PERSONAL_FIELDS})\s*:\s*[^·|\n]+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\(?\+\d{1,3}\)?[\s-]?)\d[\d\s-]{6,}\d")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|in|org|net|io)/\S+", re.I)
PIN_RE = re.compile(r"\b\d{6}\b")
# Subject pronouns become "the candidate" so the verb that follows stays grammatical.
PRONOUNS = {
    "he": "the candidate", "she": "the candidate", "him": "them", "his": "their", "her": "their",
    "hers": "theirs", "himself": "themself", "herself": "themself",
}
PRONOUN_RE = re.compile(r"\b(" + "|".join(PRONOUNS) + r")\b", re.I)


@dataclass
class Redacted:
    text: str
    removed: list[str] = field(default_factory=list)


def _pronoun(match: re.Match) -> str:
    word = match.group(0)
    replacement = PRONOUNS[word.lower()]
    return replacement.capitalize() if word[0].isupper() else replacement


def redact_cv(text: str) -> Redacted:
    lines = text.strip().splitlines()
    removed: set[str] = set()

    # The first heading is the candidate's name.
    name_parts: list[str] = []
    if lines and lines[0].startswith("#"):
        name = lines[0].lstrip("# ").strip()
        name_parts = [p for p in re.split(r"\s+", name) if len(p) > 1]
        lines = lines[1:]
        removed.add("name")

    # Everything between the name and the first section heading is the contact
    # block (email, phone, address, photo...). Drop it whole: unlabelled street
    # addresses cannot be caught reliably line by line.
    first_section = next((i for i, line in enumerate(lines) if line.startswith("#")), 0)
    for line in lines[:first_section]:
        for label, pattern in (("email", EMAIL_RE), ("profile link", URL_RE), ("phone", PHONE_RE)):
            if pattern.search(line):
                removed.add(label)
        removed.update(m.group(0).split(":")[0].strip().lower() for m in FIELD_RE.finditer(line))
        if line.strip():
            removed.add("contact block")
    lines = lines[first_section:]

    out: list[str] = []
    for line in lines:
        for label, pattern in (("email", EMAIL_RE), ("profile link", URL_RE), ("phone", PHONE_RE)):
            if pattern.search(line):
                line = pattern.sub("", line)
                removed.add(label)
        for match in FIELD_RE.finditer(line):
            removed.add(match.group(0).split(":")[0].strip().lower())
        line = FIELD_RE.sub("", line)
        if PIN_RE.search(line):
            line = PIN_RE.sub("", line)
            removed.add("postcode")
        for part in name_parts:
            if re.search(rf"\b{re.escape(part)}\b", line):
                line = re.sub(rf"\b{re.escape(part)}\b", "[candidate]", line)
                removed.add("name")
        out.append(PRONOUN_RE.sub(_pronoun, line).rstrip(" ·|,"))

    return Redacted(text="\n".join(out).strip(), removed=sorted(removed))


INCLUSIVE_LANGUAGE = {
    "rockstar": "skilled engineer", "ninja": "expert", "guru": "expert", "young": "(remove)",
    "digital native": "comfortable with modern tools", "manpower": "staffing", "he": "they", "she": "they",
    "his": "their", "aggressive": "ambitious", "dominant": "leading", "culture fit": "values alignment",
    "native english": "fluent English", "recent graduate": "early-career",
}


def lint_job_description(text: str) -> list[str]:
    """Flag wording known to put off qualified applicants (deterministic check)."""
    return [f"'{term}' → consider '{alt}'" for term, alt in INCLUSIVE_LANGUAGE.items()
            if re.search(rf"\b{re.escape(term)}\b", text, re.I)]
