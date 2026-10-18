"""Input guard: runs on every ticket before any LLM sees it (pure code, unit tested).

- Payment card numbers (Luhn-checked, so order numbers are left alone), IBANs and
  "password: ..." values are replaced with placeholders.
- Prompt-injection attempts ("ignore previous instructions", "you are now in
  admin mode", fake system tags) are detected; the ticket is still answered by a
  human, never by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentkit.safety import find_injection

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
SECRET_RE = re.compile(r"\b(password|passcode|pin|api[_ -]?key|token)\s*(?:is|:|=)\s*\S+", re.I)
CVV_RE = re.compile(r"\b(cvv|cvc|security code)\s*(?:is|:)?\s*\d{3,4}\b", re.I)
EXPIRY_RE = re.compile(r"\b(?:expiry|expires|exp\.?)\s*(?:date)?\s*:?\s*\d{2}\s*/\s*\d{2,4}\b", re.I)

@dataclass
class Guarded:
    text: str
    redactions: list[str] = field(default_factory=list)
    injection: list[str] = field(default_factory=list)

def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0

def guard_ticket(text: str) -> Guarded:
    redactions: list[str] = []

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            redactions.append("payment card")
            return f"[card ending {digits[-4:]}]"
        return match.group(0)

    text = CARD_RE.sub(card, text)
    for label, pattern, replacement in (
        ("bank account (IBAN)", IBAN_RE, "[bank account]"),
        ("card security code", CVV_RE, "[security code]"),
        ("card expiry", EXPIRY_RE, "[expiry]"),
    ):
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            redactions.append(label)
    if SECRET_RE.search(text):
        text = SECRET_RE.sub(lambda m: f"{m.group(1)}: [removed]", text)
        redactions.append("secret")
    injection = find_injection(text)
    return Guarded(text=text, redactions=sorted(set(redactions)), injection=injection)
