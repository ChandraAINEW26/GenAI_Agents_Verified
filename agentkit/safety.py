"""Prompt-injection detection shared by the agents that read untrusted text.

Detection is a tripwire, not a defence on its own: every agent also wraps untrusted
text in tags and tells the model it is data, and keeps decisions that matter
(prices, refunds, routing) in code the text cannot reach.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|prompts|rules)",
    r"disregard (?:the |your )?(?:system|previous|above)",
    r"you are now (?:in )?(?:admin|developer|god|dan) mode",
    r"(?:reveal|print|show) (?:your |the )?(?:system prompt|instructions|hidden rules)",
    r"</?(?:system|assistant|instructions?)>",
    r"act as (?:the )?(?:system|administrator|support manager) and approve",
    r"note to (?:the )?(?:system|ai|assistant|model|underwriter)\b",
    r"(?:apply|approve|confirm) (?:it|this|the discount)(?: and|,) (?:confirm|bind|approve)",
    r"(?:pre-?approved|already approved) (?:a |an )?\d+ ?% (?:discount|refund)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)


def find_injection(text: str) -> list[str]:
    return [m.group(0) for m in INJECTION_RE.finditer(text)]
