"""Numbers, amounts and dates in generated text, compared with what the sources said.

Used to catch an LLM draft that states a figure nobody gave it: "a 2-hour SLA by
Oct 23" when the call never mentioned either, or a premium the calculation never
produced. Figures are normalised so "nine hundred thousand", "900k" and
"$900,000" match, and a number is compared together with its unit, so "3 hour"
is not supported by a "4.3" elsewhere in the source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december"]
# Full month names, or 3-letter abbreviations with an optional dot ("May." is a sentence end, not a date).
MONTH_RE = "|".join(MONTH_NAMES + [m[:3] + r"\.?" for m in MONTH_NAMES if m != "may"])
FIGURE_RE = re.compile(
    rf"\b(?:{MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?\b"  # Oct 23, November 2nd
    r"|\b\d[\d,.]*(?:[\s-]*(?:%|percent|hours?|days?|weeks?|months?|k\b|million|thousand|dollars?))?",
    re.I,
)
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
# A spoken number starts with a unit or tens word ("five hundred thousand"); a bare scale word after
# digits ("500 thousand") is left for FIGURE_RE to read as a unit.
_START = "|".join([*UNITS, *TENS])
_ANY = "|".join([*UNITS, *TENS, *SCALES])
NUMBER_PHRASE = re.compile(rf"\b(?:{_START})\b(?:[\s-]+(?:{_ANY})\b)*")


def _words_to_number(phrase: str) -> str:
    total = current = 0
    for word in re.split(r"[\s-]+", phrase.strip()):
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in SCALES:
            total += max(current, 1) * SCALES[word]
            current = 0
    return str(total + current)


def figures(text: str) -> set[str]:
    """Normalised figures: '6 hour', '18 %', 'oct 23', '900000', or a version like '4.2.1'."""
    text = re.sub(r"[‐-―−]", "-", re.sub(r"[  ]", " ", text.lower()))  # unicode dashes/spaces
    text = re.sub(rf"\b({'|'.join(UNITS)})\s+point\s+({'|'.join(UNITS)})\b",  # "one point two" -> 1.2
                  lambda m: f"{UNITS[m.group(1)]}.{UNITS[m.group(2)]}", text)
    text = NUMBER_PHRASE.sub(lambda m: _words_to_number(m.group(0)) + " ", text)
    found = set()
    for match in FIGURE_RE.finditer(text):
        figure = re.sub(r"[\s-]+", " ", match.group(0)).strip(" .,").replace("percent", "%")
        figure = re.sub(r"(?<=\d),(?=\d{3})", "", figure)  # 900,000 -> 900000
        figure = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", figure)
        figure = re.sub(r"^([a-z]{3})[a-z]*\.?", r"\1", figure)  # month names -> 3 letters
        scale = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|thousand|million)", figure)
        if scale:  # 900k, 900 thousand and 900000 are the same figure
            factor = 1_000_000 if scale.group(2) == "million" else 1_000
            figure = f"{float(scale.group(1)) * factor:.0f}"
        figure = re.sub(r"\s*dollars?$", "", figure) if figure[0].isdigit() else figure
        found.add(re.sub(r"s$", "", figure.replace(" %", "%")) if not figure[-1].isdigit() else figure)
    return found


def figures_not_in(text: str, sources: Iterable[str], ignore_bare_below: int = 0) -> list[str]:
    """Figures in `text` that do not appear in any of `sources` (sorted, normalised).

    `ignore_bare_below` skips unit-less whole numbers smaller than it (step
    numbers, "the last four digits"); amounts, durations, percentages and dates
    are always checked.
    """
    text = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", "", text)  # numbered-list markers are not figures
    known: set[str] = set()
    for source in sources:
        known |= figures(source)

    def stated(figure: str) -> bool:
        if figure in known:
            return True
        # "Friday the 10th" in a source supports "Oct 10" in the text.
        date = re.fullmatch(r"[a-z]{3} (\d{1,2})", figure)
        return bool(date) and date.group(1) in known

    def small_bare(figure: str) -> bool:
        return figure.isdigit() and int(figure) < ignore_bare_below

    return sorted(f for f in figures(text) if not stated(f) and not small_bare(f))
