"""Deterministic rating and underwriting rules (pure code, Decimal arithmetic).

premium = max(minimum, sum(revenue x share x rate / 1000) x experience x claims x safety credit)
The LLM never computes a number: it only classifies the business and extracts facts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

CENT = Decimal("0.01")


@dataclass
class Category:
    code: str
    name: str
    rate: Decimal  # USD per 1,000 of revenue
    hazard: str
    status: str  # eligible | excluded
    description: str


def load_categories(path: Path) -> dict[str, Category]:
    with path.open() as fh:
        return {r["code"]: Category(r["code"], r["name"], Decimal(r["rate_per_1000"]), r["hazard"], r["status"],
                                    r["description"]) for r in csv.DictReader(fh)}


def load_rules(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, ROUND_HALF_UP)


def experience_factor(rules: dict, years: float) -> Decimal:
    for row in rules["experience"]:
        if "max_years" not in row or years <= row["max_years"]:
            return Decimal(str(row["factor"]))
    raise ValueError("experience table has no default row")


def claims_factor(rules: dict, claims: int) -> Decimal | None:
    """None means more claims than the table covers: not rateable, refer to an underwriter."""
    for row in rules["claims"]:
        if claims <= row["max_claims"]:
            return Decimal(str(row["factor"]))
    return None


def rate_quote(categories: dict[str, Category], rules: dict, revenue: float, allocations: dict[str, float],
               years: float, claims: int, safety_program: bool) -> dict:
    """allocations: category code -> share of revenue in percent (must total 100)."""
    if abs(sum(allocations.values()) - 100) > 0.01:
        raise ValueError(f"shares total {sum(allocations.values())}, not 100")
    rev = Decimal(str(revenue))
    lines, base = [], Decimal(0)
    for code, share in allocations.items():
        cat = categories[code]
        portion = rev * Decimal(str(share)) / 100
        premium = portion / 1000 * cat.rate
        base += premium
        lines.append({"code": code, "name": cat.name, "share_pct": share, "revenue_usd": float(money(portion)),
                      "rate_per_1000": float(cat.rate), "premium_usd": float(money(premium))})
    factors = {"experience": experience_factor(rules, years), "claims": claims_factor(rules, claims) or Decimal(1),
               "safety_program": Decimal(str(rules["safety_program_factor"])) if safety_program else Decimal(1)}
    modified = base
    for f in factors.values():
        modified *= f
    minimum = Decimal(str(rules["minimum_premium_usd"]))
    total = max(modified, minimum)
    return {"lines": lines, "base_premium_usd": float(money(base)), "factors": {k: float(v) for k, v in factors.items()},
            "modified_premium_usd": float(money(modified)), "minimum_applied": modified < minimum,
            "total_premium_usd": float(money(total))}


def underwrite(categories: dict[str, Category], rules: dict, facts: dict, allocations: dict[str, float],
               injection: list[str], classification_ok: bool) -> dict:
    """Decision: needs_info | decline | refer | quote, with every rule that fired."""
    missing = [f for f in rules["required_facts"] if facts.get(f) is None]
    if missing:
        return {"decision": "needs_info", "reasons": [f"missing: {', '.join(missing)}"], "missing": missing}
    excluded = [categories[c].name for c in allocations if categories[c].status == "excluded"]
    if excluded:
        return {"decision": "decline", "reasons": [f"excluded class of business: {', '.join(excluded)}"], "missing": []}
    ref, reasons = rules["referral"], []
    if facts["annual_revenue_usd"] > ref["revenue_over_usd"]:
        reasons.append(f"revenue above USD {ref['revenue_over_usd']:,}")
    if facts["employees"] > ref["employees_over"]:
        reasons.append(f"more than {ref['employees_over']} employees")
    if facts["claims_last_5y"] > ref["claims_over"]:
        reasons.append(f"more than {ref['claims_over']} claims in five years")
    high = sum(s for c, s in allocations.items() if categories[c].hazard == "high")
    if high >= ref["high_hazard_share_at_least_pct"]:
        reasons.append(f"{high:g}% of revenue is high-hazard work")
    if injection:
        reasons.append("the submission contains instructions aimed at the system; they were ignored")
    if not classification_ok:
        reasons.append("the classification could not be verified")
    return {"decision": "refer" if reasons else "quote", "reasons": reasons, "missing": []}
