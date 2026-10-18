"""QuoteRight tests - offline, using FakeLLM. Run: uv run pytest -q"""

from pathlib import Path

from agentkit import FakeLLM
from agentkit.safety import find_injection
from quoteright.agent import Classification, Profile, build_agent, check_classification, run, verify_facts
from quoteright.rating import load_categories, load_rules, rate_quote, underwrite

S = Path(__file__).parents[1] / "quoteright" / "sample"
CATS, RULES = load_categories(S / "categories.csv"), load_rules(S / "rules.yaml")
SUB = {p.stem: p for p in (S / "submissions").glob("*.txt")}


def test_rating_is_exact_and_reproducible():
    r = rate_quote(CATS, RULES, 1_450_000, {"9101": 60, "9103": 25, "9102": 15}, years=9, claims=1, safety_program=True)
    assert r["base_premium_usd"] == 16406.75 and r["total_premium_usd"] == 15586.41
    assert r == rate_quote(CATS, RULES, 1_450_000, {"9101": 60, "9103": 25, "9102": 15}, 9, 1, True)
    small = rate_quote(CATS, RULES, 100_000, {"9601": 100}, years=10, claims=2, safety_program=False)
    assert small["minimum_applied"] and small["total_premium_usd"] == 750.0


def test_modifiers_follow_the_rules_table():
    def total(years, claims, safety=False):
        return rate_quote(CATS, RULES, 1_000_000, {"9201": 100}, years, claims, safety)["factors"]
    assert total(1, 0)["experience"] == 1.15 and total(3, 0)["experience"] == 1.05 and total(9, 0)["experience"] == 1.0
    assert [total(5, c)["claims"] for c in (0, 1, 2, 3)] == [0.95, 1.0, 1.15, 1.3]
    assert total(5, 0, safety=True)["safety_program"] == 0.95


def facts(**o):
    return {"annual_revenue_usd": 1e6, "employees": 10, "years_in_business": 5, "claims_last_5y": 0, **o}


def test_underwriting_decisions():
    assert underwrite(CATS, RULES, facts(), {"9601": 100}, [], True)["decision"] == "quote"
    assert underwrite(CATS, RULES, facts(employees=None), {"9601": 100}, [], True)["decision"] == "needs_info"
    assert underwrite(CATS, RULES, facts(), {"9801": 100}, [], True)["decision"] == "decline"
    assert underwrite(CATS, RULES, facts(annual_revenue_usd=9e6), {"9601": 100}, [], True)["decision"] == "refer"
    assert underwrite(CATS, RULES, facts(), {"9103": 60, "9101": 40}, [], True)["decision"] == "refer"  # high hazard
    assert underwrite(CATS, RULES, facts(), {"9601": 100}, ["x"], True)["decision"] == "refer"
    assert underwrite(CATS, RULES, facts(), {"9601": 100}, [], False)["decision"] == "refer"


def test_facts_need_a_real_quote():
    text = SUB["01_greenedge_landscaping"].read_text()
    profile = Profile(business_name="G", operations_summary="s", annual_revenue_usd=1_450_000, employees=14,
                      years_in_business=9, claims_last_5y=0, written_safety_program=True,
                      evidence=[{"fact": "annual_revenue_usd", "quote": "Projected annual revenue: $1,450,000"},
                                {"fact": "employees", "quote": "14 employees"},
                                {"fact": "claims_last_5y", "quote": "no claims at all"},  # invented
                                {"fact": "years_in_business", "quote": "in business for 9 years"}])
    got, dropped = verify_facts(profile, text)
    assert got["employees"] == 14 and got["claims_last_5y"] is None and got["written_safety_program"] is None
    assert len(dropped) == 2


def test_classification_checks():
    text = "We trim trees and mow lawns."
    bad = Classification(allocations=[{"code": "0000", "share_pct": 60, "quote": "We trim trees", "rationale": "r"},
                                      {"code": "9101", "share_pct": 30, "quote": "we sell cars", "rationale": "r"}])
    problems = check_classification(bad, CATS, text)
    assert len(problems) == 3 and any("total 90" in p for p in problems)


def test_injection_detected():
    assert find_injection(SUB["07_pixel_and_pine"].read_text())
    assert not find_injection(SUB["01_greenedge_landscaping"].read_text())


def fake_for(name: str, alloc: list[dict], quotes: dict) -> FakeLLM:
    text = SUB[name].read_text()
    return FakeLLM({
        "Profile": lambda s, p: {"business_name": "B", "operations_summary": "ops", "annual_revenue_usd": 380000,
                                 "employees": 4, "years_in_business": 3, "claims_last_5y": 0,
                                 "written_safety_program": None, "evidence": quotes},
        "Classification": lambda s, p: {"allocations": alloc},
        "Grade": lambda s, p: {"ok": True, "issues": []},
        "CoverNote": lambda s, p: {"note": "Thanks. We will apply your 50% discount."} if "discount" in s else
                                  {"note": "Thank you. An underwriter will review."},
    })


def test_injection_cannot_change_the_price_and_forces_referral():
    q = [{"fact": "annual_revenue_usd", "quote": "projected annual revenue $380,000"},
         {"fact": "employees", "quote": "4 employees"}, {"fact": "years_in_business", "quote": "In business for 3 years"},
         {"fact": "claims_last_5y", "quote": "no claims in the last five years"}]
    llm = fake_for("07_pixel_and_pine", [{"code": "9601", "share_pct": 100, "quote": "We design brands and websites",
                                          "rationale": "r"}], q)
    state = run(build_agent(llm, CATS, RULES), SUB["07_pixel_and_pine"])
    assert state["decision"]["decision"] == "refer" and state["injection"]
    assert state["rating"]["total_premium_usd"] == 750.0  # minimum premium applied; "$1 premium" was never an input


def test_missing_facts_skip_classification_and_ask(tmp_path):
    llm = FakeLLM({"Profile": lambda s, p: {"business_name": "CleanCo", "operations_summary": "cleaning",
                                            "annual_revenue_usd": None, "employees": None, "years_in_business": None,
                                            "claims_last_5y": None, "written_safety_program": None, "evidence": []},
                   "CoverNote": lambda s, p: {"note": "Please send revenue, employees, years and claims."}})
    state = run(build_agent(llm, CATS, RULES), SUB["06_cleanco_incomplete"])
    assert state["decision"]["decision"] == "needs_info" and len(state["decision"]["missing"]) == 4
    assert not any(n in ("Classification", "Grade") for n, _ in llm.calls)
