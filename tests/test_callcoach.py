"""CallCoach tests - offline, using FakeLLM. Run: uv run pytest -q"""

import json
from pathlib import Path

from agentkit import FakeLLM
from callcoach.pipeline import analyse_call, quote_in_turn, scorecard, unsupported_figures
from callcoach.transcript import conversation_metrics, load_call, parse_transcript

CALLS = Path(__file__).parents[1] / "callcoach" / "sample_calls"


def test_parse_and_metrics():
    call = parse_transcript("---\nrep: Ann Lee\nspeakers:\n  Ann: rep\n---\n"
                            "[00:00] Ann: Hi, what is your biggest problem?\n"
                            "[00:10] Bob: Returns.\n"
                            "[00:12] Ann: How much do they cost? And how often?\n")
    assert [t.role for t in call.turns] == ["rep", "customer", "rep"]
    m = conversation_metrics(call)
    assert m["rep_questions"] == 3
    assert m["longest_rep_monologue_turn"] == "T01"
    assert 0.8 < m["rep_talk_ratio"] < 1


def test_quote_must_be_in_the_cited_turn():
    call = load_call(CALLS / "halden_renewal.txt")
    assert quote_in_turn(call, "T04", "We also haven't seen the service credit on our invoice yet.")
    assert not quote_in_turn(call, "T02", "We also haven't seen the service credit")  # wrong turn
    assert not quote_in_turn(call, "T99", "anything")


def test_scorecard_rewards_balance_and_dated_next_steps():
    metrics = {"rep_questions": 6, "rep_talk_ratio": 0.5, "longest_rep_monologue_seconds": 20}
    qual = {k: {"status": "confirmed"} for k in "abcdefgh"}
    steps = [{"owner": "rep", "due": "Friday", "verified": True}, {"owner": "customer", "due": "", "verified": True}]
    assert scorecard(metrics, [], qual, steps)["total"] == 100
    talky = {**metrics, "rep_talk_ratio": 0.8, "longest_rep_monologue_seconds": 90}
    assert scorecard(talky, [], qual, steps)["talk_balance"] == 0


def test_unsupported_figures_in_email():
    call = load_call(CALLS / "halden_renewal.txt")
    assert unsupported_figures("Sorry about the six hours of downtime; 4.2.1 fixed it.", call) == []
    assert unsupported_figures("We now offer a 3 hour SLA from Oct 23.", call) == ["3 hour", "oct 23"]
    assert unsupported_figures("I will:\n1. Check the credit.\n2) Review response times.", call) == []

    sunpeak = load_call(CALLS / "sunpeak_expansion.txt")  # says "nine hundred thousand", "five hundred thousand"
    assert unsupported_figures("Beat $900k a season; approvals over 500 thousand; 18 percent", sunpeak) == []
    assert unsupported_figures("Savings of $1.2 million", sunpeak) == ["1200000"]
    marlow = load_call(CALLS / "marlow_discovery.txt")  # says "one point two million dollars"
    assert unsupported_figures("Turnover costs you $1.2 million a year.", marlow) == []


def test_pipeline_drops_findings_whose_quotes_are_invented(tmp_path):
    llm = FakeLLM({
        "Sentiment": lambda s, p: {"trend": "declining", "segments": [
            {"start_turn": "T01", "end_turn": "T09", "score": -1, "label": "frustrated", "turn_id": "T02",
             "quote": "Six hours of downtime on a night shift"}]},
        "Objections": lambda s, p: {"objections": [
            {"objection": "missing credit", "category": "price", "turn_id": "T04",
             "quote": "We also haven't seen the service credit on our invoice yet.", "response_turn_id": "T05",
             "handling": "poorly", "better_response": "confirm the credit"},
            {"objection": "invented", "category": "price", "turn_id": "T04", "quote": "Your price is far too high",
             "response_turn_id": "", "handling": "not_addressed", "better_response": "-"}]},
        "Qualification": lambda s, p: {name: {"status": "unknown", "detail": "", "turn_id": "", "quote": ""}
                                       for name in ["metrics", "economic_buyer", "decision_criteria",
                                                    "decision_process", "paper_process", "identify_pain",
                                                    "champion", "competition"]},
        "Commitments": lambda s, p: {"next_steps": [
            {"owner": "rep", "action": "check credit", "due": "", "turn_id": "T05",
             "quote": "I'll have to check on that with finance."}],
            "competitors": [{"name": "Kinexa", "context": "4h SLA", "turn_id": "T06",
                             "quote": "Kinexa has been in to see us twice."}]},
        "Synthesis": lambda s, p: {"summary": "Risky renewal", "went_well": [], "improve": ["ask questions"],
                                   "deal_risk": "high", "risk_reasons": [], "recommended_stage": "Renewal",
                                   "follow_up_email": {"subject": "Next steps", "body": "Credit by Oct 23."}},
    })
    state = analyse_call(llm, CALLS / "halden_renewal.txt", tmp_path)

    assert len(state["unverified"]) == 1 and "invented" in state["unverified"][0]
    synthesis_prompt = next(p for name, p in llm.calls if name == "Synthesis")
    assert "far too high" not in synthesis_prompt  # dropped before the coach saw it
    crm = json.loads(Path(state["outputs"]["crm"]).read_text())
    assert crm["competitors"] == ["Kinexa"] and "no dated next step" in crm["risk_flags"]
    assert "Check before sending" in Path(state["outputs"]["email"]).read_text()
