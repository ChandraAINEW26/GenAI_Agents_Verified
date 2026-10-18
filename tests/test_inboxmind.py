"""InboxMind tests - offline, using FakeLLM. Run: uv run pytest -q"""

from pathlib import Path

import pytest

from agentkit import FakeLLM
from inboxmind.agent import (build_agent, check_draft, learn_from_feedback, load_sample, preview_rule, process,
                             times_in, worth_remembering)
from inboxmind.memory import Memory, rule_matches

SAMPLE = Path(__file__).parents[1] / "inboxmind" / "sample"
PROFILE, CALENDAR, INBOX = load_sample(SAMPLE)
EMAILS = {e["id"]: e for e in INBOX}


@pytest.fixture
def memory(tmp_path):
    return Memory(tmp_path / "m.sqlite", "jordan")


def test_rule_matching_needs_every_condition():
    ci = EMAILS["E-03"]
    assert rule_matches({"subject_contains": "build passed"}, ci)
    assert rule_matches({"sender_domain": "northwind.dev", "subject_contains": "Build passed"}, ci)
    assert not rule_matches({"sender_domain": "northwind.dev", "subject_contains": "FAILED"}, ci)
    assert not rule_matches({}, ci)  # an empty condition never matches everything
    assert rule_matches({"sender_domain": "kestrelpharma.com"}, EMAILS["T-05"])
    assert not rule_matches({"sender_domain": "pharma.com"}, EMAILS["T-05"])  # not a suffix trick


def test_most_specific_rule_wins_and_retired_rules_stop_applying(memory):
    broad = memory.add_rule("triage", {"sender_domain": "northwind.dev"}, "ignore", "ignore CI", "E-03")
    memory.add_rule("triage", {"sender_domain": "northwind.dev", "subject_contains": "FAILED"}, "notify",
                    "failed builds matter", "T-04")
    assert memory.matching_triage_rule(EMAILS["T-04"]).action == "notify"
    assert memory.matching_triage_rule(EMAILS["T-03"]).id == broad
    memory.retire_rule(broad)
    assert memory.matching_triage_rule(EMAILS["T-03"]) is None
    assert [r.status for r in memory.rules(include_retired=True)] == ["retired", "active"]  # history is kept


def test_similar_episodes_are_retrieved_for_few_shot(memory):
    for eid in ("E-01", "E-03"):
        memory.add_episode(EMAILS[eid], EMAILS[eid]["label"], EMAILS[eid]["feedback"])
    hits = memory.similar_episodes(EMAILS["T-03"])
    assert hits[0]["email_id"] == "E-03" and hits[0]["label"] == "ignore"


def test_times_and_draft_checks():
    assert times_in("Let's do 2pm or 9:30 AM or 16:30") == {"14:00", "09:30", "16:30"}
    email = EMAILS["E-05"]  # asks for Tuesday October 21
    slots = {"2026-10-21": ["09:00", "14:00", "16:30"]}
    assert check_draft("I can do 2pm on Tuesday.", email, slots, []) == []
    assert "not free" in check_draft("How about 11:00 on Tuesday?", email, slots, [])[0]
    assert "figures" in check_draft("I will send 45 pages by October 30.", email, slots, [])[0]


def test_fact_quality_filter():
    email = EMAILS["T-05"]
    assert worth_remembering("I have taken over as warehouse lead in Basel", email)
    assert not worth_remembering("From: AI Weekly", email)
    assert not worth_remembering(email["subject"], email)
    assert not worth_remembering("Basel lead", email)


def make_llm(label_for=lambda p: "notify", proposal=None, facts=None):
    return FakeLLM({
        "TriageDecision": lambda s, p: {"reasoning": "r", "label": label_for(p), "confidence": "high"},
        "Facts": lambda s, p: {"facts": facts or []},
        "DraftPlan": lambda s, p: {"meeting_requested": True, "dates": ["2026-10-21"]},
        "DraftEmail": lambda s, p: {"subject": "Re: x", "body": "I can do 2pm or 11:00."},
        "RuleProposal": lambda s, p: proposal or {"kind": "none", "condition": {}, "text": ""},
    })


def test_learned_rule_is_applied_by_code_before_the_llm(memory):
    llm = make_llm(proposal={"kind": "triage", "condition": {"subject_contains": "Build passed"},
                             "text": "Ignore successful builds"})
    result = learn_from_feedback(llm, memory, PROFILE, EMAILS["E-03"], "ignore", "noise", EMAILS,
                                 approve=lambda *a: True)
    assert result["rule_id"] == 1
    agent = build_agent(llm, memory, PROFILE, CALENDAR)
    state = process(agent, EMAILS["T-03"])
    assert state["label"] == "ignore" and state["rule_id"] == 1
    assert not any(name == "TriageDecision" for name, _ in llm.calls)  # guaranteed, not merely suggested


def test_rule_needs_approval_and_must_match_its_own_email(memory):
    proposal = {"kind": "triage", "condition": {"subject_contains": "Build passed"}, "text": "Ignore builds"}
    declined = learn_from_feedback(make_llm(proposal=proposal), memory, PROFILE, EMAILS["E-03"], "ignore", "n",
                                   EMAILS, approve=lambda *a: False)
    assert declined["rule_id"] is None and memory.rules() == []
    assert len(memory.similar_episodes(EMAILS["E-03"])) == 1  # the example is still kept

    wrong = {"kind": "triage", "condition": {"sender_domain": "other.com"}, "text": "bad rule"}
    result = learn_from_feedback(make_llm(proposal=wrong), memory, PROFILE, EMAILS["E-03"], "ignore", "n", EMAILS,
                                 approve=lambda *a: True)
    assert "does not match" in result["rejected"] and memory.rules() == []


def test_rule_preview_lists_the_examples_it_would_change():
    episodes = [{"email_id": "E-04", "subject": "FYI: Q4 volume forecast", "label": "notify"},
                {"email_id": "E-05", "subject": "Board pack", "label": "respond"}]
    changes = preview_rule({"sender_domain": "kestrelpharma.com"}, "respond", episodes, EMAILS)
    assert changes == ["E-04 (FYI: Q4 volume forecast): notify → respond"]


def test_facts_are_stored_only_when_the_email_says_them(memory):
    llm = make_llm(label_for=lambda p: "respond", facts=[
        {"subject": "Marta Keller", "fact": "I have taken over as warehouse lead in Basel"},
        {"subject": "Marta Keller", "fact": "She earns 90000 dollars a year and lives in Zurich"}])  # invented
    state = process(build_agent(llm, memory, PROFILE, CALENDAR), EMAILS["T-05"])
    assert state["facts_learned"] == ["Marta Keller: I have taken over as warehouse lead in Basel"]
    assert [f["fact"] for f in memory.facts_about("marta")] == ["I have taken over as warehouse lead in Basel"]


def test_draft_with_a_busy_time_is_flagged_not_sent(memory):
    llm = make_llm(label_for=lambda p: "respond")
    state = process(build_agent(llm, memory, PROFILE, CALENDAR), EMAILS["E-05"])
    assert state["slots"] == {"2026-10-21": ["09:00", "14:00", "16:30"]}
    assert any("11:00" in p for p in state["draft_problems"])
