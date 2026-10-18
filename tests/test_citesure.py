"""CiteSure tests - offline, using FakeLLM. Run: uv run pytest -q"""

import re
from pathlib import Path

import pytest

from agentkit import FakeLLM
from citesure.agent import ask, build_agent
from citesure.index import BM25, CorpusIndex, Passage, chunk_text, rrf, support_ratio

CORPUS = Path(__file__).parents[1] / "citesure" / "sample_corpus"


@pytest.fixture
def index(tmp_path):
    llm = FakeLLM({"SummarySchema": lambda s, p: {"summary": p[:300]}})
    return CorpusIndex.build(CORPUS, llm, cache_dir=tmp_path)


def passage_line(prompt: str, phrase: str) -> tuple[str, str]:
    """Find '[id] text' in a prompt whose text contains phrase."""
    for pid, text in re.findall(r"^\[([^\]]+)\] (.*?)(?=^\[|\n\n|\Z)", prompt, re.M | re.S):
        if phrase in text:
            return pid, text
    raise AssertionError(f"{phrase!r} not offered to the model")


def test_chunks_keep_their_heading():
    chunks = chunk_text("# Title\n\n## Kestrel Pharma\nKestrel is in Basel.\n\n## Halden\nHalden is in Oslo.")
    assert chunks == ["Kestrel Pharma\nKestrel is in Basel.", "Halden\nHalden is in Oslo."]


def test_bm25_and_rrf_rank_the_right_passage_first():
    passages = [Passage(id=str(i), doc="d", store="chunks", text=t) for i, t in
                enumerate(["firmware 4.2 memory leak froze the arms", "battery cells from a supplier", "renewal dates"])]
    bm25 = BM25(passages)
    assert bm25.search("firmware leak", 1)[0][0].id == "0"
    fused = rrf([[passages[1], passages[0]], [passages[0], passages[2]]])
    assert fused[0].id == "0"  # ranked in both lists


def test_support_ratio_rejects_invented_words():
    source = "Picking was down for 6 hours before arms were restarted manually."
    assert support_ratio("Picking was down for 6 hours", source) == 1.0
    assert support_ratio("Picking was down for 9 hours due to sabotage", source) < 0.7


def test_multi_hop_answer_with_citations_and_grounding(index):
    task_calls = []

    def decide(system, prompt):
        task_calls.append(prompt)
        if len(task_calls) == 1:
            return {"task": "Find the account manager of Kestrel Pharma", "tool": "chunks",
                    "search_queries": ["Kestrel Pharma account manager"]}
        return {"task": "Find who Ravi Menon reports to", "tool": "chunks", "search_queries": ["Ravi Menon reports to"]}

    def distill(system, prompt):
        if "Kestrel" in prompt.split("Passages:")[0]:
            pid, _ = passage_line(prompt, "Kestrel Pharma is a pharmaceutical")
            return {"extracts": [
                {"passage_id": pid, "text": "The account manager is Ravi Menon."},
                {"passage_id": pid, "text": "Kestrel pays USD 9 million penalties every quarter"},  # invented
            ]}
        pid, _ = passage_line(prompt, "He reports to Lena Fischer")
        return {"extracts": [{"passage_id": pid, "text": "He reports to Lena Fischer, Vice President of Sales for EMEA."}]}

    replans = iter([{"can_answer_now": False, "remaining_steps": ["Find who Ravi Menon reports to"]},
                    {"can_answer_now": True, "remaining_steps": []}])

    def final(system, prompt):
        ids = re.findall(r"^\[([^\]]+)\]", prompt.split("Evidence:")[1], re.M)
        return {"reasoning": "chain", "answer": "Ravi Menon, who reports to Lena Fischer.",
                "citations": [*ids, "made-up#c9"], "confidence": "high", "abstain": False}

    llm = FakeLLM({
        "Anonymized": lambda s, p: {"question": "Who does X's account manager report to?",
                                    "entities": [{"placeholder": "X", "value": "Kestrel Pharma"}]},
        "Plan": lambda s, p: {"steps": ["Find the account manager of X", "Find who that person reports to"]},
        "TaskDecision": decide,
        "Distilled": distill,
        "Replan": lambda s, p: next(replans),
        "FinalAnswer": final,
        "Grounding": lambda s, p: {"grounded": True, "unsupported_claims": []},
    })
    state = ask(build_agent(llm, index), "Who does Kestrel Pharma's account manager report to?")

    assert "Find the account manager of Kestrel Pharma" in task_calls[0]  # placeholder X replaced in code
    assert state["steps_taken"] == 2
    assert state["rejected_extracts"] == 1  # the invented extract never reached the context
    assert "made-up#c9" not in state["final"].citations
    assert {c.split("#")[0] for c in state["final"].citations} == {"customers", "team"}
    assert not state["final"].abstain


def test_unverifiable_answer_becomes_abstain(index):
    llm = FakeLLM({
        "Anonymized": lambda s, p: {"question": "What was revenue in 2025?", "entities": []},
        "Plan": lambda s, p: {"steps": ["Find 2025 revenue"]},
        "TaskDecision": lambda s, p: {"task": "Find 2025 revenue", "tool": "chunks", "search_queries": ["revenue 2025"]},
        "Distilled": lambda s, p: {"extracts": [{"passage_id": passage_line(p, "revenue")[0],
                                                 "text": "most revenue comes from multi-year contracts"}]},
        "Replan": lambda s, p: {"can_answer_now": True, "remaining_steps": []},
        "FinalAnswer": lambda s, p: {"reasoning": "", "answer": "Revenue was USD 40 million.",
                                     "citations": ["company_overview#c0"], "confidence": "high", "abstain": False},
        "Grounding": lambda s, p: {"grounded": False, "unsupported_claims": ["USD 40 million"]},
    })
    state = ask(build_agent(llm, index), "What was Northwind's revenue in 2025?")
    assert state["final"].abstain
    assert state["verify_attempts"] == 2
    assert sum(1 for name, _ in llm.calls if name == "FinalAnswer") == 2  # one rewrite, then give up


def test_extracts_from_the_same_passage_are_merged(index):
    """Regression: two facts from one passage used to overwrite each other, looping until the budget ran out."""
    tasks = iter([{"task": "Find the notice period", "tool": "chunks", "search_queries": ["Kestrel opt out notice"]},
                  {"task": "Find the renewal date", "tool": "chunks", "search_queries": ["Kestrel renews"]}])
    facts = iter(["Kestrel may opt out of automatic renewal by giving written notice at least 60 days before the "
                  "renewal date.", "Three-year contract, renews on 2026-12-15."])

    def distill(system, prompt):
        pid, _ = passage_line(prompt, "Kestrel may opt out")
        return {"extracts": [{"passage_id": pid, "text": next(facts)}]}

    replans = iter([{"can_answer_now": False, "remaining_steps": ["Find the renewal date"]},
                    {"can_answer_now": True, "remaining_steps": []}])
    seen = {}

    def final(system, prompt):
        seen["evidence"] = prompt
        return {"reasoning": "", "answer": "2026-10-16", "citations": ["contracts#c3"], "confidence": "high",
                "abstain": False}

    llm = FakeLLM({
        "Anonymized": lambda s, p: {"question": p, "entities": []},
        "Plan": lambda s, p: {"steps": ["Find the notice period", "Find the renewal date"]},
        "TaskDecision": lambda s, p: next(tasks),
        "Distilled": distill,
        "Replan": lambda s, p: next(replans),
        "FinalAnswer": final,
        "Grounding": lambda s, p: {"grounded": True, "unsupported_claims": []},
    })
    ask(build_agent(llm, index), "What is the last date Kestrel can give notice?")
    assert "60 days" in seen["evidence"] and "2026-12-15" in seen["evidence"]


def endless(tool: str) -> FakeLLM:
    return FakeLLM({
        "Anonymized": lambda s, p: {"question": "q", "entities": []},
        "Plan": lambda s, p: {"steps": ["look", "look again"]},
        "TaskDecision": lambda s, p: {"task": "look", "tool": tool, "search_queries": ["firmware"]},
        "Distilled": lambda s, p: {"extracts": []},
        "StepAnswer": lambda s, p: {"reasoning": "", "answer": "still missing", "citations": []},
        "Replan": lambda s, p: {"can_answer_now": False, "remaining_steps": ["look again"]},
        "FinalAnswer": lambda s, p: {"reasoning": "", "answer": "Not enough information.", "citations": [],
                                     "confidence": "low", "abstain": True},
    })


def test_repeated_lookups_with_no_new_evidence_stop_early(index):
    """Regression: the agent used to repeat a fruitless lookup until the step budget ran out."""
    state = ask(build_agent(endless("chunks"), index), "a question the corpus cannot answer", max_steps=6)
    assert state["steps_taken"] == 2 and state["stalled"] == 2
    assert state["final"].abstain


def test_step_budget_stops_the_loop(index):
    state = ask(build_agent(endless("answer"), index), "an endless question", max_steps=3)
    assert state["steps_taken"] == 3
    assert state["final"].abstain
