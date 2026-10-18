"""Evaluation harness (replaces the original's Ragas notebook).

For each golden question it scores:
- correct          LLM judge compares the answer with the reference answer
- source recall    share of expected documents the answer actually cited (code)
- abstain correct  unanswerable questions must abstain; answerable ones must not

Writes a markdown report so you can diff runs after changing prompts.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from pydantic import BaseModel

from agentkit import LLM, Tracer
from citesure.agent import ask, build_agent
from citesure.index import CorpusIndex


class Verdict(BaseModel):
    correct: bool
    reason: str


JUDGE_SYSTEM = (
    "You grade a question-answering system. Mark correct=true when the answer states the same key facts as the "
    "reference answer (names, dates, amounts, yes/no). Extra correct detail is fine; a missing or wrong key fact "
    "is not."
)


def run_eval(llm: LLM, corpus: Path, golden: Path, report: Path, max_steps: int = 6) -> dict:
    index = CorpusIndex.build(corpus, llm)
    agent = build_agent(llm, index, Tracer(verbose=False))
    cases = yaml.safe_load(golden.read_text())
    rows, totals = [], {"correct": 0, "recall": 0.0, "abstain_ok": 0}

    for i, case in enumerate(cases, 1):
        started = time.time()
        state = ask(agent, case["question"], max_steps)
        final = state["final"]
        expect_abstain = case["expected"] == "ABSTAIN"

        if expect_abstain:
            correct = final.abstain
            reason = "abstained as expected" if correct else "answered an unanswerable question"
        elif final.abstain:
            correct, reason = False, "abstained on an answerable question"
        else:
            verdict = llm.structured(
                JUDGE_SYSTEM,
                f"Question: {case['question']}\nReference: {case['expected']}\nAnswer: {final.answer}",
                Verdict,
                effort="low",
            )
            correct, reason = verdict.correct, verdict.reason

        cited_docs = {c.split("#")[0] for c in final.citations}
        expected_docs = set(case.get("sources") or [])
        recall = len(cited_docs & expected_docs) / len(expected_docs) if expected_docs else 1.0
        abstain_ok = final.abstain == expect_abstain

        totals["correct"] += correct
        totals["recall"] += recall
        totals["abstain_ok"] += abstain_ok
        rows.append((i, case["question"], final.answer, correct, recall, abstain_ok, reason,
                     state["steps_taken"], time.time() - started))
        print(f"[{i}/{len(cases)}] {'PASS' if correct else 'FAIL'}  recall={recall:.2f}  {case['question'][:70]}")

    n = len(cases)
    summary = {
        "accuracy": totals["correct"] / n,
        "source_recall": totals["recall"] / n,
        "abstain_accuracy": totals["abstain_ok"] / n,
        "usage": llm.usage.summary(llm.model),
    }
    lines = [
        f"# CiteSure evaluation — {llm.model}",
        "",
        f"- Accuracy: **{summary['accuracy']:.0%}** ({totals['correct']}/{n})",
        f"- Source recall: **{summary['source_recall']:.0%}**",
        f"- Abstain accuracy: **{summary['abstain_accuracy']:.0%}**",
        f"- Usage: {summary['usage']}",
        "",
        "| # | Question | Correct | Recall | Steps | Secs | Judge note |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, q, _answer, correct, recall, _ab, reason, steps, secs in rows:
        lines.append(f"| {i} | {q} | {'✅' if correct else '❌'} | {recall:.2f} | {steps} | {secs:.0f} | {reason} |")
    lines += ["", "## Answers", ""]
    for i, q, answer, *_ in rows:
        lines += [f"**{i}. {q}**", "", answer, ""]
    report.write_text("\n".join(lines))
    return summary
