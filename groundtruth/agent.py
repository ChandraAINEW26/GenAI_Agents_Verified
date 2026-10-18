"""GroundTruth: a controllable plan-and-execute RAG agent.

Rebuild of NirDiamant/Controllable-RAG-Agent with these changes:

1. Citations everywhere. Every piece of evidence keeps its passage id, every
   answer must cite ids, and code (not the LLM) checks the ids are real.
2. Deterministic grounding. Distilled extracts are checked against their source
   passage by word overlap before they enter the context.
3. Step budget and honest abstain. The original could loop forever; this one
   stops after max_steps and says "not enough information" instead of guessing.
4. Answer verification loop. A final grounding check can send the answer back
   for one rewrite, then downgrades it to an abstain if it is still unsupported.
5. Query rewrites + RRF fusion for retrieval, and de-anonymisation in code.

Graph:
  anonymize -> plan -> deanonymize -> task_handler -> {retrieve | answer_step}
  -> replan -> (task_handler | final_answer) -> verify -> (final_answer | END)
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentkit import LLM, Tracer, dumps
from groundtruth.index import CorpusIndex, support_ratio

GROUNDING_THRESHOLD = 0.7


# ---------- structured outputs ----------


class Entity(BaseModel):
    placeholder: str = Field(description="Variable name such as X or Y")
    value: str = Field(description="The named entity it replaces, exactly as written in the question")


class Anonymized(BaseModel):
    question: str
    entities: list[Entity]


class Plan(BaseModel):
    steps: list[str]


class TaskDecision(BaseModel):
    task: str = Field(description="The single next step to execute")
    tool: Literal["chunks", "quotes", "summaries", "answer"] = Field(
        description="chunks: detailed passages; quotes: exact sentences; summaries: per-document overviews; "
        "answer: reason over evidence already collected, no retrieval"
    )
    search_queries: list[str] = Field(description="2-3 keyword search rewrites of the task; empty for answer")


class Extract(BaseModel):
    passage_id: str
    text: str = Field(description="Only the relevant part, copied from the passage")


class Distilled(BaseModel):
    extracts: list[Extract]


class StepAnswer(BaseModel):
    reasoning: str
    answer: str
    citations: list[str]


class Replan(BaseModel):
    can_answer_now: bool = Field(description="True only if the evidence already answers the whole original question")
    remaining_steps: list[str]


class FinalAnswer(BaseModel):
    reasoning: str
    answer: str
    citations: list[str] = Field(description="Passage ids that support the answer")
    confidence: Literal["high", "medium", "low"]
    abstain: bool = Field(description="True when the evidence does not answer the question")


class Grounding(BaseModel):
    grounded: bool
    unsupported_claims: list[str]


# ---------- state ----------


class Evidence(TypedDict):
    id: str
    text: str


class State(TypedDict, total=False):
    question: str
    anonymized: Anonymized
    plan: list[str]
    decision: TaskDecision
    evidence: dict[str, Evidence]  # passage id -> evidence
    past_steps: list[str]
    steps_taken: int
    max_steps: int
    final: FinalAnswer
    verify_attempts: int
    verify_feedback: str
    rejected_extracts: int


# ---------- prompts ----------

SYSTEM = (
    "You are one step of a retrieval-augmented question answering agent. Use only the evidence you are given. "
    "Never use outside or prior knowledge about the entities involved."
)


def evidence_block(state: State) -> str:
    evidence = state.get("evidence") or {}
    if not evidence:
        return "(no evidence collected yet)"
    return "\n".join(f"[{e['id']}] {e['text']}" for e in evidence.values())


def build_agent(llm: LLM, index: CorpusIndex, tracer: Tracer | None = None, top_k: int = 4):
    tracer = tracer or Tracer(verbose=False)
    node = tracer.node

    @node("anonymize")
    def anonymize(state: State):
        # Planning on an anonymised question stops the planner leaning on what it
        # already "knows" about famous names - the plan must come from the data.
        result = llm.structured(
            SYSTEM,
            "Replace every named entity (people, companies, products, places) in the question with a variable "
            f"X, Y, Z... Keep everything else.\n\nQuestion: {state['question']}",
            Anonymized,
            effort="low",
        )
        return {"anonymized": result, "steps_taken": 0, "evidence": {}, "past_steps": [], "verify_attempts": 0,
                "rejected_extracts": 0}

    @node("plan")
    def plan(state: State):
        result = llm.structured(
            SYSTEM,
            "Write a short step-by-step plan to answer this question from a document collection. Each step must "
            "either look up one fact or reason over facts found by earlier steps. Use 2-5 steps.\n\n"
            f"Question: {state['anonymized'].question}",
            Plan,
        )
        return {"plan": result.steps}

    @node("deanonymize")
    def deanonymize(state: State):
        # Done in code: a string replace cannot hallucinate.
        steps = []
        for step in state["plan"]:
            for entity in state["anonymized"].entities:
                step = step.replace(entity.placeholder, entity.value)
            steps.append(step)
        tracer.print("[cyan]plan:[/cyan] " + " | ".join(steps))
        return {"plan": steps}

    @node("task_handler")
    def task_handler(state: State):
        decision = llm.structured(
            SYSTEM,
            f"Original question: {state['question']}\n\nRemaining plan:\n- " + "\n- ".join(state["plan"])
            + "\n\nSteps done so far:\n- " + "\n- ".join(state["past_steps"] or ["none"])
            + f"\n\nEvidence collected:\n{evidence_block(state)}\n\n"
            "Choose the next step from the remaining plan and the tool to run it. Prefer 'answer' only when the "
            "evidence above already contains the facts that step needs.",
            TaskDecision,
            effort="low",
        )
        tracer.print(f"[cyan]step {state['steps_taken'] + 1}:[/cyan] {decision.tool} — {decision.task}")
        return {"decision": decision}

    @node("retrieve")
    def retrieve(state: State):
        decision = state["decision"]
        queries = decision.search_queries or [decision.task]
        passages = index.search(decision.tool, [decision.task, *queries], k=top_k)
        if not passages:
            return {"past_steps": [*state["past_steps"], f"{decision.task} -> nothing found"],
                    "steps_taken": state["steps_taken"] + 1}

        distilled = llm.structured(
            SYSTEM,
            f"Task: {decision.task}\n\nPassages:\n" + "\n".join(f"[{p.id}] {p.text}" for p in passages)
            + "\n\nCopy out only the parts relevant to the task, each with the id of the passage it came from. "
            "Copy wording exactly. Return an empty list if nothing is relevant.",
            Distilled,
            effort="low",
        )

        evidence = dict(state["evidence"])
        kept = rejected = 0
        by_id = {p.id: p for p in passages}
        for extract in distilled.extracts:
            source = by_id.get(extract.passage_id)
            if source is None or support_ratio(extract.text, source.text) < GROUNDING_THRESHOLD:
                rejected += 1  # cites a passage it was not shown, or adds words the passage lacks
                continue
            evidence[extract.passage_id] = {"id": extract.passage_id, "text": extract.text}
            kept += 1
        tracer.print(f"   kept {kept} extract(s), rejected {rejected} ungrounded")
        return {
            "evidence": evidence,
            "past_steps": [*state["past_steps"], f"{decision.task} -> {kept} supporting passage(s)"],
            "steps_taken": state["steps_taken"] + 1,
            "rejected_extracts": state["rejected_extracts"] + rejected,
        }

    @node("answer_step")
    def answer_step(state: State):
        decision = state["decision"]
        result = llm.structured(
            SYSTEM,
            f"Sub-question: {decision.task}\n\nEvidence:\n{evidence_block(state)}\n\n"
            "Think step by step, then answer the sub-question using only this evidence and cite the ids you used. "
            "If the evidence is insufficient, say exactly what is missing.\n"
            "Example of good reasoning: 'Passage A says the outage lasted 6 hours; passage B says credits apply "
            "above 4 hours; so a credit applies.'\n"
            "Example of bad reasoning: 'Companies like this usually offer credits' (outside knowledge).",
            StepAnswer,
        )
        valid = [c for c in result.citations if c in state["evidence"]]
        note = f"{decision.task} -> {result.answer} (cites {', '.join(valid) or 'nothing'})"
        return {"past_steps": [*state["past_steps"], note], "steps_taken": state["steps_taken"] + 1}

    @node("replan")
    def replan(state: State):
        result = llm.structured(
            SYSTEM,
            f"Original question: {state['question']}\n\nPlan before this step:\n- " + "\n- ".join(state["plan"])
            + "\n\nSteps done:\n- " + "\n- ".join(state["past_steps"])
            + f"\n\nEvidence:\n{evidence_block(state)}\n\n"
            "Decide whether the evidence already answers the whole original question. If not, return the steps "
            "still needed (drop finished ones, add new ones if a lookup came back empty).",
            Replan,
            effort="low",
        )
        return {"plan": result.remaining_steps if not result.can_answer_now else []}

    def after_replan(state: State) -> str:
        if not state["plan"]:
            return "final_answer"
        if state["steps_taken"] >= state["max_steps"]:
            tracer.print("[yellow]step budget reached — answering with what was found[/yellow]")
            return "final_answer"
        return "task_handler"

    def after_task(state: State) -> str:
        return "answer_step" if state["decision"].tool == "answer" else "retrieve"

    @node("final_answer")
    def final_answer(state: State):
        feedback = state.get("verify_feedback")
        result = llm.structured(
            SYSTEM,
            f"Question: {state['question']}\n\nWork done:\n- " + "\n- ".join(state["past_steps"])
            + f"\n\nEvidence:\n{evidence_block(state)}\n\n"
            "Answer the question using only the evidence. Cite the passage ids for every fact. If the evidence "
            "does not answer the question, set abstain=true and say what is missing."
            + (f"\n\nA reviewer rejected your previous answer: {feedback}\nFix it." if feedback else ""),
            FinalAnswer,
        )
        return {"final": result}

    @node("verify")
    def verify(state: State):
        final = state["final"]
        known = state["evidence"]
        valid = [c for c in final.citations if c in known]
        if len(valid) != len(final.citations):
            final = final.model_copy(update={"citations": valid})
        if final.abstain:
            return {"final": final, "verify_feedback": ""}

        if not valid:
            verdict = Grounding(grounded=False, unsupported_claims=["the answer cites no collected evidence"])
        else:
            verdict = llm.structured(
                "You check answers for hallucination. A claim is supported only if the cited evidence states it "
                "or it follows by simple arithmetic or date logic from the evidence.",
                f"Answer: {final.answer}\n\nCited evidence:\n" + "\n".join(f"[{c}] {known[c]['text']}" for c in valid),
                Grounding,
            )
        if verdict.grounded:
            return {"final": final, "verify_feedback": ""}

        attempts = state["verify_attempts"] + 1
        tracer.print(f"[yellow]verification failed:[/yellow] {verdict.unsupported_claims}")
        if attempts >= 2:
            final = final.model_copy(update={
                "abstain": True,
                "confidence": "low",
                "answer": "I could not verify an answer from the documents. Unsupported: "
                + "; ".join(verdict.unsupported_claims),
            })
            return {"final": final, "verify_attempts": attempts, "verify_feedback": ""}
        return {"final": final, "verify_attempts": attempts, "verify_feedback": "; ".join(verdict.unsupported_claims)}

    def after_verify(state: State) -> str:
        return "final_answer" if state.get("verify_feedback") else END

    graph = StateGraph(State)
    for name, fn in [("anonymize", anonymize), ("plan", plan), ("deanonymize", deanonymize),
                     ("task_handler", task_handler), ("retrieve", retrieve), ("answer_step", answer_step),
                     ("replan", replan), ("final_answer", final_answer), ("verify", verify)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "anonymize")
    graph.add_edge("anonymize", "plan")
    graph.add_edge("plan", "deanonymize")
    graph.add_edge("deanonymize", "task_handler")
    graph.add_conditional_edges("task_handler", after_task, ["retrieve", "answer_step"])
    graph.add_edge("retrieve", "replan")
    graph.add_edge("answer_step", "replan")
    graph.add_conditional_edges("replan", after_replan, ["task_handler", "final_answer"])
    graph.add_edge("final_answer", "verify")
    graph.add_conditional_edges("verify", after_verify, ["final_answer", END])
    return graph.compile()


def ask(agent, question: str, max_steps: int = 6) -> State:
    # Each step is a few nodes; leave headroom so the step budget, not LangGraph, ends the run.
    return agent.invoke({"question": question, "max_steps": max_steps}, {"recursion_limit": 20 + max_steps * 4})


def render(state: State) -> str:
    final = state["final"]
    lines = [final.answer, ""]
    if final.citations:
        lines.append("Sources: " + ", ".join(final.citations))
    lines.append(f"Confidence: {final.confidence}{' (abstained)' if final.abstain else ''}"
                 f" · steps: {state['steps_taken']} · rejected extracts: {state['rejected_extracts']}")
    return "\n".join(lines)


__all__ = ["build_agent", "ask", "render", "FinalAnswer", "State", "dumps"]
