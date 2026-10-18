"""TalentScout: a blind-screening hiring pipeline with human approvals.

Rebuild of GenAI_Agents #46 (HR AI Assistant). The original scraped LinkedIn
with a session cookie and parsed scores from free text with a regex; this
version changes that and adds:

1. Blind screening - names, contact details and protected attributes are
   removed in code before any LLM scores a CV. Identity comes back only after
   the shortlist is approved, when messages are drafted.
2. Evidence-checked rubric scoring - the LLM must quote the CV for every
   criterion; code verifies each quote really appears in the CV and caps the
   score of any criterion whose evidence it cannot find.
3. Deterministic decisions - weighted totals, must-have knockouts and the
   shortlist are computed in code from the rubric in job.yaml.
4. Parallel scoring - one Send per candidate (LangGraph map step).
5. Durable human-in-the-loop - interrupt() for JD approval and shortlist
   approval, checkpointed to SQLite, so you can stop the process at a review
   and resume later with the same thread id.
6. Inclusive-language lint for the job description, and an outbox of drafted
   messages instead of auto-sending anything.

Graph:
  load -> draft_jd -> review_jd (interrupt) -> redact -> score_cv x N (parallel)
  -> rank -> review_shortlist (interrupt) -> interview_kits -> draft_messages -> report
"""

from __future__ import annotations

import operator
import re
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from agentkit import LLM, Tracer, dumps
from groundtruth.index import support_ratio
from talentscout.redact import lint_job_description, redact_cv

EVIDENCE_THRESHOLD = 0.8
UNVERIFIED_CAP = 2


# ---------- structured outputs ----------


class JobDescription(BaseModel):
    markdown: str


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=0, le=5, description="0 = no evidence, 5 = exceptional")
    evidence: list[str] = Field(description="Short quotes copied exactly from the CV")


class MustHaveCheck(BaseModel):
    requirement: str
    met: bool
    evidence: str = Field(description="Exact quote from the CV, or empty when not met")


class CVAssessment(BaseModel):
    criteria: list[CriterionScore]
    must_haves: list[MustHaveCheck]
    years_experience: float
    strengths: list[str]
    gaps: list[str]
    summary: str


class InterviewQuestion(BaseModel):
    question: str
    probes: str = Field(description="The skill or gap this question tests")
    strong_answer: str = Field(description="What a strong answer includes")


class InterviewKit(BaseModel):
    questions: list[InterviewQuestion]


class Message(BaseModel):
    subject: str
    body: str


# ---------- state ----------


class State(TypedDict, total=False):
    job_path: str
    cv_dir: str
    out_dir: str
    job: dict
    jd: str
    jd_lint: list[str]
    jd_feedback: str
    jd_approved: bool
    candidates: list[dict]  # id, file, name, blind_text, removed
    scores: Annotated[list[dict], operator.add]
    ranking: list[dict]
    shortlist: list[str]
    kits: dict[str, dict]
    outbox: list[str]
    report_path: str


class ScoreTask(TypedDict):
    job: dict
    candidate: dict


# ---------- scoring logic (pure code, unit tested) ----------


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).strip().lower()


def quote_found(quote: str, cv_text: str) -> bool:
    quote_n, cv_n = normalise(quote), normalise(cv_text)
    return bool(quote_n) and (quote_n in cv_n or support_ratio(quote_n, cv_n) >= EVIDENCE_THRESHOLD)


def finalise_score(assessment: CVAssessment, job: dict, cv_text: str) -> dict:
    """Verify evidence, apply the rubric weights and must-have knockouts."""
    weights: dict[str, float] = job["rubric"]
    criteria = []
    for c in assessment.criteria:
        quotes = [q.strip(" \"'“”‘’") for q in c.evidence]
        verified = [q for q in quotes if quote_found(q, cv_text)]
        score = c.score
        flag = ""
        if c.score > UNVERIFIED_CAP and not verified:
            score, flag = UNVERIFIED_CAP, "evidence not found in CV; score capped"
        criteria.append({"criterion": c.criterion, "score": score, "llm_score": c.score,
                         "evidence": verified, "flag": flag})

    by_name = {c["criterion"]: c for c in criteria}
    weighted = sum(weights[name] * by_name[name]["score"] for name in weights if name in by_name)

    must_haves = []
    for m in assessment.must_haves:
        # "met" only counts when the quote backing it is really in the CV.
        quote = m.evidence.strip(" \"'“”‘’")
        met = m.met and quote_found(quote, cv_text)
        must_haves.append({"requirement": m.requirement, "met": met, "evidence": quote if met else ""})
    knocked_out = [m["requirement"] for m in must_haves if not m["met"]]

    return {
        "weighted": round(weighted, 2),
        "criteria": criteria,
        "must_haves": must_haves,
        "knocked_out": knocked_out,
        "years_experience": assessment.years_experience,
        "strengths": assessment.strengths,
        "gaps": assessment.gaps,
        "summary": assessment.summary,
    }


def rank_candidates(scores: list[dict], job: dict) -> tuple[list[dict], list[str]]:
    ranking = sorted(scores, key=lambda s: (not s["knocked_out"], s["weighted"]), reverse=True)
    eligible = [s for s in ranking if not s["knocked_out"] and s["weighted"] >= job["shortlist_threshold"]]
    return ranking, [s["id"] for s in eligible[: job["shortlist_size"]]]


# ---------- graph ----------

RUBRIC_HELP = {
    "skills_match": "match between the CV's skills and the must-have and nice-to-have lists",
    "relevant_experience": "years and depth of directly relevant work, against the required years",
    "delivery_impact": "shipped work with measurable results for real users",
    "communication": "clarity of the CV and evidence of mentoring, writing or leading",
}


def build_workflow(llm: LLM, tracer: Tracer | None = None, checkpointer=None):
    tracer = tracer or Tracer(verbose=False)
    node = tracer.node

    @node("load")
    def load(state: State):
        job = yaml.safe_load(Path(state["job_path"]).read_text())
        candidates = []
        for i, path in enumerate(sorted(Path(state["cv_dir"]).glob("*.md")), 1):
            raw = path.read_text()
            name = raw.splitlines()[0].lstrip("# ").strip() if raw.startswith("#") else path.stem
            candidates.append({"id": f"C-{i:02d}", "file": path.name, "name": name, "raw": raw})
        tracer.print(f"[cyan]{len(candidates)} CVs for {job['title']}[/cyan]")
        return {"job": job, "candidates": candidates, "jd_feedback": "", "scores": []}

    @node("draft_jd")
    def draft_jd(state: State):
        job = state["job"]
        feedback = state.get("jd_feedback")
        jd = llm.structured(
            "You write clear, inclusive job descriptions. No hype words, no gendered language, no age signals. "
            "Separate must-have from nice-to-have requirements so qualified people are not put off.",
            "Write a job description in markdown with sections: About us, The role, What you will do, "
            "Must have, Nice to have, What we offer, Location and pay.\n\n" + dumps({k: v for k, v in job.items()
                                                                                      if k not in ("rubric",)})
            + (f"\n\nThe hiring manager asked for these changes to the last draft: {feedback}" if feedback else ""),
            JobDescription,
        ).markdown
        return {"jd": jd, "jd_lint": lint_job_description(jd)}

    @node("review_jd")
    def review_jd(state: State):
        decision = interrupt({"kind": "jd_review", "jd": state["jd"], "lint": state["jd_lint"]})
        return {"jd_approved": bool(decision.get("approved")), "jd_feedback": decision.get("feedback", "")}

    def after_jd_review(state: State) -> str:
        return "redact" if state["jd_approved"] else "draft_jd"

    @node("redact")
    def redact(state: State):
        candidates = []
        for c in state["candidates"]:
            blind = redact_cv(c["raw"])
            candidates.append({**c, "blind_text": blind.text, "removed": blind.removed})
        return {"candidates": candidates}

    def fan_out(state: State) -> list[Send]:
        # The scorer receives only the id and the anonymised text - never the raw CV or name.
        return [Send("score_cv", {"job": state["job"], "candidate": {"id": c["id"], "blind_text": c["blind_text"]}})
                for c in state["candidates"]]

    @node("score_cv")
    def score_cv(task: ScoreTask):
        job, candidate = task["job"], task["candidate"]
        rubric = "\n".join(f"- {name}: {RUBRIC_HELP.get(name, name)}" for name in job["rubric"])
        assessment = llm.structured(
            "You screen anonymised CVs against a job. Judge only skills, experience and results written in the "
            "CV. Every score needs short quotes copied exactly from the CV as evidence; if you cannot quote it, "
            "it is not in the CV.",
            f"Job: {job['title']} (needs {job['years_experience']}+ years)\n"
            f"Must have:\n- " + "\n- ".join(job["must_have"])
            + "\nNice to have:\n- " + "\n- ".join(job["nice_to_have"])
            + f"\n\nScore each criterion 0-5, using exactly these criterion names:\n{rubric}\n\n"
            "Check every must-have and quote the line that proves it.\n\n"
            f"Anonymised CV:\n{candidate['blind_text']}",
            CVAssessment,
        )
        result = finalise_score(assessment, job, candidate["blind_text"])
        tracer.print(f"   {candidate['id']}: {result['weighted']:.2f}"
                     + (f"  knocked out: {result['knocked_out']}" if result["knocked_out"] else ""))
        return {"scores": [{"id": candidate["id"], **result}]}

    @node("rank")
    def rank(state: State):
        ranking, shortlist = rank_candidates(state["scores"], state["job"])
        return {"ranking": ranking, "shortlist": shortlist}

    @node("review_shortlist")
    def review_shortlist(state: State):
        table = [{"id": s["id"], "weighted": s["weighted"], "knocked_out": s["knocked_out"],
                  "summary": s["summary"]} for s in state["ranking"]]
        decision = interrupt({"kind": "shortlist_review", "ranking": table, "proposed": state["shortlist"]})
        approved = decision.get("approved_ids") or state["shortlist"]
        known = {s["id"] for s in state["ranking"]}
        return {"shortlist": [i for i in approved if i in known]}

    @node("interview_kits")
    def interview_kits(state: State):
        scores = {s["id"]: s for s in state["ranking"]}
        kits = {}
        for cid in state["shortlist"]:
            s = scores[cid]
            kit = llm.structured(
                "You prepare structured interviews. Questions must be job-related and the same standard applies "
                "to every candidate; tailor follow-ups to this candidate's gaps.",
                f"Role: {state['job']['title']}\nResponsibilities: {state['job']['responsibilities']}\n"
                f"Candidate strengths: {s['strengths']}\nCandidate gaps: {s['gaps']}\n\n"
                "Write 6 questions: 2 technical deep-dives on their strongest claimed work, 2 that probe the gaps, "
                "1 system design question for this role, 1 on mentoring. For each, say what it probes and what a "
                "strong answer includes.",
                InterviewKit,
            )
            kits[cid] = kit.model_dump()
        return {"kits": kits}

    @node("draft_messages")
    def draft_messages(state: State):
        # Identity is re-attached only here, after decisions are made.
        out = Path(state["out_dir"]) / "outbox"
        out.mkdir(parents=True, exist_ok=True)
        scores = {s["id"]: s for s in state["ranking"]}
        files = []
        for c in state["candidates"]:
            invited = c["id"] in state["shortlist"]
            s = scores[c["id"]]
            message = llm.structured(
                "You write short, warm, professional candidate emails. Never mention scores, rankings or other "
                "candidates.",
                f"Candidate first name: {c['name'].split()[0]}\nRole: {state['job']['title']}\n"
                + (f"Invite them to interview between {state['job']['interview_window']}. Mention one strength: "
                   f"{s['strengths'][:1]}" if invited else
                   "Politely decline. Thank them, mention one genuine strength from their profile: "
                   f"{s['strengths'][:1]}, and keep it under 120 words."),
                Message,
                effort="low",
            )
            path = out / f"{c['id']}-{'invite' if invited else 'decline'}.md"
            path.write_text(f"To: {c['name']}\nSubject: {message.subject}\n\n{message.body}\n")
            files.append(str(path))
        return {"outbox": files}

    @node("report")
    def report(state: State):
        path = Path(state["out_dir"]) / "report.md"
        path.write_text(render_report(state))
        return {"report_path": str(path)}

    graph = StateGraph(State)
    for name, fn in [("load", load), ("draft_jd", draft_jd), ("review_jd", review_jd), ("redact", redact),
                     ("score_cv", score_cv), ("rank", rank), ("review_shortlist", review_shortlist),
                     ("interview_kits", interview_kits), ("draft_messages", draft_messages), ("report", report)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "load")
    graph.add_edge("load", "draft_jd")
    graph.add_edge("draft_jd", "review_jd")
    graph.add_conditional_edges("review_jd", after_jd_review, ["redact", "draft_jd"])
    graph.add_conditional_edges("redact", fan_out, ["score_cv"])
    graph.add_edge("score_cv", "rank")
    graph.add_edge("rank", "review_shortlist")
    graph.add_edge("review_shortlist", "interview_kits")
    graph.add_edge("interview_kits", "draft_messages")
    graph.add_edge("draft_messages", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


def render_report(state: State) -> str:
    job = state["job"]
    names = {c["id"]: c for c in state["candidates"]}
    lines = [f"# Screening report — {job['title']}", "",
             f"Rubric weights: {job['rubric']} · threshold {job['shortlist_threshold']} · "
             f"shortlist size {job['shortlist_size']}", "",
             "| Rank | Candidate | Weighted /5 | Must-haves | Decision |", "|---|---|---|---|---|"]
    for rank, s in enumerate(state["ranking"], 1):
        decision = "Interview" if s["id"] in state["shortlist"] else "Decline"
        musts = "all met" if not s["knocked_out"] else "missing: " + "; ".join(s["knocked_out"])
        lines.append(f"| {rank} | {s['id']} ({names[s['id']]['name']}) | {s['weighted']:.2f} | {musts} | {decision} |")
    lines += ["", "_Scores were produced from anonymised CVs. Names were attached after the shortlist was approved._",
              ""]
    for s in state["ranking"]:
        c = names[s["id"]]
        lines += [f"## {s['id']} — {c['name']}", "", s["summary"], "",
                  f"Hidden from the scorer: {', '.join(c['removed']) or 'nothing'}", "",
                  "| Criterion | Score | Evidence | Note |", "|---|---|---|---|"]
        for crit in s["criteria"]:
            evidence = "; ".join(f"“{q}”" for q in crit["evidence"]) or "—"
            lines.append(f"| {crit['criterion']} | {crit['score']} | {evidence} | {crit['flag']} |")
        lines += ["", f"Strengths: {'; '.join(s['strengths'])}", "", f"Gaps: {'; '.join(s['gaps'])}", ""]
        kit = (state.get("kits") or {}).get(s["id"])
        if kit:
            lines += ["### Interview kit", ""]
            for i, q in enumerate(kit["questions"], 1):
                lines += [f"{i}. **{q['question']}**  ", f"   Probes: {q['probes']}  ", f"   Strong answer: {q['strong_answer']}"]
            lines.append("")
    lines += ["## Job description (approved)", "", state["jd"]]
    return "\n".join(lines)
