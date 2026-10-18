"""DealPulse: sales call intelligence with evidence-checked findings.

Rebuild of GenAI_Agents #28 (Sales Call Analyzer). The original sent the whole
transcript to one CrewAI agent and printed whatever JSON came back. Changes:

1. Measured, not guessed: talk ratio, questions, monologues and pacing are
   computed from the transcript in code.
2. Four specialist analysers run in parallel (LangGraph fan-out/fan-in):
   sentiment arc, objections and how they were handled, MEDDPICC
   qualification, and commitments/competitors.
3. Every finding cites a turn id and a quote; code checks the quote is in that
   turn and flags anything it cannot find, so it never reaches the CRM.
4. A deterministic coaching scorecard (0-100) so scores are comparable across
   calls and reps.
5. Outputs you can act on: markdown report, CRM update JSON, follow-up email
   draft, and a team trend report across a folder of calls.

Graph:
  load -> {sentiment, objections, qualification, commitments} -> verify
  -> synthesize -> scorecard -> write_outputs
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentkit import LLM, Tracer, dumps
from dealpulse.transcript import Call, call_from_dict, call_to_dict, conversation_metrics, load_call
from groundtruth.index import support_ratio

QUOTE_THRESHOLD = 0.75


# ---------- structured outputs ----------


class Cited(BaseModel):
    turn_id: str = Field(description="Turn id such as T07")
    quote: str = Field(description="Words copied exactly from that turn")


class SentimentSegment(Cited):
    start_turn: str
    end_turn: str
    score: int = Field(ge=-2, le=2, description="Customer sentiment: -2 hostile ... +2 enthusiastic")
    label: str


class Sentiment(BaseModel):
    segments: list[SentimentSegment]
    trend: Literal["improving", "stable", "declining"]


class Objection(Cited):
    objection: str
    category: Literal["price", "trust", "timing", "competitor", "product", "authority", "other"]
    response_turn_id: str = Field(description="Rep turn that responded, or empty if none")
    handling: Literal["well", "partly", "poorly", "not_addressed"]
    better_response: str


class Objections(BaseModel):
    objections: list[Objection]


class QualField(BaseModel):
    status: Literal["confirmed", "partial", "unknown"]
    detail: str
    turn_id: str = Field(description="Supporting turn id, empty when unknown")
    quote: str = Field(description="Exact supporting words, empty when unknown")


class Qualification(BaseModel):
    metrics: QualField
    economic_buyer: QualField
    decision_criteria: QualField
    decision_process: QualField
    paper_process: QualField
    identify_pain: QualField
    champion: QualField
    competition: QualField


class NextStep(Cited):
    owner: Literal["rep", "customer"]
    action: str
    due: str = Field(description="Date or deadline as said on the call, empty if none was given")


class Competitor(Cited):
    name: str
    context: str


class Commitments(BaseModel):
    next_steps: list[NextStep]
    competitors: list[Competitor]


class Email(BaseModel):
    subject: str
    body: str


class Synthesis(BaseModel):
    summary: str
    went_well: list[str]
    improve: list[str]
    deal_risk: Literal["low", "medium", "high"]
    risk_reasons: list[str]
    recommended_stage: str
    follow_up_email: Email


# ---------- state ----------


class State(TypedDict, total=False):
    path: str
    out_dir: str
    call: dict
    metrics: dict
    sentiment: dict
    objections: dict
    qualification: dict
    commitments: dict
    unverified: list[str]
    synthesis: dict
    scorecard: dict
    outputs: dict


# ---------- evidence + scoring (pure code, unit tested) ----------


def quote_in_turn(call: Call, turn_id: str, quote: str) -> bool:
    turn = call.turn(turn_id)
    quote = quote.strip(" \"'“”")
    return bool(turn and quote) and (quote.lower() in turn.text.lower()
                                     or support_ratio(quote, turn.text) >= QUOTE_THRESHOLD)


def verify_findings(call: Call, state: State) -> tuple[dict, list[str]]:
    """Mark each cited finding verified/unverified. Returns updated sections + a list of problems."""
    problems: list[str] = []
    out = {}

    def check(section: str, items: list[dict], label) -> list[dict]:
        for item in items:
            item["verified"] = quote_in_turn(call, item["turn_id"], item["quote"])
            if not item["verified"]:
                problems.append(f"{section}: {label(item)} (cites {item['turn_id']})")
        return items

    out["sentiment"] = {**state["sentiment"], "segments": check(
        "sentiment", [dict(s) for s in state["sentiment"]["segments"]], lambda s: s["label"])}
    out["objections"] = {"objections": check(
        "objection", [dict(o) for o in state["objections"]["objections"]], lambda o: o["objection"])}
    out["commitments"] = {
        "next_steps": check("next step", [dict(n) for n in state["commitments"]["next_steps"]], lambda n: n["action"]),
        "competitors": check("competitor", [dict(c) for c in state["commitments"]["competitors"]], lambda c: c["name"]),
    }
    qual = {}
    for name, field in state["qualification"].items():
        field = dict(field)
        if field["status"] != "unknown" and not quote_in_turn(call, field["turn_id"], field["quote"]):
            problems.append(f"qualification: {name} '{field['detail']}' (cites {field['turn_id']})")
            field.update(status="unknown", detail=f"unverified: {field['detail']}")
        qual[name] = field
    out["qualification"] = qual
    return out, problems


HANDLING = {"well": 1.0, "partly": 0.6, "poorly": 0.2, "not_addressed": 0.0}
QUAL_CREDIT = {"confirmed": 1.0, "partial": 0.5, "unknown": 0.0}


def scorecard(metrics: dict, objections: list[dict], qualification: dict, next_steps: list[dict]) -> dict:
    """Four 0-25 components -> 0-100. Only verified findings count."""
    coverage = sum(QUAL_CREDIT[f["status"]] for f in qualification.values()) / max(len(qualification), 1)
    discovery = 12.5 * min(metrics["rep_questions"] / 6, 1) + 12.5 * coverage

    verified_objections = [o for o in objections if o["verified"]]
    handling = (25 * sum(HANDLING[o["handling"]] for o in verified_objections) / len(verified_objections)
                if verified_objections else 25.0)

    ratio = metrics["rep_talk_ratio"]
    distance = 0 if 0.4 <= ratio <= 0.6 else min(abs(ratio - 0.4), abs(ratio - 0.6))
    balance = max(25 - distance * 100, 0) - (5 if metrics["longest_rep_monologue_seconds"] > 45 else 0)

    steps = [n for n in next_steps if n["verified"]]
    dated_rep_step = any(n["owner"] == "rep" and n["due"] for n in steps)
    customer_step = any(n["owner"] == "customer" for n in steps)
    commitments = (15 if dated_rep_step else 5 if any(n["owner"] == "rep" for n in steps) else 0) + \
                  (10 if customer_step else 0)

    parts = {"discovery": discovery, "objection_handling": handling, "talk_balance": max(balance, 0),
             "next_steps": commitments}
    return {**{k: round(v, 1) for k, v in parts.items()}, "total": round(sum(parts.values()))}


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
NUMBER_PHRASE = re.compile(r"\b(?:(?:%s)(?:[\s-]+|\b))+" % "|".join([*UNITS, *TENS, *SCALES]))


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


def _figures(text: str) -> set[str]:
    """Normalised figures: '6 hour', '18 %', 'oct 23', '900000', or a version like '4.2.1'."""
    text = re.sub(r"[‐-―−]", "-", re.sub(r"[  ]", " ", text.lower()))  # unicode dashes/spaces
    text = NUMBER_PHRASE.sub(lambda m: _words_to_number(m.group(0)) + " ", text)
    found = set()
    for match in FIGURE_RE.finditer(text):
        figure = re.sub(r"[\s-]+", " ", match.group(0)).strip(" .,").replace("percent", "%")
        figure = re.sub(r"(?<=\d),(?=\d{3})", "", figure)  # 900,000 -> 900000
        figure = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", figure)
        figure = re.sub(r"^([a-z]{3})[a-z]*\.?", r"\1", figure)  # month names -> 3 letters
        figure = re.sub(r"\s*(dollars?|thousand|million)$", "", figure) if figure[0].isdigit() else figure
        found.add(re.sub(r"s$", "", figure.replace(" %", "%")) if not figure[-1].isdigit() else figure)
    return found


def unsupported_figures(email: str, call: Call) -> list[str]:
    """Numbers, amounts and dates in the email that never came up on the call.

    A number is compared together with its unit, so "3 hour" is flagged even
    though "4.3" was said. A drafted email that promises "a 2-hour SLA by
    Oct 23" when nobody said so is worse than no email, so these are flagged.
    """
    spoken = _figures(" ".join(t.text for t in call.turns))
    if call.meta.get("date"):  # the call's own date is known, not invented
        spoken |= _figures(f"{call.meta['date']:%B} {call.meta['date'].day}" if hasattr(call.meta["date"], "day")
                           else str(call.meta["date"]))

    def said(figure: str) -> bool:
        if figure in spoken:
            return True
        # "Friday the 10th" on the call supports "Oct 10" in the email.
        date = re.fullmatch(r"[a-z]{3} (\d{1,2})", figure)
        return bool(date) and date.group(1) in spoken

    return sorted(f for f in _figures(email) if not said(f))


def risk_flags(objections: list[dict], qualification: dict, next_steps: list[dict]) -> list[str]:
    flags = []
    open_objections = [o for o in objections if o["verified"] and o["handling"] in ("poorly", "not_addressed")]
    if open_objections:
        flags.append(f"{len(open_objections)} objection(s) left unresolved")
    if qualification["economic_buyer"]["status"] == "unknown":
        flags.append("economic buyer not identified")
    if qualification["champion"]["status"] == "unknown":
        flags.append("no champion identified")
    if not any(n["verified"] and n["due"] for n in next_steps):
        flags.append("no dated next step")
    return flags


# ---------- graph ----------

ANALYST = (
    "You analyse a B2B sales call transcript. Each line starts with a turn id (T01...), the speaker's role "
    "(rep or customer) and name. Cite the turn id and copy the exact words for every finding. Report only what "
    "was said on this call."
)


def build_pipeline(llm: LLM, tracer: Tracer | None = None):
    tracer = tracer or Tracer(verbose=False)
    node = tracer.node

    def transcript(state: State) -> str:
        call = call_from_dict(state["call"])
        meta = {k: v for k, v in call.meta.items() if k != "speakers"}
        return f"Call details: {dumps(meta)}\n\nTranscript:\n{call.as_prompt()}"

    @node("load")
    def load(state: State):
        call = load_call(Path(state["path"]))
        metrics = conversation_metrics(call)
        tracer.print(f"[cyan]{call.meta.get('call_id')} · {call.meta.get('account')} · {len(call.turns)} turns · "
                     f"rep talk {metrics['rep_talk_ratio']:.0%}[/cyan]")
        return {"call": call_to_dict(call), "metrics": metrics}

    @node("sentiment")
    def sentiment(state: State):
        result = llm.structured(ANALYST, transcript(state) + "\n\nSplit the call into 3-5 segments and rate the "
                                "customer's sentiment in each, citing the turn that best shows it.", Sentiment,
                                effort="low")
        return {"sentiment": result.model_dump()}

    @node("objections")
    def objections(state: State):
        result = llm.structured(ANALYST, transcript(state) + "\n\nList every customer objection or concern, how "
                                "the rep responded, how well it was handled, and a better response.", Objections)
        return {"objections": result.model_dump()}

    @node("qualification")
    def qualification(state: State):
        result = llm.structured(ANALYST, transcript(state) + "\n\nFill in MEDDPICC from this call: metrics, "
                                "economic buyer, decision criteria, decision process, paper process, identified "
                                "pain, champion, competition. Use 'unknown' with empty turn_id and quote when the "
                                "call did not cover it.", Qualification)
        return {"qualification": result.model_dump()}

    @node("commitments")
    def commitments(state: State):
        result = llm.structured(ANALYST, transcript(state) + "\n\nList the concrete next steps each side agreed "
                                "to (owner, action, due date as said) and every competitor mentioned.", Commitments,
                                effort="low")
        return {"commitments": result.model_dump()}

    @node("verify")
    def verify(state: State):
        checked, problems = verify_findings(call_from_dict(state["call"]), state)
        if problems:
            tracer.print(f"[yellow]   {len(problems)} finding(s) failed the quote check[/yellow]")
        return {**checked, "unverified": problems}

    @node("synthesize")
    def synthesize(state: State):
        verified = {
            "metrics": state["metrics"],
            "objections": [o for o in state["objections"]["objections"] if o["verified"]],
            "qualification": state["qualification"],
            "next_steps": [n for n in state["commitments"]["next_steps"] if n["verified"]],
            "competitors": [c for c in state["commitments"]["competitors"] if c["verified"]],
            "sentiment_trend": state["sentiment"]["trend"],
        }
        result = llm.structured(
            "You are a sales coach. Use only the verified findings and metrics given. Coaching points must be "
            "specific to this call. The follow-up email goes from the rep to the customer: confirm the next steps "
            "that were agreed, acknowledge open concerns, under 180 words, no hype. Never invent commitments, "
            "dates, prices, SLA terms or numbers that were not said on the call; where the rep must supply a "
            "detail, write a [placeholder] such as [date] or [response-time terms].",
            f"Call details: {dumps({k: v for k, v in state['call']['meta'].items() if k != 'speakers'})}\n\n"
            f"Findings:\n{dumps(verified)}",
            Synthesis,
        )
        return {"synthesis": result.model_dump()}

    @node("scorecard")
    def score(state: State):
        card = scorecard(state["metrics"], state["objections"]["objections"], state["qualification"],
                         state["commitments"]["next_steps"])
        card["risk_flags"] = risk_flags(state["objections"]["objections"], state["qualification"],
                                        state["commitments"]["next_steps"])
        tracer.print(f"   score {card['total']}/100 · risk {state['synthesis']['deal_risk']}")
        return {"scorecard": card}

    @node("write_outputs")
    def write_outputs(state: State):
        out = Path(state["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        call_id = state["call"]["meta"]["call_id"]
        report = out / f"{call_id}-report.md"
        crm = out / f"{call_id}-crm.json"
        email = out / f"{call_id}-followup-email.md"
        report.write_text(render_report(state))
        crm.write_text(json.dumps(crm_update(state), indent=2))
        mail = state["synthesis"]["follow_up_email"]
        unsupported = unsupported_figures(mail["subject"] + "\n" + mail["body"], call_from_dict(state["call"]))
        warning = (f"> ⚠️ Check before sending — not said on the call: {', '.join(unsupported)}\n\n"
                   if unsupported else "")
        email.write_text(f"{warning}Subject: {mail['subject']}\n\n{mail['body']}\n")
        return {"outputs": {"report": str(report), "crm": str(crm), "email": str(email)}}

    graph = StateGraph(State)
    analysers = ["sentiment", "objections", "qualification", "commitments"]
    for name, fn in [("load", load), ("sentiment", sentiment), ("objections", objections),
                     ("qualification", qualification), ("commitments", commitments), ("verify", verify),
                     ("synthesize", synthesize), ("scorecard", score), ("write_outputs", write_outputs)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "load")
    for name in analysers:
        graph.add_edge("load", name)
    graph.add_edge(analysers, "verify")  # fan-in: waits for all four
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", "scorecard")
    graph.add_edge("scorecard", "write_outputs")
    graph.add_edge("write_outputs", END)
    return graph.compile()


def crm_update(state: State) -> dict:
    meta = state["call"]["meta"]
    syn = state["synthesis"]
    return {
        "account": meta.get("account"),
        "call_id": meta.get("call_id"),
        "call_date": str(meta.get("date")),
        "rep": meta.get("rep"),
        "current_stage": meta.get("stage"),
        "recommended_stage": syn["recommended_stage"],
        "deal_risk": syn["deal_risk"],
        "risk_flags": state["scorecard"]["risk_flags"],
        "next_steps": [{k: n[k] for k in ("owner", "action", "due")}
                       for n in state["commitments"]["next_steps"] if n["verified"]],
        "competitors": sorted({c["name"] for c in state["commitments"]["competitors"] if c["verified"]}),
        "meddpicc": {k: {"status": v["status"], "detail": v["detail"]} for k, v in state["qualification"].items()},
        "coaching_score": state["scorecard"]["total"],
    }


def render_report(state: State) -> str:
    meta, m, syn, card = state["call"]["meta"], state["metrics"], state["synthesis"], state["scorecard"]
    lines = [
        f"# {meta.get('call_id')} — {meta.get('account')} ({meta.get('stage')})",
        f"{meta.get('date')} · rep {meta.get('rep')} · {m['duration_seconds'] // 60}m{m['duration_seconds'] % 60:02d}s",
        "", syn["summary"], "",
        f"**Coaching score {card['total']}/100** — discovery {card['discovery']}, objection handling "
        f"{card['objection_handling']}, talk balance {card['talk_balance']}, next steps {card['next_steps']} (each /25)",
        f"**Deal risk: {syn['deal_risk']}** — " + "; ".join(syn["risk_reasons"] + card["risk_flags"]),
        f"**Recommended stage:** {syn['recommended_stage']}", "",
        "## Conversation metrics", "",
        f"- Rep talk ratio: {m['rep_talk_ratio']:.0%} (healthy range 40–60%)",
        f"- Rep questions: {m['rep_questions']} · customer questions: {m['customer_questions']}",
        f"- Longest rep monologue: {m['longest_rep_monologue_seconds']}s ({m['longest_rep_monologue_turn']})", "",
        "## What went well", "", *[f"- {x}" for x in syn["went_well"]], "",
        "## Improve next time", "", *[f"- {x}" for x in syn["improve"]], "",
        "## Objections", "", "| Turn | Objection | Handling | Better response |", "|---|---|---|---|",
    ]
    for o in state["objections"]["objections"]:
        mark = "" if o["verified"] else " ⚠️ unverified"
        lines.append(f"| {o['turn_id']} | {o['objection']} ({o['category']}){mark} | {o['handling']} | {o['better_response']} |")
    lines += ["", "## MEDDPICC", "", "| Field | Status | Detail |", "|---|---|---|"]
    for name, f in state["qualification"].items():
        lines.append(f"| {name.replace('_', ' ')} | {f['status']} | {f['detail']} |")
    lines += ["", "## Next steps", ""]
    for n in state["commitments"]["next_steps"]:
        mark = "" if n["verified"] else " ⚠️ unverified"
        lines.append(f"- **{n['owner']}**: {n['action']}" + (f" — due {n['due']}" if n["due"] else " — no date") + mark)
    lines += ["", "## Sentiment arc", ""]
    for s in state["sentiment"]["segments"]:
        lines.append(f"- {s['start_turn']}–{s['end_turn']}: {s['score']:+d} {s['label']} — “{s['quote']}”")
    lines.append(f"- Trend: {state['sentiment']['trend']}")
    if state["unverified"]:
        lines += ["", "## Dropped findings (quote not found in the cited turn)", "",
                  *[f"- {p}" for p in state["unverified"]]]
    return "\n".join(lines)


def analyse_call(llm: LLM, path: Path, out_dir: Path, tracer: Tracer | None = None) -> State:
    return build_pipeline(llm, tracer).invoke({"path": str(path), "out_dir": str(out_dir)})


def team_report(results: list[State]) -> str:
    """Aggregate several analysed calls into a per-rep trend table."""
    by_rep: dict[str, list[State]] = {}
    for r in results:
        by_rep.setdefault(r["call"]["meta"].get("rep", "unknown"), []).append(r)
    lines = ["# Team call review", "", "| Rep | Calls | Avg score | Avg talk ratio | Avg questions | Open risks |",
             "|---|---|---|---|---|---|"]
    for rep, calls in sorted(by_rep.items()):
        n = len(calls)
        lines.append(
            f"| {rep} | {n} | {sum(c['scorecard']['total'] for c in calls) / n:.0f} | "
            f"{sum(c['metrics']['rep_talk_ratio'] for c in calls) / n:.0%} | "
            f"{sum(c['metrics']['rep_questions'] for c in calls) / n:.1f} | "
            f"{sum(len(c['scorecard']['risk_flags']) for c in calls)} |")
    lines += ["", "## Calls", "", "| Call | Account | Rep | Score | Risk | Top coaching point |", "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["scorecard"]["total"]):
        meta, syn = r["call"]["meta"], r["synthesis"]
        lines.append(f"| {meta['call_id']} | {meta.get('account')} | {meta.get('rep')} | {r['scorecard']['total']} | "
                     f"{syn['deal_risk']} | {(syn['improve'] or ['—'])[0]} |")
    categories: dict[str, int] = {}
    for r in results:
        for o in r["objections"]["objections"]:
            if o["verified"]:
                categories[o["category"]] = categories.get(o["category"], 0) + 1
    lines += ["", "## Objection themes", "", *[f"- {k}: {v}" for k, v in sorted(categories.items(), key=lambda x: -x[1])]]
    return "\n".join(lines)
