"""ResolveDesk CLI.

  uv run python -m resolvedesk ticket --customer C-1003 "I forgot my password and the link expired"
  uv run python -m resolvedesk batch                    # sample tickets + scored against expected outcomes
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from agentkit import Tracer, console, get_llm
from resolvedesk.agent import build_agent, cited_articles, handle, load_sample, render_ticket

HERE = Path(__file__).parent
OUT = Path("runs/resolvedesk")


def score(ticket: dict, state: dict) -> dict:
    """Compare one result with the ticket's expected outcome."""
    expect = ticket.get("expect", {})
    checks = {"route": state["route"] == expect.get("route")}
    if "category" in expect:
        checks["category"] = state["triage"]["category"] == expect["category"]
    if "kb" in expect and state["route"] == "auto":
        checks["source"] = expect["kb"] in cited_articles(state)
    if expect.get("redacted"):
        checks["redacted"] = bool(state["redactions"]) and "4111 1111" not in state["text"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(prog="resolvedesk", description="Knowledge-grounded support agent")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("ticket", help="Handle one ticket")
    one.add_argument("text")
    one.add_argument("--customer", default="C-1001")
    many = sub.add_parser("batch", help="Handle a YAML file of tickets and score them")
    many.add_argument("tickets", type=Path, nargs="?", default=HERE / "sample" / "tickets.yaml")
    many.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")
    kb, accounts = load_sample(HERE / "sample")
    OUT.mkdir(parents=True, exist_ok=True)
    agent = build_agent(llm, kb, accounts, Tracer(OUT / "trace.jsonl"))

    if args.command == "ticket":
        state = handle(agent, "T-NEW", args.customer, args.text)
        console.print(render_ticket(state))
    else:
        tickets = yaml.safe_load(args.tickets.read_text())
        with ThreadPoolExecutor(args.workers) as pool:
            states = list(pool.map(lambda t: handle(agent, t["id"], t["customer"], t["text"]), tickets))
        rows, passed, total = [], 0, 0
        for ticket, state in zip(tickets, states):
            checks = score(ticket, state)
            passed += sum(checks.values())
            total += len(checks)
            rows.append((ticket, state, checks))
            mark = "PASS" if all(checks.values()) else "FAIL"
            console.print(f"{mark} {ticket['id']} {state['route']:<8} {'; '.join(state['reasons'])[:80]}")

        auto = sum(s["route"] == "auto" for s in states)
        lines = [f"# ResolveDesk batch — {llm.model}", "",
                 f"- Tickets: {len(states)} · answered automatically: {auto} · escalated: {len(states) - auto}",
                 f"- Checks passed against expected outcomes: **{passed}/{total}**",
                 f"- Usage: {llm.usage.summary(llm.model)}", "",
                 "| Ticket | Route | Category | Sources / reasons | Checks |", "|---|---|---|---|---|"]
        for ticket, state, checks in rows:
            detail = ", ".join(cited_articles(state)) if state["route"] == "auto" else "; ".join(state["reasons"])
            verdict = " ".join(f"{k} {'✅' if v else '❌'}" for k, v in checks.items())
            lines.append(f"| {ticket['id']} | {state['route']} | {state['triage']['category']} | {detail} | {verdict} |")
        lines += ["", *[render_ticket(s) + "\n" for s in states]]
        report = OUT / "batch-report.md"
        report.write_text("\n".join(lines))
        (OUT / "results.json").write_text(json.dumps(states, indent=1, default=str))
        console.rule("Done")
        console.print(f"checks passed {passed}/{total} · report: {report}")
    console.print(f"[dim]{llm.usage.summary(llm.model)}[/dim]")


if __name__ == "__main__":
    main()
