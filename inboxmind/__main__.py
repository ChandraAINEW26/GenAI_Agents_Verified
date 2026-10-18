"""InboxMind CLI.

  uv run python -m inboxmind learn-demo          # accuracy on held-out emails before vs after learning
  uv run python -m inboxmind run                 # triage the whole sample inbox with current memory
  uv run python -m inboxmind feedback --email T-04 --label notify --note "Failed builds on main matter"
  uv run python -m inboxmind rules [--all] [--retire 3]
  uv run python -m inboxmind memory              # facts, example count, rules
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.table import Table

from agentkit import Tracer, console, get_llm
from inboxmind.agent import build_agent, learn_from_feedback, load_sample, process
from inboxmind.memory import Memory

HERE = Path(__file__).parent
OUT = Path("runs/inboxmind")


def ask_approval(auto: bool):
    def approve(proposal, condition, preview) -> bool:
        console.print(f"[cyan]Proposed {proposal.kind} rule:[/cyan] {proposal.text}"
                      + (f"  condition={condition}" if condition else ""))
        for change in preview:
            console.print(f"   [yellow]would relabel {change}[/yellow]")
        if auto:
            return True
        return console.input("Keep this rule? [Y/n] ").strip().lower() in ("", "y", "yes")
    return approve


def triage_all(agent, emails: list[dict]) -> list[dict]:
    return [process(agent, e) for e in emails]


def accuracy(states: list[dict]) -> float:
    return sum(s["label"] == s["email"]["label"] for s in states) / len(states)


def write_outputs(states: list[dict], out: Path) -> None:
    outbox = out / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    digest = ["# Notify digest", ""]
    for s in states:
        e = s["email"]
        if s["label"] == "respond" and s.get("draft"):
            warn = "".join(f"> ⚠️ {p}\n" for p in s.get("draft_problems", []))
            (outbox / f"{e['id']}.md").write_text(
                f"{warn}To: {e['from']}\nSubject: {s['draft']['subject']}\n\n{s['draft']['body']}\n")
        elif s["label"] == "notify":
            digest.append(f"- **{e['subject']}** — {e['from']}")
    (out / "digest.md").write_text("\n".join(digest) + "\n")


def table(states: list[dict], title: str) -> Table:
    t = Table("Email", "Subject", "Expected", "Got", "By", title=title)
    for s in states:
        e = s["email"]
        ok = "✅" if s["label"] == e["label"] else "❌"
        t.add_row(e["id"], e["subject"][:44], e["label"], f"{s['label']} {ok}", f"R-{s['rule_id']}" if s.get("rule_id") else "LLM")
    return t


def main() -> None:
    parser = argparse.ArgumentParser(prog="inboxmind", description="Email assistant that learns from corrections")
    parser.add_argument("--memory", type=Path, default=OUT / "memory.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("learn-demo", help="Measure learning: test accuracy before and after training corrections")
    sub.add_parser("run", help="Triage the sample inbox with current memory")
    fb = sub.add_parser("feedback", help="Correct or confirm one email")
    fb.add_argument("--email", required=True)
    fb.add_argument("--label", required=True, choices=["ignore", "notify", "respond"])
    fb.add_argument("--note", required=True)
    fb.add_argument("--yes", action="store_true", help="Keep the proposed rule without asking")
    rules = sub.add_parser("rules", help="List or retire rules")
    rules.add_argument("--all", action="store_true")
    rules.add_argument("--retire", type=int)
    sub.add_parser("memory", help="Show what has been learned")
    sub.add_parser("reset", help="Forget everything for this user")
    args = parser.parse_args()

    profile, calendar, inbox = load_sample(HERE / "sample")
    lookup = {e["id"]: e for e in inbox}
    memory = Memory(args.memory, profile["user_id"])
    OUT.mkdir(parents=True, exist_ok=True)

    if args.command in ("rules", "memory", "reset"):
        if args.command == "reset":
            memory.reset()
            console.print("memory cleared")
        elif args.command == "rules":
            if args.retire:
                memory.retire_rule(args.retire)
            for r in memory.rules(include_retired=args.all):
                console.print(f"R-{r.id} [{r.status}] {r.kind} {r.condition or ''} → {r.action or '-'} : {r.text}")
        else:
            console.print(memory.counts())
            for f in memory.facts_about(""):
                console.print(f"- {f['subject']}: {f['fact']} (from {f['source']})")
        return

    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")
    tracer = Tracer(OUT / "trace.jsonl")
    agent = build_agent(llm, memory, profile, calendar, tracer)

    if args.command == "feedback":
        result = learn_from_feedback(llm, memory, profile, lookup[args.email], args.label, args.note, lookup,
                                     ask_approval(args.yes))
        console.print(f"stored example {result['episode']}; rule: "
                      + (f"R-{result['rule_id']}" if result["rule_id"] else result.get("rejected", "none")))
    elif args.command == "run":
        states = triage_all(agent, inbox)
        write_outputs(states, OUT)
        console.print(table(states, "Triage"))
        console.print(f"accuracy {accuracy(states):.0%} · outbox: {OUT / 'outbox'} · digest: {OUT / 'digest.md'}")
    else:  # learn-demo
        memory.reset()
        test = [e for e in inbox if e["split"] == "test"]
        train = [e for e in inbox if e["split"] == "train"]

        console.rule("1. Test emails with empty memory")
        # A dry run: it uses a throwaway in-memory store so mistakes made before learning leave no trace.
        blank = build_agent(llm, Memory(Path(":memory:"), profile["user_id"]), profile, calendar, tracer)
        before = triage_all(blank, test)
        console.print(table(before, f"Before learning — {accuracy(before):.0%}"))

        console.rule("2. Train emails: triage, then the user corrects or confirms each")
        learned = []
        for email in train:
            state = process(agent, email)
            verdict = "correct" if state["label"] == email["label"] else f"wrong ({state['label']})"
            console.print(f"{email['id']} {email['subject'][:50]} → {verdict}")
            learned.append(learn_from_feedback(llm, memory, profile, email, email["label"], email["feedback"],
                                               lookup, ask_approval(True)))

        console.rule("3. Same test emails with what was learned")
        after = triage_all(agent, test)
        console.print(table(after, f"After learning — {accuracy(after):.0%}"))
        write_outputs(after, OUT)

        counts = memory.counts()
        lines = [f"# InboxMind learning demo — {llm.model}", "",
                 f"Held-out test emails: {len(test)} · training corrections: {len(train)}", "",
                 f"| | Accuracy |", "|---|---|", f"| Before learning | **{accuracy(before):.0%}** |",
                 f"| After learning | **{accuracy(after):.0%}** |", "",
                 f"Memory after training: {counts['facts']} facts, {counts['episodes']} examples, "
                 f"{counts['active_rules']} rules", "",
                 "| Email | Subject | Expected | Before | After | Decided by |", "|---|---|---|---|---|---|"]
        for b, a in zip(before, after):
            e = a["email"]
            mark = lambda s: f"{s['label']} {'✅' if s['label'] == e['label'] else '❌'}"
            lines.append(f"| {e['id']} | {e['subject']} | {e['label']} | {mark(b)} | {mark(a)} | "
                         + (f"rule R-{a['rule_id']}" if a.get("rule_id") else "LLM + memory") + " |")
        lines += ["", "## Rules learned (approved)", ""]
        for r in memory.rules():
            lines.append(f"- **R-{r.id}** ({r.kind}) {r.text}" + (f" — condition `{r.condition}` → {r.action}" if r.condition else ""))
        lines += ["", "## Facts learned", ""]
        lines += [f"- {f['subject']}: {f['fact']} (from {f['source']})" for f in memory.facts_about("")] or ["- none"]
        lines += ["", "## Drafts (not sent)", ""]
        for s in after:
            if s.get("draft"):
                flags = "".join(f"\n> ⚠️ {p}" for p in s.get("draft_problems", []))
                lines += [f"**{s['email']['id']} — {s['draft']['subject']}**{flags}", "", s["draft"]["body"], ""]
        report = OUT / "learning-report.md"
        report.write_text("\n".join(lines))
        console.print(f"before {accuracy(before):.0%} → after {accuracy(after):.0%} · report: {report}")
    console.print(f"[dim]{llm.usage.summary(llm.model)}[/dim]")


if __name__ == "__main__":
    main()
