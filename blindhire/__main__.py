"""BlindHire CLI.

  uv run python -m blindhire run --thread demo            # stops at each review for your input
  uv run python -m blindhire run --thread demo --auto-approve
  uv run python -m blindhire resume --thread demo         # continue a run you stopped at a review

Reviews are saved in runs/blindhire.sqlite, so you can quit at a review
(Ctrl+C) and resume later, even from a new terminal.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.markdown import Markdown
from rich.table import Table

from agentkit import Tracer, console, get_llm
from blindhire.workflow import build_workflow

HERE = Path(__file__).parent
DB = Path("runs/blindhire.sqlite")


def ask_human(payload: dict, auto: bool) -> dict:
    if payload["kind"] == "jd_review":
        console.rule("Job description review")
        console.print(Markdown(payload["jd"]))
        if payload["lint"]:
            console.print("[yellow]Inclusive-language check:[/yellow] " + "; ".join(payload["lint"]))
        if auto:
            return {"approved": True}
        answer = console.input("\nApprove? [y] or type the changes you want: ").strip()
        return {"approved": True} if answer.lower() in ("", "y", "yes") else {"approved": False, "feedback": answer}

    console.rule("Shortlist review")
    table = Table("ID", "Weighted /5", "Knocked out", "Summary")
    for row in payload["ranking"]:
        table.add_row(row["id"], f"{row['weighted']:.2f}", "; ".join(row["knocked_out"]) or "—", row["summary"])
    console.print(table)
    console.print(f"Proposed shortlist: {', '.join(payload['proposed']) or '(none)'}")
    if auto:
        return {"approved_ids": payload["proposed"]}
    answer = console.input("Press Enter to accept, or list IDs to interview (e.g. C-01,C-03): ").strip()
    return {"approved_ids": [i.strip() for i in answer.split(",") if i.strip()] or payload["proposed"]}


def drive(app, config: dict, first_input, auto: bool) -> dict:
    result = app.invoke(first_input, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = app.invoke(Command(resume=ask_human(payload, auto)), config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="blindhire", description="Blind-screening hiring pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--thread", default="demo", help="Run id; reuse it to resume")
        cmd.add_argument("--auto-approve", action="store_true", help="Accept every review without asking")
    run = sub.choices["run"]
    run.add_argument("--job", type=Path, default=HERE / "sample" / "job.yaml")
    run.add_argument("--cvs", type=Path, default=HERE / "sample" / "cvs")
    args = parser.parse_args()

    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")
    out_dir = Path("runs/blindhire") / args.thread
    out_dir.mkdir(parents=True, exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB, check_same_thread=False) as conn:
        app = build_workflow(llm, Tracer(out_dir / "trace.jsonl"), SqliteSaver(conn))
        config = {"configurable": {"thread_id": args.thread}}

        if args.command == "run":
            if app.get_state(config).values:
                raise SystemExit(f"Thread {args.thread!r} already exists; use `resume` or pick a new --thread.")
            first = {"job_path": str(args.job), "cv_dir": str(args.cvs), "out_dir": str(out_dir)}
        else:
            snapshot = app.get_state(config)
            if not snapshot.values:
                raise SystemExit(f"No saved run called {args.thread!r}.")
            if not snapshot.next:
                raise SystemExit(f"Run {args.thread!r} already finished: {snapshot.values.get('report_path')}")
            pending = [i for task in snapshot.tasks for i in task.interrupts]
            first = Command(resume=ask_human(pending[0].value, args.auto_approve)) if pending else None

        result = drive(app, config, first, args.auto_approve)

    console.rule("Done")
    console.print(f"Shortlist: {', '.join(result['shortlist']) or '(none)'}")
    console.print(f"Report: {result['report_path']}")
    console.print(f"Drafted messages (not sent): {out_dir / 'outbox'}")
    console.print(f"[dim]{llm.usage.summary(llm.model)}[/dim]")


if __name__ == "__main__":
    main()
