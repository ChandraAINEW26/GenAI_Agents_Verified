"""CiteSure CLI.

  uv run python -m citesure ask "Who does Kestrel's account manager report to?"
  uv run python -m citesure ask --corpus path/to/docs "your question"
  uv run python -m citesure eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentkit import Tracer, console, get_llm
from citesure.agent import ask, build_agent, render
from citesure.evaluate import run_eval
from citesure.index import CorpusIndex

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(prog="citesure", description="Controllable, citation-checked RAG agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_cmd = sub.add_parser("ask", help="Answer one question")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--corpus", type=Path, default=HERE / "sample_corpus")
    ask_cmd.add_argument("--max-steps", type=int, default=6)
    ask_cmd.add_argument("--trace", type=Path, default=Path("runs/citesure-trace.jsonl"))

    eval_cmd = sub.add_parser("eval", help="Run the golden question set")
    eval_cmd.add_argument("--corpus", type=Path, default=HERE / "sample_corpus")
    eval_cmd.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    eval_cmd.add_argument("--report", type=Path, default=Path("runs/citesure-eval.md"))
    eval_cmd.add_argument("--max-steps", type=int, default=6)

    args = parser.parse_args()
    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")

    if args.command == "ask":
        tracer = Tracer(args.trace)
        index = CorpusIndex.build(args.corpus, llm)
        state = ask(build_agent(llm, index, tracer), args.question, args.max_steps)
        console.print()
        console.print(render(state))
        console.print(f"[dim]{llm.usage.summary(llm.model)} · trace: {args.trace}[/dim]")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        summary = run_eval(llm, args.corpus, args.golden, args.report, args.max_steps)
        console.print(summary)
        console.print(f"report: {args.report}")


if __name__ == "__main__":
    main()
