"""CallCoach CLI.

  uv run python -m callcoach analyze callcoach/sample_calls/sunpeak_expansion.txt
  uv run python -m callcoach batch callcoach/sample_calls          # every call + team report
  uv run python -m callcoach transcribe call.m4a --rep "Priya Shah" --account "Acme"   # needs --extra audio
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentkit import Tracer, console, get_llm
from callcoach.pipeline import analyse_call, team_report
from callcoach.transcribe import transcribe_audio

OUT = Path("runs/callcoach")


def main() -> None:
    parser = argparse.ArgumentParser(prog="callcoach", description="Sales call intelligence")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("analyze", help="Analyse one transcript")
    one.add_argument("transcript", type=Path)
    many = sub.add_parser("batch", help="Analyse every .txt transcript in a folder and write a team report")
    many.add_argument("folder", type=Path)
    many.add_argument("--workers", type=int, default=3)
    audio = sub.add_parser("transcribe", help="Turn an audio file into a transcript, then analyse it")
    audio.add_argument("audio", type=Path)
    audio.add_argument("--rep", required=True)
    audio.add_argument("--account", required=True)
    audio.add_argument("--whisper-model", default="base")
    args = parser.parse_args()

    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")

    if args.command == "transcribe":
        text = transcribe_audio(args.audio, llm, args.rep, args.account, args.whisper_model)
        path = OUT / f"{args.audio.stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        console.print(f"transcript: {path}")
        args.command, args.transcript = "analyze", path

    if args.command == "analyze":
        state = analyse_call(llm, args.transcript, OUT, Tracer(OUT / "trace.jsonl"))
        console.rule("Done")
        for label, path in state["outputs"].items():
            console.print(f"{label}: {path}")
    else:
        files = sorted(args.folder.glob("*.txt"))
        tracer = Tracer(OUT / "trace.jsonl")
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda f: analyse_call(llm, f, OUT, tracer), files))
        report = OUT / "team-report.md"
        report.write_text(team_report(results))
        console.rule("Done")
        console.print(f"{len(results)} calls analysed · team report: {report}")
    console.print(f"[dim]{llm.usage.summary(llm.model)}[/dim]")


if __name__ == "__main__":
    main()
