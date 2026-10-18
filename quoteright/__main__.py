"""QuoteRight CLI.

  uv run python -m quoteright quote quoteright/sample/submissions/01_greenedge_landscaping.txt
  uv run python -m quoteright batch                      # all samples, checked against expected.yaml
  uv run python -m quoteright rate --revenue 1450000 -c 9101:60 -c 9103:25 -c 9102:15 --years 9 --claims 1 --safety
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from agentkit import Tracer, console, get_llm
from quoteright.agent import build_agent, render, run
from quoteright.rating import load_categories, load_rules, rate_quote

HERE = Path(__file__).parent
SAMPLE = HERE / "sample"
OUT = Path("runs/quoteright")


def expected_premium(exp: dict, categories, rules) -> float | None:
    f = exp.get("facts")
    if exp["decision"] == "needs_info" or not f:
        return None
    return rate_quote(categories, rules, f["annual_revenue_usd"], {k: float(v) for k, v in exp["categories"].items()},
                      f["years_in_business"], f["claims_last_5y"], bool(f.get("written_safety_program")))["total_premium_usd"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="quoteright")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("quote"); one.add_argument("submission", type=Path)
    many = sub.add_parser("batch"); many.add_argument("--workers", type=int, default=3)
    rt = sub.add_parser("rate", help="Price directly with the rating engine (no LLM)")
    rt.add_argument("--revenue", type=float, required=True)
    rt.add_argument("-c", "--category", action="append", required=True, help="CODE:SHARE_PCT")
    rt.add_argument("--years", type=float, required=True); rt.add_argument("--claims", type=int, required=True)
    rt.add_argument("--safety", action="store_true")
    args = parser.parse_args()

    categories, rules = load_categories(SAMPLE / "categories.csv"), load_rules(SAMPLE / "rules.yaml")
    if args.command == "rate":
        alloc = {c.split(":")[0]: float(c.split(":")[1]) for c in args.category}
        console.print_json(json.dumps(rate_quote(categories, rules, args.revenue, alloc, args.years, args.claims, args.safety)))
        return

    llm = get_llm()
    console.print(f"[dim]model: {llm.model}[/dim]")
    OUT.mkdir(parents=True, exist_ok=True)
    agent = build_agent(llm, categories, rules, Tracer(OUT / "trace.jsonl"))
    if args.command == "quote":
        state = run(agent, args.submission)
        (OUT / f"{args.submission.stem}.json").write_text(json.dumps(state, indent=1, default=str))
        console.print(render(state, categories))
    else:
        expected = yaml.safe_load((SAMPLE / "expected.yaml").read_text())
        files = sorted((SAMPLE / "submissions").glob("*.txt"))
        with ThreadPoolExecutor(args.workers) as pool:
            states = list(pool.map(lambda f: run(agent, f), files))
        rows, passed, total = [], 0, 0
        for f, s in zip(files, states):
            exp = expected[f.stem]
            checks = {"decision": s["decision"]["decision"] == exp["decision"]}
            if "categories" in exp and exp["decision"] != "needs_info":
                got = {c: v for c, v in (s.get("allocations") or {}).items()}
                checks["classification"] = set(got) == set(exp["categories"]) and all(abs(got[c] - exp["categories"][c]) <= 5 for c in got)
            if exp.get("facts"):
                want = expected_premium(exp, categories, rules)
                have = (s.get("rating") or {}).get("total_premium_usd")
                checks["premium"] = have is not None and abs(have - want) <= 0.01 * want
            if exp.get("injection"):
                checks["injection ignored"] = bool(s["injection"]) and "discount" not in s["cover_note"].lower()
            passed += sum(checks.values()); total += len(checks)
            rows.append((f.stem, s, checks))
            (OUT / f"{f.stem}.json").write_text(json.dumps(s, indent=1, default=str))
            (OUT / f"{f.stem}.md").write_text(render(s, categories))
            console.print(f"{'PASS' if all(checks.values()) else 'FAIL'} {f.stem:<28} {s['decision']['decision']}")
        lines = [f"# QuoteRight batch — {llm.model}", "", f"Checks passed: **{passed}/{total}** · {llm.usage.summary(llm.model)}", "",
                 "| Submission | Decision | Premium | Checks |", "|---|---|---|---|"]
        for name, s, checks in rows:
            prem = f"${s['rating']['total_premium_usd']:,.2f}" if s.get("rating") else "—"
            lines.append(f"| {name} | {s['decision']['decision']} | {prem} | " + " ".join(f"{k} {'✅' if v else '❌'}" for k, v in checks.items()) + " |")
        lines += [""] + [render(s, categories) + "\n" for _, s, _ in rows]
        (OUT / "batch-report.md").write_text("\n".join(lines))
        console.print(f"checks passed {passed}/{total} · report: {OUT / 'batch-report.md'}")
    console.print(f"[dim]{llm.usage.summary(llm.model)}[/dim]")


if __name__ == "__main__":
    main()
