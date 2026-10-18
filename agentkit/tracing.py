"""Per-run trace: one JSON line per graph node, so you can see what the agent did.

Wrap a node with `tracer.node("name")(fn)`; the tracer records start/end time,
the keys the node changed, and any error. `tracer.print` gives a live view.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

console = Console()


class Tracer:
    def __init__(self, path: Path | None = None, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.events: list[dict[str, Any]] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def log(self, event: str, **data: Any) -> None:
        record = {"t": round(time.time(), 3), "event": event, **data}
        self.events.append(record)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    def print(self, message: str) -> None:
        if self.verbose:
            console.print(message)

    def node(self, name: str) -> Callable:
        def decorate(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(state, *args, **kwargs):
                started = time.time()
                self.print(f"[dim]→ {name}[/dim]")
                try:
                    update = fn(state, *args, **kwargs)
                except Exception as err:
                    # LangGraph uses exceptions for interrupts; record and re-raise all of them.
                    self.log("node_error", node=name, error=f"{type(err).__name__}: {err}")
                    raise
                keys = sorted(update.keys()) if isinstance(update, dict) else type(update).__name__
                self.log("node", node=name, ms=int((time.time() - started) * 1000), updated=keys)
                return update

            return wrapper

        return decorate
