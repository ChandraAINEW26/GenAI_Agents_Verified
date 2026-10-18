"""Transcript parsing and conversation metrics (pure code).

Transcript format: optional YAML front matter, then one turn per line:
    [mm:ss] Speaker: what they said

The front matter `speakers:` map says who is the rep. Metrics that the
original project asked the LLM to guess ("call engagement") are measured here
instead: talk ratio, questions asked, longest monologue, response pacing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

TURN_RE = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*([^:]+):\s*(.+)$")


@dataclass
class Turn:
    id: str
    start: int  # seconds from call start
    speaker: str
    role: str  # rep | customer
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Call:
    meta: dict
    turns: list[Turn]

    def turn(self, turn_id: str) -> Turn | None:
        return next((t for t in self.turns if t.id == turn_id), None)

    def as_prompt(self) -> str:
        return "\n".join(f"{t.id} [{t.role}] {t.speaker}: {t.text}" for t in self.turns)


def _seconds(m: re.Match) -> int:
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def parse_transcript(text: str) -> Call:
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    roles = {name.lower(): role for name, role in (meta.get("speakers") or {}).items()}
    rep_name = str(meta.get("rep", "")).split()[0].lower() if meta.get("rep") else ""

    turns: list[Turn] = []
    for line in body.strip().splitlines():
        m = TURN_RE.match(line.strip())
        if not m:
            if turns and line.strip():  # continuation of the previous turn
                turns[-1].text += " " + line.strip()
            continue
        speaker = m.group(4).strip()
        role = roles.get(speaker.lower()) or ("rep" if speaker.lower() == rep_name else "customer")
        turns.append(Turn(id=f"T{len(turns) + 1:02d}", start=_seconds(m), speaker=speaker, role=role,
                          text=m.group(5).strip()))
    if not turns:
        raise ValueError("No turns found. Expected lines like '[01:23] Name: text'.")
    return Call(meta=meta, turns=turns)


def load_call(path: Path) -> Call:
    call = parse_transcript(path.read_text())
    call.meta.setdefault("call_id", path.stem)
    return call


def _duration(call: Call, index: int) -> int:
    turns = call.turns
    if index + 1 < len(turns):
        return max(turns[index + 1].start - turns[index].start, 1)
    return max(round(turns[index].words / 2.5), 1)  # last turn: ~150 words/minute


def conversation_metrics(call: Call) -> dict:
    rep_words = sum(t.words for t in call.turns if t.role == "rep")
    total_words = sum(t.words for t in call.turns) or 1
    rep_turns = [t for t in call.turns if t.role == "rep"]
    customer_turns = [t for t in call.turns if t.role == "customer"]
    longest = max(((t, _duration(call, i)) for i, t in enumerate(call.turns) if t.role == "rep"),
                  key=lambda x: x[1], default=(None, 0))
    end = call.turns[-1].start + _duration(call, len(call.turns) - 1)
    return {
        "duration_seconds": end,
        "turns": len(call.turns),
        "rep_talk_ratio": round(rep_words / total_words, 2),
        "rep_questions": sum(t.text.count("?") for t in rep_turns),
        "customer_questions": sum(t.text.count("?") for t in customer_turns),
        "longest_rep_monologue_seconds": longest[1],
        "longest_rep_monologue_turn": longest[0].id if longest[0] else None,
        "customer_speakers": sorted({t.speaker for t in customer_turns}),
    }


def call_to_dict(call: Call) -> dict:
    return {"meta": call.meta, "turns": [asdict(t) for t in call.turns]}


def call_from_dict(data: dict) -> Call:
    return Call(meta=data["meta"], turns=[Turn(**t) for t in data["turns"]])
