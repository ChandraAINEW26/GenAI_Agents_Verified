"""Text utilities shared by the agents: tokenising, BM25 search, rank fusion and a
cheap grounding check. Pure Python, no external services.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which who with "
    "what when does did how why".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class Passage(BaseModel):
    id: str
    doc: str
    store: str
    text: str


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        terms = tokenize(query)
        scored = []
        for passage, doc, length in zip(self.passages, self.docs, self.lengths):
            score = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            if score > 0:
                scored.append((passage, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def rrf(rankings: list[list[Passage]], k: int = 60) -> list[Passage]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across rankings."""
    scores: dict[str, float] = {}
    by_id: dict[str, Passage] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking):
            scores[passage.id] = scores.get(passage.id, 0.0) + 1 / (k + rank + 1)
            by_id[passage.id] = passage
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


def support_ratio(extract: str, source: str) -> float:
    """Share of the extract's content words that appear in the source text.

    A cheap, deterministic grounding check: a distilled extract that invents
    facts will contain words its source passage does not.
    """
    words = set(tokenize(extract))
    if not words:
        return 0.0
    return len(words & set(tokenize(source))) / len(words)
