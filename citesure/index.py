"""Corpus ingestion and retrieval.

Three stores, as in the original project:
- chunks     paragraph-sized passages     id  "<doc>#c<n>"
- quotes     single sentences              id  "<doc>#q<n>"
- summaries  one LLM summary per document  id  "<doc>#summary"

Search is BM25 (pure Python, no vector DB needed). A query is searched in
several rewrites and the rankings are merged with Reciprocal Rank Fusion, which
recovers most of what dense retrieval adds on a small corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel

from agentkit import LLM
from agentkit.text import BM25, Passage, rrf, support_ratio, tokenize  # noqa: F401 (re-exported)

STORES = ("chunks", "quotes", "summaries")


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text()


def chunk_text(text: str, max_words: int = 120) -> list[str]:
    """One chunk per heading section; long sections split by paragraph.

    Each chunk is prefixed with its heading so a passage about "Kestrel Pharma"
    still says so after splitting.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    lines: list[str] = []

    def flush() -> None:
        if lines:
            sections[-1][1].append(" ".join(lines))
            lines.clear()

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            flush()
            sections.append((line.lstrip("# ").strip(), []))
        elif not line:
            flush()
        else:
            lines.append(line)
    flush()

    chunks: list[str] = []
    for heading, paragraphs in sections:
        current: list[str] = []
        for para in paragraphs:
            if current and len(" ".join(current + [para]).split()) > max_words:
                chunks.append("\n".join(filter(None, [heading, *current])))
                current = []
            current.append(para)
        if current:
            chunks.append("\n".join(filter(None, [heading, *current])))
    return chunks


def split_sentences(text: str) -> list[str]:
    body = re.sub(r"^#.*$", "", text, flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(body.split()))
    return [s.strip() for s in parts if len(s.split()) >= 4]


class SummarySchema(BaseModel):
    summary: str


SUMMARY_SYSTEM = (
    "You summarise one document for a retrieval index. Keep every name, number, date, "
    "identifier and relationship. Do not add anything that is not in the document."
)


class CorpusIndex:
    def __init__(self, passages: list[Passage]):
        self.passages = {p.id: p for p in passages}
        self.bm25 = {s: BM25([p for p in passages if p.store == s]) for s in STORES}

    def search(self, store: str, queries: list[str], k: int = 4) -> list[Passage]:
        rankings = [[p for p, _ in self.bm25[store].search(q, k * 2)] for q in queries if q.strip()]
        return rrf(rankings)[:k]

    def get(self, passage_id: str) -> Passage | None:
        return self.passages.get(passage_id)

    @classmethod
    def build(cls, corpus_dir: Path, llm: LLM, cache_dir: Path | None = None) -> "CorpusIndex":
        files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not files:
            raise FileNotFoundError(f"No .md, .txt or .pdf files in {corpus_dir}")
        texts = {f.stem: read_document(f) for f in files}

        # Summaries cost LLM calls, so cache them by corpus content + model.
        digest = hashlib.sha256(json.dumps(texts, sort_keys=True).encode() + llm.model.encode()).hexdigest()[:16]
        cache_dir = cache_dir or corpus_dir / ".index"
        cache_file = cache_dir / f"index-{digest}.json"
        if cache_file.exists():
            return cls([Passage(**p) for p in json.loads(cache_file.read_text())])

        passages: list[Passage] = []
        for doc, text in texts.items():
            for i, chunk in enumerate(chunk_text(text)):
                passages.append(Passage(id=f"{doc}#c{i}", doc=doc, store="chunks", text=chunk))
            for i, sentence in enumerate(split_sentences(text)):
                passages.append(Passage(id=f"{doc}#q{i}", doc=doc, store="quotes", text=sentence))
            summary = llm.structured(SUMMARY_SYSTEM, text, SummarySchema, effort="low").summary
            passages.append(Passage(id=f"{doc}#summary", doc=doc, store="summaries", text=summary))

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps([p.model_dump() for p in passages], indent=1))
        return cls(passages)
