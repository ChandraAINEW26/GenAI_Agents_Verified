"""Three kinds of memory, persisted per user in SQLite.

- semantic   facts about people and accounts, each linked to the email that stated it
- episodic   past emails with the label the user confirmed or corrected, and why
- procedural rules the user approved. Triage rules have a structured condition
             that code applies before any LLM call; style rules are guidance for
             drafting. Rules are never deleted: retiring one keeps it in history.

Search over episodes uses BM25 (agentkit.text), so the few-shot examples the
triage model sees are the past emails most similar to the new one.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from agentkit.text import BM25, Passage

SCHEMA = """
create table if not exists facts (
    id integer primary key, user text, subject text, fact text, source_email text, created real);
create table if not exists episodes (
    id integer primary key, user text, email_id text, sender text, subject text, snippet text,
    label text, reason text, created real);
create table if not exists rules (
    id integer primary key, user text, kind text, condition text, action text, text text,
    status text, source_email text, created real, retired real);
"""


@dataclass
class Rule:
    id: int
    kind: str  # triage | style
    condition: dict  # sender_domain / sender / subject_contains / body_contains (all must match)
    action: str  # ignore | notify | respond (triage) or "" (style)
    text: str
    status: str
    source_email: str


def sender_address(sender: str) -> str:
    return sender.split("<")[-1].rstrip(">").strip().lower()


def rule_matches(condition: dict, email: dict) -> bool:
    """A triage rule matches when every condition it states holds (case-insensitive)."""
    if not condition:
        return False
    address = sender_address(email["from"])
    checks = {
        "sender": lambda v: address == v.lower(),
        "sender_domain": lambda v: address.endswith("@" + v.lower().lstrip("@")) or address.endswith("." + v.lower()),
        "subject_contains": lambda v: v.lower() in email["subject"].lower(),
        "body_contains": lambda v: v.lower() in email["body"].lower(),
    }
    return all(checks[key](value) for key, value in condition.items() if value and key in checks)


class Memory:
    def __init__(self, path: Path, user: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.user = user

    # --- semantic ---
    def add_fact(self, subject: str, fact: str, source_email: str) -> bool:
        exists = self.db.execute("select 1 from facts where user=? and lower(fact)=lower(?)", (self.user, fact)).fetchone()
        if exists:
            return False
        self.db.execute("insert into facts (user, subject, fact, source_email, created) values (?,?,?,?,?)",
                        (self.user, subject, fact, source_email, time.time()))
        self.db.commit()
        return True

    def facts_about(self, *keys: str) -> list[dict]:
        rows = self.db.execute("select subject, fact, source_email from facts where user=?", (self.user,)).fetchall()
        keys_l = [k.lower() for k in keys if k]
        return [{"subject": s, "fact": f, "source": e} for s, f, e in rows
                if not keys_l or any(k in s.lower() or k in f.lower() for k in keys_l)]  # no keys: everything

    # --- episodic ---
    def add_episode(self, email: dict, label: str, reason: str) -> None:
        self.db.execute("delete from episodes where user=? and email_id=?", (self.user, email["id"]))
        self.db.execute(
            "insert into episodes (user, email_id, sender, subject, snippet, label, reason, created) "
            "values (?,?,?,?,?,?,?,?)",
            (self.user, email["id"], email["from"], email["subject"], email["body"][:300], label, reason, time.time()))
        self.db.commit()

    def similar_episodes(self, email: dict, k: int = 3) -> list[dict]:
        rows = self.db.execute("select email_id, sender, subject, snippet, label, reason from episodes where user=?",
                               (self.user,)).fetchall()
        if not rows:
            return []
        episodes = {r[0]: dict(zip(("email_id", "sender", "subject", "snippet", "label", "reason"), r)) for r in rows}
        passages = [Passage(id=e["email_id"], doc=e["email_id"], store="episodes",
                            text=f"{e['sender']} {e['subject']} {e['snippet']}") for e in episodes.values()]
        hits = BM25(passages).search(f"{email['from']} {email['subject']} {email['body']}", k)
        return [episodes[p.id] for p, _ in hits]

    # --- procedural ---
    def add_rule(self, kind: str, condition: dict, action: str, text: str, source_email: str) -> int:
        cursor = self.db.execute(
            "insert into rules (user, kind, condition, action, text, status, source_email, created) "
            "values (?,?,?,?,?,'active',?,?)",
            (self.user, kind, json.dumps(condition), action, text, source_email, time.time()))
        self.db.commit()
        return cursor.lastrowid

    def rules(self, include_retired: bool = False) -> list[Rule]:
        query = "select id, kind, condition, action, text, status, source_email from rules where user=?"
        if not include_retired:
            query += " and status='active'"
        return [Rule(i, k, json.loads(c), a, t, s, e) for i, k, c, a, t, s, e in
                self.db.execute(query + " order by id", (self.user,)).fetchall()]

    def retire_rule(self, rule_id: int) -> None:
        self.db.execute("update rules set status='retired', retired=? where user=? and id=?",
                        (time.time(), self.user, rule_id))
        self.db.commit()

    def matching_triage_rule(self, email: dict) -> Rule | None:
        matches = [r for r in self.rules() if r.kind == "triage" and rule_matches(r.condition, email)]
        # Most specific rule wins (more conditions), then the newest.
        return max(matches, key=lambda r: (len([v for v in r.condition.values() if v]), r.id), default=None)

    def counts(self) -> dict:
        def n(table: str, extra: str = "") -> int:
            return self.db.execute(f"select count(*) from {table} where user=? {extra}", (self.user,)).fetchone()[0]
        return {"facts": n("facts"), "episodes": n("episodes"), "active_rules": n("rules", "and status='active'")}

    def reset(self) -> None:
        for table in ("facts", "episodes", "rules"):
            self.db.execute(f"delete from {table} where user=?", (self.user,))
        self.db.commit()
