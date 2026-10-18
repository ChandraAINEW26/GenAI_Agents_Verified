# InboxMind — email assistant that learns from your corrections

> Triages your inbox into ignore / notify / respond, drafts replies, and gets better as you correct it. What it
> learns is stored in SQLite, shown to you before it takes effect, and applied by code so the preference is guaranteed.

**Rebuilt from** GenAI_Agents #32 (Memory-Enhanced Email Agent with LangMem), which kept memories in RAM, let an
optimiser rewrite its prompts freely, and sent mail through a tool.

## Architecture

```mermaid
flowchart TD
    E([Incoming email]) --> RC["recall<br/><i>code: rule match? similar past emails, known facts</i>"]
    RC -->|approved rule matches| LB
    RC --> TG["triage<br/><i>LLM with rules + similar labelled examples</i>"]
    TG --> LB{label}
    LB -->|ignore / notify| DONE([digest or nothing])
    LB -->|respond| LF["learn_facts<br/><i>LLM proposes, code keeps only facts the email states</i>"]
    LF --> DR["draft<br/><i>calendar free slots, known facts, style rules</i>"]
    DR --> CK{{"check<br/><i>code: offered times are free,<br/>figures exist in email / memory</i>"}}
    CK --> OB(["outbox draft — never sent"])
    FB([You correct or confirm a label]) --> EP[("episodic: example + reason")]
    FB --> PR["propose rule<br/><i>LLM: structured condition + principle</i>"]
    PR --> PV{{"preview which past emails it changes<br/>you approve"}}
    PV --> RU[("procedural: versioned rules<br/>retire, never delete")]
    RU -.-> RC
    EP -.-> RC
    classDef code fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef llm fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class RC,CK,PV code
    class TG,LF,DR,PR llm
```

## Three memories

| Memory | Stores | Used for |
|---|---|---|
| Semantic | Facts about people and accounts, each linked to the email that stated it | Drafting with context |
| Episodic | Past emails with the label you confirmed and why | Few-shot examples chosen by BM25 similarity |
| Procedural | Approved rules: a structured condition (sender, domain, subject, body) plus a one-line principle | Code applies the condition before the LLM; the principle guides similar emails |

## Measured learning

Six training corrections, eight held-out emails the agent is never corrected on (`gpt-oss:120b-cloud`):

| | Accuracy |
|---|---|
| Before learning (empty memory) | **50%** |
| After learning | **100%** |

Report with rules, facts and drafts: [docs/examples/inboxmind-learning-report.md](../docs/examples/inboxmind-learning-report.md).

```bash
uv run python -m inboxmind learn-demo
uv run python -m inboxmind feedback --email T-04 --label notify --note "Failed builds on main matter"
uv run python -m inboxmind rules --all          # add --retire 3 to switch a rule off
```

Files: [`agent.py`](agent.py) graph, learning, checks · [`memory.py`](memory.py) SQLite stores · [`sample/`](sample/) inbox, calendar, profile.
