# ResolveDesk — support agent that answers from the help centre or hands off to a human

> Answers support tickets from a knowledge base and the customer's account record with checked citations. Anything
> risky, uncovered or unverified goes to the right human queue with a ready-written handoff.

**Rebuilt from** GenAI_Agents #9 (Customer Support Agent), which classified a query, wrote a generic model-knowledge
reply and escalated only on negative sentiment.

## Architecture

```mermaid
flowchart TD
    T([Ticket + customer id]) --> IN["intake<br/><i>code: redact cards, IBANs, secrets<br/>detect prompt injection, load account</i>"]
    IN --> TR["triage<br/><i>LLM: category, urgency, sentiment,<br/>legal / churn threat, refund amount</i>"]
    TR -->|security or injection| RT
    TR --> RE["retrieve<br/><i>BM25 + RRF over help-centre articles</i>"]
    RE --> DR["draft<br/><i>LLM: reply + citations from passages<br/>and account record</i>"]
    DR --> VE{{"verify<br/><i>code: cited ids were retrieved,<br/>figures exist in sources</i>"}}
    VE --> RT{{"route<br/><i>code: escalation policy</i>"}}
    RT -->|no reasons| AU(["auto reply"])
    RT -->|any reason| ES["escalate<br/><i>LLM handoff note, code sets queue + priority floor</i>"]
    ES --> HQ(["human queue + customer acknowledgement with SLA"])
    classDef code fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef llm fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class IN,VE,RT code
    class TR,DR,ES llm
```

## What changed from the original

| Original | ResolveDesk |
|---|---|
| Model-knowledge answers | Answers only from KB passages + account record; citations checked |
| Card numbers and secrets sent to the LLM | Redacted in code first (Luhn-checked cards, IBAN, CVV, passwords) |
| Escalate on negative sentiment only | Written policy: security, critical urgency, legal/churn threat, very negative sentiment, refund over USD 500, injection, no KB coverage, low confidence, failed verification |
| Escalation = a canned sentence | Handoff note (summary, context, wants, recommended action), queue, priority (policy can raise, never lower) and an SLA-aware customer message |
| No protection from hostile text | Ticket wrapped as data; injection and security tickets never reach the drafting model |

## Results (10 sample tickets, `gpt-oss:120b-cloud`)

23/23 checks passed: 5 tickets answered automatically with the right article (password reset, offline robot, duplicate
add-on charge, card update, SLA credit), 5 escalated for the right reason (refund over limit, legal threat, security
incident, prompt injection, no KB coverage). See [docs/examples/resolvedesk-batch-report.md](../docs/examples/resolvedesk-batch-report.md).

```bash
uv run python -m resolvedesk ticket --customer C-1003 "I forgot my password and the link expired"
uv run python -m resolvedesk batch
```

Files: [`agent.py`](agent.py) graph, policy, verification · [`guard.py`](guard.py) input guard · [`sample/`](sample/) KB, accounts, tickets.
