# Example outputs

Real outputs from runs with `gpt-oss:120b-cloud` on the bundled fictional data. Nothing here is hand-edited.

| File | Agent | What it shows |
|---|---|---|
| [citesure-eval.md](citesure-eval.md) | CiteSure | Golden-set evaluation: per-question verdicts, source recall, steps, judge notes, full answers |
| [citesure-trace.jsonl](citesure-trace.jsonl) | CiteSure | Node-by-node trace: node name, duration, state keys changed |
| [blindhire-report.md](blindhire-report.md) | BlindHire | Ranking, per-criterion scores with verified CV quotes, what was hidden from the scorer, interview kits, approved JD |
| [blindhire-outbox-invite.md](blindhire-outbox-invite.md) | BlindHire | Drafted interview invitation (not sent) |
| [blindhire-outbox-decline.md](blindhire-outbox-decline.md) | BlindHire | Drafted decline (not sent) |
| [callcoach-CALL-1029-report.md](callcoach-CALL-1029-report.md) | CallCoach | Weak renewal call: scorecard, risk, objections, MEDDPICC, next steps, sentiment arc |
| [callcoach-CALL-1029-crm.json](callcoach-CALL-1029-crm.json) | CallCoach | CRM update payload for the same call |
| [callcoach-CALL-1029-followup-email.md](callcoach-CALL-1029-followup-email.md) | CallCoach | Follow-up email draft with [placeholders] for details the rep must supply |
| [callcoach-CALL-1031-report.md](callcoach-CALL-1031-report.md) | CallCoach | Strong expansion call for comparison |
| [callcoach-team-report.md](callcoach-team-report.md) | CallCoach | Per-rep averages, calls ranked by score, objection themes |
