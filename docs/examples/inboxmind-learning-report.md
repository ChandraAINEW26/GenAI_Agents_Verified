# InboxMind learning demo — gpt-oss:120b-cloud

Held-out test emails: 8 · training corrections: 6

| | Accuracy |
|---|---|
| Before learning | **50%** |
| After learning | **100%** |

Memory after training: 8 facts, 6 examples, 6 rules

| Email | Subject | Expected | Before | After | Decided by |
|---|---|---|---|---|---|
| T-01 | Principal ML Engineer opportunity - fully remote | ignore | notify ❌ | ignore ✅ | LLM + memory |
| T-02 | AI Weekly #213: retrieval, robotics foundation models | notify | notify ✅ | notify ✅ | LLM + memory |
| T-03 | Build passed: fleetos release-2.9 #4420 | ignore | notify ❌ | ignore ✅ | rule R-3 |
| T-04 | Build FAILED: fleetos main #4425 | notify | notify ✅ | notify ✅ | LLM + memory |
| T-05 | FYI: new warehouse lead in Basel | respond | notify ❌ | respond ✅ | rule R-4 |
| T-06 | Lunch & learn: robotics trivia quiz | ignore | notify ❌ | ignore ✅ | rule R-6 |
| T-07 | Can you review the CallCoach rubric by Thursday? | respond | respond ✅ | respond ✅ | LLM + memory |
| T-08 | Your October invoice is available | notify | notify ✅ | notify ✅ | LLM + memory |

## Rules learned (approved)

- **R-1** (triage) Ignore recruiter outreach — condition `{'sender_domain': 'talentbridge.io'}` → ignore
- **R-2** (triage) Notify me about AI Weekly newsletters — condition `{'sender': 'newsletter@aiweekly.news'}` → notify
- **R-3** (triage) Ignore emails about successful builds. — condition `{'subject_contains': 'Build passed'}` → ignore
- **R-4** (triage) Always reply to any email from kestrelpharma.com — condition `{'sender_domain': 'kestrelpharma.com'}` → respond
- **R-5** (triage) Board requests always need a reply. — condition `{'sender_domain': 'board.northwind.com'}` → respond
- **R-6** (triage) Ignore internal social event emails — condition `{'sender': 'events@northwind.com'}` → ignore

## Facts learned

- the board: the board would like one slide on AI strategy for the November meeting (from E-05)
- Jordan: Jordan is asked to meet on Tuesday October 21 for 30 minutes to go through a first draft (from E-05)
- Helen Ortiz: Helen Ortiz asks to meet on Tuesday October 21 for 30 minutes to go through a first draft (from E-05)
- Marta Keller: I have taken over as warehouse lead in Basel from Dana's team. (from T-05)
- Marta Keller: I'm your contact for anything on the floor from now on. (from T-05)
- Priya Shah: I've updated the coaching rubric weights. (from T-07)
- Jordan: Could you review them by Thursday October 23. (from T-07)
- Priya Shah: I can roll them out to the team. (from T-07)

## Drafts (not sent)

**T-05 — Re: FYI: new warehouse lead in Basel**

Hi Marta,

Thanks for the update. Noted that you’re now the point of contact for the Basel warehouse. I’ll reach out as needed.

Best,
Jordan

**T-07 — Re: Can you review the CallCoach rubric by Thursday?**

Hi Priya,

Thanks for the update. I’ll review the rubric and get back to you by [date].

Best,
Jordan
