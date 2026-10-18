"""TalentScout tests - offline, using FakeLLM. Run: uv run pytest -q"""

from pathlib import Path

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentkit import FakeLLM
from talentscout.redact import lint_job_description, redact_cv
from talentscout.workflow import CVAssessment, build_workflow, finalise_score, rank_candidates

SAMPLE = Path(__file__).parents[1] / "talentscout" / "sample"
JOB = yaml.safe_load((SAMPLE / "job.yaml").read_text())


def test_redaction_removes_identity_and_protected_attributes():
    raw = (SAMPLE / "cvs" / "anika_rao.md").read_text()
    blind = redact_cv(raw)
    for secret in ["Anika", "Rao", "anika.rao@example.com", "555 0101", "1993", "Female", "Married", "Koramangala",
                   "linkedin.com"]:
        assert secret not in blind.text, secret
    assert " she " not in f" {blind.text.lower()} "
    assert "7 years of Python" in blind.text  # the evidence the scorer needs survives
    assert {"name", "email", "phone", "gender", "date of birth"} <= set(blind.removed)


def test_jd_lint_uses_word_boundaries():
    assert lint_job_description("We want a rockstar; he will lead.") == ["'rockstar' → consider 'skilled engineer'",
                                                                         "'he' → consider 'they'"]
    assert lint_job_description("The team builds the platform.") == []


def assessment(score=5, evidence="7 years of Python experience", met=True):
    return CVAssessment(
        criteria=[{"criterion": name, "score": score, "evidence": [f'"{evidence}"']} for name in JOB["rubric"]],
        must_haves=[{"requirement": r, "met": met, "evidence": evidence} for r in JOB["must_have"]],
        years_experience=7, strengths=["Python"], gaps=[], summary="ok",
    )


def test_unverified_evidence_is_capped_and_unproven_must_haves_knock_out():
    cv = "The candidate has 7 years of Python experience."
    good = finalise_score(assessment(), JOB, cv)
    assert good["weighted"] == 5.0 and not good["knocked_out"]

    invented = finalise_score(assessment(evidence="Led a team of 40 at Google"), JOB, cv)
    assert all(c["score"] == 2 and c["flag"] for c in invented["criteria"])
    assert invented["weighted"] == 2.0
    assert len(invented["knocked_out"]) == len(JOB["must_have"])  # "met" without real evidence does not count


def test_ranking_puts_knockouts_last_and_respects_threshold():
    scores = [{"id": "C-01", "weighted": 3.0, "knocked_out": []},
              {"id": "C-02", "weighted": 4.9, "knocked_out": ["Python"]},
              {"id": "C-03", "weighted": 4.1, "knocked_out": []}]
    ranking, shortlist = rank_candidates(scores, JOB)
    assert [s["id"] for s in ranking] == ["C-03", "C-01", "C-02"]
    assert shortlist == ["C-03"]  # C-01 is under the 3.2 threshold


def test_workflow_pauses_for_reviews_and_scorer_never_sees_names(tmp_path):
    jd_drafts = []
    llm = FakeLLM({
        "JobDescription": lambda s, p: jd_drafts.append(p) or {"markdown": f"# JD v{len(jd_drafts)}"},
        "CVAssessment": lambda s, p: assessment(evidence="Python").model_dump(),
        "InterviewKit": lambda s, p: {"questions": [{"question": "Tell me about RAG", "probes": "RAG",
                                                     "strong_answer": "specifics"}]},
        "Message": lambda s, p: {"subject": "Hello", "body": "Thanks"},
    })
    app = build_workflow(llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    result = app.invoke({"job_path": str(SAMPLE / "job.yaml"), "cv_dir": str(SAMPLE / "cvs"),
                         "out_dir": str(tmp_path)}, config)
    assert result["__interrupt__"][0].value["kind"] == "jd_review"

    result = app.invoke(Command(resume={"approved": False, "feedback": "mention the hybrid policy"}), config)
    assert "mention the hybrid policy" in jd_drafts[-1] and len(jd_drafts) == 2  # revised, asked again
    result = app.invoke(Command(resume={"approved": True}), config)
    assert result["__interrupt__"][0].value["kind"] == "shortlist_review"

    result = app.invoke(Command(resume={"approved_ids": ["C-02"]}), config)  # human override
    assert result["shortlist"] == ["C-02"]
    assert (tmp_path / "report.md").exists()
    assert sorted(p.name for p in (tmp_path / "outbox").iterdir())[1] == "C-02-invite.md"

    scorer_prompts = [p for name, p in llm.calls if name == "CVAssessment"]
    assert len(scorer_prompts) == 5
    for prompt in scorer_prompts:
        for name in ["Anika", "Tomás", "Li Wei", "Sara", "Omar", "@example", "Gender"]:
            assert name not in prompt
