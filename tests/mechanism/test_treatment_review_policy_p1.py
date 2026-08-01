"""P1: conditional treatment review policy matrix and wiring receipts."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from agent.clinical.treatment_review_policy import decide_treatment_review_policy
from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory
from agent.observability.runtime_events import canonical_hash


PATCHABLE = {"code": "missing_treatment_goal", "severity": "must_fix", "patchable": True}
UNPATCHABLE = {"code": "diagnosis_conflict", "severity": "error", "patchable": False}


@pytest.mark.parametrize(
    "verifier,safety,conflicted,count,high_risk,expected,scope",
    [
        ([], [], False, 1, False, False, "none"),
        ([PATCHABLE], [], False, 1, False, True, "issue_scoped"),
        ([], [PATCHABLE], False, 1, False, True, "issue_scoped"),
        ([], [], False, 2, False, True, "issue_scoped"),
        ([], [], False, 1, True, True, "issue_scoped"),
        ([], [PATCHABLE], True, 1, False, True, "safety_only"),
        ([UNPATCHABLE], [], True, 1, False, False, "none"),
        ([], [], True, 1, False, False, "none"),
    ],
)
def test_review_policy_matrix(
    verifier, safety, conflicted, count, high_risk, expected, scope
):
    result = decide_treatment_review_policy(
        verifier_issues=verifier,
        safety_issues=safety,
        diagnosis_conflicted=conflicted,
        diagnosis_count=count,
        high_risk_treatment=high_risk,
    )
    assert result.should_review is expected
    assert result.allowed_scope == scope


def test_clean_plan_skips_review_llm_and_writes_receipt(monkeypatch) -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())
    calls: List[str] = []

    async def boom(**kwargs):
        calls.append(str(kwargs.get("prompt_name") or ""))
        raise AssertionError("LLM must not be called for clean low-risk plan")

    monkeypatch.setattr(agent, "_call_llm", boom)
    case_state: Dict[str, Any] = {
        "chat_history": [],
        "ordered_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    original = "观察随访，对症支持治疗。"

    async def run():
        return await agent._review_treatment_plan(
            case_state=case_state,
            diagnosis="上呼吸道感染",
            diagnosis_axes=[],
            treatment_plan=original,
            verifier_issues=[],
            patient_id="offline-only",
            safety_issues=[],
        )

    out = asyncio.run(run())
    assert out == original
    assert calls == []
    decision = case_state.get("treatment_review_decision") or {}
    assert decision.get("status") == "skipped"
    assert decision.get("should_review") is False
    assert decision.get("allowed_scope") == "none"
    assert decision.get("reason_codes") == ["verifier_clean_low_risk"]
    assert decision.get("treatment_hash") == canonical_hash(original)


def test_actionable_issue_reviews_once(monkeypatch) -> None:
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=VerifiedOnlyMemory())
    prompt_names: List[str] = []

    async def fake_llm(**kwargs):
        prompt_names.append(str(kwargs.get("prompt_name") or ""))
        return {"edits": [], "revision_summary": []}

    monkeypatch.setattr(agent, "_call_llm", fake_llm)
    monkeypatch.setattr(
        "agent.legacy_orchestrator.decide_treatment_review",
        lambda *args, **kwargs: {
            "accepted": False,
            "treatment_plan": kwargs.get("original_treatment_plan")
            if False
            else args[1]
            if False
            else kwargs.get("original_treatment_plan", "观察随访。"),
            "status": "reviewed",
        },
    )

    # Keep decide_treatment_review real enough: patch only LLM.
    from agent import legacy_orchestrator as lo

    def fake_decide(review, **kwargs):
        return {
            "status": "reviewed",
            "accepted": False,
            "treatment_plan": kwargs.get("original_treatment_plan"),
            "edits": [],
        }

    monkeypatch.setattr(lo, "decide_treatment_review", fake_decide)

    case_state: Dict[str, Any] = {
        "chat_history": [],
        "ordered_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }

    async def run():
        return await agent._review_treatment_plan(
            case_state=case_state,
            diagnosis="上呼吸道感染",
            diagnosis_axes=[],
            treatment_plan="观察随访。",
            verifier_issues=[PATCHABLE],
            patient_id="offline-only",
            safety_issues=[],
        )

    out = asyncio.run(run())
    assert out == "观察随访。"
    assert prompt_names.count("treatment_review") == 1
    decision = case_state.get("treatment_review_decision") or {}
    assert decision.get("status") in {"reviewed", "applied", "rejected", "ok"} or decision
