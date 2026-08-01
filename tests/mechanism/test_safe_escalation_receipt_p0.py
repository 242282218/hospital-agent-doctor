"""P0: safe-escalation receipts must never self-certify passed=True.

`validate_safe_escalation_plan` only checks axis keyword closure. Before this
gate, finalize synthesized `{"passed": True, "issues": []}` whenever that keyword
check passed, so a red-flag plan that final_verifier could not converge was
reported as fully verified. These tests lock the two-tier contract:

- verified tier: passed=True only when final_verifier converged on the SAME text
- unverified tier: plan is still submitted (withholding a red-flag plan is worse)
  but passed=False and verification_status="axis_closure_only"
"""

from __future__ import annotations

import pytest

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    finalize_treatment_with_verified_fallback,
    unverified_safe_escalation_receipt,
    verified_safe_escalation_receipt,
)

AXIS_ID = "active_upper_gi_bleed"
AXIS_EVIDENCE = ["肝硬化门静脉高压", "突然呕血", "反复黑便"]


def _axis_closing_plan() -> str:
    """Real builder output, the only text that satisfies axis closure validation."""
    plan, _reasoning = build_safe_escalation_plan(
        axis_id=AXIS_ID,
        closure_requirement="urgent_hemostasis_and_resuscitation",
        evidence=AXIS_EVIDENCE,
        existing_treatment="",
    )
    return plan


def _red_flag_features() -> dict:
    return {
        "diagnosis_axes": [
            {
                "axis_id": "active_upper_gi_bleed",
                "source": "rule",
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "red_flag",
                "closure_requirement": "urgent_hemostasis_and_resuscitation",
                "evidence": ["肝硬化门静脉高压", "突然呕血", "反复黑便"],
                "rule_candidate_official_names": [],
                "candidate_official_names": [],
            }
        ],
        "diagnosis_candidate_records": [
            {"disease": "心律失常", "role": "current_problem", "score": 80},
        ],
        "candidate_diagnoses": ["心律失常"],
    }


def _failed_upstream() -> dict:
    return {
        "passed": False,
        "issues": [
            {
                "severity": "must_fix",
                "patchable": False,
                "problem": "未覆盖活动性上消化道出血急症处置",
            }
        ],
        "patched_treatment": "",
    }


def test_verifier_stalemate_never_yields_passed_true(monkeypatch) -> None:
    """The core regression: no verifier convergence => no passed=True."""
    monkeypatch.setattr(
        "agent.legacy_orchestrator.converge_verified_treatment",
        lambda **_kwargs: None,
    )
    _diagnosis, plan, _reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="心律失常",
        treatment_plan="仅观察心悸。",
        reasoning="误判为心律失常",
        verifier_result=_failed_upstream(),
        examinations=["全血细胞计数（CBC）"],
        official_diseases=["心律失常", "肝硬化"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=_red_flag_features(),
        safety_profiles=[],
    )
    assert receipt.get("passed") is False
    assert receipt.get("verified") is False
    assert receipt.get("verification_status") == "axis_closure_only"
    assert receipt.get("degraded") == "safe_escalation_unverified"
    # The clinical plan is still submitted; only the proof claim is withheld.
    assert plan
    assert "内镜" in plan
    codes = {i.get("code") for i in receipt.get("issues") or []}
    assert "final_verifier_not_converged" in codes


def test_unverified_receipt_requires_axis_closure() -> None:
    """A plan that fails even axis keyword closure yields no receipt at all."""
    assert (
        unverified_safe_escalation_receipt(
            axis_id="active_upper_gi_bleed",
            axis_evidence=["突然呕血"],
            escalation_plan="回家观察，无需处理。",
            aligned_diagnosis="肝硬化",
            upstream_issues=[],
        )
        is None
    )


def test_unverified_receipt_rejects_empty_plan() -> None:
    assert (
        unverified_safe_escalation_receipt(
            axis_id="active_upper_gi_bleed",
            axis_evidence=["突然呕血"],
            escalation_plan="   ",
            aligned_diagnosis="肝硬化",
            upstream_issues=[],
        )
        is None
    )


def test_verified_receipt_returns_none_when_verifier_rejects(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.legacy_orchestrator.converge_verified_treatment",
        lambda **_kwargs: {"passed": False, "issues": [], "patched_treatment": "x"},
    )
    assert (
        verified_safe_escalation_receipt(
            axis_id="active_upper_gi_bleed",
            axis_evidence=["肝硬化门静脉高压", "突然呕血"],
            escalation_plan="立即急诊住院，急诊胃镜检查并止血，液体复苏。",
            aligned_diagnosis="肝硬化",
            examinations=[],
            official_diseases=["肝硬化"],
            examination_catalog={},
            exam_plan_trace=[],
            case_features=_red_flag_features(),
            safety_profiles=[],
            upstream_issues=[],
        )
        is None
    )


def test_verified_receipt_marks_passed_when_verifier_converges(monkeypatch) -> None:
    # Build the plan with the real builder so it satisfies axis keyword closure.
    plan, _reasoning = build_safe_escalation_plan(
        axis_id="active_upper_gi_bleed",
        closure_requirement="urgent_hemostasis_and_resuscitation",
        evidence=["肝硬化门静脉高压", "突然呕血"],
        existing_treatment="",
    )
    monkeypatch.setattr(
        "agent.legacy_orchestrator.converge_verified_treatment",
        lambda **_kwargs: {"passed": True, "issues": [], "patched_treatment": plan},
    )
    receipt = verified_safe_escalation_receipt(
        axis_id="active_upper_gi_bleed",
        axis_evidence=["肝硬化门静脉高压", "突然呕血"],
        escalation_plan=plan,
        aligned_diagnosis="肝硬化",
        examinations=[],
        official_diseases=["肝硬化"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=_red_flag_features(),
        safety_profiles=[],
        upstream_issues=[],
    )
    assert receipt is not None
    assert receipt["passed"] is True
    assert receipt["degraded"] == "safe_escalation"
    assert receipt["verification_status"] != "axis_closure_only"
