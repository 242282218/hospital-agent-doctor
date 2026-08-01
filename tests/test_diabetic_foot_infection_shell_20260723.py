"""Diabetic foot infection shell must start antibiotics, not support-only."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    diagnosis_supportive_treatment_plan,
    validate_safe_escalation_plan,
)


def test_diabetic_foot_infection_shell_requires_antibiotics() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="diabetic_foot_infection",
        closure_requirement="根据药敏调整抗生素并清创",
        evidence=["糖尿病", "足部红肿", "溃破"],
        existing_treatment="",
    )
    assert "抗生素" in plan or "抗感染" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="diabetic_foot_infection",
        evidence=["糖尿病", "足部红肿", "溃破"],
    )
    support_only = (
        "针对“蜂窝织炎”：以对症支持与危险信号监测为主，尽快相关专科评估并完善关键检查后制定特异治疗。"
    )
    assert not validate_safe_escalation_plan(
        support_only,
        axis_id="diabetic_foot_infection",
        evidence=["糖尿病", "足部红肿", "溃破"],
    )


def test_cellulitis_supportive_plan_mentions_antibiotics() -> None:
    plan = diagnosis_supportive_treatment_plan("蜂窝织炎")
    assert "抗感染" in plan or "抗生素" in plan
