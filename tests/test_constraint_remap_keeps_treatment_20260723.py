"""Constraint remap must not wipe treatment to empty shell."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    CONSERVATIVE_FALLBACK_TREATMENT,
    build_conservative_fallback_plan,
    diagnosis_supportive_treatment_plan,
    reconcile_selected_diagnosis_plan,
)


def test_constraint_remap_keeps_named_supportive_plan_not_empty_shell() -> None:
    plan, reasoning = reconcile_selected_diagnosis_plan(
        {"normalized_diagnosis": "角膜炎"},
        selected_diagnosis="巩膜炎",
        treatment_plan="针对角膜炎使用抗生素滴眼液。",
        reasoning="原模型认为角膜炎",
        default_reasoning="",
    )
    assert "巩膜炎" in plan or "眼科" in plan
    assert "信息不足" not in plan
    assert "候选疾病约束" in reasoning or "巩膜炎" in reasoning
    assert "抗生素滴眼液" not in plan  # do not keep wrong-label plan


def test_diagnosis_supportive_plans_cover_common_labels() -> None:
    for name in ["慢性鼻炎", "巩膜炎", "华支睾吸虫病", "痛风", "哮喘"]:
        plan = diagnosis_supportive_treatment_plan(name)
        assert "针对" in plan
        assert "信息不足" not in plan


def test_conservative_fallback_constant_preserved_for_verifier_path() -> None:
    plan, reason = build_conservative_fallback_plan("慢性鼻炎")
    assert plan == CONSERVATIVE_FALLBACK_TREATMENT
    assert "终检" in reason or "保守" in reason
