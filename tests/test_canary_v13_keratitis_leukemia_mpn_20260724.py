"""Keratitis / leukemia / MPN shells must not be support-only."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    diagnosis_supportive_treatment_plan,
    validate_safe_escalation_plan,
)


def test_keratitis_shell_has_eye_protection() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="exposure_keratoconjunctivitis",
        closure_requirement="荧光素染色",
        evidence=["眼红", "畏光", "暴露"],
        existing_treatment="",
    )
    assert any(m in plan for m in ["眼膏", "湿房", "润滑", "荧光素"])
    assert validate_safe_escalation_plan(
        plan,
        axis_id="exposure_keratoconjunctivitis",
        evidence=["眼红", "畏光", "暴露"],
    )


def test_leukemia_shell_mentions_marrow_and_support() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="acute_leukemia_suspected",
        closure_requirement="骨髓穿刺",
        evidence=["发热", "出血", "白细胞异常"],
        existing_treatment="",
    )
    assert "骨髓" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="acute_leukemia_suspected",
        evidence=["发热", "出血", "白细胞异常"],
    )
    generic = diagnosis_supportive_treatment_plan("急性淋巴细胞白血病")
    assert "骨髓" in generic or "血液" in generic


def test_mpn_shell_has_cytoreduction() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="myeloproliferative_disorder_axis",
        closure_requirement="JAK2",
        evidence=["血小板升高", "头晕"],
        existing_treatment="",
    )
    assert "羟基脲" in plan or "降板" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="myeloproliferative_disorder_axis",
        evidence=["血小板升高", "头晕"],
    )
