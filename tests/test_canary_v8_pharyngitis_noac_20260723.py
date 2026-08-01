"""Canary v8 offline: diabetic child pharyngitis + rheumatic MS NOAC safety."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_diagnosis_specific_treatment_gate,
    build_safe_escalation_plan,
    has_acute_pharyngitis_in_diabetic_child_pattern,
    open_coverage_gaps,
    validate_safe_escalation_plan,
)


def test_diabetic_child_pharyngitis_forces_infection_labs_and_antibiotic_plan() -> None:
    text = "10岁糖尿病患儿咽痛发热两天，吞咽痛，血糖平时用胰岛素。"
    assert has_acute_pharyngitis_in_diabetic_child_pattern(text)
    case = {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": ["耳镜检查", "口咽部检查"],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    gaps = open_coverage_gaps(case)
    assert any(g["gap_id"] == "diabetic_child_pharyngitis_infection_labs" for g in gaps)
    plan, _ = build_safe_escalation_plan(
        axis_id="acute_pharyngitis_in_diabetic_child",
        closure_requirement="supported_official_diagnosis",
        evidence=["咽痛", "发热", "糖尿病"],
        existing_treatment="",
    )
    assert "抗生素" in plan or "青霉素" in plan or "抗感染" in plan
    assert "血糖" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="acute_pharyngitis_in_diabetic_child",
        evidence=["咽痛", "发热", "糖尿病"],
    )
    labs_only = (
        "针对当前高风险主轴（acute_pharyngitis_in_diabetic_child）：需完善血常规及C反应蛋白；"
        "并立即急诊或专科评估。"
    )
    assert not validate_safe_escalation_plan(
        labs_only,
        axis_id="acute_pharyngitis_in_diabetic_child",
        evidence=["咽痛", "发热", "糖尿病"],
    )


def test_rheumatic_ms_noac_default_is_patched_to_warfarin_preference() -> None:
    features = {
        "case_text": "风湿性心脏病，中度二尖瓣狭窄，左房增大，突发心悸气促。",
        "positive_findings": ["风湿性心脏病", "二尖瓣狭窄", "左房增大"],
    }
    plan = (
        "若确诊房颤，应启动抗凝治疗（如华法林，目标INR 2.0-3.0；或新型口服抗凝药，视情况而定）。"
    )
    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="风湿性心脏病",
        treatment_plan=plan,
        case_features=features,
    )
    codes = {item["code"] for item in result["issues"]}
    assert "noac_in_rheumatic_mitral_stenosis" in codes
    assert "新型口服抗凝" not in result["treatment_plan"]
    assert "华法林" in "".join(result["patches"])
