"""Acute decompensated HF: NT-proBNP + diuresis, not labs-only shell."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_name_map,
    build_safe_escalation_plan,
    flatten_examination_catalog,
    has_acute_decompensated_heart_failure_pattern,
    load_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    preferred_safe_escalation_diagnosis,
    select_exam_plan,
    validate_safe_escalation_plan,
)


def test_adhf_pattern_and_bnp_gap() -> None:
    text = "32岁长期高血压，这两天端坐呼吸、双下肢水肿加重，心率快，越来越喘。"
    assert has_acute_decompensated_heart_failure_pattern(text)
    case = {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": ["超声心动图"],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    gaps = open_coverage_gaps(case)
    assert any(g["gap_id"] == "acute_heart_failure_natriuretic_electrolytes" for g in gaps)
    catalog = load_examination_catalog()
    knowledge = load_knowledge_registry()
    plan = select_exam_plan(
        case_state=case,
        disease_candidates=[{"disease": "心力衰竭"}],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=4,
    )
    exams = set(plan["examinations"])
    assert "N末端B型利钠肽原（NT-proBNP）" in exams or "B型利钠肽（BNP）" in exams


def test_adhf_safe_escalation_requires_diuresis_not_labs_only() -> None:
    labs_only = (
        "针对当前高风险主轴（heart_failure_decompensation）：完善BNP及电解质检查以指导利尿剂使用；"
        "并立即急诊或专科评估，完成必要检查闭环后再制定特异治疗。"
    )
    assert not validate_safe_escalation_plan(
        labs_only,
        axis_id="heart_failure_decompensation",
        evidence=["端坐呼吸", "水肿", "高血压"],
    )
    plan, _ = build_safe_escalation_plan(
        axis_id="heart_failure_decompensation",
        closure_requirement="supported_official_diagnosis",
        evidence=["端坐呼吸", "水肿", "高血压"],
        existing_treatment="",
    )
    assert "利尿" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="heart_failure_decompensation",
        evidence=["端坐呼吸", "水肿", "高血压"],
    )
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="痛经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "heart_failure_decompensation",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["端坐呼吸", "水肿", "高血压"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "heart_failure_decompensation",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["端坐呼吸", "水肿", "高血压"],
        },
        official_diseases=["心力衰竭", "痛经"],
    )
    assert preferred == "心力衰竭"
