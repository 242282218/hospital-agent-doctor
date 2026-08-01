"""Canary v5 offline: neck mass / cholestasis / rib fracture coverage + treatment shells."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    has_cholestatic_liver_disease_pattern,
    has_neck_mass_b_symptoms_pattern,
    has_traumatic_rib_fracture_pattern,
    open_coverage_gaps,
    preferred_safe_escalation_diagnosis,
    required_differential_from_case,
    select_diagnosis_axes,
    validate_safe_escalation_plan,
)


def _case(text: str, ordered=None) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": ordered or [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }


def test_neck_mass_b_symptoms_forces_biopsy_not_abdomen_only() -> None:
    text = "移植后患者颈部包块进行性增大，伴盗汗低热和体重下降。"
    assert has_neck_mass_b_symptoms_pattern(text)
    gaps = open_coverage_gaps(_case(text, ordered=["腹部超声"]))
    assert any(g["gap_id"] == "neck_mass_lymphoma_workup" for g in gaps)
    exams = {e for g in gaps if g["gap_id"] == "neck_mass_lymphoma_workup" for e in g["required_exams"]}
    assert "淋巴结活检" in exams
    assert "非霍奇金淋巴瘤" in required_differential_from_case(_case(text))
    plan, _ = build_safe_escalation_plan(
        axis_id="neck_mass_b_symptoms",
        closure_requirement="supported_official_diagnosis",
        evidence=["颈部包块", "盗汗", "体重下降"],
        existing_treatment="",
    )
    assert "活检" in plan
    assert "信息不足" not in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="neck_mass_b_symptoms",
        evidence=["颈部包块", "盗汗", "体重下降"],
    )


def test_cholestasis_forces_ama_and_biliary_imaging() -> None:
    text = "皮肤发黄伴剧烈瘙痒，尿色深，肝功能提示淤胆型酶学升高。"
    assert has_cholestatic_liver_disease_pattern(text)
    gaps = open_coverage_gaps(_case(text, ordered=["全血细胞计数（CBC）", "综合代谢面板（CMP）"]))
    assert any(g["gap_id"] == "cholestasis_ama_biliary_imaging" for g in gaps)
    plan, _ = build_safe_escalation_plan(
        axis_id="cholestatic_liver_disease",
        closure_requirement="supported_official_diagnosis",
        evidence=["黄疸", "瘙痒", "淤胆"],
        existing_treatment="当前高风险主轴尚未获得可支持的官方诊断名称，需立即急诊或住院专科评估。",
    )
    assert "熊去氧胆酸" in plan or "AMA" in plan or "抗线粒体" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="cholestatic_liver_disease",
        evidence=["黄疸", "瘙痒", "淤胆", "肝酶"],
    )


def test_rib_trauma_forces_chest_xray_and_analgesia_plan() -> None:
    text = "老人摔倒后左侧肋骨处剧痛，深呼吸和咳嗽时加重。"
    assert has_traumatic_rib_fracture_pattern(text)
    gaps = {g["gap_id"] for g in open_coverage_gaps(_case(text))}
    assert "traumatic_rib_chest_radiograph" in gaps
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "traumatic_left_rib_fracture",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["外伤", "肋骨压痛", "深呼吸痛"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "traumatic_left_rib_fracture",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["外伤", "肋骨压痛", "深呼吸痛"],
        },
        official_diseases=["肋骨骨折", "闭经"],
    )
    assert preferred == "肋骨骨折"
    plan, _ = build_safe_escalation_plan(
        axis_id="traumatic_left_rib_fracture",
        closure_requirement="supported_official_diagnosis",
        evidence=["外伤", "肋骨压痛", "深呼吸痛"],
        existing_treatment="",
    )
    assert "镇痛" in plan or "止痛" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="traumatic_left_rib_fracture",
        evidence=["外伤", "肋骨压痛", "深呼吸痛"],
    )
    axes = select_diagnosis_axes({}, case_state=_case(text))
    assert any(a["axis_id"] == "traumatic_left_rib_fracture" for a in axes)


def test_select_exam_plan_prefers_coverage_gap_required_exams_over_offaxis_noise() -> None:
    from agent.legacy_orchestrator import (
        build_name_map,
        flatten_examination_catalog,
        load_examination_catalog,
        load_knowledge_registry,
        select_exam_plan,
    )

    text = "皮肤发黄伴剧烈瘙痒，尿色深，肝功能提示淤胆型酶学升高。"
    catalog = load_examination_catalog()
    knowledge = load_knowledge_registry()
    plan = select_exam_plan(
        case_state=_case(text),
        disease_candidates=[{"disease": "原发性胆汁性胆管炎"}],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=4,
    )
    exams = set(plan["examinations"])
    assert "抗线粒体抗体（AMA）" in exams or "腹部超声" in exams
    # Off-axis allergy leaves must not monopolize the first slots when AMA gap is open.
    assert not exams.issuperset({"斑贴试验", "皮肤点刺试验", "过敏原检测（生物共振）"})
    assert "coverage_gap" in plan.get("reason", "")
