"""Canary v4 offline: asthma spirometry, pituitary panel, syndactyly X-ray."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    has_congenital_syndactyly_pattern,
    has_hypothalamic_pituitary_amenorrhea_pattern,
    has_suspected_asthma_control_pattern,
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


def test_asthma_control_forces_spirometry_gap_and_label() -> None:
    text = "28岁，夜间憋醒、喘息，过敏性鼻炎和湿疹，灰尘诱发，沙丁胺醇越用越勤。"
    assert has_suspected_asthma_control_pattern(text)
    gaps = {g["gap_id"] for g in open_coverage_gaps(_case(text))}
    assert "asthma_spirometry" in gaps
    axes = select_diagnosis_axes({}, case_state=_case(text))
    axis = next(a for a in axes if a["axis_id"] == "suspected_asthma_control_issue")
    assert "哮喘" in axis["candidate_official_names"]
    assert "哮喘" in required_differential_from_case(_case(text))
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={"diagnosis_axes": [axis]},
        escalation_axis=axis,
        official_diseases=["哮喘", "闭经"],
    )
    assert preferred == "哮喘"
    plan, _ = build_safe_escalation_plan(
        axis_id="suspected_asthma_control_issue",
        closure_requirement="supported_official_diagnosis",
        evidence=axis["evidence"],
        existing_treatment="",
    )
    assert validate_safe_escalation_plan(
        plan,
        axis_id="suspected_asthma_control_issue",
        evidence=["喘息", "夜间憋醒", "过敏性鼻炎"],
    )
    assert "肺功能" in plan or "吸入" in plan


def test_secondary_amenorrhea_forces_pituitary_panel_not_electrolytes_only() -> None:
    text = "22岁，停经半年，明显体重下降并节食，怕冷乏力。"
    assert has_hypothalamic_pituitary_amenorrhea_pattern(text)
    gaps = open_coverage_gaps(_case(text, ordered=["血清电解质"]))
    assert any(g["gap_id"] == "pituitary_hormone_panel" for g in gaps)
    assert "垂体前叶功能减退" in required_differential_from_case(_case(text))
    plan, _ = build_safe_escalation_plan(
        axis_id="hypothalamic_pituitary_axis_dysfunction",
        closure_requirement="supported_official_diagnosis",
        evidence=["闭经", "体重下降", "怕冷乏力"],
        existing_treatment="当前信息不足以制定方案。",
    )
    assert "信息不足" not in plan
    assert "垂体" in plan or "内分泌" in plan


def test_syndactyly_forces_hand_xray_gap() -> None:
    text = "新生儿右手自出生起手指长在一起，拇指短小，并指畸形。"
    assert has_congenital_syndactyly_pattern(text)
    gaps = {g["gap_id"] for g in open_coverage_gaps(_case(text, ordered=["神经系统检查"]))}
    assert "syndactyly_hand_radiograph" in gaps
    assert "并指（趾）畸形" in required_differential_from_case(_case(text))


def test_unknown_high_axis_uses_closure_requirement_text_not_only_generic_shell() -> None:
    plan, reasoning = build_safe_escalation_plan(
        axis_id="neck_mass_b_symptoms",
        closure_requirement="完成颈部淋巴结活检并完善血常规、LDH与分期影像",
        evidence=["颈部包块", "盗汗", "体重下降"],
        existing_treatment="当前信息不足以制定特异性治疗方案。",
    )
    assert "信息不足" not in plan
    assert "活检" in plan
    assert "neck_mass_b_symptoms" in reasoning
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "neck_mass_b_symptoms",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["颈部包块", "盗汗", "体重下降"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "neck_mass_b_symptoms",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["颈部包块", "盗汗", "体重下降"],
        },
        official_diseases=["非霍奇金淋巴瘤", "闭经"],
    )
    assert preferred == "非霍奇金淋巴瘤"
