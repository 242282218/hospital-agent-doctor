"""Canary v9 offline: pleural effusion and immunocompromised infection shells."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    preferred_safe_escalation_diagnosis,
    validate_safe_escalation_plan,
)


def test_pleural_effusion_shell_requires_antibiotics_and_drainage_path() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="pleural_effusion_investigation",
        closure_requirement="需行胸腔穿刺术明确积液性质",
        evidence=["发热", "胸腔积液", "咳痰"],
        existing_treatment="",
    )
    assert "抗感染" in plan or "抗生素" in plan
    assert "穿刺" in plan or "引流" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="pleural_effusion_investigation",
        evidence=["发热", "胸腔积液", "咳痰"],
    )
    puncture_only = (
        "针对当前高风险主轴（pleural_effusion_investigation）：需行胸腔穿刺术明确积液性质；"
        "并立即急诊或专科评估。"
    )
    assert not validate_safe_escalation_plan(
        puncture_only,
        axis_id="pleural_effusion_investigation",
        evidence=["发热", "胸腔积液", "咳痰"],
    )


def test_immunodeficiency_infection_shell_requires_empiric_therapy() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="suspected_infection_immunodeficiency",
        closure_requirement="完善病原学检测",
        evidence=["免疫抑制", "发热", "气促"],
        existing_treatment="",
    )
    assert "抗感染" in plan or "抗生素" in plan
    assert "隔离" in plan or "住院" in plan or "急诊" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="suspected_infection_immunodeficiency",
        evidence=["免疫抑制", "发热", "气促"],
    )
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "suspected_infection_immunodeficiency",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["免疫抑制", "发热", "气促"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "suspected_infection_immunodeficiency",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["免疫抑制", "发热", "气促"],
        },
        official_diseases=["肺炎", "闭经"],
    )
    assert preferred == "肺炎"


def test_bartholin_default_label_and_treatment() -> None:
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "right_bartholin_cyst_infection",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["外阴肿块", "压痛", "黄分泌物"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "right_bartholin_cyst_infection",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["外阴肿块", "压痛", "黄分泌物"],
        },
        official_diseases=["巴氏腺囊肿", "闭经"],
    )
    assert preferred == "巴氏腺囊肿"
    plan, _ = build_safe_escalation_plan(
        axis_id="right_bartholin_cyst_infection",
        closure_requirement="supported_official_diagnosis",
        evidence=["外阴肿块", "压痛", "黄分泌物"],
        existing_treatment="",
    )
    assert "抗生素" in plan or "引流" in plan or "坐浴" in plan
