"""Canary v10 offline: gout acute therapy shell must not be labs-only."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    preferred_safe_escalation_diagnosis,
    validate_safe_escalation_plan,
)


def test_gout_shell_requires_acute_antiinflammatory_therapy() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="suspected_gouty_arthritis",
        closure_requirement="需行关节液穿刺检查以确认尿酸盐结晶",
        evidence=["夜间足趾剧痛", "红肿", "高尿酸"],
        existing_treatment="",
    )
    assert any(marker in plan for marker in ["NSAIDs", "非甾体", "秋水仙碱", "糖皮质激素", "布洛芬", "抗炎"])
    assert validate_safe_escalation_plan(
        plan,
        axis_id="suspected_gouty_arthritis",
        evidence=["夜间足趾剧痛", "红肿", "高尿酸"],
    )
    labs_only = (
        "针对当前高风险主轴（suspected_gouty_arthritis）：需行关节液穿刺检查以确认尿酸盐结晶，"
        "并完善血尿酸检测；若确诊需启动降尿酸治疗；并立即急诊或专科评估。"
    )
    assert not validate_safe_escalation_plan(
        labs_only,
        axis_id="suspected_gouty_arthritis",
        evidence=["夜间足趾剧痛", "红肿", "高尿酸"],
    )
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="闭经",
        case_features={
            "diagnosis_axes": [
                {
                    "axis_id": "suspected_gouty_arthritis",
                    "priority": "high",
                    "candidate_official_names": [],
                    "evidence": ["夜间足趾剧痛", "红肿", "高尿酸"],
                }
            ]
        },
        escalation_axis={
            "axis_id": "suspected_gouty_arthritis",
            "priority": "high",
            "candidate_official_names": [],
            "evidence": ["夜间足趾剧痛", "红肿", "高尿酸"],
        },
        official_diseases=["痛风", "闭经"],
    )
    assert preferred == "痛风"


def test_hunters_syndrome_shell_requires_antiviral() -> None:
    plan, _ = build_safe_escalation_plan(
        axis_id="suspected_hunters_syndrome_neurologic",
        closure_requirement="完善VZV血清学",
        evidence=["面瘫", "耳痛", "听力下降"],
        existing_treatment="",
    )
    assert "抗病毒" in plan or "阿昔洛韦" in plan or "伐昔洛韦" in plan
    assert validate_safe_escalation_plan(
        plan,
        axis_id="suspected_hunters_syndrome_neurologic",
        evidence=["面瘫", "耳痛", "听力下降"],
    )
