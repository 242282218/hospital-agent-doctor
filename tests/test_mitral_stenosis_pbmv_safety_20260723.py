"""Mitral stenosis: no PBMV before LA thrombus exclusion."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_diagnosis_specific_treatment_gate,
    build_safe_escalation_plan,
    preferred_safe_escalation_diagnosis,
    treatment_recommends_pbmv_without_thrombus_exclusion,
    validate_safe_escalation_plan,
)


def test_detects_pbmv_without_thrombus_exclusion() -> None:
    plan = "建议尽快转诊评估经皮球囊二尖瓣成形术（PBMV）可行性。"
    assert treatment_recommends_pbmv_without_thrombus_exclusion(plan)
    safe = "先经食道超声排除左房血栓后，再由心内科评估是否PBMV。"
    assert treatment_recommends_pbmv_without_thrombus_exclusion(safe) is False


def test_mitral_stenosis_gate_blocks_pbmv_before_tee() -> None:
    features = {
        "case_text": "79岁，呼吸困难，超声示二尖瓣口面积0.9cm²，肺动脉高压。",
        "positive_findings": ["二尖瓣狭窄", "肺动脉高压"],
        "diagnosis_axes": [
            {
                "axis_id": "mitral_stenosis_hemodynamics",
                "priority": "high",
                "evidence": ["二尖瓣口面积0.9", "肺高压", "心衰症状"],
            }
        ],
    }
    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="二尖瓣狭窄",
        treatment_plan="利尿并尽快行经皮球囊二尖瓣成形术。",
        case_features=features,
    )
    codes = {item["code"] for item in result["issues"]}
    assert "mitral_stenosis_pbmv_before_la_thrombus_exclusion" in codes
    assert "球囊" not in result["treatment_plan"] or "血栓" in "".join(result["patches"])
    assert "食道超声" in "".join(result["patches"]) or "左房" in "".join(result["patches"])


def test_mitral_axis_default_label_and_safe_plan() -> None:
    axis = {
        "axis_id": "mitral_stenosis_hemodynamics",
        "priority": "high",
        "candidate_official_names": [],
        "evidence": ["二尖瓣狭窄", "肺高压", "水肿"],
    }
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="老视",
        case_features={"diagnosis_axes": [axis]},
        escalation_axis=axis,
        official_diseases=["二尖瓣狭窄", "老视"],
    )
    assert preferred == "二尖瓣狭窄"
    plan, _ = build_safe_escalation_plan(
        axis_id="mitral_stenosis_hemodynamics",
        closure_requirement="supported_official_diagnosis",
        evidence=axis["evidence"],
        existing_treatment="",
    )
    assert validate_safe_escalation_plan(
        plan,
        axis_id="mitral_stenosis_hemodynamics",
        evidence=axis["evidence"],
    )
    assert "血栓" in plan or "食道超声" in plan
    unsafe = plan + "尽快行经皮球囊二尖瓣成形术。"
    assert not validate_safe_escalation_plan(
        unsafe,
        axis_id="mitral_stenosis_hemodynamics",
        evidence=axis["evidence"],
    )
