"""Stroke secondary prevention: do not stop aspirin without major bleed."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_diagnosis_specific_treatment_gate,
    final_verifier,
    has_stroke_secondary_prevention_context,
    sanitize_unindicated_aspirin_discontinuation,
    treatment_stops_aspirin_without_active_bleed,
)


def test_detects_unindicated_aspirin_stop_after_stroke() -> None:
    features = {
        "case_text": "患者有卒中史及焦虑史，左肩剧痛肿胀发紫。",
        "patient_text": "有卒中史，现在左肩剧痛。",
        "positive_findings": ["卒中史", "左肩剧痛"],
    }
    plan = "建议立即停用阿司匹林，改用氯吡格雷抗血小板治疗以防卒中复发；给予加巴喷丁。"
    assert has_stroke_secondary_prevention_context(features)
    assert treatment_stops_aspirin_without_active_bleed(plan, features)
    cleaned = sanitize_unindicated_aspirin_discontinuation(plan)
    assert "停用阿司匹林" not in cleaned
    assert "加巴喷丁" in cleaned


def test_active_gi_bleed_allows_aspirin_stop() -> None:
    features = {
        "case_text": "既往脑梗死，今日呕血黑便，血红蛋白急剧下降。",
        "positive_findings": ["呕血", "黑便", "脑梗死史"],
    }
    plan = "立即停用阿司匹林并急诊止血。"
    assert treatment_stops_aspirin_without_active_bleed(plan, features) is False


def test_diagnosis_gate_patches_unindicated_aspirin_stop() -> None:
    features = {
        "case_text": "患者有卒中史及焦虑史。",
        "patient_text": "有卒中史。",
        "positive_findings": ["卒中史"],
    }
    result = apply_diagnosis_specific_treatment_gate(
        diagnosis="缺血性心肌病",
        treatment_plan="立即停用阿司匹林，改用氯吡格雷；加巴喷丁止痛。",
        case_features=features,
    )
    codes = {item["code"] for item in result["issues"]}
    assert "stroke_secondary_prevention_aspirin_discontinued" in codes
    assert "停用阿司匹林" not in result["treatment_plan"]
    assert "继续阿司匹林" in "".join(result["patches"])


def test_final_verifier_blocks_unindicated_aspirin_stop_after_stroke() -> None:
    features = {
        "case_text": "有缺血性卒中史，当前左上肢疼痛。",
        "patient_text": "有缺血性卒中史。",
        "positive_findings": ["缺血性卒中史"],
        "candidate_diagnoses": ["缺血性心肌病"],
        "diagnosis_axes": [],
    }
    result = final_verifier(
        diagnosis="缺血性心肌病",
        examinations=["四肢血管超声"],
        treatment_plan="立即停用阿司匹林，改用氯吡格雷以防卒中复发。",
        official_diseases=["缺血性心肌病", "复杂性区域疼痛综合征"],
        examination_catalog={"影像": ["四肢血管超声"]},
        exam_plan_trace=[],
        case_features=features,
        safety_profiles=[],
    )
    codes = {item.get("code") for item in result.get("issues") or []}
    assert "stroke_secondary_prevention_aspirin_discontinued" in codes
    patched = result.get("patched_treatment") or ""
    # Original unindicated stop-as-regimen is removed; safety patch may still mention 停用 as forbidden action.
    assert "立即停用阿司匹林" not in patched
    assert "继续阿司匹林" in patched
    assert result.get("passed") is False or "stroke_secondary_prevention_aspirin_discontinued" in codes
