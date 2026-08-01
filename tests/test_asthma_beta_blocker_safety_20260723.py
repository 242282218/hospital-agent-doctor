"""Asthma + AF RVR: beta-blockers must not be recommended (Canary 01329)."""

from __future__ import annotations

from agent.legacy_orchestrator import apply_treatment_safety, has_asthma_or_reactive_airway


def test_has_asthma_detects_history_in_case_text() -> None:
    assert has_asthma_or_reactive_airway(
        {
            "case_text": "有高血压、哮喘和下肢水肿的病史。",
            "positive_findings": [],
        }
    )
    assert not has_asthma_or_reactive_airway(
        {
            "case_text": "阵发性心悸，无呼吸道基础病。",
            "positive_findings": ["心悸"],
        }
    )


def test_asthma_blocks_iv_beta_blocker_rate_control_prefers_ccb_path() -> None:
    plan = (
        "首选静脉给予β受体阻滞剂（如艾司洛尔）或非二氢吡啶类钙通道阻滞剂（如地尔硫卓）"
        "控制心室率，若血流动力学不稳定则考虑同步电复律。"
    )
    features = {
        "case_text": (
            "气短和喘鸣反复出现。今年76岁。平时吃利尿剂，哮喘药用得不太规律。"
            "有高血压、哮喘和下肢水肿的病史。"
        ),
        "positive_findings": ["哮喘", "喘鸣", "心房颤动伴快速心室反应"],
    }
    result = apply_treatment_safety(
        plan,
        diagnosis="心律失常",
        case_features=features,
        safety_profiles=[],
    )
    codes = {item["code"] for item in result["issues"]}
    assert "contraindicated_drug_recommended" in codes
    assert result["patched"] is True
    # Must not keep a positive first-line beta-blocker recommendation.
    assert "首选静脉给予β受体阻滞剂" not in result["treatment_plan"]
    assert "给予艾司洛尔" not in result["treatment_plan"]
    # Safety patch must name the risk and alternative.
    assert "哮喘" in result["treatment_plan"] or "气道" in result["treatment_plan"]
    assert "避免" in result["treatment_plan"] and "β" in result["treatment_plan"]
    assert "地尔硫卓" in result["treatment_plan"] or "钙通道" in result["treatment_plan"]


def test_no_asthma_does_not_force_beta_blocker_removal() -> None:
    plan = "给予美托洛尔控制心室率并监测血压。"
    features = {
        "case_text": "阵发性心悸，心电图示窦性心动过速，无哮喘史。",
        "positive_findings": ["心悸"],
    }
    result = apply_treatment_safety(
        plan,
        diagnosis="心律失常",
        case_features=features,
        safety_profiles=[],
    )
    codes = {item["code"] for item in result["issues"]}
    assert "contraindicated_drug_recommended" not in codes
    assert "美托洛尔" in result["treatment_plan"]
