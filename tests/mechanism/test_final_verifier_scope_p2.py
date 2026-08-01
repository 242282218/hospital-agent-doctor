"""P2-1: FinalVerifier validates; it never reselects a diagnosis."""
from __future__ import annotations

import inspect

from agent.legacy_orchestrator import final_verifier


def test_final_verifier_does_not_call_diagnosis_selector() -> None:
    source = inspect.getsource(final_verifier)
    assert "enforce_selected_diagnosis_consistency" not in source
    assert '"selected_diagnosis"' not in source


def test_final_verifier_reports_axis_conflict_without_rewriting_diagnosis() -> None:
    result = final_verifier(
        diagnosis="膝关节炎",
        examinations=[],
        treatment_plan="对症观察并随访。",
        official_diseases=["膝关节炎", "蜂窝织炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["膝关节炎", "蜂窝织炎"],
            "diagnosis_candidate_records": [
                {"disease": "膝关节炎", "role": "background_condition"},
                {"disease": "蜂窝织炎", "role": "current_problem"},
            ],
            "diagnosis_axes": [
                {
                    "axis_id": "acute_lower_extremity_soft_tissue_infection",
                    "source": "rule",
                    "status": "suspected",
                    "priority": "high",
                    "clinical_role": "current_problem",
                    "closure_requirement": "supported_official_diagnosis",
                    "evidence": ["下肢红肿热痛", "发热寒战"],
                    "candidate_official_names": ["蜂窝织炎"],
                    "rule_candidate_official_names": ["蜂窝织炎"],
                }
            ],
        },
        safety_profiles=[],
    )
    assert "selected_diagnosis" not in result
    issue = next(
        item
        for item in result["issues"]
        if item.get("code") == "diagnosis_conflicts_with_high_risk_axis"
    )
    assert issue["edit"] == "蜂窝织炎"
    assert issue["patchable"] is False


def test_unresolved_llm_high_axis_without_candidate_does_not_override_diagnosis() -> None:
    result = final_verifier(
        diagnosis="慢性根尖周炎",
        examinations=[],
        treatment_plan="针对慢性根尖周炎进行根管治疗，治疗后随访复查。",
        official_diseases=["慢性根尖周炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["慢性根尖周炎"],
            "diagnosis_axes": [
                {
                    "axis_id": "unsupported_llm_high_axis",
                    "source": "llm",
                    "validated": True,
                    "status": "suspected",
                    "priority": "high",
                    "clinical_role": "current_problem",
                    "evidence": ["非特异性症状", "未闭合的模型提示"],
                    "candidate_official_names": [],
                    "promotable_candidate_official_names": [],
                }
            ],
        },
        safety_profiles=[],
    )

    assert not any(
        item.get("code") == "diagnosis_conflicts_with_high_risk_axis"
        for item in result["issues"]
    )


def test_llm_raw_candidate_without_promotion_does_not_override_diagnosis() -> None:
    result = final_verifier(
        diagnosis="幻肢痛",
        examinations=[],
        treatment_plan="进行神经病理性疼痛管理，治疗后随访复查。",
        official_diseases=["幻肢痛"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["幻肢痛"],
            "diagnosis_axes": [
                {
                    "axis_id": "phantom_limb_pain_axis",
                    "source": "llm",
                    "validated": True,
                    "status": "confirmed",
                    "priority": "high",
                    "clinical_role": "current_problem",
                    "evidence": ["截肢后出现幻觉性肢体疼痛", "残端诱发放射痛"],
                    "candidate_official_names": ["幻肢痛"],
                    "promotable_candidate_official_names": [],
                }
            ],
        },
        safety_profiles=[],
    )

    assert not any(
        item.get("code") == "diagnosis_conflicts_with_high_risk_axis"
        for item in result["issues"]
    )


def test_verified_skin_myiasis_prior_coexists_with_secondary_soft_tissue_axis() -> None:
    result = final_verifier(
        diagnosis="皮肤蝇蛆病",
        examinations=["体格检查", "全血细胞计数（CBC）"],
        treatment_plan=(
            "皮肤蝇蛆病：采用封闭疗法并完整取出幼虫，进行局部伤口护理和止痒；"
            "复查血常规并完善贫血与铁代谢评估，安排复诊观察。"
        ),
        official_diseases=["皮肤蝇蛆病", "蜂窝织炎"],
        examination_catalog={
            "皮肤科": ["体格检查", "全血细胞计数（CBC）"],
        },
        exam_plan_trace=[],
        case_features={
            "case_text": "皮肤蝇蛆病病灶伴局部红肿和低热，尚无培养阳性证据。",
            "candidate_diagnoses": ["皮肤蝇蛆病", "蜂窝织炎"],
            "diagnosis_candidate_records": [
                {
                    "disease": "皮肤蝇蛆病",
                    "source": "verified_case_prior",
                    "score": 1000,
                },
                {
                    "disease": "蜂窝织炎",
                    "source": "diagnosis_axis",
                    "role": "differential",
                },
            ],
            "diagnosis_axes": [
                {
                    "axis_id": "acute_lower_extremity_soft_tissue_infection",
                    "source": "rule",
                    "status": "suspected",
                    "priority": "high",
                    "clinical_role": "current_problem",
                    "evidence": ["局部红肿", "低热"],
                    "candidate_official_names": ["蜂窝织炎"],
                    "rule_candidate_official_names": ["蜂窝织炎"],
                }
            ],
        },
        safety_profiles=[],
    )

    assert result["passed"] is True
    assert not any(
        item.get("code") == "diagnosis_conflicts_with_high_risk_axis"
        for item in result["issues"]
    )
