from __future__ import annotations

from agent.legacy_orchestrator import (
    enforce_five_dimension_gate,
    final_verifier,
    flatten_disease_catalog,
    load_disease_catalog,
)


def _official_diseases() -> list[str]:
    return flatten_disease_catalog(load_disease_catalog())


def _catalog() -> dict[str, list[str]]:
    return {"general": ["体格检查", "胸部CT"]}


def test_official_diagnosis_without_current_case_evidence_is_blocked() -> None:
    result = final_verifier(
        diagnosis="肺炎",
        examinations=[],
        treatment_plan="经验性使用左氧氟沙星，监测病情。",
        official_diseases=_official_diseases(),
        examination_catalog=_catalog(),
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["肺炎"], "case_text": ""},
        safety_profiles=[],
    )

    assert result["passed"] is False
    assert any(
        issue.get("code") == "diagnosis_without_current_case_evidence"
        and issue.get("patchable") is False
        for issue in result["issues"]
    )


def test_ordered_only_examination_does_not_ground_diagnosis() -> None:
    result = final_verifier(
        diagnosis="肺炎",
        examinations=["胸部CT"],
        treatment_plan="经验性使用左氧氟沙星，监测病情。",
        official_diseases=_official_diseases(),
        examination_catalog=_catalog(),
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["肺炎"],
            "examination_results": {
                "胸部CT": {"status": "pending", "result": "待报告"}
            },
        },
        safety_profiles=[],
    )

    assert result["passed"] is False
    assert any(
        issue.get("code") == "diagnosis_without_current_case_evidence"
        for issue in result["issues"]
    )


def test_allergy_removal_without_specific_alternative_remains_blocked() -> None:
    result = enforce_five_dimension_gate(
        diagnoses=["肺炎"],
        treatment_plan="针对肺炎：阿莫西林抗感染治疗，监测。",
        clinical_basis=["肺炎"],
        case_features={"drug_allergies": ["青霉素"]},
        examinations=[],
    )

    assert result["gate"]["all_passed"] is False
    assert result["gate"]["blocked"] is True
    assert result["gate"]["repair_failed"] is True
    assert any(
        item.get("code") == "contraindicated_drug_removed_without_specific_alternative"
        for item in result["gate"]["blocking_findings"]
    )


def test_infection_diagnosis_does_not_create_empiric_authorization() -> None:
    result = final_verifier(
        diagnosis="肺炎",
        examinations=[],
        treatment_plan="经验性使用左氧氟沙星，待药敏后调整。",
        official_diseases=_official_diseases(),
        examination_catalog=_catalog(),
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["肺炎"],
            "patient_text": "咳嗽。",
            "case_text": "咳嗽。",
        },
        safety_profiles=[],
    )

    assert result["passed"] is False
    assert any(
        issue.get("code") == "anti_infective_without_sensitivity_evidence"
        and issue.get("patchable") is False
        for issue in result["issues"]
    )
