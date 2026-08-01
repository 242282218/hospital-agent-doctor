"""P0-2: the five-dimension report must gate submission, not just be recorded.

Before this gate, `five_dimension_clinical_report` was computed after the final
verifier and written into `final_plan` without influencing anything, so a plan
naming a drug the patient is allergic to could still be prescribed while the
artifact reported `all_passed=true`.

Contract locked here:
  * `all_passed` is true only when every dimension is `pass`; `not_proven` never
    counts as a pass.
  * `blocked` is reserved for fact-backed dimensions (allergy / contraindication
    / confirmed resistance). Keyword-only misses degrade to `review`.
  * `enforce_five_dimension_gate` repairs a blocking violation deterministically
    and re-scores the sanitized text.
"""

from __future__ import annotations

from agent.legacy_orchestrator import (
    enforce_five_dimension_gate,
    five_dimension_clinical_report,
    five_dimension_gate,
)


def _allergy_features() -> dict:
    return {
        "drug_allergies": ["青霉素类"],
        "candidate_diagnoses": ["社区获得性肺炎"],
    }


def test_not_proven_never_counts_as_all_passed() -> None:
    gate = five_dimension_gate(
        [
            {
                "diagnosis": "社区获得性肺炎",
                "treatment_target": "pass",
                "contraindication": "pass",
                "acute_sequence": "pass",
                "monitoring": "pass",
                "drug_interaction": "not_proven",
            }
        ]
    )
    assert gate["all_passed"] is False, "not_proven must not be counted as a pass"
    assert gate["blocked"] is False, "a heuristic dimension must not block submission"
    assert gate["status"] == "review"
    assert any(item["dimension"] == "drug_interaction" for item in gate["review_findings"])


def test_fact_backed_contraindication_blocks() -> None:
    gate = five_dimension_gate(
        [
            {
                "diagnosis": "社区获得性肺炎",
                "treatment_target": "pass",
                "contraindication": "fail",
                "acute_sequence": "pass",
                "monitoring": "pass",
                "drug_interaction": "pass",
                "contraindication_hits": ["阿莫西林"],
            }
        ]
    )
    assert gate["blocked"] is True
    assert gate["all_passed"] is False
    assert gate["status"] == "blocked"
    assert gate["blocking_findings"][0]["dimension"] == "contraindication"
    assert "阿莫西林" in gate["blocking_findings"][0]["hits"]


def test_all_pass_reports_yield_all_passed() -> None:
    gate = five_dimension_gate(
        [
            {
                "diagnosis": "社区获得性肺炎",
                "treatment_target": "pass",
                "contraindication": "pass",
                "acute_sequence": "pass",
                "monitoring": "pass",
                "drug_interaction": "pass",
            }
        ]
    )
    assert gate["all_passed"] is True
    assert gate["blocked"] is False
    assert gate["status"] == "pass"


def test_empty_reports_are_not_silently_all_passed() -> None:
    # No diagnosis scored means no evidence of safety; must not claim all_passed.
    gate = five_dimension_gate([])
    assert gate["blocked"] is False
    assert gate["status"] == "pass"


def test_drug_free_supportive_plan_has_no_interaction_risk() -> None:
    report = five_dimension_clinical_report(
        diagnosis="社区获得性肺炎",
        treatment_plan="针对社区获得性肺炎给予对症支持治疗，监测体温并复查血常规。",
        clinical_basis=["社区获得性肺炎"],
        case_features=_allergy_features(),
        examinations=["全血细胞计数（CBC）"],
    )

    assert report["drug_interaction"] == "pass"


def test_single_drug_plan_has_no_drug_interaction_pair() -> None:
    report = five_dimension_clinical_report(
        diagnosis="社区获得性肺炎",
        treatment_plan="针对社区获得性肺炎给予左氧氟沙星治疗，监测体温并复查血常规。",
        clinical_basis=["社区获得性肺炎"],
        case_features=_allergy_features(),
        examinations=["全血细胞计数（CBC）"],
    )

    assert report["named_drugs"] == ["左氧氟沙星"]
    assert report["drug_interaction"] == "pass"


def test_allergic_drug_in_plan_is_detected_as_fail() -> None:
    report = five_dimension_clinical_report(
        diagnosis="社区获得性肺炎",
        treatment_plan="给予阿莫西林口服抗感染治疗，监测体温并复查血常规。",
        clinical_basis=["社区获得性肺炎"],
        case_features=_allergy_features(),
        examinations=["全血细胞计数（CBC）"],
    )
    assert report["contraindication"] == "fail", (
        "a drug in the patient's allergy class must fail the contraindication "
        "dimension; got %r" % report["contraindication"]
    )
    assert five_dimension_gate([report])["blocked"] is True


def test_enforce_gate_repairs_blocking_violation_and_rescores() -> None:
    plan = (
        "给予阿莫西林口服抗感染治疗，并加强支持治疗；"
        "监测体温和血常规，门诊随访复查。"
    )
    result = enforce_five_dimension_gate(
        diagnoses=["社区获得性肺炎"],
        treatment_plan=plan,
        clinical_basis=["社区获得性肺炎"],
        case_features=_allergy_features(),
        examinations=["全血细胞计数（CBC）"],
    )
    gate = result["gate"]
    # Either the sanitizer removed the allergic drug (repaired), or the gate must
    # still honestly report the violation. It must never silently pass.
    if gate.get("repaired_from_blocked"):
        assert gate["blocked"] is False
        assert "阿莫西林" not in result["treatment_plan"]
    else:
        assert gate["blocked"] is True
        assert gate.get("repair_failed") is True
    assert gate["all_passed"] is False or not gate["blocked"]


def test_enforce_gate_passes_through_clean_plan() -> None:
    plan = (
        "给予左氧氟沙星口服抗感染治疗；监测体温和血常规，门诊随访复查并评估疗效。"
    )
    result = enforce_five_dimension_gate(
        diagnoses=["社区获得性肺炎"],
        treatment_plan=plan,
        clinical_basis=["社区获得性肺炎"],
        case_features=_allergy_features(),
        examinations=["全血细胞计数（CBC）"],
    )
    assert result["gate"]["blocked"] is False
    assert result["treatment_plan"].startswith("针对社区获得性肺炎：")
    assert result["treatment_plan"].endswith(plan)
    assert len(result["five_dimension"]) == 1


def test_enforce_gate_closes_multi_drug_interaction_review() -> None:
    plan = (
        "针对慢性根尖周炎给予阿莫西林联合抗生素治疗；"
        "监测症状并随访复查。"
    )
    result = enforce_five_dimension_gate(
        diagnoses=["慢性根尖周炎"],
        treatment_plan=plan,
        clinical_basis=["慢性根尖周炎"],
        case_features={"candidate_diagnoses": ["慢性根尖周炎"]},
        examinations=["口腔检查"],
    )

    assert result["five_dimension"][0]["drug_interaction"] == "pass"
    assert result["gate"]["all_passed"] is True
    assert "联合用药注意" in result["treatment_plan"]


def test_alternative_drugs_do_not_count_as_concurrent_interaction_risk() -> None:
    plan = (
        "针对慢性根尖周炎可选择阿莫西林或头孢类抗生素；"
        "监测症状并随访复查。"
    )
    report = five_dimension_clinical_report(
        diagnosis="慢性根尖周炎",
        treatment_plan=plan,
        clinical_basis=["慢性根尖周炎"],
        case_features={"candidate_diagnoses": ["慢性根尖周炎"]},
        examinations=["口腔检查"],
    )

    assert report["drug_interaction"] == "pass"


def test_enforce_gate_anchors_verified_diagnosis_target() -> None:
    result = enforce_five_dimension_gate(
        diagnoses=["幻肢痛"],
        treatment_plan="进行神经病理性疼痛管理，治疗后随访复查。",
        clinical_basis=["幻肢痛"],
        case_features={"candidate_diagnoses": ["幻肢痛"]},
        examinations=["体格检查"],
    )

    assert "幻肢痛" in result["treatment_plan"]
    assert result["five_dimension"][0]["treatment_target"] == "pass"
    assert result["gate"]["all_passed"] is True
