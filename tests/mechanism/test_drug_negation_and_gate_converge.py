"""Regression: negated drug mentions and anti-infective gate convergence."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    drug_mention_is_negated,
    find_anti_infective_evidence_gaps,
    normalize_name,
    treatment_recommends_drug,
)


def test_conjunctive_class_negation_not_a_recommendation() -> None:
    plan = normalize_name("因有过敏史，磺胺类抗生素和磺胺嘧啶禁用，应开始使用乙胺嘧啶联合克林霉素")
    assert drug_mention_is_negated(plan, plan.find(normalize_name("磺胺")), len(normalize_name("磺胺")))
    assert not treatment_recommends_drug(plan, ["磺胺"])


def test_scoped_avoid_list_not_a_recommendation() -> None:
    plan = normalize_name(
        "避免使用新生儿禁忌药物（尤其是头孢曲松和阿司匹林）；按小儿心脏外科方案选择抗菌药物。"
    )
    assert not treatment_recommends_drug(plan, ["头孢曲松"])
    assert not treatment_recommends_drug(plan, ["头孢"])


def test_replacement_drug_after_negation_stays_positive() -> None:
    plan = normalize_name("禁用青霉素，改用头孢曲松经验性抗感染。")
    assert not treatment_recommends_drug(plan, ["青霉素"])
    assert treatment_recommends_drug(plan, ["头孢曲松"])


def test_allergy_assessment_is_not_a_recommendation() -> None:
    plan = normalize_name(
        "评估是否存在磺胺类药物敏感性；如怀疑磺胺类药物过敏则避免使用乙酰唑胺。"
    )
    assert not treatment_recommends_drug(plan, ["磺胺"])


def test_gate_patch_converges_with_infection_diagnosis_empiric() -> None:
    # Infection-shaped diagnosis authorizes conditional empiric; one patch cleans.
    plan = "开始局部使用氟喹诺酮类抗生素滴眼液（如 0.5% 莫西沙星或 1.5% 左氧氟沙星）"
    case_features = {
        "candidate_diagnoses": ["角膜病"],
        "case_text": "眼痛、畏光，检查提示角膜上皮缺损。",
        "patient_text": "眼痛、畏光，检查提示角膜上皮缺损。",
        "positive_findings": ["角膜上皮缺损"],
    }
    first = find_anti_infective_evidence_gaps(plan, case_features)
    assert first["issues"], "expected a missing-conditional-language issue"
    assert all(issue.get("patchable") for issue in first["issues"])
    assert all(
        issue.get("code") == "anti_infective_empiric_missing_conditional_language"
        for issue in first["issues"]
    )
    patched = plan + " " + " ".join(first["patches"])
    second = find_anti_infective_evidence_gaps(patched, case_features)
    assert second["issues"] == [], second


def test_empty_features_named_drug_stays_unpatchable() -> None:
    plan = "立即使用环丙沙星治疗。"
    first = find_anti_infective_evidence_gaps(plan, {})
    assert first["issues"]
    assert all(issue.get("patchable") is False for issue in first["issues"])
    patched = plan + " " + " ".join(first["patches"])
    second = find_anti_infective_evidence_gaps(patched, {})
    assert second["issues"], "disclaimer must not launder without structured evidence"
