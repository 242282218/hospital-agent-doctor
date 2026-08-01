"""P0: anti-infective gate must use structured provenance, not self-written text."""
from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_treatment_safety,
    converge_verified_treatment,
    find_anti_infective_evidence_gaps,
    treatment_recommends_drug,
    ANTIBIOTIC_ALIASES,
    normalize_name,
)


def _infection_features(**extra):
    base = {
        "positive_findings": ["发热", "会阴痛", "尿频"],
        "candidate_diagnoses": ["急性细菌性前列腺炎"],
        "case_text": "发热会阴痛尿频疑前列腺炎",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
    }
    base.update(extra)
    return base


def test_plan_text_markers_alone_do_not_count_as_provenance() -> None:
    # No structured infection diagnosis / AST / empiric object — only plan text.
    plan = "立即静脉使用环丙沙星。药敏结果示敏感。经验用药。"
    result = find_anti_infective_evidence_gaps(plan, {})
    assert result["issues"], "text-only markers must not clear the gate"
    assert any(
        i.get("code")
        in {
            "anti_infective_without_sensitivity_evidence",
            "anti_infective_sensitivity_claim_without_ast",
        }
        for i in result["issues"]
    )


def test_structured_ast_clears_named_drug() -> None:
    plan = "静脉使用环丙沙星抗感染。"
    features = _infection_features(
        anti_infective_provenance={
            "ast": [{"drug": "环丙沙星", "result": "S", "source": "urine_culture"}],
            "cultures": [{"source": "urine", "result": "pos"}],
        }
    )
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"]


def test_structured_resistance_blocks_named_drug() -> None:
    plan = "静脉使用环丙沙星抗感染。"
    features = _infection_features(
        anti_infective_provenance={
            "confirmed_resistance": ["环丙沙星"],
            "ast": [{"drug": "环丙沙星", "result": "R", "source": "urine_culture"}],
        }
    )
    result = find_anti_infective_evidence_gaps(plan, features)
    assert result["issues"]
    assert any(i.get("patchable") is False for i in result["issues"])
    assert any(i.get("code") == "anti_infective_confirmed_resistance" for i in result["issues"])


def test_structured_empiric_requires_conditional_language() -> None:
    bare = "立即静脉使用环丙沙星抗感染治疗。"
    features = _infection_features(
        anti_infective_provenance={
            "empiric": {
                "allowed": True,
                "indication": "急性细菌性前列腺炎，血流动力学稳定",
                "must_reassess_on_ast": True,
            }
        }
    )
    result = find_anti_infective_evidence_gaps(bare, features)
    assert result["issues"], "empiric without conditional language must still flag"
    safety = apply_treatment_safety(
        bare,
        diagnosis="急性细菌性前列腺炎",
        case_features=features,
        safety_profiles=[],
    )
    plan = safety.get("treatment_plan") or ""
    assert "待" in plan or "药敏" in plan
    assert "条件" in plan or "经验" in plan


def test_second_pass_self_patch_cannot_clear_without_structured_evidence() -> None:
    # Empty features: no structured diagnosis/AST/empiric. Gate patch text must not launder.
    plan = "立即使用环丙沙星治疗。"
    features = {}
    first = apply_treatment_safety(
        plan,
        diagnosis="未分化",
        case_features=features,
        safety_profiles=[],
    )
    assert first["issues"], "first pass must flag bare named antibiotic"
    second = find_anti_infective_evidence_gaps(first["treatment_plan"], features)
    still_names_drug = treatment_recommends_drug(
        normalize_name(first["treatment_plan"]),
        ["环丙沙星"],
    )
    if still_names_drug:
        assert second["issues"], "self-written patch text must not clear provenance"


def test_converge_cannot_launder_named_drug_via_disclaimer() -> None:
    plan = "立即静脉环丙沙星抗感染。"
    features = {}  # no structured provenance
    report = converge_verified_treatment(
        diagnosis="未分化",
        examinations=["尿常规"],
        treatment_plan=plan,
        official_diseases=["未分化"],
        examination_catalog={"检验": ["尿常规"]},
        exam_plan_trace=[],
        case_features=features,
        safety_profiles=[],
        max_rounds=3,
    )
    if report is None:
        return
    final_plan = report.get("patched_treatment") or ""
    if treatment_recommends_drug(normalize_name(final_plan), ["环丙沙星"]):
        assert report.get("passed") is not True


# ---- Real-examination-results provenance tests (Round 2) ----

def _real_exam_case_state():
    return {
        "patient_id": "REAL_PROBE",
        "mode": "test",
        "memory_notes": [],
        "chat_history": [],
        "ordered_examinations": ["尿常规", "尿培养+药敏"],
        "invalid_examinations": [],
        "examination_results": {
            "尿培养+药敏": {
                "ast": [
                    {"drug": "环丙沙星", "result": "R", "source": "尿培养+药敏", "status": "resulted"},
                    {"drug": "头孢曲松", "result": "S", "source": "尿培养+药敏"},
                ],
                "cultures": [{"source": "尿培养", "organism": "大肠埃希菌", "status": "resulted"}],
            }
        },
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def test_real_examination_results_ast_r_builds_provenance() -> None:
    """AST=R from real examination_results must become structured provenance."""
    from agent.legacy_orchestrator import extract_case_features
    features = extract_case_features(_real_exam_case_state())
    prov = features["anti_infective_provenance"]
    ast = prov["ast"]
    assert len(ast) == 2, "both AST rows must be parsed"
    by_drug = {a["drug"]: a["result"] for a in ast}
    assert by_drug.get("环丙沙星") == "R"
    assert by_drug.get("头孢曲松") == "S"
    assert ast[0]["source"] == "尿培养+药敏"


def test_real_ast_r_blocks_named_drug_with_patchable_false() -> None:
    """The exact Codex counter-example: ciprofloxacin AST=R must block ciprofloxacin."""
    from agent.legacy_orchestrator import (
        extract_case_features,
        find_anti_infective_evidence_gaps,
    )
    features = extract_case_features(_real_exam_case_state())
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = [i["code"] for i in result["issues"]]
    assert "anti_infective_confirmed_resistance" in codes, result["issues"]
    assert any(i.get("patchable") is False for i in result["issues"])


def test_real_ast_r_resists_conditional_laundering() -> None:
    """A conditional patch text must NOT launder an AST=R drug back to clean."""
    from agent.legacy_orchestrator import (
        extract_case_features,
        apply_treatment_safety,
        find_anti_infective_evidence_gaps,
        treatment_recommends_drug,
        normalize_name,
    )
    features = extract_case_features(_real_exam_case_state())
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    first = apply_treatment_safety(plan, diagnosis="尿路感染", case_features=features, safety_profiles=[])
    second = apply_treatment_safety(
        first.get("treatment_plan") or "", diagnosis="尿路感染",
        case_features=features, safety_profiles=[],
    )
    final_plan = second.get("treatment_plan") or ""
    assert treatment_recommends_drug(normalize_name(final_plan), ["环丙沙星"])
    gaps = find_anti_infective_evidence_gaps(final_plan, features)
    assert any(i["code"] == "anti_infective_confirmed_resistance" for i in gaps["issues"])


def test_real_ast_s_clears_named_drug() -> None:
    """AST=S must clear the named drug (susceptible)."""
    from agent.legacy_orchestrator import (
        extract_case_features,
        find_anti_infective_evidence_gaps,
    )
    state = _real_exam_case_state()
    plan = "静脉使用头孢曲松抗感染。"
    features = extract_case_features(state)
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"], "AST=S must clear susceptible drug"
