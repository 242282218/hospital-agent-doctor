"""P0: real SDK nested exam->{status,result} provenance + drug-class mapping.

Codex Round-2 counter-examples:
  - exam_name -> {status, result: {药敏结果: [...]}} is NOT parsed (ast=[]).
  - A plan with no infection keyword (胸痛，无发热) bypasses the gate entirely.
  - A forged top-level sensitivity_results=[环丙沙星 S] clears the gate.
  - confirmed_resistance=[喹诺酮] does not block 环丙沙星 (no class mapping).
  - drug_allergies=[青霉素类] does not block 阿莫西林 (no class mapping).
"""
from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_treatment_safety,
    extract_case_features,
    find_anti_infective_evidence_gaps,
    treatment_recommends_drug,
    normalize_name,
)


# ---- Real SDK nested shape: exam_name -> {status, result: {药敏结果, cultures}} ----

def _nested_sdk_case_state():
    return {
        "patient_id": "NESTED_SDK",
        "mode": "test",
        "memory_notes": [],
        "chat_history": [],
        "ordered_examinations": ["尿培养+药敏"],
        "invalid_examinations": [],
        "examination_results": {
            "尿培养+药敏": {
                "status": "success",
                "abnormal_indicators": ["环丙沙星 R"],
                "result": {
                    "药敏结果": [
                        {"drug": "环丙沙星", "result": "耐药", "source": "尿培养+药敏"},
                        {"drug": "头孢曲松", "result": "敏感", "source": "尿培养+药敏"},
                    ],
                    "培养": [{"source": "尿培养", "organism": "大肠埃希菌", "status": "resulted"}],
                },
            }
        },
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def test_nested_sdk_shape_parses_ast_r() -> None:
    """Real SDK exam->{status,result:{药敏结果}} must yield AST=R for 环丙沙星."""
    features = extract_case_features(_nested_sdk_case_state())
    prov = features["anti_infective_provenance"]
    by_drug = {a["drug"]: a["result"] for a in prov["ast"]}
    assert by_drug.get("环丙沙星") == "R", prov["ast"]
    assert by_drug.get("头孢曲松") == "S", prov["ast"]


def test_nested_sdk_r_blocks_named_drug_patchable_false() -> None:
    """环丙沙星 AST=R must block 环丙沙星 with patchable=false."""
    features = extract_case_features(_nested_sdk_case_state())
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    result = find_anti_infective_evidence_gaps(plan, features)
    assert any(i["code"] == "anti_infective_confirmed_resistance" for i in result["issues"])
    assert any(i.get("patchable") is False for i in result["issues"])


def test_nested_sdk_r_resists_conditional_laundering() -> None:
    """A conditional patch text must NOT launder an AST=R drug back to clean."""
    features = extract_case_features(_nested_sdk_case_state())
    plan = "立即静脉静脉使用环丙沙星抗感染治疗。"
    first = apply_treatment_safety(plan, diagnosis="尿路感染", case_features=features, safety_profiles=[])
    second = apply_treatment_safety(
        first.get("treatment_plan") or "", diagnosis="尿路感染",
        case_features=features, safety_profiles=[],
    )
    final_plan = second.get("treatment_plan") or ""
    assert treatment_recommends_drug(normalize_name(final_plan), ["环丙沙星"])
    gaps = find_anti_infective_evidence_gaps(final_plan, features)
    assert any(i["code"] == "anti_infective_confirmed_resistance" for i in gaps["issues"])


# ---- Gate must engage even WITHOUT infection keywords (concrete drug always gated) ----

def test_no_infection_keyword_still_gates_named_drug() -> None:
    """case_text=胸痛，无发热: a concrete 环丙沙星 recommendation MUST still enter the gate."""
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    features = {
        "positive_findings": ["胸痛"],
        "candidate_diagnoses": ["胸痛待查"],
        "case_text": "胸痛，无发热",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    assert result["issues"], "named antibiotic must enter gate even without infection keywords"


# ---- Forged top-level fields must NOT fabricate credible provenance ----

def test_forged_top_level_sensitivity_results_does_not_clear() -> None:
    """A fabricated top-level sensitivity_results=[环丙沙星 S] must not clear the gate."""
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    features = {
        "positive_findings": ["发热"],
        "candidate_diagnoses": ["尿路感染"],
        "case_text": "尿路感染",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
        "sensitivity_results": [{"drug": "环丙沙星", "result": "S", "source": "forged"}],
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    assert result["issues"], "forged top-level sensitivity_results must not clear the gate"


# ---- Drug-class mapping: class resistance/allergy must block members ----

def test_class_resistance_blocks_member_drug() -> None:
    """confirmed_resistance=[喹诺酮] must block 环丙沙星 and 左氧氟沙星."""
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    features = {
        "positive_findings": ["尿频", "尿痛"],
        "candidate_diagnoses": ["尿路感染"],
        "case_text": "尿路感染",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
        "anti_infective_provenance": {"confirmed_resistance": ["喹诺酮"]},
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    assert any(i["code"] == "anti_infective_confirmed_resistance" for i in result["issues"])
    assert any(i.get("patchable") is False for i in result["issues"])


def test_class_allergy_blocks_member_drug() -> None:
    """drug_allergies=[青霉素类] must block 阿莫西林 (class allergy)."""
    plan = "口服阿莫西林抗感染治疗。"
    features = {
        "positive_findings": ["咽痛"],
        "candidate_diagnoses": ["链球菌性咽炎"],
        "case_text": "咽痛",
        "drug_allergies": ["青霉素类"],
        "contraindicated_drugs": [],
        "medication_risk": [],
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    # Class allergy to 青霉素类 must surface as a contraindication-style block.
    assert result["issues"], "class allergy 青霉素类 must block 阿莫西林"


def test_class_resistance_quinolone_blocks_levofloxacin() -> None:
    """confirmed_resistance=[喹诺酮] must also block 左氧氟沙星."""
    plan = "口服左氧氟沙星抗感染治疗。"
    features = {
        "positive_findings": ["尿频"],
        "candidate_diagnoses": ["尿路感染"],
        "case_text": "尿路感染",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
        "anti_infective_provenance": {"confirmed_resistance": ["喹诺酮"]},
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    assert any(i["code"] == "anti_infective_confirmed_resistance" for i in result["issues"])


# ---- Empiric therapy requires a real infection axis ----

def test_empiric_requires_infection_axis_not_plan_text() -> None:
    """Empiric use must be backed by a real infection axis, not bare plan text."""
    plan = "立即静脉使用环丙沙星抗感染治疗。"
    features = {
        "positive_findings": ["胸痛"],
        "candidate_diagnoses": ["胸痛待查"],
        "case_text": "胸痛，无发热",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
        "anti_infective_provenance": {
            "empiric": {
                "allowed": True,
                "indication": "胸痛待查",
                "must_reassess_on_ast": True,
            }
        },
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    # An empiric object without a real infection diagnosis axis must not clear a named drug.
    assert result["issues"], "empiric without real infection axis must not clear named drug"
