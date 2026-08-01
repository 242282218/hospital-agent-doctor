"""Test the anti-infectious evidence gate (I3)."""
from __future__ import annotations

import pytest

from agent.legacy_orchestrator import find_anti_infective_evidence_gaps, SENSITIVITY_MARKERS


def _plan(antibiotic=True, sensitivity=False, empiric=False):
    if antibiotic:
        plan = "立即静脉使用左氧氟沙星抗感染治疗。"
    else:
        plan = "完善检查后制定治疗。"
    features = {"drug_allergies": [], "contraindicated_drugs": [], "medication_risk": []}
    if sensitivity:
        # Structured AST only — plan text markers are not provenance.
        features["anti_infective_provenance"] = {
            "ast": [{"drug": "左氧氟沙星", "result": "S", "source": "sputum"}]
        }
    if empiric:
        plan += " 经验用药，待药敏结果后调整。"
        features["anti_infective_provenance"] = {
            "empiric": {
                "allowed": True,
                "indication": "社区获得性肺炎",
                "must_reassess_on_ast": True,
                "source": "infection_diagnosis",
                "evidence_ref": "infection_diagnosis:社区获得性肺炎",
            }
        }
    return plan, features


def test_antibiotic_without_sensitivity_is_flagged():
    plan, features = _plan(True, False, False)
    result = find_anti_infective_evidence_gaps(plan, features)
    assert result["issues"], "expected issue without sensitivity"
    assert result["patches"], "expected patch"


def test_antibiotic_with_sensitivity_is_clean():
    plan, features = _plan(True, True)
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"], "structured sensitivity must clear"


def test_antibiotic_with_empiric_is_clean():
    plan, features = _plan(True, False, True)
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"], "structured empiric + conditional language must clear"


def test_no_antibiotic_is_clean():
    plan, features = _plan(False)
    result = find_anti_infective_evidence_gaps(plan, features)
    assert not result["issues"], "no antibiotic must not fire"


def test_sensitivity_markers_imported():
    assert "药敏" in SENSITIVITY_MARKERS
    assert "敏感" in SENSITIVITY_MARKERS


# ---- A-line leak guard: the frozen 300 registry is high-quality GT (P6/P7
# 0.966-1.0 treatment). Any I3 change must NOT increase its false-positive count. ----

HIT_CASES = ["Patient_00061", "Patient_00144", "Patient_Comorbid-01712"]


@pytest.mark.parametrize("patient_id", HIT_CASES)
def test_hit_case_does_not_get_new_false_positive(patient_id):
    """A-line exact-memory hits must stay clinically valid. We do not require
    zero issues (a flagged issue is not a correctness failure), but we lock the
    CURRENT per-case issue count so a regression adds new false positives."""
    import json
    from pathlib import Path
    reg = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "releases/release_C_case_memory_20260724_v_final_300cases/verified_registry.json"
        ).read_text(encoding="utf-8")
    )
    tx = next(
        a["content"]["treatment_plan"]
        for a in reg["assets"]
        if a["content"].get("patient_id") == patient_id
    )
    result = find_anti_infective_evidence_gaps(tx, {})
    # Baseline: Patient_00061/Comorbid-01712 clean; Patient_00144 has 1 known
    # limitation (negated drug-name substring in "磺胺...禁用" is not syntax-aware).
    baseline = {"Patient_00061": 0, "Patient_00144": 1, "Patient_Comorbid-01712": 0}
    assert len(result["issues"]) <= baseline[patient_id] + 0, (
        "%s regressed: issue count rose above baseline %d" % (patient_id, baseline[patient_id])
    )


def test_negated_contraindicated_drug_is_clean():
    # Both drug mentions describe a contraindication, not an anti-infective prescription.
    plan = "患者青霉素过敏，禁用青霉素，改用对乙酰氨基酚退热。"
    result = find_anti_infective_evidence_gaps(plan, {})
    assert not result["issues"], "negated drug mentions must clear, got %r" % result["issues"]


def test_registry_false_positive_count_locked():
    """Lock the whole-registry FP count: any code change that raises this number
    is a regression. Known limitation: substring negation is not syntax-aware."""
    import json
    from pathlib import Path
    reg = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "releases/release_C_case_memory_20260724_v_final_300cases/verified_registry.json"
        ).read_text(encoding="utf-8")
    )
    flagged = sum(
        1
        for a in reg["assets"]
        if find_anti_infective_evidence_gaps(a["content"]["treatment_plan"], {})["issues"]
    )
    # Baseline measured 2026-07-25 after structured-provenance rewrite: 64.
    # Text-only markers no longer clear the gate (intentional correctness fix).
    assert flagged <= 64, "A-line registry FP count regressed: %d (baseline 64)" % flagged


def test_empiric_without_closed_enum_source_is_flagged():
    """T02: the pre-closed-enum empiric shape can no longer authorize a named drug."""
    plan = "立即静脉使用左氧氟沙星抗感染治疗。 经验用药，待药敏结果后调整。"
    features = {
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
        "anti_infective_provenance": {
            "empiric": {
                "allowed": True,
                "indication": "社区获得性肺炎",
                "must_reassess_on_ast": True,
            }
        },
    }
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = {issue.get("code") for issue in result["issues"]}
    assert "anti_infective_without_sensitivity_evidence" in codes
