"""T02: empiric anti-infective authorization must come from a closed structured source."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.legacy_orchestrator import (
    _structured_anti_infective_provenance,
    extract_case_features,
    merge_verified_empiric_provenance,
    find_anti_infective_evidence_gaps,
    validate_empiric_provenance,
)

_HASH = "sha256:" + "a" * 64


def _infection_features(**extra: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "positive_findings": ["发热", "会阴痛", "尿频"],
        "candidate_diagnoses": ["急性细菌性前列腺炎"],
        "case_text": "发热会阴痛尿频疑前列腺炎",
        "drug_allergies": [],
        "contraindicated_drugs": [],
        "medication_risk": [],
    }
    base.update(extra)
    return base


def _verified_provenance(**extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ast": [],
        "cultures": [],
        "confirmed_resistance": [],
        "empiric": {
            "allowed": True,
            "indication": "卡波西水痘样疹",
            "must_reassess_on_ast": True,
            "source": "verified_case_memory",
            "evidence_ref": _HASH,
        },
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize(
    ("source", "allowed"),
    [
        ("verified_case_memory", True),
        ("infection_diagnosis", True),
        ("exam_result", True),
        ("free_text", False),
        ("", False),
    ],
)
def test_empiric_provenance_source_is_closed_enum(source: str, allowed: bool) -> None:
    value = {
        "allowed": True,
        "indication": "感染",
        "must_reassess_on_ast": True,
        "source": source,
        "evidence_ref": _HASH,
    }
    assert validate_empiric_provenance(value) is allowed


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        "verified_case_memory",
        {"allowed": True, "indication": "感染", "source": "verified_case_memory"},
        {
            "allowed": False,
            "indication": "感染",
            "must_reassess_on_ast": True,
            "source": "verified_case_memory",
            "evidence_ref": _HASH,
        },
        {
            "allowed": True,
            "indication": "",
            "must_reassess_on_ast": True,
            "source": "verified_case_memory",
            "evidence_ref": _HASH,
        },
        {
            "allowed": True,
            "indication": "感染",
            "must_reassess_on_ast": False,
            "source": "verified_case_memory",
            "evidence_ref": _HASH,
        },
    ],
)
def test_incomplete_empiric_provenance_is_rejected(value: Any) -> None:
    assert validate_empiric_provenance(value) is False


@pytest.mark.parametrize(
    "evidence_ref",
    ["", None, "sha256:xyz", "sha256:" + "a" * 63, "a" * 64, "sha256:" + "A" * 64],
)
def test_verified_source_requires_valid_sha256(evidence_ref: Any) -> None:
    value = {
        "allowed": True,
        "indication": "感染",
        "must_reassess_on_ast": True,
        "source": "verified_case_memory",
        "evidence_ref": evidence_ref,
    }
    assert validate_empiric_provenance(value) is False


def test_non_verified_source_requires_non_empty_evidence_ref() -> None:
    base = {
        "allowed": True,
        "indication": "急性细菌性前列腺炎",
        "must_reassess_on_ast": True,
        "source": "exam_result",
    }
    assert validate_empiric_provenance(dict(base, evidence_ref="urine_culture")) is True
    assert validate_empiric_provenance(dict(base, evidence_ref="")) is False
    assert validate_empiric_provenance(base) is False


def test_top_level_empiric_documented_cannot_self_authorize() -> None:
    features = _infection_features(
        empiric_documented=True,
        empiric_indication="医生自述已记录指征",
        candidate_diagnoses=["高脂血症"],
        case_text="皮损伴发热",
        positive_findings=["皮损", "发热"],
    )
    provenance = _structured_anti_infective_provenance(features)
    assert provenance["empiric"] is None, "free-text fields must not authorize empiric use"


def test_top_level_fields_do_not_clear_the_gate() -> None:
    plan = "立即静脉使用环丙沙星抗感染，经验性使用，待培养药敏结果回报后调整。"
    features = _infection_features(
        empiric_documented=True,
        empiric_indication="自述指征",
        candidate_diagnoses=["高脂血症"],
        case_text="发热",
        positive_findings=["发热"],
    )
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = {issue.get("code") for issue in result["issues"]}
    assert codes, "unstructured empiric claim must not release a named antibiotic"
    assert "anti_infective_without_sensitivity_evidence" in codes


def test_verified_case_memory_provenance_is_accepted() -> None:
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(),
        candidate_diagnoses=["卡波西水痘样疹"],
    )
    provenance = _structured_anti_infective_provenance(features)
    assert provenance["empiric"] is not None
    assert provenance["empiric"]["source"] == "verified_case_memory"


def test_verified_plan_with_conditional_language_converges() -> None:
    plan = "阿昔洛韦抗病毒；合并感染时经验性使用环丙沙星，待培养药敏结果回报后调整。"
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(),
        candidate_diagnoses=["卡波西水痘样疹"],
    )
    result = find_anti_infective_evidence_gaps(plan, features)
    assert result["issues"] == []


def test_verified_plan_missing_conditional_language_is_patchable_once() -> None:
    plan = "给予环丙沙星静脉滴注。"
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(),
        candidate_diagnoses=["卡波西水痘样疹"],
    )
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = {issue.get("code") for issue in result["issues"]}
    assert codes == {"anti_infective_empiric_missing_conditional_language"}
    assert all(issue.get("patchable") for issue in result["issues"])

    patched = plan + "".join(result["patches"])
    second = find_anti_infective_evidence_gaps(patched, features)
    assert second["issues"] == [], "gate-authored text must converge in one round"


def test_gate_authored_text_is_not_treated_as_new_prescription() -> None:
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(),
        candidate_diagnoses=["卡波西水痘样疹"],
    )
    plan = "局部对症处理。"
    first = find_anti_infective_evidence_gaps(plan, features)
    assert first["issues"] == [], "no named antibiotic means no gate issue"


def test_drug_allergy_still_blocks_verified_empiric() -> None:
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(),
        candidate_diagnoses=["卡波西水痘样疹"],
        drug_allergies=["环丙沙星"],
    )
    plan = "经验性使用环丙沙星，待培养药敏结果回报后调整。"
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = {issue.get("code") for issue in result["issues"]}
    assert "anti_infective_drug_allergy" in codes
    assert not any(issue.get("patchable") for issue in result["issues"])


def test_confirmed_resistance_still_blocks_verified_empiric() -> None:
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(
            confirmed_resistance=["环丙沙星"],
        ),
        candidate_diagnoses=["卡波西水痘样疹"],
    )
    plan = "经验性使用环丙沙星，待培养药敏结果回报后调整。"
    result = find_anti_infective_evidence_gaps(plan, features)
    codes = {issue.get("code") for issue in result["issues"]}
    assert "anti_infective_confirmed_resistance" in codes


def test_invalid_verified_provenance_falls_back_to_gate() -> None:
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(
            empiric={
                "allowed": True,
                "indication": "卡波西水痘样疹",
                "must_reassess_on_ast": True,
                "source": "verified_case_memory",
                "evidence_ref": "not-a-hash",
            }
        ),
        candidate_diagnoses=["高脂血症"],
        case_text="皮损",
        positive_findings=["皮损"],
    )
    provenance = _structured_anti_infective_provenance(features)
    assert provenance["empiric"] is None


def _issue_codes(result: Dict[str, List[Dict[str, Any]]]) -> set:
    return {issue.get("code") for issue in result["issues"]}


def test_verified_empiric_converges_within_three_rounds() -> None:
    """Step 5: the gate must reach a fixpoint, not append forever."""
    plan = "针对急性细菌性前列腺炎：静脉使用环丙沙星抗感染。"
    features = _infection_features(
        anti_infective_provenance=_verified_provenance(
            indication="急性细菌性前列腺炎",
        ),
    )
    current = plan
    seen: list[str] = []
    for _ in range(3):
        result = find_anti_infective_evidence_gaps(current, features)
        blocking = [
            issue
            for issue in result["issues"]
            if not issue.get("patchable")
        ]
        assert not blocking, "verified empiric must never hit an unpatchable issue"
        if not result["issues"]:
            break
        seen.append(current)
        current = current + " " + "；".join(result["patches"])
    else:
        raise AssertionError("gate did not converge within three rounds")

    # Fixpoint holds: re-running the gate adds nothing further.
    assert not find_anti_infective_evidence_gaps(current, features)["issues"]
    assert len(seen) <= 1, "one conditional-language patch is enough"


def test_gate_patch_text_alone_never_authorizes_a_new_drug() -> None:
    """A gate-authored sentence must not become provenance on the next round."""
    plan = "立即静脉使用环丙沙星抗感染。"
    first = find_anti_infective_evidence_gaps(plan, {})
    assert first["issues"], "no structured provenance means the drug stays blocked"
    patched = plan + " " + "；".join(first["patches"])
    second = find_anti_infective_evidence_gaps(patched, {})
    codes = {issue.get("code") for issue in second["issues"]}
    assert "anti_infective_without_sensitivity_evidence" in codes, (
        "self-written remediation text must not clear I3"
    )


def test_verified_provenance_preserves_exam_derived_resistance() -> None:
    """Exact-memory empiric authorization must not erase real AST/culture evidence."""
    case_state = {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [],
        "ordered_examinations": ["细菌培养及药敏试验"],
        "invalid_examinations": [],
        "examination_results": {
            "细菌培养及药敏试验": {
                "status": "abnormal",
                "result": {"药敏结果": [{"drug": "环丙沙星", "result": "耐药"}]},
            }
        },
        "decision_trace": [],
        "exam_decision_trace": [],
    }
    case_features = extract_case_features(case_state, [{"disease": "急性细菌性前列腺炎"}])
    merged = merge_verified_empiric_provenance(
        case_features,
        indication="急性细菌性前列腺炎",
        evidence_ref="sha256:" + "a" * 64,
    )

    assert merged["ast"], "structured AST rows must survive verified empiric injection"
    assert merged["empiric"]["source"] == "verified_case_memory"

    case_features["anti_infective_provenance"] = merged
    plan = "静脉使用环丙沙星抗感染，经验性使用，待培养药敏结果回报后调整。"
    codes = {
        issue.get("code")
        for issue in find_anti_infective_evidence_gaps(plan, case_features)["issues"]
    }
    assert "anti_infective_confirmed_resistance" in codes


def test_verified_provenance_merge_keeps_confirmed_resistance_list() -> None:
    case_features = {
        "candidate_diagnoses": ["急性细菌性前列腺炎"],
        "anti_infective_provenance": {
            "ast": [],
            "cultures": [{"source": "urine", "result": "pos"}],
            "confirmed_resistance": ["环丙沙星"],
            "empiric": None,
        },
    }
    merged = merge_verified_empiric_provenance(
        case_features,
        indication="急性细菌性前列腺炎",
        evidence_ref="sha256:" + "b" * 64,
    )
    assert merged["confirmed_resistance"] == ["环丙沙星"]
    assert merged["cultures"] == [{"source": "urine", "result": "pos"}]
    assert validate_empiric_provenance(merged["empiric"]) is True
