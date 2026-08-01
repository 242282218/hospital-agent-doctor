"""A3: exact-memory safety facts must be hash-bound and fail closed."""
from __future__ import annotations

from copy import deepcopy

import pytest

from agent.clinical.final_submission import FinalAuthorizationRegistry, FinalPayload, LoadedRuntimeIdentity
from agent.clinical.final_submission_adapters import build_case_coordinator, make_clinical_context
from agent.clinical.safety_facts import (
    SafetyFact,
    canonical_safety_facts_hash,
    validate_case_memory_safety_facts,
)
from agent.legacy_orchestrator import validate_runtime_case_memory


BASE_MEMORY = {
    "patient_id": "Patient_01061",
    "diagnoses": ["三房心"],
    "examinations": ["体格检查"],
    "treatment_plan": "转心血管外科评估手术指征。",
    "clinical_basis": ["先天性心脏结构异常"],
    "provenance": {
        "source": "train_evaluation",
        "evaluation_hash": "sha256:" + "a" * 64,
    },
}
OFFICIAL_DISEASES = ["三房心"]
EXAMINATION_CATALOG = {"影像学检查": ["体格检查"]}


def _facts() -> list[dict[str, object]]:
    return [
        {
            "fact_id": "allergy-1",
            "kind": "allergy",
            "value": "磺胺类药物",
            "polarity": "present",
            "source_ref": "structured_safety_intake",
            "source_evidence_ids": ["obs-1"],
            "temporality": "current",
        }
    ]


def _v2_memory() -> dict[str, object]:
    facts = _facts()
    parsed = validate_case_memory_safety_facts(
        facts,
        "sha256:" + "0" * 64,
    )
    assert parsed is None
    typed = (
        SafetyFact(
            fact_id="allergy-1",
            kind="allergy",
            value="磺胺类药物",
            polarity="present",
            source_ref="structured_safety_intake",
            source_evidence_ids=("obs-1",),
        ),
    )
    return {
        **BASE_MEMORY,
        "safety_facts": facts,
        "safety_facts_hash": canonical_safety_facts_hash(typed),
    }


def _validate(memory: object):
    return validate_runtime_case_memory(
        memory,
        patient_id="Patient_01061",
        official_diseases=OFFICIAL_DISEASES,
        examination_catalog=EXAMINATION_CATALOG,
    )


def test_legacy_memory_is_prior_only_not_safety_complete() -> None:
    validated = _validate(deepcopy(BASE_MEMORY))
    assert validated is not None
    assert validated["safety_facts_complete"] is False
    assert validated["safety_facts"] == []


def test_complete_v2_memory_exposes_hash_bound_safety_facts() -> None:
    validated = _validate(_v2_memory())
    assert validated is not None
    assert validated["safety_facts_complete"] is True
    assert [fact.kind for fact in validated["safety_facts"]] == ["allergy"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["safety_facts"].append({"unknown": "fact"}),
        lambda value: value["safety_facts"].__setitem__(0, {**value["safety_facts"][0], "kind": "made_up"}),
        lambda value: value["safety_facts"].__setitem__(0, {**value["safety_facts"][0], "source_evidence_ids": []}),
        lambda value: value["safety_facts"].__setitem__(0, {**value["safety_facts"][0], "temporality": "historical"}),
        lambda value: value.update(safety_facts_hash="sha256:" + "f" * 64),
    ],
)
def test_invalid_v2_safety_facts_fail_closed(mutate) -> None:
    memory = _v2_memory()
    mutate(memory)
    assert _validate(memory) is None


def test_duplicate_id_and_polarity_conflict_fail_closed() -> None:
    memory = _v2_memory()
    duplicate = deepcopy(memory["safety_facts"][0])
    duplicate["polarity"] = "absent"
    memory["safety_facts"].append(duplicate)
    assert _validate(memory) is None


def test_treatment_or_basis_text_cannot_upgrade_legacy_memory() -> None:
    memory = deepcopy(BASE_MEMORY)
    memory["treatment_plan"] = "患者对磺胺类药物过敏。"
    memory["clinical_basis"] = ["当前服用华法林且肾功能不全"]
    validated = _validate(memory)
    assert validated is not None
    assert validated["safety_facts_complete"] is False


def test_adapter_snapshots_safety_facts() -> None:
    facts = _v2_memory()["safety_facts"]
    context = make_clinical_context(
        diagnoses=["三房心"],
        examinations=["体格检查"],
        official_diseases=OFFICIAL_DISEASES,
        examination_catalog=EXAMINATION_CATALOG,
        exam_plan_trace=[],
        case_features={"safety_facts": facts},
        safety_profiles=[],
        clinical_basis=[],
        safety_facts=facts,
    )
    identity = LoadedRuntimeIdentity(
        identity_hash="sha256:" + "a" * 64,
        status="strict_verified",
    )
    coordinator = build_case_coordinator(
        registry=FinalAuthorizationRegistry(
            release_identity_hash=identity.identity_hash,
        ),
        runtime_identity=identity,
        clinical_context=context,
    )
    facts[0]["value"] = "MUTATED"
    assert coordinator._apply_safety.__self__._features["safety_facts"][0]["value"] == "磺胺类药物"
