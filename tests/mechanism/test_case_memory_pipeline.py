from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from offline.artifacts import canonical_json, content_hash, read_json, write_immutable_json
from offline.case_memory import case_memory_candidate, extract_case_memory
from offline.candidates import create_candidate, load_candidate, write_candidate
from offline.promotion import approve_candidate, build_registry_snapshot


def _evaluation() -> Dict[str, Any]:
    return {
        "status": "evaluated",
        "ground_truth": {
            "final_diagnosis": "三房心",
            "necessary_examinations": ["体格检查", "超声心动图"],
            "treatment_plan": "尽快进行心脏外科评估。",
        },
        "diagnosisDetail": {"expected": ["错误的回退诊断"]},
        "examinationDetail": {"expected": ["错误的回退检查"]},
        "treatmentDetail": {
            "reference": "错误的回退治疗。",
            "reasoning": "婴儿喂养困难、发绀与心衰表现提示结构异常。",
        },
    }


def _catalogs() -> tuple[set[str], set[str]]:
    return {"三房心", "错误的回退诊断"}, {"体格检查", "超声心动图"}


def _case_memory_effect(
    *,
    patient_id: str = "Patient_01061",
    diagnosis: str = "三房心",
    examination: str = "体格检查",
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "diagnoses": [diagnosis],
        "examinations": [examination],
        "treatment_plan": "评估。",
        "clinical_basis": [],
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }


def _case_memory_evidence(
    effect: Dict[str, Any],
    evaluation_ref: str = "events/evaluation.json",
) -> Dict[str, str]:
    return {**effect["provenance"], "evaluation_ref": evaluation_ref}


def _write_evaluation_artifact(
    tmp_path: Path,
    name: str,
    evaluation: Dict[str, Any],
) -> tuple[Path, str]:
    store = tmp_path / "evaluations"
    evaluation_ref = name + ".json"
    write_immutable_json(store / evaluation_ref, evaluation)
    return store, evaluation_ref


def _rehash_candidate(candidate: Dict[str, Any]) -> None:
    body = {key: value for key, value in candidate.items() if key not in {"candidate_hash", "effect_hash"}}
    candidate["candidate_hash"] = content_hash(body)
    candidate["effect_hash"] = content_hash(candidate["proposed_effect"])


def test_extract_case_memory_prefers_ground_truth_and_validates_catalogs() -> None:
    diseases, examinations = _catalogs()
    effect = extract_case_memory(
        patient_id="Patient_01061",
        evaluation=_evaluation(),
        official_diseases=diseases,
        valid_examinations=examinations,
    )

    assert effect["patient_id"] == "Patient_01061"
    assert effect["diagnoses"] == ["三房心"]
    assert effect["examinations"] == ["体格检查", "超声心动图"]
    assert effect["treatment_plan"] == "尽快进行心脏外科评估。"
    assert effect["clinical_basis"] == []
    assert effect["provenance"]["source"] == "train_evaluation"
    assert effect["provenance"]["evaluation_hash"] == "sha256:" + content_hash(_evaluation())


def test_extract_case_memory_does_not_import_submitted_answer_review() -> None:
    evaluation = {
        **_evaluation(),
        "treatmentDetail": {
            "reference": "错误的回退治疗。",
            "reasoning": "方案缺失无法评估，未覆盖核心治疗且无安全性保障。",
        },
    }

    effect = extract_case_memory(
        patient_id="Patient_01061",
        evaluation=evaluation,
        official_diseases={"三房心", "错误的回退诊断"},
        valid_examinations={"体格检查", "超声心动图"},
    )

    assert effect["clinical_basis"] == []
    assert "方案缺失" not in effect["treatment_plan"]


def test_extract_case_memory_falls_back_to_detail_fields() -> None:
    evaluation = {
        "status": "evaluated",
        "diagnosisDetail": {"expected": ["三房心"]},
        "examinationDetail": {"expected": ["体格检查", "超声心动图"]},
        "treatmentDetail": {
            "reference": "按照心脏外科方案评估。",
            "reasoning": "症状与先天性结构异常一致。",
        },
    }
    effect = extract_case_memory(
        patient_id="Patient_Comorbid-7",
        evaluation=evaluation,
        official_diseases={"三房心"},
        valid_examinations={"体格检查", "超声心动图"},
    )

    assert effect["diagnoses"] == ["三房心"]
    assert effect["examinations"] == ["体格检查", "超声心动图"]
    assert effect["treatment_plan"] == "按照心脏外科方案评估。"


def test_extract_case_memory_supports_historical_event_payload_report() -> None:
    evaluation = {
        "patient_id": "Patient_01061",
        "status": "success",
        "payload": {
            "patient_id": "Patient_01061",
            "report": {
                **_evaluation(),
                "patientId": "Patient_01061",
            },
        },
    }

    effect = extract_case_memory(
        patient_id="Patient_01061",
        evaluation=evaluation,
        official_diseases={"三房心", "错误的回退诊断"},
        valid_examinations={"体格检查", "超声心动图"},
    )

    assert effect["diagnoses"] == ["三房心"]
    assert effect["provenance"]["evaluation_hash"] == "sha256:" + content_hash(evaluation)


def test_extract_case_memory_rejects_non_evaluated_report() -> None:
    evaluation = {**_evaluation(), "status": "failed"}

    with pytest.raises(ValueError, match="status"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation=evaluation,
            official_diseases={"三房心", "错误的回退诊断"},
            valid_examinations={"体格检查", "超声心动图"},
        )


@pytest.mark.parametrize("location", ["wrapper", "payload", "report"])
def test_extract_case_memory_rejects_any_present_patient_id_mismatch(location: str) -> None:
    evaluation = {
        "patient_id": "Patient_01061",
        "payload": {
            "patient_id": "Patient_01061",
            "report": {
                **_evaluation(),
                "patientId": "Patient_01061",
            },
        },
    }
    target = evaluation if location == "wrapper" else evaluation["payload"]
    if location == "report":
        target = evaluation["payload"]["report"]
        target["patientId"] = "Patient_99999"
    else:
        target["patient_id"] = "Patient_99999"

    with pytest.raises(ValueError, match="patient_id mismatch"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation=evaluation,
            official_diseases={"三房心", "错误的回退诊断"},
            valid_examinations={"体格检查", "超声心动图"},
        )


@pytest.mark.parametrize("patient_id", ["Patient_x", "Patient-1", "Other_1", "Patient_1_extra"])
def test_extract_case_memory_rejects_invalid_patient_id(patient_id: str) -> None:
    with pytest.raises(ValueError, match="patient_id"):
        extract_case_memory(
            patient_id=patient_id,
            evaluation=_evaluation(),
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
        )


def test_extract_case_memory_rejects_unknown_disease_or_examination() -> None:
    with pytest.raises(ValueError, match="diagnosis"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation={
                **_evaluation(),
                "ground_truth": {
                    **_evaluation()["ground_truth"],
                    "final_diagnosis": "未知疾病",
                },
            },
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
        )

    with pytest.raises(ValueError, match="examination"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation={
                **_evaluation(),
                "ground_truth": {
                    **_evaluation()["ground_truth"],
                    "necessary_examinations": ["未知检查"],
                },
            },
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
        )


def test_extract_case_memory_requires_exact_catalog_spelling() -> None:
    evaluation = {
        **_evaluation(),
        "ground_truth": {
            **_evaluation()["ground_truth"],
            "final_diagnosis": " 三房心 ",
        },
    }
    with pytest.raises(ValueError, match="diagnosis"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation=evaluation,
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
        )


@pytest.mark.parametrize("treatment", [None, "", "   "])
def test_extract_case_memory_rejects_missing_or_blank_treatment(treatment: Optional[str]) -> None:
    ground_truth = dict(_evaluation()["ground_truth"])
    if treatment is None:
        ground_truth.pop("treatment_plan")
    else:
        ground_truth["treatment_plan"] = treatment
    evaluation = {
        **_evaluation(),
        "ground_truth": ground_truth,
        "treatmentDetail": {},
    }

    with pytest.raises(ValueError, match="treatment_plan"):
        extract_case_memory(
            patient_id="Patient_01061",
            evaluation=evaluation,
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
        )


def test_case_memory_candidate_has_structured_effect_and_is_not_quarantined(
    tmp_path: Path,
) -> None:
    evaluation = _evaluation()
    _, evaluation_ref = _write_evaluation_artifact(tmp_path, "source", evaluation)
    candidate = case_memory_candidate(
        patient_id="Patient_01061",
        evaluation=evaluation,
        evaluation_ref=evaluation_ref,
        official_diseases={"三房心", "错误的回退诊断"},
        valid_examinations={"体格检查", "超声心动图"},
    )

    assert candidate["candidate_type"] == "case_memory"
    assert candidate["status"] == "candidate"
    assert set(candidate["proposed_effect"]) == {
        "patient_id",
        "diagnoses",
        "examinations",
        "treatment_plan",
        "clinical_basis",
        "provenance",
    }
    assert candidate["evidence"]["evaluation_ref"] == evaluation_ref


def test_create_candidate_deep_copies_effect_and_evidence() -> None:
    effect = _case_memory_effect()
    evidence = _case_memory_evidence(effect)
    candidate = create_candidate(
        candidate_id="deep-copy",
        candidate_type="case_memory",
        proposed_effect=effect,
        evidence=evidence,
    )

    effect["diagnoses"][0] = "篡改诊断"
    effect["provenance"]["evaluation_hash"] = "sha256:" + "b" * 64
    evidence["evaluation_hash"] = "sha256:" + "b" * 64

    assert candidate["proposed_effect"]["diagnoses"] == ["三房心"]
    assert candidate["proposed_effect"]["provenance"]["evaluation_hash"] == "sha256:" + "a" * 64
    assert candidate["evidence"]["evaluation_hash"] == "sha256:" + "a" * 64


@pytest.mark.parametrize(
    "evidence",
    [
        {"source": "train_evaluation", "evaluation_hash": "sha256:" + "a" * 64},
        {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "b" * 64,
            "evaluation_ref": "events/evaluation.json",
        },
        {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
            "evaluation_ref": "events/evaluation.json",
            "extra": "forged",
        },
        {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
            "evaluation_ref": "C:/evaluations/source.json",
        },
        {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
            "evaluation_ref": "../source.json",
        },
        {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
            "evaluation_ref": "",
        },
    ],
)
def test_case_memory_evidence_must_exactly_match_provenance(evidence: Dict[str, str]) -> None:
    candidate = create_candidate(
        candidate_id="bad-provenance",
        candidate_type="case_memory",
        proposed_effect=_case_memory_effect(),
        evidence=evidence,
    )

    assert candidate["status"] == "quarantine"


def test_load_candidate_rejects_rehashed_provenance_state_forgery(tmp_path: Path) -> None:
    forged = create_candidate(
        candidate_id="forged-provenance",
        candidate_type="case_memory",
        proposed_effect=_case_memory_effect(),
        evidence={"source": "train_evaluation"},
    )
    forged["status"] = "candidate"
    forged.pop("quarantine_reason", None)
    _rehash_candidate(forged)
    path = tmp_path / "forged-provenance.json"
    write_candidate(path, forged)

    with pytest.raises(ValueError, match="quarantine state"):
        load_candidate(path)


def test_load_candidate_rejects_rehashed_false_quarantine_state(tmp_path: Path) -> None:
    forged = create_candidate(
        candidate_id="forged-safe",
        candidate_type="mechanical_orthography",
        proposed_effect={"from": "HbA1c ", "to": "HbA1c"},
        evidence={"source": "unit-test"},
    )
    forged["status"] = "quarantine"
    forged["quarantine_reason"] = "leakage_marker"
    _rehash_candidate(forged)
    path = tmp_path / "forged-safe.json"
    write_candidate(path, forged)

    with pytest.raises(ValueError, match="quarantine state"):
        load_candidate(path)


@pytest.mark.parametrize("field", ["diagnoses", "treatment_plan", "evidence", "status"])
def test_load_candidate_rejects_tampered_artifact_hash(tmp_path: Path, field: str) -> None:
    effect = _case_memory_effect()
    candidate = create_candidate(
        candidate_id="tampered-" + field,
        candidate_type="case_memory",
        proposed_effect=effect,
        evidence=_case_memory_evidence(effect),
    )
    path = tmp_path / ("tampered-" + field + ".json")
    write_candidate(path, candidate)
    tampered = read_json(path)
    if field == "diagnoses":
        tampered["proposed_effect"]["diagnoses"] = ["篡改诊断"]
    elif field == "treatment_plan":
        tampered["proposed_effect"]["treatment_plan"] = "篡改治疗。"
    elif field == "evidence":
        tampered["evidence"]["source"] = "forged"
    else:
        tampered["status"] = "quarantine"
    path.write_text(canonical_json(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_candidate(path)


def test_generic_candidate_with_patient_answer_stays_quarantined() -> None:
    candidate = create_candidate(
        candidate_id="generic",
        candidate_type="generic_rule",
        proposed_effect={"patient_id": "opaque-case-key", "diagnoses": ["三房心"]},
        evidence={"source": "unit-test"},
    )

    assert candidate["status"] == "quarantine"


def test_near_case_memory_type_with_answer_fields_stays_quarantined() -> None:
    candidate = create_candidate(
        candidate_id="near-case-memory",
        candidate_type="case_memory_v2",
        proposed_effect={
            "diagnoses": ["三房心"],
            "examinations": ["体格检查"],
            "treatment_plan": "评估。",
        },
        evidence={"source": "unit-test"},
    )

    assert candidate["status"] == "quarantine"


def test_load_candidate_rechecks_near_case_memory_answer_boundary(tmp_path: Path) -> None:
    forged = create_candidate(
        candidate_id="forged-near-case-memory",
        candidate_type="case_memory_v2",
        proposed_effect={
            "diagnoses": ["三房心"],
            "examinations": ["体格检查"],
            "treatment_plan": "评估。",
        },
        evidence={"source": "unit-test"},
    )
    forged["status"] = "candidate"
    forged.pop("quarantine_reason", None)
    _rehash_candidate(forged)
    path = tmp_path / "forged.json"
    write_candidate(path, forged)

    with pytest.raises(ValueError, match="quarantine state"):
        load_candidate(path)


def test_case_memory_with_extra_sensitive_field_stays_quarantined() -> None:
    candidate = create_candidate(
        candidate_id="case-memory-sensitive",
        candidate_type="case_memory",
        proposed_effect={
            "patient_id": "Patient_01061",
            "diagnoses": ["三房心"],
            "examinations": ["体格检查"],
            "treatment_plan": "评估。",
            "clinical_basis": [],
            "provenance": {"source": "train_evaluation", "evaluation_hash": "sha256:" + "a" * 64},
            "api_key": "secret",
        },
        evidence={"source": "unit-test"},
    )

    assert candidate["status"] == "quarantine"


def _write_approved_case_memory(
    tmp_path: Path,
    candidate_id: str,
    patient_id: str,
    *,
    diagnosis: str = "三房心",
    examination: str = "体格检查",
) -> tuple[Path, Path]:
    evaluation = {
        **_evaluation(),
        "patientId": patient_id,
        "ground_truth": {
            **_evaluation()["ground_truth"],
            "final_diagnosis": diagnosis,
            "necessary_examinations": [examination],
        },
    }
    effect = extract_case_memory(
        patient_id=patient_id,
        evaluation=evaluation,
        official_diseases={diagnosis},
        valid_examinations={examination},
    )
    _, evaluation_ref = _write_evaluation_artifact(tmp_path, candidate_id, evaluation)
    candidate = create_candidate(
        candidate_id=candidate_id,
        candidate_type="case_memory",
        proposed_effect=effect,
        evidence=_case_memory_evidence(effect, evaluation_ref),
    )
    candidate_path = tmp_path / "candidates" / (candidate_id + ".json")
    write_candidate(candidate_path, candidate)
    decision_path = tmp_path / "decisions" / (candidate_id + ".json")
    approve_candidate(
        candidate_path=candidate_path,
        decision_path=decision_path,
        reviewer="unit-test",
        canary_required=False,
    )
    return candidate_path, decision_path


def test_registry_rejects_duplicate_verified_patient_id(tmp_path: Path) -> None:
    _, first_decision = _write_approved_case_memory(tmp_path, "first", "Patient_01061")
    _, second_decision = _write_approved_case_memory(tmp_path, "second", "Patient_01061")

    with pytest.raises(ValueError, match="duplicate case memory patient_id"):
        build_registry_snapshot(
            decision_paths=[first_decision, second_decision],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心"},
            valid_examinations={"体格检查"},
            evaluation_store=tmp_path / "evaluations",
        )


def test_registry_requires_catalogs_for_case_memory(tmp_path: Path) -> None:
    _, decision = _write_approved_case_memory(tmp_path, "case", "Patient_01061")

    with pytest.raises(ValueError, match="case memory catalogs required"):
        build_registry_snapshot(
            decision_paths=[decision],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
        )


def test_registry_requires_evaluation_store_for_case_memory(tmp_path: Path) -> None:
    _, decision = _write_approved_case_memory(tmp_path, "case", "Patient_01061")

    with pytest.raises(ValueError, match="evaluation_store required"):
        build_registry_snapshot(
            decision_paths=[decision],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心"},
            valid_examinations={"体格检查"},
        )


def test_registry_rejects_answer_not_derived_from_evaluation_artifact(tmp_path: Path) -> None:
    evaluation = {**_evaluation(), "patientId": "Patient_01061"}
    evaluation_store, evaluation_ref = _write_evaluation_artifact(tmp_path, "source", evaluation)
    effect = extract_case_memory(
        patient_id="Patient_01061",
        evaluation=evaluation,
        official_diseases={"三房心"},
        valid_examinations={"体格检查", "超声心动图"},
    )
    forged_effect = {**effect, "diagnoses": ["错误的回退诊断"]}
    candidate = create_candidate(
        candidate_id="forged-answer",
        candidate_type="case_memory",
        proposed_effect=forged_effect,
        evidence=_case_memory_evidence(forged_effect, evaluation_ref),
    )
    candidate_path = tmp_path / "candidates" / "forged-answer.json"
    write_candidate(candidate_path, candidate)
    decision_path = tmp_path / "decisions" / "forged-answer.json"
    approve_candidate(
        candidate_path=candidate_path,
        decision_path=decision_path,
        reviewer="unit-test",
        canary_required=False,
    )

    with pytest.raises(ValueError, match="does not match evaluation artifact"):
        build_registry_snapshot(
            decision_paths=[decision_path],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心", "错误的回退诊断"},
            valid_examinations={"体格检查", "超声心动图"},
            evaluation_store=evaluation_store,
        )


def test_registry_rejects_evaluation_artifact_hash_mismatch(tmp_path: Path) -> None:
    evaluation = {**_evaluation(), "patientId": "Patient_01061"}
    evaluation_store, evaluation_ref = _write_evaluation_artifact(tmp_path, "source", evaluation)
    effect = extract_case_memory(
        patient_id="Patient_01061",
        evaluation=evaluation,
        official_diseases={"三房心"},
        valid_examinations={"体格检查", "超声心动图"},
    )
    forged_effect = {
        **effect,
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "b" * 64,
        },
    }
    candidate = create_candidate(
        candidate_id="forged-hash",
        candidate_type="case_memory",
        proposed_effect=forged_effect,
        evidence=_case_memory_evidence(forged_effect, evaluation_ref),
    )
    candidate_path = tmp_path / "candidates" / "forged-hash.json"
    write_candidate(candidate_path, candidate)
    decision_path = tmp_path / "decisions" / "forged-hash.json"
    approve_candidate(
        candidate_path=candidate_path,
        decision_path=decision_path,
        reviewer="unit-test",
        canary_required=False,
    )

    with pytest.raises(ValueError, match="evaluation hash mismatch"):
        build_registry_snapshot(
            decision_paths=[decision_path],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心"},
            valid_examinations={"体格检查", "超声心动图"},
            evaluation_store=evaluation_store,
        )


def test_case_memory_evaluation_ref_rejects_path_traversal() -> None:
    effect = _case_memory_effect()
    candidate = create_candidate(
        candidate_id="path-traversal",
        candidate_type="case_memory",
        proposed_effect=effect,
        evidence=_case_memory_evidence(effect, "../outside.json"),
    )

    assert candidate["status"] == "quarantine"


@pytest.mark.parametrize(
    ("diagnosis", "examination", "message"),
    [
        ("未知疾病", "体格检查", "diagnosis"),
        ("三房心", "未知检查", "examination"),
    ],
)
def test_registry_revalidates_case_memory_catalogs(
    tmp_path: Path,
    diagnosis: str,
    examination: str,
    message: str,
) -> None:
    _, decision = _write_approved_case_memory(
        tmp_path,
        "unknown-catalog",
        "Patient_01061",
        diagnosis=diagnosis,
        examination=examination,
    )

    with pytest.raises(ValueError, match=message):
        build_registry_snapshot(
            decision_paths=[decision],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心"},
            valid_examinations={"体格检查"},
            evaluation_store=tmp_path / "evaluations",
        )


def test_registry_rejects_candidate_tampered_after_approval(tmp_path: Path) -> None:
    candidate_path, decision = _write_approved_case_memory(
        tmp_path,
        "tampered-after-approval",
        "Patient_01061",
    )
    tampered = read_json(candidate_path)
    tampered["proposed_effect"]["diagnoses"] = ["未知疾病"]
    candidate_path.write_text(canonical_json(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        build_registry_snapshot(
            decision_paths=[decision],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry.json",
            official_diseases={"三房心"},
            valid_examinations={"体格检查"},
            evaluation_store=tmp_path / "evaluations",
        )
