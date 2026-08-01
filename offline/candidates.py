from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from agent.clinical.safety_facts import validate_case_memory_safety_facts
from offline.artifacts import content_hash, read_json, write_immutable_json


LEAKAGE_MARKERS = (
    "patient_",
    "patient_id",
    "diagnoses",
    "final_diagnosis",
    "examinations",
    "expected_examinations",
    "necessary_examinations",
    "treatment_plan",
    "clinical_basis",
    "provenance",
    "ground_truth",
    "expected",
    "reference",
    "reference_treatment",
    "evaluator_reasoning",
)
PROFILE_CANDIDATE_TYPES = frozenset(
    {"disease_exam_profile", "disease_treatment_profile", "reflection_rule"}
)
# Values inside a profile are catalog names, closed codes and counts. These
# markers may never appear in a value, regardless of the surrounding key names.
PROFILE_VALUE_LEAKAGE_MARKERS = (
    "patient_",
    "patient_id",
    "ground_truth",
    "reference",
    "expected",
    "evaluator_reasoning",
    "clinical_basis",
)
PROFILE_MAX_VALUE_LENGTH = 80

CASE_MEMORY_FIELDS = frozenset(
    {
        "patient_id",
        "diagnoses",
        "examinations",
        "treatment_plan",
        "clinical_basis",
        "provenance",
    }
)
CASE_MEMORY_FIELDS_V2 = CASE_MEMORY_FIELDS | frozenset(
    {"safety_facts", "safety_facts_hash"}
)
_PATIENT_ID_PATTERN = re.compile(r"Patient_(?:Comorbid-)?\d+")
_EVALUATION_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_ENGLISH_ANSWER_SECTIONS = (
    re.compile(r"(?<![a-z])diagnosis\s*[:：]"),
    re.compile(r"(?<![a-z])(?:examination|exam)\s*[:：]"),
    re.compile(r"(?<![a-z])(?:treatment|management|therapy)\s*[:：]"),
)
_CHINESE_ANSWER_LABEL = re.compile(r"(?:标准答案|参考答案|正确答案|答案)\s*[:：]")
_CHINESE_SECTION_TERMS = (
    re.compile(r"诊断"),
    re.compile(r"(?:检查|检验)"),
    re.compile(r"治疗"),
)
_CHINESE_ANSWER_SECTIONS = (
    re.compile(r"诊断\s*[:：]"),
    re.compile(r"(?:检查|检验)\s*[:：]"),
    re.compile(r"治疗\s*[:：]"),
)


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_case_memory_effect(effect: Mapping[str, Any]) -> bool:
    fields = set(effect)
    if fields != CASE_MEMORY_FIELDS and fields != CASE_MEMORY_FIELDS_V2:
        return False
    if fields == CASE_MEMORY_FIELDS_V2 and validate_case_memory_safety_facts(
        effect.get("safety_facts"),
        effect.get("safety_facts_hash"),
    ) is None:
        return False
    patient_id = effect.get("patient_id")
    treatment = effect.get("treatment_plan")
    clinical_basis = effect.get("clinical_basis")
    provenance = effect.get("provenance")
    return bool(
        isinstance(patient_id, str)
        and _PATIENT_ID_PATTERN.fullmatch(patient_id)
        and _nonempty_string_list(effect.get("diagnoses"))
        and _nonempty_string_list(effect.get("examinations"))
        and isinstance(treatment, str)
        and treatment.strip()
        and isinstance(clinical_basis, list)
        and all(isinstance(item, str) and bool(item.strip()) for item in clinical_basis)
        and isinstance(provenance, Mapping)
        and set(provenance) == {"source", "evaluation_hash"}
        and provenance.get("source") == "train_evaluation"
        and isinstance(provenance.get("evaluation_hash"), str)
        and _EVALUATION_HASH_PATTERN.fullmatch(provenance["evaluation_hash"])
    )


def _valid_case_memory_evidence(
    proposed_effect: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    provenance = proposed_effect.get("provenance")
    evaluation_ref = evidence.get("evaluation_ref")
    valid_ref = bool(
        isinstance(evaluation_ref, str)
        and evaluation_ref.strip() == evaluation_ref
        and evaluation_ref
        and not Path(evaluation_ref).is_absolute()
        and ".." not in Path(evaluation_ref).parts
    )
    return bool(
        isinstance(provenance, Mapping)
        and set(evidence) == {"source", "evaluation_hash", "evaluation_ref"}
        and evidence.get("source") == provenance.get("source")
        and evidence.get("evaluation_hash") == provenance.get("evaluation_hash")
        and valid_ref
    )


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return []


def _answer_shape(value: str) -> bool:
    text = unicodedata.normalize("NFKC", value).casefold()
    if all(pattern.search(text) for pattern in _ENGLISH_ANSWER_SECTIONS):
        return True
    label = _CHINESE_ANSWER_LABEL.search(text)
    if label and all(pattern.search(text, label.end()) for pattern in _CHINESE_SECTION_TERMS):
        return True
    return all(pattern.search(text) for pattern in _CHINESE_ANSWER_SECTIONS)


def leakage_reason(
    proposed_effect: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    blob = unicodedata.normalize("NFKC", str(proposed_effect) + str(evidence)).casefold()
    marker_leak = any(marker in blob for marker in LEAKAGE_MARKERS)
    texts = [
        text
        for value in (proposed_effect, evidence)
        for text in _text_values(value)
    ]
    answer_leak = any(_answer_shape(text) for text in texts) or _answer_shape(" ".join(texts))
    return "leakage_marker" if marker_leak or answer_leak else ""


def _registered_profile_code_values() -> frozenset[str]:
    """Every value a closed codebook may legitimately emit.

    Registered codes are already constrained by the schema validators, so they
    must not be re-scanned for substrings: `inpatient_monitoring` legitimately
    contains `patient_`.
    """
    from offline.profile_candidates import (
        CONTRAINDICATION_CODEBOOK,
        GOAL_CODEBOOK,
        RISK_CODEBOOK,
    )
    from offline.reflection_sources import ALLOWED_STAGES, ALLOWED_TRIGGER_CODES

    return frozenset(
        set(GOAL_CODEBOOK)
        | set(RISK_CODEBOOK)
        | set(CONTRAINDICATION_CODEBOOK)
        | set(ALLOWED_TRIGGER_CODES)
        | set(ALLOWED_STAGES)
    )


def _profile_value_leakage(value: Any) -> str:
    """Scan profile values only, so legitimate schema keys never self-trip."""
    allowed = _registered_profile_code_values()
    for text in _text_values(value):
        if text in allowed:
            continue
        normalized = unicodedata.normalize("NFKC", text).casefold()
        if _PATIENT_ID_PATTERN.search(text):
            return "leakage_patient_id"
        for marker in PROFILE_VALUE_LEAKAGE_MARKERS:
            if marker in normalized:
                return "leakage_marker"
        if _answer_shape(text):
            return "leakage_answer_shape"
        if len(text) > PROFILE_MAX_VALUE_LENGTH:
            return "leakage_long_text"
    return ""


def _profile_quarantine_reason(
    candidate_type: str,
    proposed_effect: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    """Whitelist the schema first, then scan values; never str() the whole effect."""
    from offline.profile_candidates import (
        valid_exam_profile_effect,
        valid_treatment_profile_effect,
    )
    from offline.reflection_sources import valid_reflection_rule_effect

    validators = {
        "disease_exam_profile": valid_exam_profile_effect,
        "disease_treatment_profile": valid_treatment_profile_effect,
        "reflection_rule": valid_reflection_rule_effect,
    }
    validator = validators[candidate_type]
    if not validator(proposed_effect):
        return "%s_schema" % candidate_type
    for payload in (proposed_effect, evidence):
        reason = _profile_value_leakage(payload)
        if reason:
            return reason
    return ""


def _quarantine_reason(
    candidate_type: Any,
    proposed_effect: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    if candidate_type in PROFILE_CANDIDATE_TYPES:
        return _profile_quarantine_reason(candidate_type, proposed_effect, evidence)
    exact_case_memory = candidate_type == "case_memory"
    if exact_case_memory and not _valid_case_memory_effect(proposed_effect):
        return "case_memory_schema"
    if exact_case_memory and not _valid_case_memory_evidence(proposed_effect, evidence):
        return "case_memory_provenance"
    if exact_case_memory:
        return ""
    return leakage_reason(proposed_effect, evidence)


def create_candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    proposed_effect: Mapping[str, Any],
    evidence: Mapping[str, Any],
    project_root: Path | None = None,
) -> Dict[str, Any]:
    from offline.knowledge_rules import (
        KNOWLEDGE_CANDIDATE_TYPES,
        is_unknown_clinical_candidate_type,
        validate_knowledge_effect,
    )

    effect = deepcopy(dict(proposed_effect))
    evidence_copy = deepcopy(dict(evidence))
    # Profile types own a typed schema + value-level scan, so the generic
    # key-substring gate must not run on them: their legitimate schema keys
    # (diagnosis_name, exam_items) would otherwise self-trip.
    prevalidation_quarantine = (
        "" if candidate_type in PROFILE_CANDIDATE_TYPES else leakage_reason(effect, evidence_copy)
    )
    if candidate_type in KNOWLEDGE_CANDIDATE_TYPES:
        if project_root is None:
            raise ValueError("project_root required for knowledge candidate")
        if not prevalidation_quarantine:
            validate_knowledge_effect(candidate_type, effect, project_root=project_root)
    elif is_unknown_clinical_candidate_type(candidate_type):
        raise ValueError("unknown clinical candidate type: %s" % candidate_type)
    body = {
        "schema_version": "candidate/v1",
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "proposed_effect": effect,
        "evidence": evidence_copy,
        "status": "candidate",
    }
    quarantine_reason = _quarantine_reason(candidate_type, effect, evidence_copy)
    if quarantine_reason:
        body["status"] = "quarantine"
        body["quarantine_reason"] = quarantine_reason
    body["candidate_hash"] = content_hash(body)
    body["effect_hash"] = content_hash(deepcopy(effect))
    return body


def write_candidate(path: Path, candidate: Mapping[str, Any]) -> str:
    return write_immutable_json(path, dict(candidate))


def load_candidate(path: Path, *, project_root: Path | None = None) -> Dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("invalid candidate structure")
    if not data.get("candidate_hash") or not data.get("effect_hash"):
        raise ValueError("candidate missing hashes")
    proposed_effect = data.get("proposed_effect")
    evidence = data.get("evidence")
    if not isinstance(proposed_effect, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("invalid candidate structure")

    from offline.knowledge_rules import (
        KNOWLEDGE_CANDIDATE_TYPES,
        is_unknown_clinical_candidate_type,
        validate_knowledge_effect,
    )

    candidate_type = data.get("candidate_type")
    if candidate_type in KNOWLEDGE_CANDIDATE_TYPES:
        if project_root is None:
            raise ValueError("project_root required for knowledge candidate")
        validate_knowledge_effect(candidate_type, proposed_effect, project_root=project_root)
    elif is_unknown_clinical_candidate_type(candidate_type):
        raise ValueError("unknown clinical candidate type: %s" % candidate_type)

    effect_hash = content_hash(deepcopy(proposed_effect))
    if effect_hash != data.get("effect_hash"):
        raise ValueError("effect_hash mismatch")
    body = {key: value for key, value in data.items() if key not in {"candidate_hash", "effect_hash"}}
    if content_hash(body) != data.get("candidate_hash"):
        raise ValueError("candidate_hash mismatch")

    quarantine_reason = _quarantine_reason(data.get("candidate_type"), proposed_effect, evidence)
    expected_status = "quarantine" if quarantine_reason else "candidate"
    reason_matches = (
        data.get("quarantine_reason") == quarantine_reason
        if quarantine_reason
        else "quarantine_reason" not in data
    )
    if data.get("status") != expected_status or not reason_matches:
        raise ValueError("candidate quarantine state mismatch")
    return data


def profile_value_leakage(value):
    """Public alias for value-level profile leakage scanning."""
    return _profile_value_leakage(value)
