from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any, Dict, List, Tuple

from offline.artifacts import content_hash
from offline.candidates import create_candidate


CASE_MEMORY_TYPE = "case_memory"
_PATIENT_ID_PATTERN = re.compile(r"Patient_(?:Comorbid-)?\d+")


def _evaluation_report(
    evaluation: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]]]:
    layers = [evaluation]
    payload = evaluation.get("payload")
    top_report = evaluation.get("report")
    if isinstance(payload, Mapping):
        layers.append(payload)
    if isinstance(top_report, Mapping):
        layers.append(top_report)
    payload_report = payload.get("report") if isinstance(payload, Mapping) else None
    if isinstance(payload_report, Mapping):
        layers.append(payload_report)
        return payload_report, layers
    if isinstance(top_report, Mapping):
        return top_report, layers
    return evaluation, layers


def _validate_evaluation_binding(
    patient_id: str,
    report: Mapping[str, Any],
    layers: List[Mapping[str, Any]],
) -> None:
    if report.get("status") != "evaluated":
        raise ValueError("evaluation report status must be evaluated")
    for layer in layers:
        for key in ("patient_id", "patientId"):
            if key in layer and layer[key] != patient_id:
                raise ValueError("patient_id mismatch")


def _value_or_fallback(
    report: Mapping[str, Any],
    primary_section: str,
    primary_key: str,
    fallback_section: str,
    fallback_key: str,
) -> Any:
    primary = report.get(primary_section)
    if isinstance(primary, Mapping):
        value = primary.get(primary_key)
        if value not in (None, "", []):
            return value
    fallback = report.get(fallback_section)
    if isinstance(fallback, Mapping):
        return fallback.get(fallback_key)
    return None


def _as_names(value: Any, field: str, *, deduplicate: bool = True) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError("%s must be a non-empty string or list" % field)
    names: List[str] = []
    seen = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("%s contains an empty or invalid name" % field)
        name = item
        if not deduplicate or name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        raise ValueError("%s is required" % field)
    return names


def _basis(report: Mapping[str, Any]) -> List[str]:
    ground_truth = report.get("ground_truth")
    if isinstance(ground_truth, Mapping):
        # Ground-truth memory must not inherit criticism of the submitted answer.
        return _basis_from_sections(ground_truth, ())

    return _basis_from_sections(
        report,
        ("diagnosisDetail", "examinationDetail", "treatmentDetail"),
    )


def _basis_from_sections(
    report: Mapping[str, Any],
    detail_sections: tuple[str, ...],
) -> List[str]:
    values: List[str] = []
    for section_name in ("clinical_basis", "reasoning", "reflection"):
        value = report.get(section_name)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            values.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    for section_name in detail_sections:
        section = report.get(section_name)
        if not isinstance(section, Mapping):
            continue
        value = section.get("reasoning")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            values.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    # Keep the audit payload concise and deterministic.
    return list(dict.fromkeys(values))[:3]


def extract_case_memory(
    *,
    patient_id: str,
    evaluation: Mapping[str, Any],
    official_diseases: Collection[str],
    valid_examinations: Collection[str],
) -> Dict[str, Any]:
    if not isinstance(patient_id, str) or _PATIENT_ID_PATTERN.fullmatch(patient_id) is None:
        raise ValueError("invalid patient_id")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")

    report, layers = _evaluation_report(evaluation)
    _validate_evaluation_binding(patient_id, report, layers)
    diagnoses = _as_names(
        _value_or_fallback(
            report,
            "ground_truth",
            "final_diagnosis",
            "diagnosisDetail",
            "expected",
        ),
        "diagnosis",
    )
    unknown_diagnoses = [name for name in diagnoses if name not in set(official_diseases)]
    if unknown_diagnoses:
        raise ValueError("diagnosis not in official catalog: %s" % unknown_diagnoses[0])

    examinations = _as_names(
        _value_or_fallback(
            report,
            "ground_truth",
            "necessary_examinations",
            "examinationDetail",
            "expected",
        ),
        "examination",
    )
    unknown_examinations = [name for name in examinations if name not in set(valid_examinations)]
    if unknown_examinations:
        raise ValueError("examination not in valid catalog: %s" % unknown_examinations[0])

    treatment = _value_or_fallback(
        report,
        "ground_truth",
        "treatment_plan",
        "treatmentDetail",
        "reference",
    )
    if not isinstance(treatment, str) or not treatment.strip():
        raise ValueError("treatment_plan is required")

    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses,
        "examinations": examinations,
        "treatment_plan": treatment.strip(),
        "clinical_basis": _basis(report),
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + content_hash(evaluation),
        },
    }


def case_memory_candidate(
    *,
    patient_id: str,
    evaluation: Mapping[str, Any],
    evaluation_ref: str,
    official_diseases: Collection[str],
    valid_examinations: Collection[str],
) -> Dict[str, Any]:
    effect = extract_case_memory(
        patient_id=patient_id,
        evaluation=evaluation,
        official_diseases=official_diseases,
        valid_examinations=valid_examinations,
    )
    return create_candidate(
        candidate_id="case-memory-%s" % patient_id,
        candidate_type=CASE_MEMORY_TYPE,
        proposed_effect=effect,
        evidence={
            "source": "train_evaluation",
            "evaluation_hash": effect["provenance"]["evaluation_hash"],
            "evaluation_ref": evaluation_ref,
        },
    )
