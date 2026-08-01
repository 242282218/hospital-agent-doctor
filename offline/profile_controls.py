"""Held-out controls for disease profiles.

A profile is an aggregate claim built from the build partition. These controls
re-measure that claim on records that never entered the aggregation, so a
profile that only memorized its own build set cannot be promoted. Controls never
mutate candidates: a failure only sets ``passed=False``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from offline.artifacts import content_hash, write_immutable_json
from offline.ground_truth_profiles import GroundTruthRecord
from offline.profile_candidates import (
    CONTRAINDICATION_CODEBOOK,
    GOAL_CODEBOOK,
    codes_for_text,
)

PROFILE_CONTROL_REPORT_SCHEMA = "profile-control-report/v1"

PROFILE_CONTROL_CANDIDATE_TYPES = frozenset(
    {"disease_exam_profile", "disease_treatment_profile"}
)

# A candidate may lose at most this much precision against the held-out
# no-profile baseline; recall and goal recall may not drop at all.
MAX_PRECISION_REGRESSION = 0.005


@dataclass(frozen=True)
class ProfileControlReport:
    candidate_type: Literal["disease_exam_profile", "disease_treatment_profile"]
    candidate_hash: str
    source_receipt_hash: str
    held_out_partition_hash: str
    exam_macro_recall_at_12: float
    exam_macro_precision_at_12: float
    baseline_exam_macro_recall_at_12: float
    baseline_exam_macro_precision_at_12: float
    treatment_goal_macro_recall: float
    baseline_treatment_goal_macro_recall: float
    contraindication_false_positive_count: int
    held_out_case_count: int
    evaluated_diagnosis_count: int
    leakage_count: int
    passed: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["schema_version"] = PROFILE_CONTROL_REPORT_SCHEMA
        return body


def _require_held_out(records: Sequence[GroundTruthRecord]) -> None:
    for record in records:
        if record.partition != "held_out":
            raise ValueError("control metrics require held-out records only")


def _held_out_by_diagnosis(
    records: Sequence[GroundTruthRecord],
) -> dict[str, list[GroundTruthRecord]]:
    grouped: dict[str, list[GroundTruthRecord]] = {}
    for record in records:
        for diagnosis in set(record.diagnosis_items):
            grouped.setdefault(diagnosis, []).append(record)
    return grouped


def _finalize(body: dict[str, Any]) -> ProfileControlReport:
    body = dict(body)
    body.pop("report_hash", None)
    schema_body = dict(body)
    schema_body["schema_version"] = PROFILE_CONTROL_REPORT_SCHEMA
    body["report_hash"] = "sha256:" + content_hash(schema_body)
    return ProfileControlReport(**body)


def build_exam_profile_control_report(
    *,
    profiles: Sequence[Mapping[str, Any]],
    held_out_records: Sequence[GroundTruthRecord],
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
    exam_catalog_order: Sequence[str],
    max_examinations: int = 12,
) -> ProfileControlReport:
    """Measure profile-driven examination recall/precision on held-out cases."""
    _require_held_out(held_out_records)
    catalog = set(exam_catalog_order)
    grouped = _held_out_by_diagnosis(held_out_records)
    profile_by_diagnosis = {
        str(profile.get("diagnosis_name")): profile for profile in profiles
    }

    recalls: list[float] = []
    precisions: list[float] = []
    baseline_recalls: list[float] = []
    baseline_precisions: list[float] = []
    evaluated = 0

    for diagnosis, profile in sorted(profile_by_diagnosis.items()):
        records = grouped.get(diagnosis)
        if not records:
            continue
        evaluated += 1
        predicted = [
            str(item.get("name"))
            for item in list(profile.get("exam_items") or [])[: int(max_examinations)]
            if str(item.get("name")) in catalog
        ]
        predicted_set = set(predicted)
        for record in records:
            expected = {name for name in record.exam_items if name in catalog}
            if not expected:
                continue
            hits = len(expected & predicted_set)
            recalls.append(hits / len(expected))
            precisions.append(hits / len(predicted_set) if predicted_set else 0.0)
            # No-profile baseline orders nothing from a profile, so it neither
            # recalls nor mis-orders anything.
            baseline_recalls.append(0.0)
            baseline_precisions.append(1.0)

    recall = sum(recalls) / len(recalls) if recalls else 0.0
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    baseline_recall = sum(baseline_recalls) / len(baseline_recalls) if baseline_recalls else 0.0
    baseline_precision = (
        sum(baseline_precisions) / len(baseline_precisions) if baseline_precisions else 0.0
    )
    leakage = _profile_leakage_count(profiles)
    passed = bool(
        recalls
        and leakage == 0
        and recall >= baseline_recall
        and precision >= 1.0 - MAX_PRECISION_REGRESSION
    )
    return _finalize(
        {
            "candidate_type": "disease_exam_profile",
            "candidate_hash": str(candidate_hash),
            "source_receipt_hash": str(source_receipt_hash),
            "held_out_partition_hash": str(held_out_partition_hash),
            "exam_macro_recall_at_12": round(recall, 6),
            "exam_macro_precision_at_12": round(precision, 6),
            "baseline_exam_macro_recall_at_12": round(baseline_recall, 6),
            "baseline_exam_macro_precision_at_12": round(baseline_precision, 6),
            "treatment_goal_macro_recall": 0.0,
            "baseline_treatment_goal_macro_recall": 0.0,
            "contraindication_false_positive_count": 0,
            "held_out_case_count": len(held_out_records),
            "evaluated_diagnosis_count": evaluated,
            "leakage_count": leakage,
            "passed": passed,
        }
    )


def build_treatment_profile_control_report(
    *,
    profiles: Sequence[Mapping[str, Any]],
    held_out_records: Sequence[GroundTruthRecord],
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
) -> ProfileControlReport:
    """Measure goal recall and contraindication false positives on held-out cases."""
    _require_held_out(held_out_records)
    grouped = _held_out_by_diagnosis(held_out_records)
    profile_by_diagnosis = {
        str(profile.get("diagnosis_name")): profile for profile in profiles
    }

    recalls: list[float] = []
    false_positives = 0
    evaluated = 0

    for diagnosis, profile in sorted(profile_by_diagnosis.items()):
        records = grouped.get(diagnosis)
        if not records:
            continue
        evaluated += 1
        goal_codes = {str(code) for code in profile.get("goal_codes") or ()}
        contraindication_codes = {
            str(code) for code in profile.get("contraindication_codes") or ()
        }
        for record in records:
            observed_goals, _ = codes_for_text(record.treatment_text, GOAL_CODEBOOK)
            if observed_goals:
                hits = len(observed_goals & goal_codes)
                recalls.append(hits / len(observed_goals))
            observed_contraindications: set[str] = set()
            for item in record.contraindication_items:
                codes, _ = codes_for_text(item, CONTRAINDICATION_CODEBOOK)
                observed_contraindications.update(codes)
            # A profile-asserted contraindication the held-out case contradicts
            # is a false positive: it would trigger a needless safety edit.
            false_positives += len(contraindication_codes - observed_contraindications)

    recall = sum(recalls) / len(recalls) if recalls else 0.0
    # No-profile baseline supplies no goal evidence at all.
    baseline_recall = 0.0
    leakage = _profile_leakage_count(profiles)
    passed = bool(
        recalls
        and leakage == 0
        and false_positives == 0
        and recall >= baseline_recall
    )
    return _finalize(
        {
            "candidate_type": "disease_treatment_profile",
            "candidate_hash": str(candidate_hash),
            "source_receipt_hash": str(source_receipt_hash),
            "held_out_partition_hash": str(held_out_partition_hash),
            "exam_macro_recall_at_12": 0.0,
            "exam_macro_precision_at_12": 0.0,
            "baseline_exam_macro_recall_at_12": 0.0,
            "baseline_exam_macro_precision_at_12": 0.0,
            "treatment_goal_macro_recall": round(recall, 6),
            "baseline_treatment_goal_macro_recall": round(baseline_recall, 6),
            "contraindication_false_positive_count": false_positives,
            "held_out_case_count": len(held_out_records),
            "evaluated_diagnosis_count": evaluated,
            "leakage_count": leakage,
            "passed": passed,
        }
    )


def _profile_leakage_count(profiles: Sequence[Mapping[str, Any]]) -> int:
    from offline.candidates import profile_value_leakage

    return sum(1 for profile in profiles if profile_value_leakage(profile))


def write_control_report(path: Path, report: ProfileControlReport) -> Path:
    write_immutable_json(Path(path), report.to_dict())
    return Path(path)


def validate_profile_control_report(
    stored: Mapping[str, Any],
    *,
    candidate_type: str,
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
    require_passed: bool = False,
) -> bool:
    """Recompute the report hash and bind it to candidate, source and partition."""
    if not isinstance(stored, Mapping):
        raise ValueError("control report must be an object")
    if stored.get("schema_version") != PROFILE_CONTROL_REPORT_SCHEMA:
        raise ValueError("unexpected control report schema_version")
    if stored.get("candidate_type") != candidate_type:
        raise ValueError("control report candidate_type mismatch")
    body = {
        key: value
        for key, value in stored.items()
        if key != "report_hash"
    }
    expected = "sha256:" + content_hash(body)
    if expected != stored.get("report_hash"):
        raise ValueError("control report report_hash mismatch")
    if stored.get("candidate_hash") != candidate_hash:
        raise ValueError("control report candidate_hash mismatch")
    if stored.get("source_receipt_hash") != source_receipt_hash:
        raise ValueError("control report source_receipt_hash mismatch")
    if stored.get("held_out_partition_hash") != held_out_partition_hash:
        raise ValueError("control report held_out_partition_hash mismatch")
    if require_passed and stored.get("passed") is not True:
        raise ValueError("control report did not pass")
    return True


REFLECTION_CONTROL_REPORT_SCHEMA = "reflection-control-report/v1"

REFLECTION_POSITIVE_MIN = 2
REFLECTION_NEAR_NEIGHBOR_MIN = 2


@dataclass(frozen=True)
class ReflectionControlReport:
    """Reflection rules get their own metrics; never borrow exam/treatment fields."""

    candidate_type: Literal["reflection_rule"]
    candidate_hash: str
    source_receipt_hash: str
    held_out_partition_hash: str
    positive_pass_count: int
    near_neighbor_pass_count: int
    false_positive_count: int
    leakage_count: int
    passed: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["schema_version"] = REFLECTION_CONTROL_REPORT_SCHEMA
        return body


def _finalize_reflection(body: dict[str, Any]) -> ReflectionControlReport:
    body = dict(body)
    body.pop("report_hash", None)
    payload = dict(body)
    payload["schema_version"] = REFLECTION_CONTROL_REPORT_SCHEMA
    report_hash = "sha256:" + content_hash(payload)
    return ReflectionControlReport(**body, report_hash=report_hash)


def validate_reflection_control_report(
    stored: Any,
    *,
    candidate_type: str,
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
    require_passed: bool = False,
) -> bool:
    """Recompute the reflection report hash and bind it to its candidate/source."""
    if not isinstance(stored, Mapping):
        raise ValueError("control report must be an object")
    if stored.get("schema_version") != REFLECTION_CONTROL_REPORT_SCHEMA:
        raise ValueError("unexpected control report schema_version")
    if stored.get("candidate_type") != candidate_type:
        raise ValueError("control report candidate_type mismatch")
    if stored.get("candidate_hash") != candidate_hash:
        raise ValueError("control report candidate_hash mismatch")
    if stored.get("source_receipt_hash") != source_receipt_hash:
        raise ValueError("control report source_receipt_hash mismatch")
    if stored.get("held_out_partition_hash") != held_out_partition_hash:
        raise ValueError("control report held_out_partition_hash mismatch")

    body = {key: value for key, value in stored.items() if key != "report_hash"}
    if "sha256:" + content_hash(body) != stored.get("report_hash"):
        raise ValueError("control report report_hash mismatch")
    if require_passed and stored.get("passed") is not True:
        raise ValueError("control report did not pass")
    return True


def write_reflection_control_report(path: Path, report: ReflectionControlReport) -> Path:
    write_immutable_json(Path(path), report.to_dict())
    return Path(path)
