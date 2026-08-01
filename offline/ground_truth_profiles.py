"""Auditable ground-truth harvest accounting for offline profile building.

Harvest runs are append-only evidence produced by earlier authorized sessions.
This module only reads them: it deduplicates by patient, refuses conflicting
answers for the same patient, and splits records into a build partition and a
never-aggregated held-out partition so downstream controls stay honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from offline.artifacts import content_hash, file_hash

SOURCE_RECEIPT_SCHEMA = "ground-truth-source-receipt/v1"
HELD_OUT_MODULUS = 5
_EVALUATION_FILE_NAME = "evaluation_results.jsonl"

Partition = Literal["build", "held_out"]

REJECT_REASONS = (
    "malformed_row",
    "missing_ground_truth",
    "missing_patient_id",
    "diagnosis_not_official",
    "examination_not_catalog_leaf",
    "empty_treatment_plan",
)


@dataclass(frozen=True)
class GroundTruthRecord:
    patient_id: str
    diagnosis_items: tuple[str, ...]
    exam_items: tuple[str, ...]
    treatment_text: str
    contraindication_items: tuple[str, ...]
    source_run: str
    evaluation_hash: str
    partition: Partition

    def answer_key(self) -> tuple[Any, ...]:
        """Identity of the answer itself, ignoring which run produced it."""
        return (
            self.diagnosis_items,
            self.exam_items,
            self.treatment_text,
            self.contraindication_items,
        )


@dataclass(frozen=True)
class LoadedGroundTruth:
    records: tuple[GroundTruthRecord, ...]
    source_files: tuple[str, ...]
    source_file_hashes: Mapping[str, str]
    raw_row_count: int
    identical_duplicate_count: int
    rejected_count: int
    rejected_reasons: Mapping[str, int]
    conflicting_patient_ids: tuple[str, ...] = field(default=())

    @property
    def build_records(self) -> tuple[GroundTruthRecord, ...]:
        return tuple(item for item in self.records if item.partition == "build")

    @property
    def held_out_records(self) -> tuple[GroundTruthRecord, ...]:
        return tuple(item for item in self.records if item.partition == "held_out")


def partition_for_patient(patient_id: str) -> Partition:
    """Stable content-addressed split; never depends on input order."""
    digest = sha256(str(patient_id).encode("utf-8")).hexdigest()
    return "held_out" if int(digest, 16) % HELD_OUT_MODULUS == 0 else "build"


def _clean_items(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if text:
            items.append(text)
    return tuple(items)


_CONTRAINDICATION_KEYS = ("drugs", "treatments")


def _clean_contraindications(value: Any) -> tuple[str, ...] | None:
    """Harvest rows store contraindications as a closed {drugs, treatments} object."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, str)):
        return _clean_items(value)
    if not isinstance(value, Mapping):
        return None
    if set(value) - set(_CONTRAINDICATION_KEYS):
        return None
    items: list[str] = []
    for key in _CONTRAINDICATION_KEYS:
        if key not in value:
            continue
        cleaned = _clean_items(value.get(key))
        if cleaned is None:
            return None
        items.extend(cleaned)
    ordered: list[str] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _evaluation_hash(report: Mapping[str, Any]) -> str:
    return "sha256:" + content_hash(report.get("ground_truth"))


def _iter_source_files(source_root: Path) -> list[Path]:
    root = Path(source_root)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("*/" + _EVALUATION_FILE_NAME) if path.is_file()),
        key=lambda path: path.parent.name,
    )


def _record_from_row(
    row: Mapping[str, Any],
    *,
    source_run: str,
    official_diseases: frozenset[str],
    exam_leaf_names: frozenset[str],
) -> tuple[GroundTruthRecord | None, str]:
    patient_id = row.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.strip():
        return None, "missing_patient_id"
    report = row.get("report")
    if not isinstance(report, Mapping):
        return None, "missing_ground_truth"
    ground_truth = report.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        return None, "missing_ground_truth"

    diagnoses = _clean_items(ground_truth.get("final_diagnosis"))
    exams = _clean_items(ground_truth.get("necessary_examinations"))
    contraindications = _clean_contraindications(ground_truth.get("contraindications"))
    treatment = ground_truth.get("treatment_plan")
    if diagnoses is None or exams is None or contraindications is None:
        return None, "malformed_row"
    if not diagnoses or any(item not in official_diseases for item in diagnoses):
        return None, "diagnosis_not_official"
    if any(item not in exam_leaf_names for item in exams):
        return None, "examination_not_catalog_leaf"
    if not isinstance(treatment, str) or not treatment.strip():
        return None, "empty_treatment_plan"

    return (
        GroundTruthRecord(
            patient_id=patient_id.strip(),
            diagnosis_items=diagnoses,
            exam_items=exams,
            treatment_text=treatment.strip(),
            contraindication_items=contraindications,
            source_run=source_run,
            evaluation_hash=_evaluation_hash(report),
            partition=partition_for_patient(patient_id.strip()),
        ),
        "",
    )


def load_ground_truth_records(
    *,
    source_root: Path,
    official_diseases: Iterable[str],
    exam_leaf_names: Iterable[str],
) -> LoadedGroundTruth:
    """Read every harvest run once; dedupe by patient and fail on conflicts."""
    official = frozenset(official_diseases)
    leaves = frozenset(exam_leaf_names)

    source_files = _iter_source_files(Path(source_root))
    raw_count = 0
    identical_duplicate_count = 0
    rejected_reasons: dict[str, int] = {}
    by_patient: dict[str, GroundTruthRecord] = {}
    conflicts: list[str] = []

    def reject(reason: str) -> None:
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for path in source_files:
        source_run = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                raw_count += 1
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    reject("malformed_row")
                    continue
                if not isinstance(row, Mapping):
                    reject("malformed_row")
                    continue
                record, reason = _record_from_row(
                    row,
                    source_run=source_run,
                    official_diseases=official,
                    exam_leaf_names=leaves,
                )
                if record is None:
                    reject(reason or "malformed_row")
                    continue
                existing = by_patient.get(record.patient_id)
                if existing is None:
                    by_patient[record.patient_id] = record
                    continue
                if existing.answer_key() == record.answer_key():
                    identical_duplicate_count += 1
                    continue
                conflicts.append(record.patient_id)

    if conflicts:
        raise ValueError(
            "conflicting ground truth for patients: %s"
            % ", ".join(sorted(set(conflicts))[:5])
        )

    records = tuple(by_patient[key] for key in sorted(by_patient))
    file_hashes = {
        path.parent.name + "/" + path.name: "sha256:" + file_hash(path)
        for path in source_files
    }
    return LoadedGroundTruth(
        records=records,
        source_files=tuple(sorted(file_hashes)),
        source_file_hashes=dict(sorted(file_hashes.items())),
        raw_row_count=raw_count,
        identical_duplicate_count=identical_duplicate_count,
        rejected_count=sum(rejected_reasons.values()),
        rejected_reasons=dict(sorted(rejected_reasons.items())),
        conflicting_patient_ids=(),
    )


def build_source_receipt(
    loaded: LoadedGroundTruth,
    *,
    declared_pool_count: int,
    rejected_ledger_count: int = 0,
    rejected_ledger_ref: str = "",
) -> dict[str, Any]:
    """Close the books: raw = unique + identical duplicates + rejected."""
    build_ids = sorted(record.patient_id for record in loaded.build_records)
    held_out_ids = sorted(record.patient_id for record in loaded.held_out_records)
    unique_count = len(loaded.records)

    accounted = unique_count + loaded.identical_duplicate_count + loaded.rejected_count
    if accounted != loaded.raw_row_count:
        raise ValueError(
            "raw rows do not reconcile: raw=%d accounted=%d"
            % (loaded.raw_row_count, accounted)
        )
    if unique_count != len(build_ids) + len(held_out_ids):
        raise ValueError("partition counts do not reconcile")

    declared = int(declared_pool_count)
    total_rejected = loaded.rejected_count + max(0, int(rejected_ledger_count))
    missing_count = declared - unique_count - total_rejected

    body = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "declared_pool_count": declared,
        "raw_row_count": loaded.raw_row_count,
        "unique_count": unique_count,
        "identical_duplicate_count": loaded.identical_duplicate_count,
        "conflict_count": len(loaded.conflicting_patient_ids),
        "rejected_count": loaded.rejected_count,
        "rejected_reasons": dict(loaded.rejected_reasons),
        "rejected_ledger_count": max(0, int(rejected_ledger_count)),
        "rejected_ledger_ref": str(rejected_ledger_ref),
        "total_rejected_count": total_rejected,
        "missing_count": missing_count,
        "build_count": len(build_ids),
        "held_out_count": len(held_out_ids),
        "source_file_hashes": dict(loaded.source_file_hashes),
        "source_hash": "sha256:" + content_hash(sorted(loaded.source_file_hashes.items())),
        "build_partition_hash": "sha256:" + content_hash(build_ids),
        "held_out_partition_hash": "sha256:" + content_hash(held_out_ids),
        "reconciled": missing_count >= 0,
    }
    body["receipt_hash"] = "sha256:" + content_hash(body)
    return body
