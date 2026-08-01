"""Curated reflection sources and the reflection-rule candidate schema.

Reflections are never mined from harvest runs, ground truth, grader reasoning or
online logs. The only accepted origin is a human-reviewed, read-only JSONL file
inside the workspace, so this module is deliberately strict about the path it is
willing to open and about every field it accepts.
"""
from __future__ import annotations

import json
import os
import stat
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from agent.knowledge.profile_schema import valid_reflection_rule_effect
from offline.artifacts import content_hash, file_hash, write_immutable_json

REFLECTION_SOURCE_SCHEMA = "reflection-source/v1"
REFLECTION_SOURCE_RECEIPT_SCHEMA = "reflection-source-receipt/v1"
REFLECTION_RULE_SCHEMA = "reflection-rule/v1"

# Distinct curated sources required before a rule becomes a candidate.
REFLECTION_MIN_SUPPORT_COUNT = 3

CURATED_SOURCE_REF = PurePosixPath("data/knowledge_sources/reflection_sources.jsonl")

HELD_OUT_MODULUS = 5

NOTE_MAX_UNICODE_CHARS = 160
NOTE_MAX_CJK_CHARS = 80

REFLECTION_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "source_id",
        "trigger_codes",
        "stages",
        "note",
        "positive_controls",
        "near_neighbor_controls",
        "provenance",
    }
)
CONTROL_FIELDS = frozenset({"control_id", "fact_codes"})
PROVENANCE_FIELDS = frozenset({"source_type", "source_ref", "reviewer"})

ALLOWED_STAGES = frozenset({"diagnosis", "examination", "treatment"})

# Closed trigger/fact vocabulary. Anything else is rejected rather than learned.
ALLOWED_TRIGGER_CODES = frozenset(
    {
        "immunosuppressed_infection",
        "vesicular_rash",
        "fever",
        "noninfectious_eczema",
        "isolated_vesicle_without_systemic_risk",
        "neonate",
        "infant",
        "intrauterine_viral_exposure",
        "acute_limb_soft_tissue_infection",
        "hyperlipidemia_with_xanthelasma",
        "severe_pneumonia_aerosol_exposure",
        "pediatric_congenital_glaucoma",
        "high_energy_hindfoot_trauma",
        "suspected_sepsis",
        "bleeding_tendency",
        "confirmed_resistance",
        "drug_allergy",
    }
)

REFLECTION_RULE_FIELDS = frozenset(
    {
        "schema_version",
        "trigger_codes",
        "stages",
        "note",
        "source_refs",
        "support_count",
        "source_receipt_hash",
    }
)

Partition = Literal["build", "held_out"]

_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class ReflectionSourcePathError(ValueError):
    """Raised when the curated source path is not a trusted in-workspace file."""


@dataclass(frozen=True)
class ReflectionSourceRecord:
    source_id: str
    trigger_codes: tuple[str, ...]
    stages: tuple[str, ...]
    note: str
    positive_controls: tuple[Mapping[str, Any], ...]
    near_neighbor_controls: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    partition: Partition

    def rule_key(self) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        """Normalized identity used to merge identical curated rules."""
        return (self.trigger_codes, self.stages, self.note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFLECTION_SOURCE_SCHEMA,
            "source_id": self.source_id,
            "trigger_codes": list(self.trigger_codes),
            "stages": list(self.stages),
            "note": self.note,
            "positive_controls": [dict(item) for item in self.positive_controls],
            "near_neighbor_controls": [dict(item) for item in self.near_neighbor_controls],
            "provenance": dict(self.provenance),
            "partition": self.partition,
        }


def partition_for_source_id(source_id: str) -> Partition:
    digest = sha256(str(source_id).encode("utf-8")).hexdigest()
    return "held_out" if int(digest, 16) % HELD_OUT_MODULUS == 0 else "build"


def _is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def resolve_curated_source_path(
    *,
    project_root: Path,
    source_ref: PurePosixPath | str = CURATED_SOURCE_REF,
) -> Path:
    """Only the fixed in-workspace curated file is openable.

    Aliases, absolute paths, symlinks and Windows junctions are refused so a
    redirected path can never smuggle outside content into the knowledge chain.
    """
    # Literal comparison: PurePosixPath() would silently normalize "./x" and
    # "a/../a/x" into the canonical value, so an alias must be refused on the
    # raw text before any path object is built from it.
    literal = source_ref if isinstance(source_ref, str) else str(source_ref)
    if literal != CURATED_SOURCE_REF.as_posix():
        raise ReflectionSourcePathError("source_ref must equal %s" % CURATED_SOURCE_REF)

    root = Path(project_root)
    source_path = root.joinpath(*CURATED_SOURCE_REF.parts)

    # Check every existing component from project_root/data downwards.
    current = root
    for part in CURATED_SOURCE_REF.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink() or _is_reparse_point(current):
            raise ReflectionSourcePathError("refusing reparse point: %s" % current)

    if source_path.exists():
        resolved = source_path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ReflectionSourcePathError("curated source escapes project root")
    return source_path


def _clean_code_tuple(value: Any, allowed: frozenset[str]) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if not text or text not in allowed:
            return None
        items.append(text)
    if len(set(items)) != len(items):
        return None
    return tuple(sorted(items))


def _valid_note(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text != value:
        return False
    if len(text) > NOTE_MAX_UNICODE_CHARS:
        return False
    cjk = sum(1 for char in text if unicodedata.east_asian_width(char) in {"W", "F"})
    return cjk <= NOTE_MAX_CJK_CHARS


def _clean_controls(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, list) or not value:
        return None
    controls: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != CONTROL_FIELDS:
            return None
        control_id = item.get("control_id")
        if not isinstance(control_id, str) or not control_id.strip():
            return None
        if control_id in seen:
            return None
        seen.add(control_id)
        fact_codes = _clean_code_tuple(item.get("fact_codes"), ALLOWED_TRIGGER_CODES)
        if fact_codes is None:
            return None
        controls.append({"control_id": control_id, "fact_codes": list(fact_codes)})
    return tuple(controls)


def _valid_provenance(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != PROVENANCE_FIELDS:
        return False
    if value.get("source_type") != "curated_offline":
        return False
    source_ref = value.get("source_ref")
    reviewer = value.get("reviewer")
    if not isinstance(source_ref, str) or not source_ref.strip():
        return False
    if Path(source_ref).is_absolute() or ".." in PurePosixPath(source_ref).parts:
        return False
    return isinstance(reviewer, str) and bool(reviewer.strip())


_PATIENT_ID_MARKERS = ("patient_", "ground_truth", "reference", "expected")


def parse_reflection_source_row(row: Any) -> tuple[ReflectionSourceRecord | None, str]:
    """Strictly validate one curated row; unknown fields are rejected."""
    if not isinstance(row, Mapping) or set(row) != REFLECTION_SOURCE_FIELDS:
        return None, "schema"
    if row.get("schema_version") != REFLECTION_SOURCE_SCHEMA:
        return None, "schema_version"
    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return None, "source_id"
    lowered = source_id.casefold()
    if any(marker in lowered for marker in _PATIENT_ID_MARKERS):
        return None, "source_id_leakage"
    trigger_codes = _clean_code_tuple(row.get("trigger_codes"), ALLOWED_TRIGGER_CODES)
    if trigger_codes is None:
        return None, "trigger_codes"
    stages = _clean_code_tuple(row.get("stages"), ALLOWED_STAGES)
    if stages is None:
        return None, "stages"
    if not _valid_note(row.get("note")):
        return None, "note"
    note = str(row["note"])
    if any(marker in note.casefold() for marker in _PATIENT_ID_MARKERS):
        return None, "note_leakage"
    positive = _clean_controls(row.get("positive_controls"))
    if positive is None:
        return None, "positive_controls"
    near = _clean_controls(row.get("near_neighbor_controls"))
    if near is None:
        return None, "near_neighbor_controls"
    if not _valid_provenance(row.get("provenance")):
        return None, "provenance"

    return (
        ReflectionSourceRecord(
            source_id=source_id.strip(),
            trigger_codes=trigger_codes,
            stages=stages,
            note=note,
            positive_controls=positive,
            near_neighbor_controls=near,
            provenance=dict(row["provenance"]),
            partition=partition_for_source_id(source_id.strip()),
        ),
        "",
    )


@dataclass(frozen=True)
class LoadedReflectionSources:
    status: Literal["ready", "no_curated_source"]
    source_path: Path
    records: tuple[ReflectionSourceRecord, ...]
    raw_count: int
    rejected_count: int
    rejected_reasons: Mapping[str, int]


def load_reflection_sources(
    *,
    project_root: Path,
    source_ref: PurePosixPath = CURATED_SOURCE_REF,
) -> LoadedReflectionSources:
    """Read the curated file if it exists; never invent content when it does not."""
    source_path = resolve_curated_source_path(
        project_root=project_root, source_ref=source_ref
    )
    if not source_path.is_file():
        return LoadedReflectionSources(
            status="no_curated_source",
            source_path=source_path,
            records=(),
            raw_count=0,
            rejected_count=0,
            rejected_reasons={},
        )

    raw_count = 0
    rejected_count = 0
    reasons: dict[str, int] = {}
    by_id: dict[str, ReflectionSourceRecord] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            raw_count += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                rejected_count += 1
                reasons["json"] = reasons.get("json", 0) + 1
                continue
            record, reason = parse_reflection_source_row(row)
            if record is None:
                rejected_count += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            if record.source_id in by_id:
                rejected_count += 1
                reasons["duplicate_source_id"] = reasons.get("duplicate_source_id", 0) + 1
                continue
            by_id[record.source_id] = record

    records = tuple(by_id[key] for key in sorted(by_id))
    return LoadedReflectionSources(
        status="ready",
        source_path=source_path,
        records=records,
        raw_count=raw_count,
        rejected_count=rejected_count,
        rejected_reasons=dict(sorted(reasons.items())),
    )


def build_reflection_source_receipt(loaded: "LoadedReflectionSources") -> dict[str, Any]:
    """Recomputable receipt over the curated source, reconciled by construction."""
    source_path = loaded.source_path
    records = loaded.records
    raw_count = int(loaded.raw_count)
    rejected_count = int(loaded.rejected_count)
    normalized = [record.to_dict() for record in sorted(records, key=lambda item: item.source_id)]
    build_ids = sorted(record.source_id for record in records if record.partition == "build")
    held_out_ids = sorted(
        record.source_id for record in records if record.partition == "held_out"
    )
    has_file = source_path is not None and Path(source_path).is_file()
    body = {
        "schema_version": REFLECTION_SOURCE_RECEIPT_SCHEMA,
        "status": "ready" if has_file else "no_curated_source",
        "source_ref": CURATED_SOURCE_REF.as_posix(),
        "source_file_hash": ("sha256:" + file_hash(Path(source_path))) if has_file else None,
        "normalized_source_hash": "sha256:" + content_hash(normalized),
        "raw_count": raw_count,
        "unique_count": len(records),
        "rejected_count": rejected_count,
        "build_count": len(build_ids),
        "held_out_count": len(held_out_ids),
        "build_partition_hash": "sha256:" + content_hash(build_ids),
        "held_out_partition_hash": "sha256:" + content_hash(held_out_ids),
    }
    if body["raw_count"] != body["unique_count"] + body["rejected_count"]:
        raise ValueError("reflection source rows do not reconcile")
    if body["unique_count"] != body["build_count"] + body["held_out_count"]:
        raise ValueError("reflection partitions do not reconcile")
    body["receipt_hash"] = "sha256:" + content_hash(body)
    return body


def aggregate_reflection_rules(
    source: Any,
    *,
    source_receipt_hash: str,
    partition: Literal["build"] = "build",
    min_support_count: int = REFLECTION_MIN_SUPPORT_COUNT,
) -> list[dict[str, Any]]:
    """Only identical curated rules backed by enough distinct sources become candidates."""
    if partition != "build":
        raise ValueError("reflection rules may only be built from the build partition")
    loaded_records = getattr(source, "records", None)
    if loaded_records is None:
        # An explicit record sequence must already be build-only.
        records = list(source)
        for record in records:
            if record.partition != "build":
                raise ValueError("held-out record passed to reflection builder")
    else:
        # A loaded source carries both partitions; held-out never feeds a candidate.
        records = [record for record in loaded_records if record.partition == "build"]

    grouped: dict[tuple[tuple[str, ...], tuple[str, ...], str], list[str]] = {}
    for record in records:
        grouped.setdefault(record.rule_key(), []).append(record.source_id)

    candidates: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[2], item[0], item[1])):
        source_ids = sorted(set(grouped[key]))
        if len(source_ids) < int(min_support_count):
            continue
        trigger_codes, stages, note = key
        candidates.append(
            {
                "schema_version": REFLECTION_RULE_SCHEMA,
                "trigger_codes": list(trigger_codes),
                "stages": list(stages),
                "note": note,
                "source_refs": source_ids,
                "support_count": len(source_ids),
                "source_receipt_hash": str(source_receipt_hash),
            }
        )
    return candidates

REFLECTION_CONTROL_REPORT_SCHEMA = "reflection-control-report/v1"

REFLECTION_CONTROL_MIN_POSITIVE = 2
REFLECTION_CONTROL_MIN_NEAR_NEIGHBOR = 2


@dataclass(frozen=True)
class ReflectionControlReport:
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

    def to_dict(self) -> dict:
        return {
            "schema_version": REFLECTION_CONTROL_REPORT_SCHEMA,
            "candidate_type": self.candidate_type,
            "candidate_hash": self.candidate_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "held_out_partition_hash": self.held_out_partition_hash,
            "positive_pass_count": self.positive_pass_count,
            "near_neighbor_pass_count": self.near_neighbor_pass_count,
            "false_positive_count": self.false_positive_count,
            "leakage_count": self.leakage_count,
            "passed": self.passed,
            "report_hash": self.report_hash,
        }


def build_reflection_control_report(
    *,
    rule_effect=None,
    held_out_records=None,
    rule=None,
    loaded=None,
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
):
    """Score a reflection rule on held-out curated sources only.

    A rule earns promotion only when an independent held-out source states the
    same normalized rule, its positive controls fire, and its near-neighbour
    controls stay silent. Near neighbours are the whole point: a note that fires
    on everything is worse than no note.
    """
    from agent.knowledge.profile_schema import valid_reflection_rule_effect

    if rule_effect is None:
        rule_effect = rule
    if held_out_records is None:
        # A LoadedReflectionSources carries both partitions; controls read held-out only.
        records = getattr(loaded, "records", loaded) or ()
        held_out_records = [
            record for record in records if getattr(record, "partition", None) == "held_out"
        ]

    for record in held_out_records or ():
        if getattr(record, "partition", None) != "held_out":
            raise ValueError("build-partition record passed to reflection controls")

    if not valid_reflection_rule_effect(rule_effect):
        raise ValueError("invalid reflection rule effect")

    trigger_codes = frozenset(rule_effect["trigger_codes"])
    stages = frozenset(rule_effect["stages"])
    note = rule_effect["note"]

    matching = [
        record
        for record in held_out_records or ()
        if frozenset(record.trigger_codes) == trigger_codes
        and frozenset(record.stages) == stages
        and record.note == note
    ]

    positive_pass = 0
    near_neighbor_pass = 0
    false_positive = 0
    for record in matching:
        for control in record.positive_controls:
            fired = bool(trigger_codes.intersection(control.get("fact_codes") or ()))
            if fired:
                positive_pass += 1
        for control in record.near_neighbor_controls:
            fired = bool(trigger_codes.intersection(control.get("fact_codes") or ()))
            if fired:
                false_positive += 1
            else:
                near_neighbor_pass += 1

    leakage_count = 1 if _reflection_leakage(rule_effect) else 0
    passed = bool(
        matching
        and positive_pass >= REFLECTION_CONTROL_MIN_POSITIVE
        and near_neighbor_pass >= REFLECTION_CONTROL_MIN_NEAR_NEIGHBOR
        and false_positive == 0
        and leakage_count == 0
    )
    body = {
        "schema_version": REFLECTION_CONTROL_REPORT_SCHEMA,
        "candidate_type": "reflection_rule",
        "candidate_hash": str(candidate_hash),
        "source_receipt_hash": str(source_receipt_hash),
        "held_out_partition_hash": str(held_out_partition_hash),
        "positive_pass_count": positive_pass,
        "near_neighbor_pass_count": near_neighbor_pass,
        "false_positive_count": false_positive,
        "leakage_count": leakage_count,
        "passed": passed,
    }
    report_hash = "sha256:" + content_hash(body)
    return ReflectionControlReport(
        candidate_type="reflection_rule",
        candidate_hash=body["candidate_hash"],
        source_receipt_hash=body["source_receipt_hash"],
        held_out_partition_hash=body["held_out_partition_hash"],
        positive_pass_count=positive_pass,
        near_neighbor_pass_count=near_neighbor_pass,
        false_positive_count=false_positive,
        leakage_count=leakage_count,
        passed=passed,
        report_hash=report_hash,
    )


def _reflection_leakage(rule_effect) -> bool:
    from offline.candidates import profile_value_leakage

    return bool(profile_value_leakage(dict(rule_effect)))


def write_reflection_control_report(path: Path, report: ReflectionControlReport) -> Path:
    write_immutable_json(Path(path), report.to_dict())
    return Path(path)


def validate_reflection_control_report(
    stored,
    *,
    candidate_type: str,
    candidate_hash: str,
    source_receipt_hash: str,
    held_out_partition_hash: str,
    require_passed: bool = False,
) -> bool:
    """Recompute the reflection report hash and bind it to candidate/source."""
    if not isinstance(stored, Mapping):
        raise ValueError("control report must be an object")
    if stored.get("schema_version") != REFLECTION_CONTROL_REPORT_SCHEMA:
        raise ValueError("unexpected control report schema_version")
    if stored.get("candidate_type") != candidate_type:
        raise ValueError("control report candidate_type mismatch")
    body = {key: value for key, value in stored.items() if key != "report_hash"}
    if "sha256:" + content_hash(body) != stored.get("report_hash"):
        raise ValueError("control report report_hash mismatch")
    if stored.get("candidate_hash") != candidate_hash:
        raise ValueError("control report candidate_hash mismatch")
    if stored.get("source_receipt_hash") != source_receipt_hash:
        raise ValueError("control report source_receipt_hash mismatch")
    if stored.get("held_out_partition_hash") != held_out_partition_hash:
        raise ValueError("control report held_out_partition_hash mismatch")
    if require_passed and stored.get("passed") is not True:
        raise ValueError("control report did not pass; cannot be approved")
    return True
