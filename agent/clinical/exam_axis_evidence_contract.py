"""Strict, versioned examination-finding contract for A4 cross-system review."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONTRACT_VERSION = "exam-axis-evidence-contract/v1"
FINDING_FIELDS = frozenset(
    {
        "schema_version",
        "finding_code",
        "polarity",
        "target_system_id",
        "source_evidence_id",
    }
)
FINDING_POLARITIES = frozenset({"present", "absent"})
USABLE_STATUSES = frozenset({"normal", "abnormal"})


@dataclass(frozen=True)
class AxisContradiction:
    axis_id: str
    axis_system_id: str


@dataclass(frozen=True)
class FindingContract:
    finding_code: str
    target_system_id: str
    polarities: tuple[str, ...]
    contradictions: tuple[AxisContradiction, ...]


@dataclass(frozen=True)
class ExamAxisEvidenceContract:
    findings: tuple[FindingContract, ...]

    def finding(self, finding_code: str) -> Optional[FindingContract]:
        return next(
            (item for item in self.findings if item.finding_code == finding_code),
            None,
        )


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def parse_exam_axis_evidence_contract(raw: Any) -> Optional[ExamAxisEvidenceContract]:
    """Parse only a complete, closed-schema contract; malformed assets fail closed."""
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "findings"}:
        return None
    if raw.get("schema_version") != CONTRACT_VERSION:
        return None
    entries = raw.get("findings")
    if not isinstance(entries, list) or not entries:
        return None
    findings = []
    codes = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "finding_code",
            "target_system_id",
            "polarities",
            "contradictions",
        }:
            return None
        code = _text(entry.get("finding_code"))
        system = _text(entry.get("target_system_id"))
        polarities = entry.get("polarities")
        contradictions = entry.get("contradictions")
        if (
            not code
            or not system
            or code in codes
            or not isinstance(polarities, list)
            or not polarities
            or not isinstance(contradictions, list)
            or not contradictions
        ):
            return None
        parsed_polarities = tuple(_text(item) for item in polarities)
        if (
            any(item not in FINDING_POLARITIES for item in parsed_polarities)
            or len(set(parsed_polarities)) != len(parsed_polarities)
        ):
            return None
        parsed_contradictions = []
        axis_ids = set()
        for contradiction in contradictions:
            if not isinstance(contradiction, Mapping) or set(contradiction) != {
                "axis_id",
                "axis_system_id",
            }:
                return None
            axis_id = _text(contradiction.get("axis_id"))
            axis_system = _text(contradiction.get("axis_system_id"))
            if (
                not axis_id
                or not axis_system
                or axis_system == system
                or axis_id in axis_ids
            ):
                return None
            axis_ids.add(axis_id)
            parsed_contradictions.append(AxisContradiction(axis_id, axis_system))
        codes.add(code)
        findings.append(
            FindingContract(code, system, parsed_polarities, tuple(parsed_contradictions))
        )
    return ExamAxisEvidenceContract(tuple(findings))


class ExamAxisEvidenceContractUnavailable(RuntimeError):
    """Raised when a required safety contract cannot be verified."""


@lru_cache(maxsize=1)
def load_exam_axis_evidence_contract() -> ExamAxisEvidenceContract:
    path = Path(__file__).resolve().parents[1] / "knowledge" / "exam_axis_evidence_contract.json"
    try:
        with path.open(encoding="utf-8") as file:
            contract = parse_exam_axis_evidence_contract(json.load(file))
    except (OSError, ValueError, TypeError) as exc:
        raise ExamAxisEvidenceContractUnavailable(
            "exam axis evidence contract unavailable: %s" % path
        ) from exc
    if contract is None:
        raise ExamAxisEvidenceContractUnavailable(
            "exam axis evidence contract is invalid: %s" % path
        )
    return contract


def _valid_finding(raw: Any, contract: ExamAxisEvidenceContract) -> Optional[FindingContract]:
    if not isinstance(raw, Mapping) or set(raw) != FINDING_FIELDS:
        return None
    if raw.get("schema_version") != CONTRACT_VERSION:
        return None
    code = _text(raw.get("finding_code"))
    polarity = _text(raw.get("polarity"))
    system = _text(raw.get("target_system_id"))
    source = _text(raw.get("source_evidence_id"))
    finding = contract.finding(code or "")
    if (
        not finding
        or not polarity
        or not system
        or not source
        or polarity not in finding.polarities
        or system != finding.target_system_id
    ):
        return None
    return finding


def has_specific_cross_system_conflict(
    examination_results: Any,
    diagnosis_axes: Sequence[Mapping[str, Any]],
    *,
    contract: Optional[ExamAxisEvidenceContract] = None,
) -> bool:
    """Return true only for a contract-bound finding against an active other system."""
    active_axis_ids = {
        axis_id
        for axis in diagnosis_axes
        if isinstance(axis, Mapping)
        for axis_id in (_text(axis.get("axis_id")),)
        if axis_id
    }
    if not active_axis_ids or not isinstance(examination_results, Mapping):
        return False
    resolved = contract if contract is not None else load_exam_axis_evidence_contract()
    for payload in examination_results.values():
        if not isinstance(payload, Mapping):
            continue
        if _text(payload.get("status")) not in USABLE_STATUSES:
            continue
        raw_findings = payload.get("structured_findings")
        if not isinstance(raw_findings, list):
            continue
        for raw_finding in raw_findings:
            finding = _valid_finding(raw_finding, resolved)
            if finding is None:
                continue
            if any(
                contradiction.axis_id in active_axis_ids
                and contradiction.axis_system_id != finding.target_system_id
                for contradiction in finding.contradictions
            ):
                return True
    return False
