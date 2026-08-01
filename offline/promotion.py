from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Optional, Sequence

from offline.artifacts import content_hash, read_json, write_immutable_json
from offline.case_memory import extract_case_memory
from offline.candidates import load_candidate


def normalize_orthography(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def is_mechanical_orthography(before: str, after: str) -> bool:
    return normalize_orthography(before) == normalize_orthography(after) and before != after


def _resolve_control_report(
    *,
    control_store: Path,
    control_report_ref: str,
) -> tuple[Path, Dict[str, Any]]:
    """Resolve a control report strictly inside the immutable control store."""
    ref = str(control_report_ref or "")
    if not ref:
        raise ValueError("control_report_ref required for profile candidates")
    relative = Path(ref)
    if relative.is_absolute() or ".." in relative.parts or ref.startswith(("/", "\\")):
        raise ValueError("invalid control_report_ref: %s" % ref)
    root = Path(control_store).resolve()
    report_path = (root / relative).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("control_report_ref escapes control store") from exc
    if not report_path.is_file():
        raise ValueError("control report missing: %s" % ref)
    return report_path, read_json(report_path)


CONTROL_REPORT_CANDIDATE_TYPES = frozenset(
    {"disease_exam_profile", "disease_treatment_profile", "reflection_rule"}
)


def _control_report_validator(candidate_type: Any):
    """Discriminated union: profile reports and reflection reports never mix."""
    if candidate_type in {"disease_exam_profile", "disease_treatment_profile"}:
        from offline.profile_controls import validate_profile_control_report

        return validate_profile_control_report
    if candidate_type == "reflection_rule":
        from offline.profile_controls import validate_reflection_control_report

        return validate_reflection_control_report
    raise ValueError("no control report validator for %s" % candidate_type)


def _validated_control_report(
    *,
    candidate: Mapping[str, Any],
    control_store: Optional[Path],
    control_report_ref: str,
) -> Dict[str, Any]:
    """Profile and reflection candidates need a passing, hash-bound report."""
    candidate_type = candidate.get("candidate_type")
    if control_store is None:
        raise ValueError("control report required for %s" % candidate_type)
    report_path, report = _resolve_control_report(
        control_store=control_store,
        control_report_ref=control_report_ref,
    )
    evidence = candidate.get("evidence") or {}
    source_receipt_hash = str(report.get("source_receipt_hash") or "")
    if isinstance(evidence, Mapping) and evidence.get("source_receipt_hash"):
        source_receipt_hash = str(evidence["source_receipt_hash"])
    _control_report_validator(candidate_type)(
        report,
        candidate_type=candidate_type,
        candidate_hash=str(candidate.get("candidate_hash") or ""),
        source_receipt_hash=source_receipt_hash,
        held_out_partition_hash=str(report.get("held_out_partition_hash") or ""),
        require_passed=True,
    )
    return {
        "path": report_path,
        "report": report,
        "hash": str(report.get("report_hash") or ""),
    }


def approve_candidate(
    *,
    candidate_path: Path,
    decision_path: Path,
    reviewer: str,
    decision: str = "approved",
    canary_required: bool = True,
    required_gate_ids: Sequence[str] = ("gate0_schema", "gate1_safety"),
    rationale: str = "",
    control_store: Optional[Path] = None,
    control_report_ref: str = "",
) -> Dict[str, Any]:
    """Require a passing immutable control report for profile candidates only."""
    from offline.candidates import PROFILE_CANDIDATE_TYPES

    candidate = load_candidate(candidate_path)
    candidate_hash = str(candidate.get("candidate_hash") or "")
    effect_hash = str(candidate.get("effect_hash") or "")
    if not candidate_hash or not effect_hash:
        raise ValueError("candidate hashes required")
    if candidate.get("status") == "quarantine":
        raise ValueError("quarantined candidate cannot be approved")

    candidate_type = candidate.get("candidate_type")
    needs_control = candidate_type in PROFILE_CANDIDATE_TYPES
    if not needs_control and (control_store is not None or control_report_ref):
        raise ValueError("control reports only apply to profile candidates")

    payload = {
        "schema_version": "promotion-decision/v1",
        "candidate_id": candidate.get("candidate_id"),
        "candidate_hash": candidate_hash,
        "candidate_effect_hash": effect_hash,
        "reviewer": reviewer,
        "rationale": rationale,
        "decision": decision,
        "canary_required": bool(canary_required),
        "required_gate_ids": list(required_gate_ids),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    if needs_control:
        resolved = _validated_control_report(
            candidate=candidate,
            control_store=control_store,
            control_report_ref=control_report_ref,
        )
        payload["control_report_ref"] = str(control_report_ref)
        payload["control_report_hash"] = resolved["hash"]
    payload["decision_hash"] = content_hash(payload)
    write_immutable_json(decision_path, payload)
    return payload


def _load_evaluation_artifact(
    *,
    evaluation_store: Path,
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Any:
    root = Path(evaluation_store).resolve()
    evaluation_ref = evidence.get("evaluation_ref")
    if not isinstance(evaluation_ref, str):
        raise ValueError("invalid evaluation_ref")
    relative_path = Path(evaluation_ref)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("invalid evaluation_ref")
    artifact_path = (root / relative_path).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("evaluation_ref escapes evaluation_store") from exc
    artifact = read_json(artifact_path)
    artifact_hash = "sha256:" + content_hash(artifact)
    if artifact_hash != evidence.get("evaluation_hash") or artifact_hash != provenance.get(
        "evaluation_hash"
    ):
        raise ValueError("evaluation hash mismatch")
    return artifact


def _reverify_control_report(
    *,
    decision: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control_store: Path,
) -> Dict[str, Any]:
    """Reload the report by ref and recompute its hash, so tampering fails closed."""
    ref = decision.get("control_report_ref")
    report_path, stored = _resolve_control_report(
        control_store=control_store,
        control_report_ref=ref,
    )
    report_hash = "sha256:" + content_hash(
        {key: value for key, value in stored.items() if key != "report_hash"}
    )
    if report_hash != stored.get("report_hash"):
        raise ValueError("control report_hash mismatch")
    if report_hash != decision.get("control_report_hash"):
        raise ValueError("control report hash does not match decision")
    validator = _control_report_validator(candidate.get("candidate_type"))
    validator(
        stored,
        candidate_type=candidate.get("candidate_type"),
        candidate_hash=candidate.get("candidate_hash"),
        source_receipt_hash=(candidate.get("evidence") or {}).get("source_receipt_hash"),
        held_out_partition_hash=stored.get("held_out_partition_hash"),
        require_passed=True,
    )
    return stored


def build_registry_snapshot(
    *,
    decision_paths: Sequence[Path],
    candidate_store: Path,
    output_path: Path,
    official_diseases: Optional[Collection[str]] = None,
    valid_examinations: Optional[Collection[str]] = None,
    evaluation_store: Optional[Path] = None,
    control_store: Optional[Path] = None,
) -> Dict[str, Any]:
    assets = []
    case_patient_ids = set()
    for decision_path in decision_paths:
        decision = read_json(decision_path)
        if decision.get("decision") != "approved":
            raise ValueError("decision not approved: %s" % decision_path)
        if not decision.get("decision_hash"):
            raise ValueError("missing decision_hash")
        # Recompute decision hash without self field.
        body = {k: v for k, v in decision.items() if k != "decision_hash"}
        if content_hash(body) != decision.get("decision_hash"):
            raise ValueError("decision_hash mismatch")
        candidate_path = Path(candidate_store) / ("%s.json" % decision.get("candidate_id"))
        if not candidate_path.exists():
            raise FileNotFoundError("candidate missing for decision")
        candidate = load_candidate(candidate_path)
        if candidate.get("candidate_hash") != decision.get("candidate_hash"):
            raise ValueError("candidate_hash does not match decision")
        if candidate.get("effect_hash") != decision.get("candidate_effect_hash"):
            raise ValueError("effect_hash does not match decision")
        candidate_type = candidate.get("candidate_type")
        if candidate_type in CONTROL_REPORT_CANDIDATE_TYPES:
            if control_store is None:
                raise ValueError("control_store required for profile/reflection decisions")
            _reverify_control_report(
                decision=decision,
                candidate=candidate,
                control_store=Path(control_store),
            )
        elif "control_report_ref" in decision or "control_report_hash" in decision:
            raise ValueError("control report fields are only valid for profile candidates")
        if candidate.get("candidate_type") == "case_memory":
            content = candidate.get("proposed_effect") or {}
            if official_diseases is None or valid_examinations is None:
                raise ValueError("case memory catalogs required")
            if evaluation_store is None:
                raise ValueError("evaluation_store required for case memory")
            unknown_diagnoses = [
                name for name in content.get("diagnoses", []) if name not in official_diseases
            ]
            if unknown_diagnoses:
                raise ValueError("case memory diagnosis not in official catalog")
            unknown_examinations = [
                name for name in content.get("examinations", []) if name not in valid_examinations
            ]
            if unknown_examinations:
                raise ValueError("case memory examination not in valid catalog")
            evidence = candidate.get("evidence") or {}
            artifact = _load_evaluation_artifact(
                evaluation_store=evaluation_store,
                evidence=evidence,
                provenance=content.get("provenance") or {},
            )
            extracted = extract_case_memory(
                patient_id=content.get("patient_id"),
                evaluation=artifact,
                official_diseases=official_diseases,
                valid_examinations=valid_examinations,
            )
            if extracted != content:
                raise ValueError("case memory does not match evaluation artifact")
            patient_id = content.get("patient_id") if isinstance(content, Mapping) else None
            if patient_id in case_patient_ids:
                raise ValueError("duplicate case memory patient_id")
            case_patient_ids.add(patient_id)
        # approval_ref non-empty alone is insufficient — decision file+hash required (checked above).
        assets.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": candidate.get("candidate_type"),
                "content": candidate.get("proposed_effect"),
                "approval_ref": str(decision_path),
                "decision_hash": decision.get("decision_hash"),
                "effect_hash": candidate.get("effect_hash"),
            }
        )
    registry = {
        "schema_version": "verified-registry/v1",
        "assets": assets,
    }
    registry["registry_hash"] = content_hash(registry)
    write_immutable_json(output_path, registry)
    return registry
