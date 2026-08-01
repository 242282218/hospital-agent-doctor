from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, Set, Tuple, Union

from agent.clinical.model import (
    ClinicalBlackboard,
    RejectedProposal,
    SkillOperation,
    SkillProposal,
    ValidatedDelta,
)

SKILL_PERMISSIONS: Mapping[str, Set[str]] = {
    "IntakeExtractor": {"add_evidence", "add_or_update_gap", "close_gap"},
    "QuestionPlanner": set(),
    "HypothesisBuilder": {"add_or_update_hypothesis", "add_or_update_gap"},
    "DifferentialCritic": {"add_or_update_hypothesis"},
    "ExamIntentPlanner": {"add_or_update_exam_intent", "add_or_update_gap"},
    "ResultInterpreter": {
        "add_evidence",
        "attach_exam_result",
        "close_gap",
        "add_or_update_hypothesis",
    },
    "DiagnosisSelector": {"select_hypothesis"},
    "TreatmentPlanner": {"update_treatment_draft"},
    "TreatmentRefiner": {"update_treatment_draft"},
    "TreatmentDraftSanitizer": {"update_treatment_draft"},
    "FinalVerifier": {"replace_verifier_issues"},
}


class DeltaValidator:
    def validate(
        self,
        proposal: SkillProposal,
        snapshot: ClinicalBlackboard,
    ) -> Union[ValidatedDelta, RejectedProposal]:
        issues = []
        if proposal.input_revision != snapshot.revision:
            issues.append("stale_revision")
        allowed = SKILL_PERMISSIONS.get(proposal.skill_name)
        if allowed is None:
            issues.append("unknown_skill")
        else:
            for operation in proposal.operations:
                if operation.operation not in allowed:
                    issues.append("unauthorized_operation:%s" % operation.operation)
                issues.extend(self._validate_operation(operation, snapshot))
        if issues:
            return RejectedProposal(
                proposal_id=proposal.proposal_id,
                input_revision=proposal.input_revision,
                issues=tuple(issues),
            )
        content = {
            "proposal_id": proposal.proposal_id,
            "input_revision": proposal.input_revision,
            "operations": [
                {"operation": op.operation, "payload": _jsonable(op.payload)}
                for op in proposal.operations
            ],
        }
        digest = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return ValidatedDelta(
            proposal_id=proposal.proposal_id,
            input_revision=proposal.input_revision,
            operations=proposal.operations,
            validator_decisions=("schema:accepted",),
            content_hash=digest,
        )

    def _validate_operation(
        self, operation: SkillOperation, snapshot: ClinicalBlackboard
    ) -> Sequence[str]:
        payload = dict(operation.payload)
        if operation.operation == "add_evidence":
            item = payload.get("item") or payload
            if isinstance(item, Mapping):
                if not str(item.get("source_ref") or "").strip() and not item.get(
                    "source_evidence_ids"
                ):
                    return ("evidence_missing_source",)
                if str(item.get("subject") or "patient") not in {
                    "patient",
                    "family",
                    "other",
                }:
                    return ("invalid_subject",)
        if operation.operation == "add_or_update_hypothesis":
            item = payload.get("item") or payload
            if isinstance(item, Mapping):
                if not str(item.get("official_disease_name") or "").strip():
                    return ("hypothesis_missing_name",)
                refs = item.get("supporting_evidence_ids") or ()
                if not refs:
                    return ("hypothesis_missing_evidence_refs",)
                known = {ev.evidence_id for ev in snapshot.evidence_ledger}
                if known and not set(str(x) for x in refs) & known:
                    # Allow when board empty (fresh propose before evidence applied in same batch).
                    if snapshot.evidence_ledger:
                        return ("hypothesis_unknown_evidence_refs",)
        return ()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
