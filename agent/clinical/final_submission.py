"""Clinical final submission primitives (A1 dormant layer).

This module introduces a case-scoped, in-process, single-use capability
registry (``FinalAuthorizationRegistry``) and the only coordinator allowed to
register verified authorization tickets (``FinalSubmissionCoordinator``).

A1 scope:
  * The registry + coordinator + dedicated prescribe command factory exist as
    dormant APIs exercised by new tests only.
  * Existing legacy prescribe call sites and the public ``build_action_command``
    path are intentionally unchanged; the real production switch happens in A2.

Security invariant enforced here:
  * A prescribe command can only be created by ``build_prescribe_command`` after
    atomically consuming a verified ticket from the SAME case registry.
  * The registry consumes under a single ``threading.Lock`` critical section:
    lookup -> full-field match -> delete. Mismatched tickets are also deleted so
    they cannot be probed repeatedly.
  * Only ``FinalSubmissionCoordinator`` may call ``registry._register_verified``;
    only the dedicated prescribe factory may call ``registry.consume``. AST
    architecture tests enforce that boundary.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple


# --- Static payload shapes -------------------------------------------------

PRESCRIBE_PAYLOAD_KEYS = {"diagnosis", "treatment_plan", "reasoning"}


@dataclass(frozen=True)
class FinalPayload:
    """Verified final payload shape for a clinical submission."""

    diagnoses: Tuple[str, ...]
    treatment_plan: str
    reasoning: str


@dataclass(frozen=True)
class FinalSubmissionTicket:
    """Single-use capability returned by ``FinalSubmissionCoordinator.authorize``.

    `authorization_id` is a random, unforgeable token that the prescribe
    factory must consume (atomically) to build a command. It is NOT a boolean
    ``passed`` flag and cannot be reconstructed from prior state.
    """

    authorization_id: str
    payload_hash: str


@dataclass(frozen=True)
class PreparedFinalSubmission:
    """Verified payload bound to a fresh ticket, ready for command construction."""

    payload: FinalPayload
    ticket: FinalSubmissionTicket


@dataclass(frozen=True)
class VerifiedFinalResult:
    """Immutable result of ``FinalSubmissionCoordinator.verify``.

    Hashes bind the SAME payload object to its verifier / gate outputs; the
    coordinator stores them on the authorization record and the prescribe
    factory re-checks the payload hash before consuming the ticket.
    """

    payload: FinalPayload
    verification_id: str
    payload_hash: str
    legacy_verifier_hash: str
    five_dimension_gate_hash: str
    issue_codes: Tuple[str, ...] = ()
    patch_count: int = 0


@dataclass(frozen=True)
class _AuthorizationRecord:
    """Per-case, in-memory capability record kept inside the registry."""

    case_run_id: str
    input_revision: int
    snapshot_hash: str
    payload_hash: str
    legacy_verifier_hash: str
    five_dimension_gate_hash: str
    release_identity_hash: str
    issue_codes: Tuple[str, ...]


@dataclass(frozen=True)
class _VerificationRecord:
    """Trusted output of one coordinator verification run."""

    coordinator_id: str
    release_identity_hash: str
    payload: FinalPayload
    payload_hash: str
    legacy_verifier_hash: str
    five_dimension_gate_hash: str
    issue_codes: Tuple[str, ...]
    patch_count: int


# --- Canonical hashing helpers --------------------------------------------

def canonical_final_payload_hash(payload: FinalPayload) -> str:
    """Stable hash over the FinalPayload tuple for cross-stage identity checks."""
    projection = {
        "diagnoses": tuple(str(item) for item in payload.diagnoses),
        "treatment_plan": str(payload.treatment_plan),
        "reasoning": str(payload.reasoning),
    }
    return hashlib.sha256(_canonical_json(projection).encode("utf-8")).hexdigest()


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    """Stable hash over a verifier / gate report mapping."""
    return hashlib.sha256(_canonical_json(dict(report)).encode("utf-8")).hexdigest()


def report_passed(report: Mapping[str, Any]) -> bool:
    """A report passes only when it carries an explicit boolean ``True``.

    Missing field, string ``"true"``, falsy-but-not-True booleans, and empty
    reports all return False so that absence cannot be interpreted as a pass.
    """
    value = report.get("passed") if isinstance(report, Mapping) else None
    return value is True  # explicit True only; bool(True) is True, "true" str fails


def report_issue_codes(report: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return stable, deduplicated issue codes from a verifier/gate report."""
    raw = report.get("issue_codes") if isinstance(report, Mapping) else None
    if not raw:
        return ()
    seen: List[str] = []
    for item in raw:
        code = str(item).strip()
        if code and code not in seen:
            seen.append(code)
    return tuple(seen)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- Registry + coordinator ------------------------------------------------

class FinalAuthorizationRegistry:
    """Case-scoped, in-process, single-use capability registry.

    Holds verified authorization records keyed by a random authorization ID.
    The coordinator registers a record under a lock; the dedicated prescribe
    factory consumes a ticket under the SAME lock, atomically matching every
    expected field and then deleting the record.

    Mismatched tickets are also removed so they cannot be probed repeatedly.
    Cross-case, duplicate, or field-mismatch consumption all raise ValueError.
    """

    def __init__(self, *, release_identity_hash: str) -> None:
        if not release_identity_hash:
            raise ValueError("release_identity_hash required for registry")
        self._release_identity_hash = str(release_identity_hash)
        self._records: dict[str, _AuthorizationRecord] = {}
        self._verified_records: dict[str, _VerificationRecord] = {}
        self._lock = threading.Lock()

    @property
    def release_identity_hash(self) -> str:
        return self._release_identity_hash

    def _register_verified(self, record: _AuthorizationRecord) -> FinalSubmissionTicket:
        """Register a verified payload and return its single-use ticket.

        Only ``FinalSubmissionCoordinator`` is permitted to call this. We do not
        expose a public ``register``; the underscore-prefixed name plus AST
        checks keep the trust boundary local to this module.
        """
        if record.release_identity_hash != self._release_identity_hash:
            raise ValueError("release identity mismatch during registration")
        if not record.case_run_id:
            raise ValueError("case_run_id required")
        if record.input_revision < 0:
            raise ValueError("input_revision must be non-negative")
        authorization_id = secrets.token_urlsafe(32)
        with self._lock:
            if authorization_id in self._records:
                # Astronomically unlikely; refuse to overwrite silently.
                raise ValueError("authorization id collision; retry registration")
            self._records[authorization_id] = record
        return FinalSubmissionTicket(
            authorization_id=authorization_id,
            payload_hash=record.payload_hash,
        )

    def _register_verification(self, record: _VerificationRecord) -> str:
        """Persist a successful verification as a single-use capability."""
        verification_id = secrets.token_urlsafe(32)
        with self._lock:
            if verification_id in self._verified_records:
                raise ValueError("verification id collision; retry verification")
            self._verified_records[verification_id] = record
        return verification_id

    def _consume_verification(
        self,
        *,
        verification_id: str,
        coordinator_id: str,
        release_identity_hash: str,
        verified: VerifiedFinalResult,
    ) -> None:
        """Atomically consume a verified result after matching every field."""
        with self._lock:
            record = self._verified_records.pop(verification_id, None)
            if record is None:
                raise ValueError("verification not found or already consumed")
            expected = _VerificationRecord(
                coordinator_id=coordinator_id,
                release_identity_hash=str(release_identity_hash),
                payload=verified.payload,
                payload_hash=str(verified.payload_hash),
                legacy_verifier_hash=str(verified.legacy_verifier_hash),
                five_dimension_gate_hash=str(verified.five_dimension_gate_hash),
                issue_codes=tuple(str(item) for item in verified.issue_codes),
                patch_count=int(verified.patch_count),
            )
            if record != expected:
                raise ValueError("verification field mismatch")

    def consume(
        self,
        *,
        authorization_id: str,
        case_run_id: str,
        input_revision: int,
        snapshot_hash: str,
        payload_hash: str,
        legacy_verifier_hash: str,
        five_dimension_gate_hash: str,
        issue_codes: Sequence[str] = (),
    ) -> _AuthorizationRecord:
        """Atomically consume a verified ticket.

        Lookup, full-field match and delete happen under the same lock. On any
        mismatch the record is still deleted (no repeat probing). On miss we
        raise ValueError; the caller (the dedicated prescribe factory) bubbles
        it up so no command is built.
        """
        with self._lock:
            record = self._records.pop(authorization_id, None)
            if record is None:
                raise ValueError("authorization ticket not found or already consumed")
            expected = _AuthorizationRecord(
                case_run_id=str(case_run_id),
                input_revision=int(input_revision),
                snapshot_hash=str(snapshot_hash),
                payload_hash=str(payload_hash),
                legacy_verifier_hash=str(legacy_verifier_hash),
                five_dimension_gate_hash=str(five_dimension_gate_hash),
                release_identity_hash=self._release_identity_hash,
                issue_codes=tuple(str(item) for item in issue_codes),
            )
            if record != expected:
                # Mismatch means tampering or stale ticket; ticket is already
                # removed above, so it cannot be retried.
                raise ValueError("authorization ticket field mismatch")
            return record

    def outstanding_count(self) -> int:
        with self._lock:
            return len(self._records)


@dataclass(frozen=True)
class LoadedRuntimeIdentity:
    """Read-only runtime identity produced by the release loader.

    A1 only needs the hash so the registry can bind tickets to a specific
    release identity. The loader expands the meaning of ``status`` in A2/A6:
      * ``legacy_unverified`` — assets not closed, history-only, no tickets.
      * ``asset_verified``   — A2-A5 offline compatibility.
      * ``strict_verified``  — adds runtime code hash, used from A6 onward.
    """

    status: str
    identity_hash: str


class FinalVerificationError(RuntimeError):
    """Raised when verification cannot produce a sound VerifiedFinalResult."""


class FinalSubmissionCoordinator:
    """Owns the verify/authorize two-phase submission for a case.

    A1 scope: only the ``authorize`` path is exercised in tests so the registry
    can mint verified tickets. The full verify pipeline (apply_safety ->
    legacy verifier -> optional one-shot revise -> converge -> gate -> re-run
    verifier/gate) lands in A2, so the constructor already binds the same
    trusted callables and identity the A2 pipeline will use.

    The coordinator never trusts a ``passed=True`` dataclass field; only its own
    verifier/gate/hashes + the registry's random authorization ID count.
    """

    def __init__(
        self,
        *,
        registry: FinalAuthorizationRegistry,
        runtime_identity: LoadedRuntimeIdentity,
        apply_safety: Callable[[FinalPayload, Mapping[str, Any]], FinalPayload],
        run_legacy_verifier: Callable[[FinalPayload], Mapping[str, Any]],
        run_five_dimension_gate: Callable[
            [FinalPayload], Tuple[FinalPayload, Mapping[str, Any]]
        ],
        converge: Callable[[FinalPayload], FinalPayload],
        revise_once: Optional[
            Callable[[FinalPayload, Sequence[str]], FinalPayload]
        ] = None,
    ) -> None:
        if runtime_identity.status != "strict_verified":
            raise ValueError("strict_verified runtime identity required")
        if not getattr(runtime_identity, "identity_hash", ""):
            raise ValueError("runtime identity hash required")
        self._registry = registry
        self._identity = runtime_identity
        self._apply_safety = apply_safety
        self._run_legacy_verifier = run_legacy_verifier
        self._run_gate = run_five_dimension_gate
        self._converge = converge
        self._revise_once = revise_once
        self._coordinator_id = secrets.token_urlsafe(32)

    @property
    def registry(self) -> FinalAuthorizationRegistry:
        return self._registry

    @property
    def runtime_identity(self) -> LoadedRuntimeIdentity:
        return self._identity

    def verify(
        self,
        *,
        payload: FinalPayload,
        clinical_context: Mapping[str, Any],
    ) -> VerifiedFinalResult:
        """Run the A2 fixed submission pipeline and bind hashes to the SAME output.

        Fixed tunnel (do not reorder — see plan §A2):
          apply_safety -> legacy_verifier -> (one-shot revise when issues)
          -> converge -> five-dim gate -> legacy_verifier -> five-dim gate
          -> hashes bound only when both final reports explicitly report
          ``passed=True`` and the second gate does not mutate the payload.

        The coordinator treats ``passed`` as truth only when the report carries
        an explicit boolean True via ``report_passed``; missing field, string
        "true", or empty report fail-closed.
        """
        current = self._apply_safety(payload, clinical_context)
        patch_count = int(current != payload)
        initial = self._run_legacy_verifier(current)
        initial_issues = report_issue_codes(initial)
        if initial_issues and self._revise_once is not None:
            revised = self._revise_once(current, initial_issues)
            patch_count += int(revised != current)
            current = revised
        converged = self._converge(current)
        patch_count += int(converged != current)
        current = converged
        gated, _initial_gate_report = self._run_gate(current)
        patch_count += int(gated != current)
        current = gated
        final_report = self._run_legacy_verifier(current)
        final_current, final_gate_report = self._run_gate(current)
        # The five-dimension gate must be idempotent on the final text.
        if final_current != current:
            raise FinalVerificationError("five-dimension gate did not converge")
        if not report_passed(final_report) or not report_passed(final_gate_report):
            raise FinalVerificationError("final payload failed verification")
        verification_record = _VerificationRecord(
            coordinator_id=self._coordinator_id,
            release_identity_hash=self._identity.identity_hash,
            payload=current,
            payload_hash=canonical_final_payload_hash(current),
            legacy_verifier_hash=canonical_report_hash(final_report),
            five_dimension_gate_hash=canonical_report_hash(final_gate_report),
            issue_codes=(),
            patch_count=patch_count,
        )
        verification_id = self._registry._register_verification(verification_record)
        return VerifiedFinalResult(
            payload=current,
            verification_id=verification_id,
            payload_hash=verification_record.payload_hash,
            legacy_verifier_hash=verification_record.legacy_verifier_hash,
            five_dimension_gate_hash=verification_record.five_dimension_gate_hash,
            issue_codes=verification_record.issue_codes,
            patch_count=verification_record.patch_count,
        )

    def authorize(
        self,
        *,
        verified: VerifiedFinalResult,
        case_run_id: str,
        revision: int,
        snapshot_hash: str,
    ) -> PreparedFinalSubmission:
        """Bind a verified result to a fresh single-use authorization ticket."""
        if not snapshot_hash:
            raise ValueError("snapshot hash required")
        self._registry._consume_verification(
            verification_id=verified.verification_id,
            coordinator_id=self._coordinator_id,
            release_identity_hash=self._identity.identity_hash,
            verified=verified,
        )
        record = _AuthorizationRecord(
            case_run_id=str(case_run_id),
            input_revision=int(revision),
            snapshot_hash=str(snapshot_hash),
            payload_hash=str(verified.payload_hash),
            legacy_verifier_hash=str(verified.legacy_verifier_hash),
            five_dimension_gate_hash=str(verified.five_dimension_gate_hash),
            release_identity_hash=self._identity.identity_hash,
            issue_codes=tuple(str(item) for item in verified.issue_codes),
        )
        ticket = self._registry._register_verified(record)
        return PreparedFinalSubmission(payload=verified.payload, ticket=ticket)


# --- Dedicated prescribe command factory (dormant) ------------------------

def build_prescribe_command(
    *,
    registry: FinalAuthorizationRegistry,
    ticket: FinalSubmissionTicket,
    case_run_id: str,
    blackboard_revision: int,
    action_sequence: int,
    payload: Mapping[str, Any],
    snapshot_hash: str,
    legacy_verifier_hash: str,
    five_dimension_gate_hash: str,
    issue_codes: Sequence[str] = (),
):
    """Build a prescribe_treatment command by consuming the verified ticket.

    Re-computes the actual payload hash from the supplied payload mapping,
    then atomically consumes the ticket from ``registry``. Only after the
    ticket is consumed is the validated ``ActionCommand`` constructed via the
    existing private ``_build_validated_command`` in action_gateway.

    Import is local so this module stays import-safe when action_gateway is
    not yet imported (keeps the dormant layer light to import in tests).
    """
    if set(payload) != PRESCRIBE_PAYLOAD_KEYS:
        raise ValueError("prescribe payload keys must be %s" % PRESCRIBE_PAYLOAD_KEYS)
    diagnoses = tuple(str(item).strip() for item in payload["diagnosis"] if str(item).strip())
    if not diagnoses:
        raise ValueError("prescribe_treatment diagnosis must not be empty")
    if not str(payload["treatment_plan"]).strip():
        raise ValueError("prescribe_treatment treatment_plan must not be empty")
    final_payload = FinalPayload(
        diagnoses=diagnoses,
        treatment_plan=str(payload["treatment_plan"]),
        reasoning=str(payload["reasoning"]),
    )
    actual_payload_hash = canonical_final_payload_hash(final_payload)
    # The registry consume is the trust boundary: if the ticket is forged the
    # record is gone ("not found"); if any binding field (payload/case/revision/
    # snapshot/verifier/gate/release) does not match the registered record the
    # ticket is also removed so it cannot be probed repeatedly. We deliberately
    # do NOT short-circuit on a local payload-hash comparison before consume:
    # consume is the only actor allowed to side-effect the registry.
    registry.consume(
        authorization_id=ticket.authorization_id,
        case_run_id=case_run_id,
        input_revision=int(blackboard_revision),
        snapshot_hash=str(snapshot_hash),
        payload_hash=actual_payload_hash,
        legacy_verifier_hash=str(legacy_verifier_hash),
        five_dimension_gate_hash=str(five_dimension_gate_hash),
        issue_codes=issue_codes,
    )
    # Build the validated command with the normalized payload ordering.
    from agent.runtime.action_gateway import (
        _PRESCRIBE_CAPABILITY,
        _build_validated_command,
    )

    return _build_validated_command(
        case_run_id=case_run_id,
        blackboard_revision=blackboard_revision,
        action_sequence=action_sequence,
        action_type="prescribe_treatment",
        payload={
            "diagnosis": list(diagnoses),
            "treatment_plan": final_payload.treatment_plan,
            "reasoning": final_payload.reasoning,
        },
        reason_evidence_ids=(),
        authorization_capability=_PRESCRIBE_CAPABILITY,
    )


__all__ = [
    "FinalPayload",
    "FinalSubmissionTicket",
    "PreparedFinalSubmission",
    "VerifiedFinalResult",
    "FinalAuthorizationRegistry",
    "FinalSubmissionCoordinator",
    "FinalVerificationError",
    "LoadedRuntimeIdentity",
    "build_prescribe_command",
    "canonical_final_payload_hash",
    "canonical_report_hash",
    "report_passed",
    "report_issue_codes",
]
