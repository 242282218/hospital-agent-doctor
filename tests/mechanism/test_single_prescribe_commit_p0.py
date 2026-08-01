"""A2 tests for single-use verified-final authorization and prescribe commands.

The verification receipt and authorization ticket are both one-shot
capabilities. Every bound final-payload, case, revision, snapshot, and release
identity field must match; forgery, drift, cross-coordinator use, and replay
must fail closed.
"""

from __future__ import annotations

from dataclasses import replace
import threading
from typing import Any, Mapping

import pytest

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    FinalSubmissionCoordinator,
    FinalSubmissionTicket,
    LoadedRuntimeIdentity,
    VerifiedFinalResult,
    build_prescribe_command,
    canonical_final_payload_hash,
    canonical_report_hash,
)


RELEASE_ID = "release-identity-sha256-fixed-for-tests"
VERIFIER_HASH = canonical_report_hash({"passed": True})
GATE_HASH = canonical_report_hash({"passed": True})


def _make_registry() -> FinalAuthorizationRegistry:
    return FinalAuthorizationRegistry(release_identity_hash=RELEASE_ID)


def _make_coordinator(
    registry: FinalAuthorizationRegistry,
) -> FinalSubmissionCoordinator:
    # A1 coordinator construction binds the trusted callables; verify() is not
    # called here. Tests mint VerifiedFinalResult directly to drive authorize().
    identity = LoadedRuntimeIdentity(status="strict_verified", identity_hash=RELEASE_ID)
    return FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=identity,
        apply_safety=lambda payload, ctx: payload,
        run_legacy_verifier=lambda payload: {"passed": True},
        run_five_dimension_gate=lambda payload: (payload, {"passed": True}),
        converge=lambda payload: payload,
    )


def _verified(
    coordinator: FinalSubmissionCoordinator, payload: FinalPayload
) -> VerifiedFinalResult:
    return coordinator.verify(payload=payload, clinical_context={})


def _authorize(
    coordinator: FinalSubmissionCoordinator,
    payload: FinalPayload,
    *,
    case_run_id: str,
    revision: int,
    snapshot_hash: str,
) -> FinalSubmissionTicket:
    verified = _verified(coordinator, payload)
    prepared = coordinator.authorize(
        verified=verified,
        case_run_id=case_run_id,
        revision=revision,
        snapshot_hash=snapshot_hash,
    )
    return prepared.ticket


def test_verify_registers_random_verification_id() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    first = coordinator.verify(payload=_base_payload(), clinical_context={})
    second = coordinator.verify(payload=_base_payload(), clinical_context={})

    assert first.verification_id
    assert first.verification_id != second.verification_id


def test_authorize_rejects_forged_verified_result() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    forged = VerifiedFinalResult(
        payload=_base_payload(),
        verification_id="forged-verification-id",
        payload_hash=canonical_final_payload_hash(_base_payload()),
        legacy_verifier_hash=VERIFIER_HASH,
        five_dimension_gate_hash=GATE_HASH,
    )

    with pytest.raises(ValueError, match="verification not found or already consumed"):
        coordinator.authorize(
            verified=forged,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )


def test_authorize_consumes_verification_and_rejects_tampering() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    verified = coordinator.verify(payload=_base_payload(), clinical_context={})
    tampered = replace(verified, payload_hash="tampered")

    with pytest.raises(ValueError, match="verification field mismatch"):
        coordinator.authorize(
            verified=tampered,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )
    with pytest.raises(ValueError, match="verification not found or already consumed"):
        coordinator.authorize(
            verified=verified,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )


def test_authorize_rejects_cross_coordinator_verified_result() -> None:
    registry = _make_registry()
    source = _make_coordinator(registry)
    other = _make_coordinator(registry)
    verified = source.verify(payload=_base_payload(), clinical_context={})

    with pytest.raises(ValueError, match="verification field mismatch"):
        other.authorize(
            verified=verified,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )
    with pytest.raises(ValueError, match="verification not found or already consumed"):
        source.authorize(
            verified=verified,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )


def test_authorize_rejects_verification_replay() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    verified = coordinator.verify(payload=_base_payload(), clinical_context={})
    coordinator.authorize(
        verified=verified,
        case_run_id="run-1",
        revision=7,
        snapshot_hash="snapshot-fixed",
    )

    with pytest.raises(ValueError, match="verification not found or already consumed"):
        coordinator.authorize(
            verified=verified,
            case_run_id="run-1",
            revision=7,
            snapshot_hash="snapshot-fixed",
        )


def _base_payload() -> FinalPayload:
    return FinalPayload(
        diagnoses=("肺炎",),
        treatment_plan="抗感染并监测。",
        reasoning="结合病史和检查。",
    )


def _base_kw(ticket: FinalSubmissionTicket) -> Mapping[str, Any]:
    return {
        "registry_ticket": ticket,
        "case_run_id": "run-1",
        "blackboard_revision": 7,
        "action_sequence": 3,
        "snapshot_hash": "snapshot-fixed",
        "legacy_verifier_hash": VERIFIER_HASH,
        "five_dimension_gate_hash": GATE_HASH,
    }


def test_ticket_consumed_once_and_field_bound() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    payload = _base_payload()
    ticket = _authorize(
        coordinator, payload, case_run_id="run-1", revision=7, snapshot_hash="snapshot-fixed"
    )
    cmd = build_prescribe_command(
        registry=registry,
        ticket=ticket,
        case_run_id="run-1",
        blackboard_revision=7,
        action_sequence=3,
        payload={
            "diagnosis": list(payload.diagnoses),
            "treatment_plan": payload.treatment_plan,
            "reasoning": payload.reasoning,
        },
        snapshot_hash="snapshot-fixed",
        legacy_verifier_hash=VERIFIER_HASH,
        five_dimension_gate_hash=GATE_HASH,
    )
    assert cmd.action_type == "prescribe_treatment"
    assert cmd.payload["diagnosis"] == ("肺炎",)
    assert cmd.payload["treatment_plan"] == "抗感染并监测。"
    assert cmd.payload["reasoning"] == "结合病史和检查。"
    # Consumed: registry no longer holds this ticket.
    assert registry.outstanding_count() == 0
    # A second consumption attempt with the same ticket must fail.
    with pytest.raises(ValueError, match="not found or already consumed"):
        build_prescribe_command(
            registry=registry,
            ticket=ticket,
            case_run_id="run-1",
            blackboard_revision=7,
            action_sequence=3,
            payload={
                "diagnosis": list(payload.diagnoses),
                "treatment_plan": payload.treatment_plan,
                "reasoning": payload.reasoning,
            },
            snapshot_hash="snapshot-fixed",
            legacy_verifier_hash=VERIFIER_HASH,
            five_dimension_gate_hash=GATE_HASH,
        )


@pytest.mark.parametrize(
    "diff,reason",
    [
        ("diagnosis", "diagnosis"),
        ("treatment_plan", "treatment_plan"),
        ("reasoning", "reasoning"),
    ],
)
def test_payload_drift_burns_ticket_on_factory_side(diff: str, reason: str) -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    original = _base_payload()
    ticket = _authorize(
        coordinator, original, case_run_id="run-1", revision=7, snapshot_hash="snapshot-fixed"
    )
    drifted_payload = {
        "diagnosis": list(original.diagnoses),
        "treatment_plan": original.treatment_plan,
        "reasoning": original.reasoning,
    }
    drift_map = {
        "diagnosis": (["肺脓肿"], "肺脓肿"),
        "treatment_plan": ("修改治疗。", "修改治疗。"),
        "reasoning": ("改 reasoning。", "改 reasoning。"),
    }
    drift_value, _ = drift_map[diff]
    drifted_payload[diff] = drift_value
    with pytest.raises(ValueError, match="(?:field mismatch|payload hash)"):
        build_prescribe_command(
            registry=registry,
            ticket=ticket,
            case_run_id="run-1",
            blackboard_revision=7,
            action_sequence=3,
            payload=drifted_payload,
            snapshot_hash="snapshot-fixed",
            legacy_verifier_hash=VERIFIER_HASH,
            five_dimension_gate_hash=GATE_HASH,
        )
    # Ticket was burned by the consume mis-match: a second attempt to use any
    # payload still fails because the ticket record is gone from the registry.
    assert registry.outstanding_count() == 0


def test_case_run_id_drift_destroys_ticket() -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    payload = _base_payload()
    ticket = _authorize(
        coordinator, payload, case_run_id="run-1", revision=7, snapshot_hash="snapshot-fixed"
    )
    with pytest.raises(ValueError, match="field mismatch"):
        build_prescribe_command(
            registry=registry,
            ticket=ticket,
            case_run_id="run-OTHER",
            blackboard_revision=7,
            action_sequence=3,
            payload={
                "diagnosis": list(payload.diagnoses),
                "treatment_plan": payload.treatment_plan,
                "reasoning": payload.reasoning,
            },
            snapshot_hash="snapshot-fixed",
            legacy_verifier_hash=VERIFIER_HASH,
            five_dimension_gate_hash=GATE_HASH,
        )
    # Cross-case use burns the ticket.
    assert registry.outstanding_count() == 0


@pytest.mark.parametrize("delta", ["revision", "snapshot", "gate", "legacy", "issue"])
def test_revision_snapshot_or_verifier_hash_drift_burns_ticket(delta: str) -> None:
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    payload = _base_payload()
    ticket = _authorize(
        coordinator, payload, case_run_id="run-1", revision=7, snapshot_hash="snapshot-fixed"
    )
    kwargs: dict[str, Any] = dict(
        registry=registry,
        ticket=ticket,
        case_run_id="run-1",
        blackboard_revision=7,
        action_sequence=3,
        payload={
            "diagnosis": list(payload.diagnoses),
            "treatment_plan": payload.treatment_plan,
            "reasoning": payload.reasoning,
        },
        snapshot_hash="snapshot-fixed",
        legacy_verifier_hash=VERIFIER_HASH,
        five_dimension_gate_hash=GATE_HASH,
        issue_codes=(),
    )
    if delta == "revision":
        kwargs["blackboard_revision"] = 99
    elif delta == "snapshot":
        kwargs["snapshot_hash"] = "different-snapshot"
    elif delta == "gate":
        kwargs["five_dimension_gate_hash"] = "different-gate"
    elif delta == "legacy":
        kwargs["legacy_verifier_hash"] = "different-legacy"
    elif delta == "issue":
        kwargs["issue_codes"] = ("FUZZ",)
    with pytest.raises(ValueError, match="field mismatch"):
        build_prescribe_command(**kwargs)
    assert registry.outstanding_count() == 0


def test_forged_authorization_id_cannot_consume() -> None:
    registry = _make_registry()
    # No registration ever happened for this id.
    forged_ticket = FinalSubmissionTicket(
        authorization_id="not-an-actual-id",
        payload_hash="anything",
    )
    with pytest.raises(ValueError, match="not found or already consumed"):
        build_prescribe_command(
            registry=registry,
            ticket=forged_ticket,
            case_run_id="run-1",
            blackboard_revision=1,
            action_sequence=1,
            payload={
                "diagnosis": ["肺炎"],
                "treatment_plan": "x",
                "reasoning": "y",
            },
            snapshot_hash="snapshot",
            legacy_verifier_hash="l",
            five_dimension_gate_hash="g",
        )
    assert registry.outstanding_count() == 0


def test_cross_case_registry_cannot_authorize_other_case_ticket() -> None:
    a = _make_registry()
    b = _make_registry()
    coord_a = _make_coordinator(a)
    payload = _base_payload()
    ticket_a = _authorize(
        coord_a, payload, case_run_id="case-a", revision=1, snapshot_hash="s-a"
    )
    # Even though registry b has the same release identity, ticket from a is
    # unknown there.
    with pytest.raises(ValueError, match="not found or already consumed"):
        build_prescribe_command(
            registry=b,
            ticket=ticket_a,
            case_run_id="case-a",
            blackboard_revision=1,
            action_sequence=1,
            payload={
                "diagnosis": list(payload.diagnoses),
                "treatment_plan": payload.treatment_plan,
                "reasoning": payload.reasoning,
            },
            snapshot_hash="s-a",
            legacy_verifier_hash=VERIFIER_HASH,
            five_dimension_gate_hash=GATE_HASH,
        )
    # And the artifact ticket is still consumable from its own registry.
    cmd = build_prescribe_command(
        registry=a,
        ticket=ticket_a,
        case_run_id="case-a",
        blackboard_revision=1,
        action_sequence=1,
        payload={
            "diagnosis": list(payload.diagnoses),
            "treatment_plan": payload.treatment_plan,
            "reasoning": payload.reasoning,
        },
        snapshot_hash="s-a",
        legacy_verifier_hash=VERIFIER_HASH,
        five_dimension_gate_hash=GATE_HASH,
    )
    assert cmd.action_type == "prescribe_treatment"


def test_release_identity_drift_blocks_authorize() -> None:
    # A registry committed to one release identity cannot authorize a ticket
    # whose record claims a different identity; the coordinator refuses at
    # register time.
    registry = _make_registry()  # released identity RELEASE_ID
    identity_other = LoadedRuntimeIdentity(
        status="strict_verified", identity_hash="other-release-identity"
    )
    coord_other = FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=identity_other,
        apply_safety=lambda payload, ctx: payload,
        run_legacy_verifier=lambda payload: {"passed": True},
        run_five_dimension_gate=lambda payload: (payload, {"passed": True}),
        converge=lambda payload: payload,
    )
    with pytest.raises(ValueError, match="release identity mismatch"):
        coord_other.authorize(
            verified=_verified(coord_other, _base_payload()),
            case_run_id="run-1",
            revision=1,
            snapshot_hash="snapshot",
        )


def test_coordinator_requires_verified_identity() -> None:
    registry = _make_registry()
    for status in ("legacy_unverified", "asset_verified", "", "not_implemented"):
        with pytest.raises(ValueError, match="strict_verified runtime identity required"):
            FinalSubmissionCoordinator(
                registry=registry,
                runtime_identity=LoadedRuntimeIdentity(
                    status=status, identity_hash=RELEASE_ID
                ),
                apply_safety=lambda p, c: p,
                run_legacy_verifier=lambda p: {"passed": True},
                run_five_dimension_gate=lambda p: (p, {"passed": True}),
                converge=lambda p: p,
            )


def test_registry_consume_is_atomic_under_concurrency() -> None:
    """Two concurrent consume() callers: only one wins, registry empty after."""
    registry = _make_registry()
    coordinator = _make_coordinator(registry)
    payload = _base_payload()
    ticket = _authorize(
        coordinator, payload, case_run_id="run-1", revision=7, snapshot_hash="snapshot-fixed"
    )
    results: list[Any] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def try_consume() -> None:
        barrier.wait()
        try:
            cmd = build_prescribe_command(
                registry=registry,
                ticket=ticket,
                case_run_id="run-1",
                blackboard_revision=7,
                action_sequence=3,
                payload={
                    "diagnosis": list(payload.diagnoses),
                    "treatment_plan": payload.treatment_plan,
                    "reasoning": payload.reasoning,
                },
                snapshot_hash="snapshot-fixed",
                legacy_verifier_hash=VERIFIER_HASH,
                five_dimension_gate_hash=GATE_HASH,
            )
            results.append(cmd)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=try_consume)
    t2 = threading.Thread(target=try_consume)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 1, "only one concurrent consumer may build a command"
    assert len(errors) == 1, "the other must fail with a ValueError"
    assert isinstance(errors[0], ValueError)
    assert registry.outstanding_count() == 0
