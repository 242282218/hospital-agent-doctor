from __future__ import annotations

from agent.clinical.model import (
    ClinicalBlackboard,
    ControlOperation,
    EvidenceItem,
    RuntimeEvent,
    SkillOperation,
    SkillProposal,
    ValidatedControlDelta,
    ValidatedDelta,
)
from agent.clinical.orchestrator import ClinicalOrchestrator
from agent.clinical.transitions import TransitionGuard, transition_event
from agent.validators.delta_validator import DeltaValidator


def test_validated_delta_returns_new_revision_without_mutating_input() -> None:
    board = ClinicalBlackboard()
    evidence = EvidenceItem(
        evidence_id="ev-1",
        concept="chief_complaint",
        value="发热三天",
        kind="patient_statement",
        source_ref="message://patient/1",
        status="validated",
    )
    delta = ValidatedDelta(
        proposal_id="proposal-1",
        input_revision=0,
        operations=(SkillOperation("add_evidence", {"item": evidence}),),
        validator_decisions=("schema:accepted",),
        content_hash="a" * 64,
    )

    updated = board.apply_validated_delta(delta)

    assert board.revision == 0
    assert board.evidence_ledger == ()
    assert updated.revision == 1
    assert updated.evidence_ledger[0].evidence_id == "ev-1"


def test_stale_delta_is_rejected() -> None:
    board = ClinicalBlackboard(revision=2)
    delta = ValidatedDelta(
        proposal_id="p",
        input_revision=0,
        operations=(),
        validator_decisions=(),
        content_hash="b" * 64,
    )
    try:
        board.apply_validated_delta(delta)
        assert False, "expected stale failure"
    except ValueError as exc:
        assert "stale" in str(exc)


def test_orchestrator_applies_proposal_via_validator_only() -> None:
    orch = ClinicalOrchestrator()
    proposal = SkillProposal(
        proposal_id="p1",
        skill_name="IntakeExtractor",
        input_revision=0,
        purpose="facts",
        operations=(
            SkillOperation(
                "add_evidence",
                {
                    "item": EvidenceItem(
                        evidence_id="ev-1",
                        concept="chief_complaint",
                        value="咳嗽",
                        kind="patient_statement",
                        source_ref="message://1",
                    )
                },
            ),
        ),
    )
    result = orch.apply_proposal(proposal)
    assert isinstance(result, ClinicalBlackboard)
    assert result.revision == 1
    assert len(result.evidence_ledger) == 1


def test_transition_guard_allows_legal_edges_only() -> None:
    orch = ClinicalOrchestrator()
    orch.apply_runtime_event(transition_event(event_id="t1", snapshot=orch.snapshot, target_state="HYPOTHESIS"))
    assert orch.snapshot.workflow_state.execution_state == "HYPOTHESIS"
    try:
        orch.apply_runtime_event(
            transition_event(event_id="t2", snapshot=orch.snapshot, target_state="VERIFY")
        )
        assert False, "illegal transition should fail"
    except ValueError:
        pass


def test_control_delta_records_llm_reservation() -> None:
    board = ClinicalBlackboard()
    delta = ValidatedControlDelta(
        event_id="e1",
        input_revision=0,
        operations=(ControlOperation("record_llm_call_started", {"skill_name": "HypothesisBuilder"}),),
        guard_decisions=("ok",),
        content_hash="c" * 64,
    )
    updated = board.apply_validated_control_delta(delta)
    assert updated.budget_state.llm_calls_used == 1
    assert updated.budget_state.calls_for("HypothesisBuilder") == 1
