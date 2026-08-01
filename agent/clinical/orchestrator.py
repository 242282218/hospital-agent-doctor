from __future__ import annotations

from typing import Optional, Tuple, Union
from uuid import uuid4

from agent.clinical.model import (
    ClinicalBlackboard,
    QuestionDecision,
    RejectedProposal,
    RuntimeEvent,
    SkillProposal,
    ValidatedControlDelta,
    ValidatedDelta,
)
from agent.clinical.transitions import TransitionGuard
from agent.runtime.action_gateway import ActionCommand, build_action_command
from agent.validators.delta_validator import DeltaValidator


class ClinicalOrchestrator:
    """Single clinical controller: only ValidatedDelta / ValidatedControlDelta mutate board."""

    def __init__(
        self,
        *,
        case_run_id: str = "",
        initial: Optional[ClinicalBlackboard] = None,
        validator: Optional[DeltaValidator] = None,
        guard: Optional[TransitionGuard] = None,
    ) -> None:
        self._case_run_id = case_run_id or uuid4().hex
        self._snapshot = initial or ClinicalBlackboard()
        self._validator = validator or DeltaValidator()
        self._guard = guard or TransitionGuard()

    @property
    def snapshot(self) -> ClinicalBlackboard:
        return self._snapshot

    @property
    def case_run_id(self) -> str:
        return self._case_run_id

    def apply_proposal(
        self, proposal: SkillProposal
    ) -> Union[ClinicalBlackboard, RejectedProposal]:
        result = self._validator.validate(proposal, self._snapshot)
        if isinstance(result, RejectedProposal):
            return result
        self._snapshot = self._snapshot.apply_validated_delta(result)
        return self._snapshot

    def apply_runtime_event(self, event: RuntimeEvent) -> ClinicalBlackboard:
        validated = self._guard.validate(event, self._snapshot)
        self._snapshot = self._snapshot.apply_validated_control_delta(validated)
        return self._snapshot

    def apply_validated_delta(self, delta: ValidatedDelta) -> ClinicalBlackboard:
        self._snapshot = self._snapshot.apply_validated_delta(delta)
        return self._snapshot

    def apply_validated_control_delta(
        self, delta: ValidatedControlDelta
    ) -> ClinicalBlackboard:
        self._snapshot = self._snapshot.apply_validated_control_delta(delta)
        return self._snapshot

    def build_ask_command(
        self, decision: QuestionDecision, *, action_sequence: int
    ) -> ActionCommand:
        if decision.input_revision != self._snapshot.revision:
            raise ValueError("stale question decision")
        open_ids = {
            gap.gap_id for gap in self._snapshot.information_gaps if gap.status == "open"
        }
        if decision.gap_id not in open_ids:
            raise ValueError("question gap is not open: %s" % decision.gap_id)
        return build_action_command(
            case_run_id=self._case_run_id,
            blackboard_revision=self._snapshot.revision,
            action_sequence=action_sequence,
            action_type="ask_patient",
            payload={"question": decision.question},
        )

    def build_exam_command(
        self, *, items: Tuple[str, ...], reason: str, action_sequence: int
    ) -> ActionCommand:
        return build_action_command(
            case_run_id=self._case_run_id,
            blackboard_revision=self._snapshot.revision,
            action_sequence=action_sequence,
            action_type="order_examination",
            payload={"items": list(items), "reason": reason},
        )
