from __future__ import annotations

import hashlib
import json
from typing import Dict, Set, Tuple

from agent.clinical.model import (
    ClinicalBlackboard,
    ControlOperation,
    RuntimeEvent,
    ValidatedControlDelta,
)

# Seven clinical execution states + FINISH
EXECUTION_STATES = (
    "INTAKE",
    "HYPOTHESIS",
    "EVIDENCE_ACQUIRE",
    "RESULT_INTERPRET",
    "DIAGNOSIS_SELECT",
    "FINAL_PLAN",
    "VERIFY",
    "FINISH",
)

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    "INTAKE": {"HYPOTHESIS", "EVIDENCE_ACQUIRE", "FINISH"},
    "HYPOTHESIS": {"EVIDENCE_ACQUIRE", "DIAGNOSIS_SELECT", "FINAL_PLAN"},
    "EVIDENCE_ACQUIRE": {"RESULT_INTERPRET", "HYPOTHESIS", "DIAGNOSIS_SELECT", "FINAL_PLAN"},
    "RESULT_INTERPRET": {"HYPOTHESIS", "EVIDENCE_ACQUIRE", "DIAGNOSIS_SELECT", "FINAL_PLAN"},
    "DIAGNOSIS_SELECT": {"FINAL_PLAN", "EVIDENCE_ACQUIRE"},
    "FINAL_PLAN": {"VERIFY"},
    "VERIFY": {"FINAL_PLAN", "FINISH"},
    "FINISH": set(),
}


class TransitionGuard:
    def validate(
        self, event: RuntimeEvent, snapshot: ClinicalBlackboard
    ) -> ValidatedControlDelta:
        if event.input_revision != snapshot.revision:
            raise ValueError("stale runtime event")
        decisions = []
        for operation in event.control_operations:
            if operation.operation == "set_execution_state":
                target = str(operation.payload.get("execution_state") or "")
                current = snapshot.workflow_state.execution_state
                if target not in EXECUTION_STATES:
                    raise ValueError("unknown execution state: %s" % target)
                allowed = ALLOWED_TRANSITIONS.get(current, set())
                if target != current and target not in allowed:
                    raise ValueError("illegal transition %s -> %s" % (current, target))
                decisions.append("transition:%s->%s" % (current, target))
            else:
                decisions.append("control:%s" % operation.operation)
        content = {
            "event_id": event.event_id,
            "input_revision": event.input_revision,
            "operations": [
                {"operation": op.operation, "payload": dict(op.payload)}
                for op in event.control_operations
            ],
        }
        digest = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return ValidatedControlDelta(
            event_id=event.event_id,
            input_revision=event.input_revision,
            operations=event.control_operations,
            guard_decisions=tuple(decisions),
            content_hash=digest,
        )


def transition_event(
    *,
    event_id: str,
    snapshot: ClinicalBlackboard,
    target_state: str,
    finish_reason: str = "",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        event_type="transition",
        input_revision=snapshot.revision,
        control_operations=(
            ControlOperation(
                "set_execution_state",
                {"execution_state": target_state, "finish_reason": finish_reason},
            ),
        ),
    )
