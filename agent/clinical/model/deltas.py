from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class SkillOperation:
    operation: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ControlOperation:
    operation: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedDelta:
    proposal_id: str
    input_revision: int
    operations: Tuple[SkillOperation, ...]
    validator_decisions: Tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class ValidatedControlDelta:
    event_id: str
    input_revision: int
    operations: Tuple[ControlOperation, ...]
    guard_decisions: Tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class SkillProposal:
    proposal_id: str
    skill_name: str
    input_revision: int
    purpose: str
    evidence_refs: Tuple[str, ...] = ()
    operations: Tuple[SkillOperation, ...] = ()
    confidence: str = "medium"
    estimated_cost: int = 0


@dataclass(frozen=True)
class RejectedProposal:
    proposal_id: str
    input_revision: int
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    event_type: str
    input_revision: int
    control_operations: Tuple[ControlOperation, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionDecision:
    gap_id: str
    question: str
    expected_information_gain: str
    input_revision: int
