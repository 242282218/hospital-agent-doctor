from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from agent.clinical.model import ClinicalBlackboard, SafeEscalationPlan

ClosureNext = Literal[
    "evidence_acquire",
    "ready_for_diagnosis_selection",
    "final_plan",
    "safe_escalation",
]


@dataclass(frozen=True)
class ClosureDecision:
    next_step: ClosureNext
    reason: str
    safe_escalation: Optional[SafeEscalationPlan] = None


class EvidenceClosurePolicy:
    def decide(self, snapshot: ClinicalBlackboard) -> ClosureDecision:
        open_must = [
            gap
            for gap in snapshot.information_gaps
            if gap.status == "open" and gap.priority in {"high", "must_fix"}
        ]
        mandatory_open = [
            exam
            for exam in snapshot.examination_state
            if exam.requirement == "mandatory"
            and exam.status not in {"resulted", "waived"}
        ]
        if open_must or mandatory_open:
            if snapshot.budget_state.llm_calls_used >= snapshot.budget_state.llm_hard_cap:
                return ClosureDecision(
                    "safe_escalation",
                    "budget_exhausted_with_open_must_fix",
                    safe_escalation=_build_safe_escalation(snapshot, open_must, mandatory_open),
                )
            return ClosureDecision("evidence_acquire", "open_must_fix_or_mandatory")
        return ClosureDecision("ready_for_diagnosis_selection", "evidence_ready")


class FinalClosurePolicy:
    def decide(self, snapshot: ClinicalBlackboard) -> ClosureDecision:
        selected = [h for h in snapshot.hypothesis_set if h.status == "selected"]
        if len(selected) == 1 and selected[0].supporting_evidence_ids:
            return ClosureDecision("final_plan", "selected_with_evidence")
        return ClosureDecision(
            "safe_escalation",
            "missing_selected_or_evidence",
            safe_escalation=_build_safe_escalation(snapshot, [], []),
        )


def _build_safe_escalation(
    snapshot: ClinicalBlackboard,
    open_must,
    mandatory_open,
) -> SafeEscalationPlan:
    candidates = sorted(
        [h for h in snapshot.hypothesis_set if h.supporting_evidence_ids],
        key=lambda h: (h.status == "selected", len(h.supporting_evidence_ids)),
        reverse=True,
    )
    if candidates:
        best = candidates[0]
        diagnosis = best.official_disease_name
        refs = best.supporting_evidence_ids
    elif snapshot.evidence_ledger:
        diagnosis = "未分化临床综合征"
        refs = (snapshot.evidence_ledger[0].evidence_id,)
    else:
        diagnosis = "未分化临床综合征"
        refs = ()
    unresolved = tuple(gap.gap_id for gap in open_must) + tuple(
        exam.exam_intent_id for exam in mandatory_open
    )
    if not diagnosis.strip():
        diagnosis = "未分化临床综合征"
    return SafeEscalationPlan(
        submission_diagnosis=diagnosis,
        supporting_evidence_ids=refs if refs else (("ev-placeholder",) if not refs else refs),
        unresolved_ids=unresolved,
        disposition="conservative_outpatient_or_urgent_referral",
        reason="safe_escalation_for_unresolved_items",
    )
