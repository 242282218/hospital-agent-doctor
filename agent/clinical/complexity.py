from __future__ import annotations

from agent.clinical.model import ClinicalBlackboard


class ComplexityPolicy:
    SIMPLE_CAP = 5
    COMPLEX_CAP = 8

    def hard_cap(self, snapshot: ClinicalBlackboard) -> int:
        if snapshot.workflow_state.complexity == "complex":
            return self.COMPLEX_CAP
        return self.SIMPLE_CAP

    def should_upgrade(self, snapshot: ClinicalBlackboard) -> bool:
        if snapshot.workflow_state.complexity == "complex":
            return False
        open_must = [
            gap
            for gap in snapshot.information_gaps
            if gap.status == "open" and gap.priority in {"high", "must_fix"}
        ]
        multi_hyp = len([h for h in snapshot.hypothesis_set if h.status == "active"]) >= 3
        return bool(open_must) or multi_hyp

    def can_call_llm(self, snapshot: ClinicalBlackboard) -> bool:
        return snapshot.budget_state.llm_calls_used < self.hard_cap(snapshot)

    def should_run_critic(self, snapshot: ClinicalBlackboard) -> bool:
        return snapshot.workflow_state.complexity == "complex"
