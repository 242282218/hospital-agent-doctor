from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from agent.clinical.model import ClinicalBlackboard


@dataclass(frozen=True)
class LegacyCaseView:
    """One-way Blackboard + runtime projection for unmigrated legacy helpers."""

    snapshot: ClinicalBlackboard
    runtime_state: Mapping[str, Any]

    @classmethod
    def from_sources(
        cls,
        snapshot: ClinicalBlackboard,
        runtime_state: Optional[Mapping[str, Any]] = None,
    ) -> "LegacyCaseView":
        return cls(snapshot=snapshot, runtime_state=dict(runtime_state or {}))

    def to_case_state(self) -> Dict[str, Any]:
        # Projection only: validated evidence wins over legacy intake fields.
        state = deepcopy(dict(self.runtime_state))
        state["chat_history"] = list(state.get("chat_history") or [])
        state["ordered_examinations"] = [
            exam.catalog_leaf_name
            for exam in self.snapshot.examination_state
            if exam.catalog_leaf_name
        ] or list(state.get("ordered_examinations") or [])
        state["examination_results"] = dict(state.get("examination_results") or {})
        state["coverage_gaps"] = [
            {
                "gap_id": gap.gap_id,
                "intent": gap.intent,
                "status": gap.status,
                "priority": gap.priority,
            }
            for gap in self.snapshot.information_gaps
        ]
        selected = [h for h in self.snapshot.hypothesis_set if h.status == "selected"]
        if selected:
            state["final_plan"] = {
                "diagnosis": selected[0].official_disease_name,
                "treatment_plan": self.snapshot.treatment_state.draft_text,
                "reasoning": "",
            }
        state["blackboard_revision"] = self.snapshot.revision
        return state


def legacy_intake_question(case_state: Mapping[str, Any]) -> str:
    """Deterministic replacement for select_required_intake_question."""
    history = case_state.get("chat_history") or []
    texts = " ".join(
        str(item.get("text") or "")
        for item in history
        if isinstance(item, Mapping)
    )
    asked = {
        str(item.get("text") or "").strip()
        for item in history
        if isinstance(item, Mapping) and item.get("from") == "doctor"
    }

    def _available(question: str) -> str:
        return "" if question in asked else question

    if any(token in texts for token in ("发热", "出血", "反复")) and "暴露" not in texts:
        q = "近期是否有明确感染、旅行或环境暴露史？"
        return _available(q)
    if any(token in texts for token in ("尿频", "排尿", "烧灼")) and "绝经" not in texts:
        q = "是否已绝经？有无外阴干涩？"
        return _available(q)
    if len([i for i in history if isinstance(i, Mapping) and i.get("from") == "patient"]) == 0:
        return "请您按时间顺序描述这次最主要的不适、何时开始、如何变化，以及伴随症状。"
    return ""
