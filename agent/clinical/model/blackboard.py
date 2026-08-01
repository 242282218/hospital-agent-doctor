from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Tuple

from .deltas import ControlOperation, SkillOperation, ValidatedControlDelta, ValidatedDelta
from .evidence import EvidenceItem
from .examination import ExamIntent
from .gaps import InformationGap
from .hypothesis import HypothesisItem
from .treatment import TreatmentItem, TreatmentState


@dataclass(frozen=True)
class WorkflowState:
    execution_state: str = "INTAKE"
    complexity: str = "simple"
    selected_hypothesis_id: str = ""
    finish_reason: str = ""


@dataclass(frozen=True)
class BudgetState:
    llm_calls_used: int = 0
    llm_calls_reserved: int = 0
    llm_hard_cap: int = 5
    patient_questions_used: int = 0
    examination_actions_used: int = 0
    llm_calls_by_skill: Tuple[Tuple[str, int], ...] = ()

    def calls_for(self, skill_name: str) -> int:
        for name, count in self.llm_calls_by_skill:
            if name == skill_name:
                return count
        return 0


@dataclass(frozen=True)
class ClinicalBlackboard:
    revision: int = 0
    evidence_ledger: Tuple[EvidenceItem, ...] = ()
    hypothesis_set: Tuple[HypothesisItem, ...] = ()
    information_gaps: Tuple[InformationGap, ...] = ()
    examination_state: Tuple[ExamIntent, ...] = ()
    treatment_state: TreatmentState = field(default_factory=TreatmentState)
    workflow_state: WorkflowState = field(default_factory=WorkflowState)
    budget_state: BudgetState = field(default_factory=BudgetState)

    def snapshot_hash(self) -> str:
        payload = asdict(self)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def apply_validated_delta(self, delta: ValidatedDelta) -> "ClinicalBlackboard":
        if delta.input_revision != self.revision:
            raise ValueError("stale validated delta: expected revision %s" % self.revision)
        board: ClinicalBlackboard = self
        for operation in delta.operations:
            board = board._apply_skill_operation(operation)
        return replace(board, revision=self.revision + 1)

    def apply_validated_control_delta(
        self, delta: ValidatedControlDelta
    ) -> "ClinicalBlackboard":
        if delta.input_revision != self.revision:
            raise ValueError("stale validated control delta: expected revision %s" % self.revision)
        board: ClinicalBlackboard = self
        for operation in delta.operations:
            board = board._apply_control_operation(operation)
        return replace(board, revision=self.revision + 1)

    def _apply_skill_operation(self, operation: SkillOperation) -> "ClinicalBlackboard":
        name = operation.operation
        payload = dict(operation.payload)
        if name == "add_evidence":
            item = _coerce_evidence(payload.get("item") or payload)
            return replace(self, evidence_ledger=self.evidence_ledger + (item,))
        if name == "add_or_update_hypothesis":
            item = _coerce_hypothesis(payload.get("item") or payload)
            others = tuple(
                hyp
                for hyp in self.hypothesis_set
                if hyp.hypothesis_id != item.hypothesis_id
                and hyp.official_disease_name != item.official_disease_name
            )
            return replace(self, hypothesis_set=others + (item,))
        if name == "add_or_update_gap":
            item = _coerce_gap(payload.get("item") or payload)
            others = tuple(gap for gap in self.information_gaps if gap.gap_id != item.gap_id)
            return replace(self, information_gaps=others + (item,))
        if name == "close_gap":
            gap_id = str(payload.get("gap_id") or "")
            closure = tuple(str(x) for x in payload.get("closure_evidence_ids") or ())
            updated = []
            for gap in self.information_gaps:
                if gap.gap_id == gap_id:
                    updated.append(
                        replace(gap, status="closed", closure_evidence_ids=closure)
                    )
                else:
                    updated.append(gap)
            return replace(self, information_gaps=tuple(updated))
        if name == "add_or_update_exam_intent":
            item = _coerce_exam(payload.get("item") or payload)
            others = tuple(
                exam
                for exam in self.examination_state
                if exam.exam_intent_id != item.exam_intent_id
                and exam.catalog_leaf_name != item.catalog_leaf_name
            )
            return replace(self, examination_state=others + (item,))
        if name == "attach_exam_result":
            leaf = str(payload.get("catalog_leaf_name") or "")
            result_ids = tuple(str(x) for x in payload.get("result_evidence_ids") or ())
            updated = []
            for exam in self.examination_state:
                if exam.catalog_leaf_name == leaf or exam.exam_intent_id == leaf:
                    updated.append(
                        replace(exam, status="resulted", result_evidence_ids=result_ids)
                    )
                else:
                    updated.append(exam)
            return replace(self, examination_state=tuple(updated))
        if name == "update_treatment_draft":
            items = payload.get("treatment_items") or ()
            treatment_items = tuple(
                item if isinstance(item, TreatmentItem) else _coerce_treatment_item(item)
                for item in items
            )
            state = TreatmentState(
                urgency_and_disposition=str(payload.get("urgency_and_disposition") or ""),
                treatment_items=treatment_items,
                draft_text=str(payload.get("draft_text") or ""),
                verifier_issues=self.treatment_state.verifier_issues,
            )
            return replace(self, treatment_state=state)
        if name == "replace_verifier_issues":
            issues = payload.get("issues")
            if issues is None:
                # single issue form
                issue = {
                    "severity": payload.get("severity"),
                    "field": payload.get("field"),
                    "code": payload.get("code"),
                    "problem": payload.get("problem"),
                }
                issues = (issue,)
            serialized = tuple(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if not isinstance(item, str)
                else item
                for item in issues
            )
            state = replace(self.treatment_state, verifier_issues=serialized)
            return replace(self, treatment_state=state)
        if name == "select_hypothesis":
            hyp_id = str(payload.get("hypothesis_id") or "")
            updated = []
            for hyp in self.hypothesis_set:
                if hyp.hypothesis_id == hyp_id or hyp.official_disease_name == hyp_id:
                    updated.append(replace(hyp, status="selected", role="selected"))
                elif hyp.status == "selected":
                    updated.append(replace(hyp, status="active", role="differential"))
                else:
                    updated.append(hyp)
            workflow = replace(self.workflow_state, selected_hypothesis_id=hyp_id)
            return replace(self, hypothesis_set=tuple(updated), workflow_state=workflow)
        raise ValueError("unsupported skill operation: %s" % name)

    def _apply_control_operation(self, operation: ControlOperation) -> "ClinicalBlackboard":
        name = operation.operation
        payload = dict(operation.payload)
        if name == "set_execution_state":
            workflow = replace(
                self.workflow_state,
                execution_state=str(payload.get("execution_state") or self.workflow_state.execution_state),
                finish_reason=str(payload.get("finish_reason") or self.workflow_state.finish_reason),
            )
            return replace(self, workflow_state=workflow)
        if name == "set_complexity":
            workflow = replace(self.workflow_state, complexity=str(payload.get("complexity") or "simple"))
            hard_cap = 8 if workflow.complexity == "complex" else 5
            budget = replace(self.budget_state, llm_hard_cap=hard_cap)
            return replace(self, workflow_state=workflow, budget_state=budget)
        if name == "record_llm_call_started":
            skill = str(payload.get("skill_name") or "unknown")
            budget = self.budget_state
            counts: Dict[str, int] = {k: v for k, v in budget.llm_calls_by_skill}
            counts[skill] = counts.get(skill, 0) + 1
            budget = replace(
                budget,
                llm_calls_used=budget.llm_calls_used + 1,
                llm_calls_reserved=budget.llm_calls_reserved + 1,
                llm_calls_by_skill=tuple(sorted(counts.items())),
            )
            return replace(self, budget_state=budget)
        if name == "record_llm_call_finished":
            budget = replace(
                self.budget_state,
                llm_calls_reserved=max(0, self.budget_state.llm_calls_reserved - 1),
            )
            return replace(self, budget_state=budget)
        if name == "increment_patient_questions":
            budget = replace(
                self.budget_state,
                patient_questions_used=self.budget_state.patient_questions_used + 1,
            )
            return replace(self, budget_state=budget)
        if name == "increment_examination_actions":
            budget = replace(
                self.budget_state,
                examination_actions_used=self.budget_state.examination_actions_used + 1,
            )
            return replace(self, budget_state=budget)
        if name == "mark_exam_ordered":
            leaf = str(payload.get("catalog_leaf_name") or "")
            updated = []
            found = False
            for exam in self.examination_state:
                if exam.catalog_leaf_name == leaf:
                    updated.append(replace(exam, status="ordered"))
                    found = True
                else:
                    updated.append(exam)
            if not found and leaf:
                updated.append(
                    ExamIntent(
                        exam_intent_id="exam-%s" % (len(updated) + 1),
                        catalog_leaf_name=leaf,
                        status="ordered",
                    )
                )
            return replace(self, examination_state=tuple(updated))
        raise ValueError("unsupported control operation: %s" % name)


def _coerce_evidence(value: Any) -> EvidenceItem:
    if isinstance(value, EvidenceItem):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("evidence payload must be mapping or EvidenceItem")
    return EvidenceItem(
        evidence_id=str(value.get("evidence_id") or ""),
        concept=str(value.get("concept") or ""),
        value=str(value.get("value") or ""),
        kind=str(value.get("kind") or "patient_statement"),
        subject=str(value.get("subject") or "patient"),
        temporality=str(value.get("temporality") or "current"),
        polarity=str(value.get("polarity") or "unknown"),
        source_ref=str(value.get("source_ref") or ""),
        source_evidence_ids=tuple(str(x) for x in value.get("source_evidence_ids") or ()),
        confidence=str(value.get("confidence") or "medium"),
        status=str(value.get("status") or "raw"),
        created_by=str(value.get("created_by") or ""),
    )


def _coerce_hypothesis(value: Any) -> HypothesisItem:
    if isinstance(value, HypothesisItem):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("hypothesis payload must be mapping or HypothesisItem")
    return HypothesisItem(
        hypothesis_id=str(value.get("hypothesis_id") or ""),
        official_disease_name=str(value.get("official_disease_name") or ""),
        role=str(value.get("role") or "differential"),
        confidence=str(value.get("confidence") or "low"),
        supporting_evidence_ids=tuple(str(x) for x in value.get("supporting_evidence_ids") or ()),
        opposing_evidence_ids=tuple(str(x) for x in value.get("opposing_evidence_ids") or ()),
        open_gap_ids=tuple(str(x) for x in value.get("open_gap_ids") or ()),
        required_exam_intents=tuple(str(x) for x in value.get("required_exam_intents") or ()),
        treatment_risk_tags=tuple(str(x) for x in value.get("treatment_risk_tags") or ()),
        status=str(value.get("status") or "active"),
    )


def _coerce_gap(value: Any) -> InformationGap:
    if isinstance(value, InformationGap):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("gap payload must be mapping or InformationGap")
    return InformationGap(
        gap_id=str(value.get("gap_id") or ""),
        intent=str(value.get("intent") or ""),
        related_hypothesis_ids=tuple(str(x) for x in value.get("related_hypothesis_ids") or ()),
        decision_impact=str(value.get("decision_impact") or "diagnosis"),
        acquisition_route=str(value.get("acquisition_route") or "ask"),
        priority=str(value.get("priority") or "normal"),
        status=str(value.get("status") or "open"),
        closure_evidence_ids=tuple(str(x) for x in value.get("closure_evidence_ids") or ()),
    )


def _coerce_exam(value: Any) -> ExamIntent:
    if isinstance(value, ExamIntent):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("exam payload must be mapping or ExamIntent")
    return ExamIntent(
        exam_intent_id=str(value.get("exam_intent_id") or ""),
        related_gap_ids=tuple(str(x) for x in value.get("related_gap_ids") or ()),
        related_hypothesis_ids=tuple(str(x) for x in value.get("related_hypothesis_ids") or ()),
        requirement=str(value.get("requirement") or "optional"),
        catalog_leaf_name=str(value.get("catalog_leaf_name") or ""),
        reason=str(value.get("reason") or ""),
        status=str(value.get("status") or "proposed"),
        result_evidence_ids=tuple(str(x) for x in value.get("result_evidence_ids") or ()),
        waived_by=str(value.get("waived_by") or ""),
        waiver_reason=str(value.get("waiver_reason") or ""),
        waiver_evidence_ids=tuple(str(x) for x in value.get("waiver_evidence_ids") or ()),
    )


def _coerce_treatment_item(value: Any) -> TreatmentItem:
    if isinstance(value, TreatmentItem):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("treatment item payload must be mapping or TreatmentItem")
    return TreatmentItem(
        item_id=str(value.get("item_id") or ""),
        category=str(value.get("category") or ""),
        stage=str(value.get("stage") or ""),
        intent=str(value.get("intent") or ""),
        related_hypothesis_ids=tuple(str(x) for x in value.get("related_hypothesis_ids") or ()),
        evidence_ids=tuple(str(x) for x in value.get("evidence_ids") or ()),
        constraint_ids=tuple(str(x) for x in value.get("constraint_ids") or ()),
        status=str(value.get("status") or "proposed"),
    )
