"""Offline-only projection of legacy case traces onto immutable Blackboard snapshots."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model import (
    BudgetState,
    ClinicalBlackboard,
    EvidenceItem,
    ExamIntent,
    HypothesisItem,
    InformationGap,
    TreatmentState,
    WorkflowState,
)

NEGATIVE_MARKERS = ("无", "没有", "否认", "未", "不是", "阴性")
POSITIVE_MARKERS = ("有", "确诊", "升高", "阳性", "出现", "服用")


@dataclass(frozen=True)
class ShadowSnapshot:
    blackboard: ClinicalBlackboard
    source_trace_hash: str


@dataclass(frozen=True)
class ShadowConflict:
    concept: str
    evidence_ids: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ShadowDiff:
    source_trace_hash: str
    snapshot_hash: str
    missing_fact_keys: Tuple[str, ...] = ()
    conflicts: Tuple[ShadowConflict, ...] = ()
    final_field_differences: Tuple[str, ...] = ()

    @property
    def final_submission_changed(self) -> bool:
        return bool(self.final_field_differences)


class ShadowBlackboardProjector:
    def project(self, trace: Mapping[str, Any]) -> ShadowSnapshot:
        canonical_trace = _canonical_copy(trace)
        evidence = _project_evidence(canonical_trace)
        evidence, _conflicts = _mark_conflicts(evidence)
        board = ClinicalBlackboard(
            revision=int(canonical_trace.get("trace_revision") or 0),
            evidence_ledger=evidence,
            hypothesis_set=_project_hypotheses(canonical_trace, evidence),
            information_gaps=_project_gaps(canonical_trace),
            examination_state=_project_examinations(canonical_trace, evidence),
            treatment_state=_project_treatment(canonical_trace),
            workflow_state=_project_workflow(canonical_trace),
            budget_state=_project_budget(canonical_trace),
        )
        return ShadowSnapshot(board, _content_hash(canonical_trace))

    def compare(
        self,
        trace: Mapping[str, Any],
        snapshot: ShadowSnapshot,
        *,
        final_payload: Optional[Mapping[str, Any]] = None,
    ) -> ShadowDiff:
        required = tuple(str(item) for item in trace.get("required_fact_keys", ()))
        expressed = {item.concept for item in snapshot.blackboard.evidence_ledger}
        missing = tuple(item for item in required if item not in expressed)
        expected = _normalized_final(trace.get("final_plan", {}))
        submitted = _normalized_final(final_payload or {})
        fields = tuple(
            name
            for name in ("diagnosis", "treatment_plan", "reasoning")
            if expected.get(name) != submitted.get(name)
        )
        return ShadowDiff(
            source_trace_hash=snapshot.source_trace_hash,
            snapshot_hash=snapshot.blackboard.snapshot_hash(),
            missing_fact_keys=missing,
            conflicts=_conflict_report(snapshot.blackboard.evidence_ledger),
            final_field_differences=fields,
        )


def _canonical_copy(value: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _content_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _polarity(text: str, *, structured: bool = False) -> str:
    value = str(text or "").strip()
    if not value:
        return "unknown"
    if any(marker in value for marker in NEGATIVE_MARKERS):
        return "negative"
    if any(marker in value for marker in POSITIVE_MARKERS):
        return "positive"
    # Structured short facts without markers (e.g. "高血压") are affirmative.
    if structured:
        return "positive"
    return "unknown"


def _project_evidence(trace: Mapping[str, Any]) -> Tuple[EvidenceItem, ...]:
    items: List[EvidenceItem] = []
    counter = 0

    def add(
        *,
        concept: str,
        value: str,
        kind: str,
        subject: str = "patient",
        temporality: str = "current",
        source_ref: str = "",
        status: str = "raw",
    ) -> None:
        nonlocal counter
        counter += 1
        items.append(
            EvidenceItem(
                evidence_id="ev-%s" % counter,
                concept=concept,
                value=str(value),
                kind=kind,
                subject=subject,
                temporality=temporality,
                polarity=_polarity(str(value), structured=(kind == "structured_fact")),
                source_ref=source_ref,
                status=status,
                created_by="shadow_projector",
            )
        )

    for index, message in enumerate(trace.get("chat_history") or []):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("from") or "") != "patient":
            continue
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        add(
            concept="patient_statement",
            value=text,
            kind="patient_statement",
            source_ref="message://patient/%s" % (index + 1),
        )

    features = trace.get("case_features") or {}
    if isinstance(features, Mapping):
        for value in features.get("family_history") or []:
            add(
                concept="family_history",
                value=str(value),
                kind="structured_fact",
                subject="family",
                temporality="historical",
                source_ref="case_features://family_history",
            )
        for value in features.get("personal_history") or []:
            add(
                concept="personal_history",
                value=str(value),
                kind="structured_fact",
                subject="patient",
                temporality="historical",
                source_ref="case_features://personal_history",
            )
        for value in features.get("medications") or []:
            add(
                concept="medications",
                value=str(value),
                kind="structured_fact",
                subject="patient",
                temporality="current",
                source_ref="case_features://medications",
            )
        for value in features.get("drug_allergies") or []:
            add(
                concept="drug_allergies",
                value=str(value),
                kind="structured_fact",
                subject="patient",
                temporality="current",
                source_ref="case_features://drug_allergies",
            )

    results = trace.get("examination_results") or {}
    invalid = {str(item) for item in (trace.get("invalid_examinations") or [])}
    if isinstance(results, Mapping):
        for name, payload in results.items():
            leaf = str(name)
            if leaf in invalid:
                continue
            if not isinstance(payload, Mapping):
                continue
            status = str(payload.get("status") or "")
            if status == "invalid":
                continue
            add(
                concept="exam_result",
                value=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                kind="exam_result",
                source_ref="runtime://legacy-exam/%s" % leaf,
                status="validated",
            )
    return tuple(items)


def _mark_conflicts(
    evidence: Sequence[EvidenceItem],
) -> Tuple[Tuple[EvidenceItem, ...], Tuple[ShadowConflict, ...]]:
    groups: Dict[Tuple[str, str, str], List[EvidenceItem]] = {}
    for item in evidence:
        key = (item.concept, item.subject, item.temporality)
        groups.setdefault(key, []).append(item)

    conflicted_ids = set()
    conflicts: List[ShadowConflict] = []
    for key, group in groups.items():
        polarities = {item.polarity for item in group}
        if "positive" in polarities and "negative" in polarities:
            ids = tuple(item.evidence_id for item in group)
            conflicted_ids.update(ids)
            conflicts.append(
                ShadowConflict(
                    concept=key[0],
                    evidence_ids=ids,
                    reason="positive_and_negative_same_slot",
                )
            )

    updated = tuple(
        replace(item, status="conflicted") if item.evidence_id in conflicted_ids else item
        for item in evidence
    )
    return updated, tuple(conflicts)


def _project_examinations(
    trace: Mapping[str, Any],
    evidence: Sequence[EvidenceItem],
) -> Tuple[ExamIntent, ...]:
    ordered = [str(item) for item in (trace.get("ordered_examinations") or []) if str(item).strip()]
    invalid = {str(item) for item in (trace.get("invalid_examinations") or [])}
    results = trace.get("examination_results") or {}
    result_names = set(results.keys()) if isinstance(results, Mapping) else set()
    result_evidence = {
        item.source_ref.split("/")[-1]: item.evidence_id
        for item in evidence
        if item.kind == "exam_result" and item.source_ref.startswith("runtime://legacy-exam/")
    }

    seen = []
    for name in ordered:
        if name not in seen:
            seen.append(name)
    for name in result_names:
        leaf = str(name)
        if leaf not in seen:
            seen.append(leaf)

    intents: List[ExamIntent] = []
    for index, name in enumerate(seen, start=1):
        if name in invalid:
            status = "invalid"
            result_ids: Tuple[str, ...] = ()
        elif name in result_names:
            status = "resulted"
            result_ids = (result_evidence[name],) if name in result_evidence else ()
        else:
            status = "ordered"
            result_ids = ()
        intents.append(
            ExamIntent(
                exam_intent_id="exam-%s" % index,
                catalog_leaf_name=name,
                status=status,
                result_evidence_ids=result_ids,
            )
        )
    return tuple(intents)


def _project_hypotheses(
    trace: Mapping[str, Any],
    evidence: Sequence[EvidenceItem],
) -> Tuple[HypothesisItem, ...]:
    final_plan = trace.get("final_plan") or {}
    selected_name = ""
    if isinstance(final_plan, Mapping):
        selected_name = str(final_plan.get("diagnosis") or "").strip()

    axes = trace.get("diagnosis_axes") or []
    candidates = trace.get("disease_candidates") or []
    items: List[HypothesisItem] = []
    seen = set()

    def add_item(name: str, *, selected: bool, intents: Sequence[str] = (), risks: Sequence[str] = ()) -> None:
        clean = str(name or "").strip()
        if not clean or clean in seen:
            return
        seen.add(clean)
        items.append(
            HypothesisItem(
                hypothesis_id="hyp-%s" % (len(items) + 1),
                official_disease_name=clean,
                role="selected" if selected else "differential",
                confidence="high" if selected else "medium",
                supporting_evidence_ids=tuple(item.evidence_id for item in evidence[:3]),
                required_exam_intents=tuple(str(x) for x in intents if str(x).strip()),
                treatment_risk_tags=tuple(str(x) for x in risks if str(x).strip()),
                status="selected" if selected else "active",
            )
        )

    if selected_name:
        add_item(selected_name, selected=True)

    if isinstance(axes, list):
        for axis in axes:
            if not isinstance(axis, Mapping):
                continue
            names = axis.get("candidate_official_names") or []
            intents = axis.get("exam_intents") or []
            risks = axis.get("treatment_risks") or []
            for name in names:
                add_item(str(name), selected=str(name) == selected_name, intents=intents, risks=risks)

    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                add_item(str(candidate.get("disease") or ""), selected=False)
            else:
                add_item(str(candidate), selected=False)
    return tuple(items)


def _project_gaps(trace: Mapping[str, Any]) -> Tuple[InformationGap, ...]:
    gaps = trace.get("coverage_gaps") or []
    items: List[InformationGap] = []
    if not isinstance(gaps, list):
        return ()
    for index, gap in enumerate(gaps, start=1):
        if not isinstance(gap, Mapping):
            continue
        items.append(
            InformationGap(
                gap_id=str(gap.get("gap_id") or "gap-%s" % index),
                intent=str(gap.get("intent") or ""),
                decision_impact=str(gap.get("decision_impact") or "diagnosis"),
                acquisition_route=str(gap.get("acquisition_route") or "ask"),
                priority=str(gap.get("priority") or "normal"),
                status=str(gap.get("status") or "open"),
            )
        )
    return tuple(items)


def _project_treatment(trace: Mapping[str, Any]) -> TreatmentState:
    plan = trace.get("final_plan") or {}
    if not isinstance(plan, Mapping):
        return TreatmentState()
    return TreatmentState(
        draft_text=str(plan.get("treatment_plan") or "").strip(),
        urgency_and_disposition=str(plan.get("department") or "").strip(),
    )


def _project_workflow(trace: Mapping[str, Any]) -> WorkflowState:
    has_final = bool(trace.get("final_plan"))
    return WorkflowState(
        execution_state="finish" if has_final else "intake",
        complexity=str(trace.get("complexity") or "simple"),
        selected_hypothesis_id="",
        finish_reason="legacy_final" if has_final else "",
    )


def _project_budget(trace: Mapping[str, Any]) -> BudgetState:
    return BudgetState(
        llm_calls_used=int(trace.get("llm_calls_used") or 0),
        llm_hard_cap=int(trace.get("llm_hard_cap") or 5),
        patient_questions_used=len(
            [
                item
                for item in (trace.get("chat_history") or [])
                if isinstance(item, Mapping) and item.get("from") == "doctor"
            ]
        ),
        examination_actions_used=len(list(trace.get("ordered_examinations") or [])),
    )


def _normalized_final(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"diagnosis": (), "treatment_plan": "", "reasoning": ""}
    diagnosis = payload.get("diagnosis")
    if isinstance(diagnosis, str):
        diagnosis_tuple = (diagnosis.strip(),) if diagnosis.strip() else ()
    elif isinstance(diagnosis, Iterable) and not isinstance(diagnosis, (bytes, bytearray)):
        diagnosis_tuple = tuple(str(item).strip() for item in diagnosis if str(item).strip())
    else:
        diagnosis_tuple = ()
    return {
        "diagnosis": diagnosis_tuple,
        "treatment_plan": str(payload.get("treatment_plan") or "").strip(),
        "reasoning": str(payload.get("reasoning") or "").strip(),
    }


def _conflict_report(evidence: Sequence[EvidenceItem]) -> Tuple[ShadowConflict, ...]:
    groups: Dict[str, List[str]] = {}
    for item in evidence:
        if item.status != "conflicted":
            continue
        groups.setdefault(item.concept, []).append(item.evidence_id)
    return tuple(
        ShadowConflict(concept=concept, evidence_ids=tuple(ids), reason="positive_and_negative_same_slot")
        for concept, ids in sorted(groups.items())
    )


def deep_equal_trace(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical_copy(left) == _canonical_copy(right)


def project_and_diff(
    trace: Mapping[str, Any],
    *,
    final_payload: Optional[Mapping[str, Any]] = None,
) -> Tuple[ShadowSnapshot, ShadowDiff]:
    projector = ShadowBlackboardProjector()
    working = deepcopy(dict(trace))
    snapshot = projector.project(working)
    assert deep_equal_trace(working, dict(trace))
    payload = final_payload
    if payload is None and isinstance(trace.get("final_plan"), Mapping):
        plan = trace["final_plan"]
        payload = {
            "diagnosis": [plan.get("diagnosis")] if isinstance(plan.get("diagnosis"), str) else plan.get("diagnosis"),
            "treatment_plan": plan.get("treatment_plan"),
            "reasoning": plan.get("reasoning"),
        }
    diff = projector.compare(trace, snapshot, final_payload=payload)
    return snapshot, diff
