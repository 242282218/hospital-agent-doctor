"""Strict A5 projection from legacy exam plans to executable catalog leaves."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.clinical.exam_budget_policy import assess_exam_plan_value


STRUCTURED_SOURCES = frozenset(
    {
        "coverage_gap_required",
        "required_intent",
        "typed_rule_intent",
        "axis_intent",
        "diagnosis_profile",
        "verified_case_prior",
        "verified_disease_profile",
    }
)


@dataclass(frozen=True)
class ExamPlanResult:
    examinations: tuple[str, ...]
    gap_ids: tuple[str, ...]
    intent_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    fallback_kind: str
    status: str


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _gap_ids_for_authorized(
    accepted: Mapping[str, Mapping[str, Any]],
    authorized: Sequence[str],
    open_gaps: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    known_gaps = {
        _text(gap.get("gap_id"))
        for gap in open_gaps
        if isinstance(gap, Mapping) and _text(gap.get("gap_id"))
    }
    return _unique(
        _text(accepted[name].get("axis_id"))
        for name in authorized
        if _text(accepted[name].get("source")) == "coverage_gap_required"
        and _text(accepted[name].get("axis_id")) in known_gaps
    )


def _accepted_by_name(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in plan.get("accepted") or ():
        if not isinstance(row, Mapping):
            continue
        name = _text(row.get("name"))
        source = _text(row.get("source"))
        if name and source in STRUCTURED_SOURCES:
            rows.setdefault(name, row)
    return rows


def plan_examinations(
    *,
    raw_plan: Mapping[str, Any],
    open_gaps: Sequence[Mapping[str, Any]],
    ordered_examinations: Sequence[str],
    allowed_catalog_leaves: Iterable[str],
    semantic_key_fn: Callable[[str], str],
) -> ExamPlanResult:
    """Allow only a new catalog leaf with an exact structured provenance row."""
    planned = tuple(_text(item) for item in raw_plan.get("examinations") or ())
    planned = tuple(item for item in planned if item)
    accepted = _accepted_by_name(raw_plan)
    authorized = tuple(item for item in planned if item in accepted)
    gaps = _gap_ids_for_authorized(accepted, authorized, open_gaps)
    intents = _unique(_text(accepted[item].get("intent")) for item in authorized)
    sources = _unique(_text(accepted[item].get("source")) for item in authorized)
    value = assess_exam_plan_value(
        planned_examinations=authorized,
        ordered_examinations=ordered_examinations,
        plan_reason_codes=sources,
        allowed_catalog_leaves=allowed_catalog_leaves,
        semantic_key_fn=semantic_key_fn,
    )
    if not value.has_value:
        return ExamPlanResult(
            examinations=(),
            gap_ids=gaps,
            intent_ids=intents,
            reason_codes=_unique((*value.reason_codes, "no_supported_exam")),
            fallback_kind="none",
            status="no_supported_exam",
        )
    return ExamPlanResult(
        examinations=value.new_examinations,
        gap_ids=gaps,
        intent_ids=intents,
        reason_codes=_unique((*sources, *value.reason_codes)),
        fallback_kind="structured",
        status="ready",
    )


def candidate_summary_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Hash only a compact candidate summary; never retain patient text in trace."""
    summary = [
        {"disease": _text(item.get("disease")), "score": int(item.get("score") or 0)}
        for item in candidates
        if isinstance(item, Mapping) and _text(item.get("disease"))
    ]
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def exam_value_record(
    result: ExamPlanResult,
    *,
    candidates: Sequence[Mapping[str, Any]],
    semantic_key_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Emit the bounded pre-order value record; outcome values are unknown yet."""
    return {
        "gap_ids": list(result.gap_ids),
        "intent_ids": list(result.intent_ids),
        "semantic_keys": [semantic_key_fn(name) for name in result.examinations],
        "candidate_hash_before": candidate_summary_hash(candidates),
        "candidate_hash_after": "unknown",
        "treatment_changed": "unknown",
        "urgency_changed": "unknown",
        "cost": None,
        "duration_ms": None,
    }
