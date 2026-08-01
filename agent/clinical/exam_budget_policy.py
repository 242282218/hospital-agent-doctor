"""Pure exam budget and plan-value decisions (no LLM / file I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple


STRUCTURED_REASON_CODES = frozenset(
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
class ExamBudgetDecision:
    should_stop: bool
    stop_kind: str  # "continue" | "hard"
    reason_codes: Tuple[str, ...]
    raw_action_count: int
    effective_action_count: int
    open_high_value_gap_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ExamPlanValue:
    has_value: bool
    stop_kind: str  # "continue" | "soft"
    reason_codes: Tuple[str, ...]
    new_examinations: Tuple[str, ...]
    new_semantic_keys: Tuple[str, ...]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_text_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = _clean(value)
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            text = _clean(item)
            if text and text not in out:
                out.append(text)
        return tuple(out)
    text = _clean(value)
    return (text,) if text else ()


def _high_value_gap_ids(open_gaps: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    ids: list[str] = []
    for gap in open_gaps or ():
        if not isinstance(gap, Mapping):
            continue
        gap_id = _clean(gap.get("gap_id"))
        if not gap_id:
            continue
        required = _as_text_list(gap.get("required_exams"))
        intents = _as_text_list(gap.get("exam_intents"))
        if not required and not intents:
            continue
        if gap_id not in ids:
            ids.append(gap_id)
    return tuple(ids)


def _plan_has_new_leaf(
    plan: Mapping[str, Any],
    ordered_examinations: Sequence[str],
    *,
    semantic_key_fn: Optional[Callable[[str], str]] = None,
) -> bool:
    ordered = {_clean(x) for x in ordered_examinations if _clean(x)}
    ordered_keys = set()
    if semantic_key_fn is not None:
        ordered_keys = {semantic_key_fn(x) for x in ordered if x}
    for name in _as_text_list(plan.get("examinations")):
        if name not in ordered:
            if semantic_key_fn is None:
                return True
            key = semantic_key_fn(name)
            if key not in ordered_keys:
                return True
    return False


def decide_exam_budget(
    *,
    exam_trace: Sequence[Mapping[str, Any]],
    open_gaps: Sequence[Mapping[str, Any]],
    ordered_examinations: Sequence[str],
    hard_cap: int,
    semantic_key_fn: Optional[Callable[[str], str]] = None,
) -> ExamBudgetDecision:
    raw_count = len(list(exam_trace or ()))
    ordered = list(ordered_examinations or ())
    effective = 0
    seen_ordered: list[str] = []
    for plan in exam_trace or ():
        if not isinstance(plan, Mapping):
            continue
        if _plan_has_new_leaf(plan, seen_ordered, semantic_key_fn=semantic_key_fn):
            effective += 1
            for name in _as_text_list(plan.get("examinations")):
                if name not in seen_ordered:
                    seen_ordered.append(name)
        # Keep raw history visibility even when plan adds nothing.
    # Prefer actual ordered list for effective baseline if trace incomplete.
    if ordered and effective == 0:
        # No successful new-leaf plans observed; keep effective at 0.
        pass
    high_value = _high_value_gap_ids(open_gaps)
    if raw_count >= int(hard_cap):
        return ExamBudgetDecision(
            should_stop=True,
            stop_kind="hard",
            reason_codes=("exam_hard_cap",),
            raw_action_count=raw_count,
            effective_action_count=effective,
            open_high_value_gap_ids=high_value,
        )
    return ExamBudgetDecision(
        should_stop=False,
        stop_kind="continue",
        reason_codes=("exam_budget_open",),
        raw_action_count=raw_count,
        effective_action_count=effective,
        open_high_value_gap_ids=high_value,
    )


def assess_exam_plan_value(
    *,
    planned_examinations: Sequence[str],
    ordered_examinations: Sequence[str],
    plan_reason_codes: Sequence[str],
    allowed_catalog_leaves: Iterable[str],
    semantic_key_fn: Optional[Callable[[str], str]] = None,
) -> ExamPlanValue:
    catalog = {_clean(x) for x in allowed_catalog_leaves if _clean(x)}
    ordered = {_clean(x) for x in ordered_examinations if _clean(x)}
    ordered_keys = set()
    if semantic_key_fn is not None:
        ordered_keys = {semantic_key_fn(x) for x in ordered if x}

    new_exams: list[str] = []
    new_keys: list[str] = []
    for name in planned_examinations or ():
        clean_name = _clean(name)
        if not clean_name:
            continue
        if catalog and clean_name not in catalog:
            continue
        if clean_name in ordered:
            continue
        key = semantic_key_fn(clean_name) if semantic_key_fn is not None else clean_name
        if key in ordered_keys or key in new_keys:
            continue
        new_exams.append(clean_name)
        new_keys.append(key)

    structured = []
    for code in plan_reason_codes or ():
        text = _clean(code)
        if not text:
            continue
        if text in STRUCTURED_REASON_CODES and text not in structured:
            structured.append(text)

    if new_exams and structured:
        return ExamPlanValue(
            has_value=True,
            stop_kind="continue",
            reason_codes=tuple(structured),
            new_examinations=tuple(new_exams),
            new_semantic_keys=tuple(new_keys),
        )
    reasons = []
    if not new_exams:
        reasons.append("exam_no_new_catalog_leaf")
    if not structured:
        reasons.append("exam_no_structured_reason")
    reasons.append("exam_no_structured_gain")
    return ExamPlanValue(
        has_value=False,
        stop_kind="soft",
        reason_codes=tuple(dict.fromkeys(reasons)),
        new_examinations=(),
        new_semantic_keys=(),
    )
