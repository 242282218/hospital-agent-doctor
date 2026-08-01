"""Exam-stage clinical_closure intent extraction (no diagnosis reordering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from agent.knowledge.typed_rule_engine import CompiledRulePack, RuleContext, apply_rules


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _clean_axis_id(value: object) -> str:
    text = _clean_text(value)
    return text.replace(" ", "_") if text else ""


@dataclass(frozen=True)
class ExamRuleClosureResult:
    exam_intent_ids: Tuple[str, ...]
    matched_rule_ids: Tuple[str, ...]
    excluded_rule_ids: Tuple[str, ...]
    fact_codes: Tuple[str, ...]


def evaluate_exam_rule_closure(
    *,
    rule_pack: CompiledRulePack,
    fact_codes: Iterable[str],
    diagnostic_axis_ids: Iterable[str],
) -> ExamRuleClosureResult:
    """Run clinical_closure only; never return or rewrite diagnosis candidates."""
    normalized_axes = tuple(
        dict.fromkeys(
            _clean_axis_id(item) for item in diagnostic_axis_ids if _clean_axis_id(item)
        )
    )
    normalized_facts = tuple(
        dict.fromkeys(_clean_text(item) for item in fact_codes if _clean_text(item))
    )
    context = RuleContext(
        diagnostic_axis_ids=normalized_axes,
        fact_codes=normalized_facts,
    )
    result = apply_rules(rule_pack, "clinical_closure", context)
    intent_ids = tuple(
        dict.fromkeys(
            _clean_text(item)
            for item in result.output_context.exam_intent_ids
            if _clean_text(item)
        )
    )
    matched: list[str] = []
    excluded: list[str] = []
    for decision in result.decisions:
        rule_id = _clean_text(decision.rule_id)
        if not rule_id:
            continue
        outcome = str(decision.outcome or "")
        if outcome in {"applied", "matched"}:
            matched.append(rule_id)
        elif outcome in {"excluded", "blocked", "not_matched"}:
            # Keep excluded_rule_ids for rules that actually hit exclusion; still
            # record not_matched separately only when reason implies exclusion.
            reason = _clean_text(getattr(decision, "reason_code", ""))
            if "exclud" in reason or outcome == "excluded":
                excluded.append(rule_id)
    return ExamRuleClosureResult(
        exam_intent_ids=intent_ids,
        matched_rule_ids=tuple(dict.fromkeys(matched)),
        excluded_rule_ids=tuple(dict.fromkeys(excluded)),
        fact_codes=tuple(result.output_context.fact_codes) or normalized_facts,
    )
