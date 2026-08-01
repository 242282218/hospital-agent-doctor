"""P1: exam soft/hard stop and plan value matrix."""

from __future__ import annotations

from agent.clinical.exam_budget_policy import (
    assess_exam_plan_value,
    decide_exam_budget,
)


def _key(name: str) -> str:
    return name.strip().lower()


def test_hard_cap_stops_even_with_open_gap() -> None:
    decision = decide_exam_budget(
        exam_trace=[{}, {}, {}, {}, {}, {}],
        open_gaps=[
            {
                "gap_id": "g1",
                "required_exams": ["血常规"],
                "exam_intents": ["感染评估"],
            }
        ],
        ordered_examinations=[],
        hard_cap=6,
        semantic_key_fn=_key,
    )
    assert decision.should_stop is True
    assert decision.stop_kind == "hard"
    assert "exam_hard_cap" in decision.reason_codes
    assert decision.open_high_value_gap_ids == ("g1",)


def test_below_cap_without_gap_still_continues_for_planner() -> None:
    decision = decide_exam_budget(
        exam_trace=[{}],
        open_gaps=[],
        ordered_examinations=["血常规"],
        hard_cap=6,
        semantic_key_fn=_key,
    )
    assert decision.should_stop is False
    assert decision.stop_kind == "continue"


def test_effective_count_only_increments_for_new_leaf_plans() -> None:
    decision = decide_exam_budget(
        exam_trace=[
            {"examinations": ["血常规"]},
            {"examinations": ["血常规"]},  # no new leaf
            {"examinations": []},  # empty
            {"examinations": ["C反应蛋白（CRP）"]},
        ],
        open_gaps=[],
        ordered_examinations=["血常规", "C反应蛋白（CRP）"],
        hard_cap=6,
        semantic_key_fn=_key,
    )
    assert decision.raw_action_count == 4
    assert decision.effective_action_count == 2


def test_plan_value_soft_stops_without_new_leaf() -> None:
    value = assess_exam_plan_value(
        planned_examinations=["血常规"],
        ordered_examinations=["血常规"],
        plan_reason_codes=["typed_rule_intent"],
        allowed_catalog_leaves=["血常规", "软组织超声"],
        semantic_key_fn=_key,
    )
    assert value.has_value is False
    assert value.stop_kind == "soft"
    assert "exam_no_structured_gain" in value.reason_codes


def test_plan_value_soft_stops_without_structured_reason() -> None:
    value = assess_exam_plan_value(
        planned_examinations=["软组织超声"],
        ordered_examinations=[],
        plan_reason_codes=["llm_free_text"],
        allowed_catalog_leaves=["血常规", "软组织超声"],
        semantic_key_fn=_key,
    )
    assert value.has_value is False
    assert value.stop_kind == "soft"


def test_free_text_structured_token_does_not_authorize_plan_value() -> None:
    value = assess_exam_plan_value(
        planned_examinations=["软组织超声"],
        ordered_examinations=[],
        plan_reason_codes=["coverage_gap_required:soft_tissue_gap"],
        allowed_catalog_leaves=["血常规", "软组织超声"],
        semantic_key_fn=_key,
    )
    assert value.has_value is False
    assert "exam_no_structured_reason" in value.reason_codes


def test_plan_value_continues_for_new_leaf_with_structured_reason() -> None:
    value = assess_exam_plan_value(
        planned_examinations=["软组织超声"],
        ordered_examinations=["血常规"],
        plan_reason_codes=["typed_rule_intent", "coverage_gap_required"],
        allowed_catalog_leaves=["血常规", "软组织超声"],
        semantic_key_fn=_key,
    )
    assert value.has_value is True
    assert value.stop_kind == "continue"
    assert value.new_examinations == ("软组织超声",)


def test_invalid_catalog_leaf_is_ignored() -> None:
    value = assess_exam_plan_value(
        planned_examinations=["不存在检查"],
        ordered_examinations=[],
        plan_reason_codes=["axis_intent"],
        allowed_catalog_leaves=["血常规"],
        semantic_key_fn=_key,
    )
    assert value.has_value is False
    assert value.new_examinations == ()
