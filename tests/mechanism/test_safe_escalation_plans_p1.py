"""P1: safe-escalation closure text must live in data, not in a 281-line if/elif chain.

`build_safe_escalation_plan` used to carry 46 hardcoded Chinese closure paragraphs
inline, mirrored by `validate_safe_escalation_plan`. Adding an axis meant editing
two places, and the closure text could not be reviewed as knowledge.

The closure text now comes from `agent/knowledge/safe_escalation_plans.json`. These
tests pin that contract:
- the table is the single source of the per-axis closure text,
- every table axis still round-trips through validate (build -> validate is closed),
- an unknown axis still produces a conservative closure instead of an empty plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.knowledge import safe_escalation
from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    validate_safe_escalation_plan,
)

PLANS_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent"
    / "knowledge"
    / "safe_escalation_plans.json"
)

EVIDENCE = ("证据一", "证据二")


def _table() -> dict:
    payload = json.loads(PLANS_PATH.read_text(encoding="utf-8"))
    return {row["axis_id"]: row["closure"] for row in payload["plans"]}


def test_plan_table_is_the_source_of_truth() -> None:
    table = _table()
    assert len(table) == 45, "expected 45 axis closures, got %d" % len(table)
    for axis_id, closure in table.items():
        assert axis_id.strip(), "axis_id must not be blank"
        assert len(closure) >= 20, "closure for %s is suspiciously short" % axis_id


def test_loader_returns_exactly_the_table_text() -> None:
    for axis_id, closure in _table().items():
        assert safe_escalation.closure_for_axis(axis_id) == closure


def test_build_uses_the_table_text_for_every_axis() -> None:
    for axis_id, closure in _table().items():
        plan, reason = build_safe_escalation_plan(
            axis_id=axis_id,
            closure_requirement="supported_official_diagnosis",
            evidence=EVIDENCE,
            existing_treatment="",
        )
        assert closure in plan, "built plan for %s dropped the table closure" % axis_id
        assert axis_id in reason


def test_every_table_axis_round_trips_through_validate() -> None:
    """build -> validate must be closed: a generated plan is always acceptable.

    This is the guard against mirror drift between the two functions.
    """
    failures = []
    for axis_id in _table():
        plan, _reason = build_safe_escalation_plan(
            axis_id=axis_id,
            closure_requirement="supported_official_diagnosis",
            evidence=EVIDENCE,
            existing_treatment="",
        )
        if not validate_safe_escalation_plan(plan, axis_id=axis_id, evidence=EVIDENCE):
            failures.append(axis_id)
    assert not failures, "generated plans rejected by validate: %s" % failures


def test_unknown_axis_still_gets_a_conservative_closure() -> None:
    assert safe_escalation.closure_for_axis("no_such_axis") == ""
    plan, _reason = build_safe_escalation_plan(
        axis_id="no_such_axis",
        closure_requirement="supported_official_diagnosis",
        evidence=EVIDENCE,
        existing_treatment="",
    )
    # Must never be empty: an unmapped axis falls back to emergency/specialist review.
    assert plan.strip()
    assert "急诊" in plan or "住院" in plan


def test_unknown_axis_with_specific_requirement_names_the_requirement() -> None:
    plan, _reason = build_safe_escalation_plan(
        axis_id="no_such_axis",
        closure_requirement="close_the_bleeding_source",
        evidence=EVIDENCE,
        existing_treatment="",
    )
    assert "close_the_bleeding_source" in plan


@pytest.mark.parametrize("evidence", [(), ("只有一条",)])
def test_validate_requires_at_least_two_evidence_items(evidence) -> None:
    plan, _reason = build_safe_escalation_plan(
        axis_id="active_upper_gi_bleed",
        closure_requirement="urgent_hemostasis_and_resuscitation",
        evidence=EVIDENCE,
        existing_treatment="",
    )
    assert (
        validate_safe_escalation_plan(
            plan, axis_id="active_upper_gi_bleed", evidence=evidence
        )
        is False
    )
