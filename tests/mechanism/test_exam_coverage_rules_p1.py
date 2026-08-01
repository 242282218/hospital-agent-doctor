"""P1: exam coverage predicates must be data, not 27 hand-written functions.

The `exams_cover_*` family was 27 near-identical functions, each a keyword list
plus `any()`. That shape made every new coverage requirement a code change, so
reflection output (Reflexion `next_action_rule`) had no data landing zone.

The markers now live in `agent/knowledge/exam_coverage_rules.json` and the
predicates are generated from it. These tests pin the contract so the family
cannot drift back into code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.knowledge import exam_coverage
from agent.legacy_orchestrator import (
    exams_cover_cbc,
    exams_cover_chest_imaging,
    exams_cover_glucose,
    exams_cover_renal_urine,
    exams_cover_upper_arm_fracture,
)

RULES_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "knowledge" / "exam_coverage_rules.json"
)


def test_rules_file_is_the_source_of_truth() -> None:
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules = payload["rules"]
    assert len(rules) == 26, "expected 26 data-driven coverage rules, got %d" % len(rules)
    for rule in rules:
        rule_id = rule.get("id")
        assert rule_id, "every rule must declare an id"
        assert rule.get("markers") or rule.get("groups"), (
            "rule %s must declare markers or groups" % rule_id
        )


def test_loaded_rule_ids_match_the_file() -> None:
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert set(exam_coverage.rule_ids()) == {rule["id"] for rule in payload["rules"]}


@pytest.mark.parametrize(
    "predicate,positive,negative",
    [
        (exams_cover_cbc, "全血细胞计数（CBC）", "尿液分析"),
        (exams_cover_chest_imaging, "胸部CT", "腹部超声"),
        (exams_cover_upper_arm_fracture, "四肢X线检查", "胸部X线"),
    ],
)
def test_generated_predicates_discriminate(predicate, positive, negative) -> None:
    assert predicate([positive]) is True
    assert predicate([negative]) is False
    assert predicate([]) is False


def test_exclusion_semantics_are_preserved() -> None:
    # 糖化血红蛋白 mentions 血糖-adjacent text but must NOT count as a glucose test.
    assert exams_cover_glucose(["空腹血糖（FBG）"]) is True
    assert exams_cover_glucose(["糖化血红蛋白（HbA1c）"]) is False


def test_two_group_or_semantics_are_preserved() -> None:
    # renal_urine passes on EITHER a urine study OR a renal function panel.
    assert exams_cover_renal_urine(["尿液分析"]) is True
    assert exams_cover_renal_urine(["肾功能检查"]) is True
    assert exams_cover_renal_urine(["胸部CT"]) is False


def test_unknown_rule_id_fails_closed() -> None:
    # A typo in a rule id must raise, never silently report "not covered".
    with pytest.raises(exam_coverage.ExamCoverageRuleError):
        exam_coverage.covers("no_such_rule", ["体格检查"])
