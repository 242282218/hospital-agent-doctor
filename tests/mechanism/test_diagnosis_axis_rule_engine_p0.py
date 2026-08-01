"""P0-4 Step1: diagnosis_axis_rule / emit_diagnosis_axis engine expansion."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from offline.artifacts import content_hash
from agent.knowledge.typed_rule_engine import (
    DIAGNOSIS_AXIS_OPCODE,
    RuleContext,
    apply_rules,
    parse_compiled_rule_pack,
)
from agent.legacy_orchestrator import materialize_diagnosis_rule_axes


def _axis_rule(
    *,
    rule_id: str = "acute_lower_extremity_soft_tissue_infection_axis",
    axis_id: str = "acute_lower_extremity_soft_tissue_infection",
    all_groups=None,
    excluded_groups=None,
    candidates=None,
) -> dict:
    effect = {
        "axis_id": axis_id,
        "evidence": ["下肢红肿热痛", "发热"],
        "missing_evidence": ["感染严重度客观证据"],
        "candidate_official_names": candidates or ["蜂窝织炎"],
        "exam_intents": ["下肢软组织感染严重度评估"],
        "treatment_risks": [],
        "clinical_role": "current_problem",
        "priority": "urgent",
        "closure_requirement": "objective_exam_support",
    }
    return {
        "rule_id": rule_id,
        "candidate_type": "diagnosis_axis_rule",
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["下肢红肿热痛伴发热"],
        "required_evidence": ["肢体感染征象"],
        "exclusions": ["药物过敏"],
        "effect": effect,
        "positive_controls": [
            {
                "control_id": "p1",
                "kind": "positive",
                "facts": ["小腿红肿热痛发热"],
                "assertions": ["fires"],
            }
        ],
        "negative_controls": [
            {
                "control_id": "n1",
                "kind": "near_neighbor",
                "facts": ["仅下肢肿胀"],
                "assertions": ["bounded"],
            }
        ],
        "source_refs": [
            {"path": "agent/legacy_orchestrator.py", "sha256": "c" * 64}
        ],
        "test_refs": [
            {
                "path": "tests/mechanism/test_diagnosis_axis_rule_engine_p0.py",
                "sha256": "d" * 64,
            }
        ],
        "priority": 10,
        "scope": {"phase": "diagnosis", "application": "trigger_bound"},
        "runtime": {
            "status": "active",
            "stage": "diagnosis_candidates",
            "opcode": DIAGNOSIS_AXIS_OPCODE,
            "parameters": {
                "all_groups": all_groups
                or [["limb_swelling"], ["skin_redness_heat"], ["fever"]],
                "any_groups": [],
                "excluded_groups": excluded_groups or [],
            },
        },
    }


def _pack(rules: list[dict]) -> dict:
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": rules,
        "rule_count": len(rules),
        "rules_hash": content_hash(rules),
    }


def test_emit_diagnosis_axis_appends_typed_axis() -> None:
    pack = parse_compiled_rule_pack(_pack([_axis_rule()]))
    result = apply_rules(
        pack,
        "diagnosis_candidates",
        RuleContext(fact_codes=("limb_swelling", "skin_redness_heat", "fever")),
    )
    assert result.decisions[0].outcome == "applied"
    assert result.decisions[0].opcode == DIAGNOSIS_AXIS_OPCODE
    assert result.output_context.diagnostic_axis_ids == (
        "acute_lower_extremity_soft_tissue_infection",
    )
    axis = result.output_context.diagnosis_axes[0]
    assert axis.candidate_official_names == ("蜂窝织炎",)
    legacy = materialize_diagnosis_rule_axes([], result)
    assert legacy[0]["axis_id"] == "acute_lower_extremity_soft_tissue_infection"
    assert legacy[0]["candidate_official_names"] == ["蜂窝织炎"]


def test_emit_diagnosis_axis_respects_excluded_groups() -> None:
    rule = _axis_rule(excluded_groups=[["drug_allergy"]])
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(
        pack,
        "diagnosis_candidates",
        RuleContext(
            fact_codes=(
                "limb_swelling",
                "skin_redness_heat",
                "fever",
                "drug_allergy",
            )
        ),
    )
    assert result.decisions[0].outcome == "excluded"
    assert result.output_context.diagnosis_axes == ()


def test_toxic_inputs_rejected_for_diagnosis_axis_rule() -> None:
    rule = _axis_rule()
    toxic = deepcopy(rule)
    toxic["runtime"]["opcode"] = "exec_python"
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([toxic]))

    toxic2 = deepcopy(rule)
    toxic2["runtime"]["parameters"]["all_groups"] = [["__import__('os')"]]
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([toxic2]))

    toxic3 = deepcopy(rule)
    toxic3["effect"]["axis_id"] = "Not A Snake"
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([toxic3]))

    toxic4 = deepcopy(rule)
    toxic4["effect"]["candidate_official_names"] = ["不存在的虚构疾病"]
    with pytest.raises(ValueError, match="non-catalog diseases"):
        parse_compiled_rule_pack(_pack([toxic4]))


def test_diagnosis_axis_rule_stage_is_diagnosis_candidates() -> None:
    pack = parse_compiled_rule_pack(_pack([_axis_rule()]))
    # Wrong stage must not evaluate the rule.
    result = apply_rules(
        pack,
        "clinical_closure",
        RuleContext(fact_codes=("limb_swelling", "skin_redness_heat", "fever")),
    )
    assert result.decisions == ()
    assert result.output_context.diagnosis_axes == ()
