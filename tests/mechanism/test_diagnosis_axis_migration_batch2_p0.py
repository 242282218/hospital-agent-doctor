"""P0-4 batch 2: ten more diagnosis-axis rules preserve legacy helper parity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pytest

from agent.knowledge.typed_rule_engine import RuleContext, apply_rules, parse_compiled_rule_pack
from agent.legacy_orchestrator import (
    apply_diagnosis_candidate_rules,
    diagnosis_rule_fact_codes,
    has_active_upper_gi_bleed_pattern,
    has_acute_ear_pain_after_instrumentation_pattern,
    has_corneal_infection_target_rash_pattern,
    has_high_risk_pediatric_lower_respiratory_infection_pattern,
    has_immunosuppressed_acute_infection_pattern,
    has_infant_congenital_structural_heart_pattern,
    has_migraine_reproductive_travel_pattern,
    has_post_traumatic_cognitive_vestibular_pattern,
    has_sle_axis_pattern,
    has_upper_arm_trauma_pattern,
    materialize_diagnosis_rule_axes,
    normalize_name,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "agent" / "knowledge" / "clinical_pattern_rules.json"

HELPERS: Dict[str, Callable[[str], bool]] = {
    "active_upper_gi_bleed_axis": has_active_upper_gi_bleed_pattern,
    "immunosuppressed_acute_infection_axis": has_immunosuppressed_acute_infection_pattern,
    "sle_organ_thrombosis_reproductive_risk_axis": has_sle_axis_pattern,
    "corneal_infection_with_target_rash_axis": has_corneal_infection_target_rash_pattern,
    "migraine_reproductive_travel_trigger_axis": has_migraine_reproductive_travel_pattern,
    "post_traumatic_headache_cognitive_vestibular_axis": has_post_traumatic_cognitive_vestibular_pattern,
    "infant_congenital_structural_heart_disease_axis": has_infant_congenital_structural_heart_pattern,
    "high_risk_pediatric_lower_respiratory_infection_axis": has_high_risk_pediatric_lower_respiratory_infection_pattern,
    "acute_ear_pain_after_instrumentation_axis": has_acute_ear_pain_after_instrumentation_pattern,
    "upper_arm_trauma_fracture_axis": has_upper_arm_trauma_pattern,
}


def _state(text: str) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
        "diagnosis_axes": [],
    }


def _payload() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _batch_rules() -> list[dict]:
    return [rule for rule in _payload()["rules"] if rule["rule_id"] in HELPERS]


def _typed_fires(pack, rule_id: str, text: str) -> bool:
    facts = diagnosis_rule_fact_codes(_state(text))
    result = apply_rules(pack, "diagnosis_candidates", RuleContext(fact_codes=facts))
    decisions = {decision.rule_id: decision for decision in result.decisions}
    return decisions[rule_id].outcome in {"applied", "matched_no_change"}


def test_batch2_has_exactly_ten_axis_rules_with_three_controls() -> None:
    rules = _batch_rules()
    assert len(rules) == 10
    for rule in rules:
        assert len(rule["positive_controls"]) >= 3
        assert len(rule["negative_controls"]) >= 3


def test_batch2_shadow_parity_all_controls() -> None:
    pack = parse_compiled_rule_pack(_payload())
    mismatches = []
    for rule in _batch_rules():
        helper = HELPERS[rule["rule_id"]]
        for control in [*rule["positive_controls"], *rule["negative_controls"]]:
            text = " ".join(control["facts"])
            legacy = bool(helper(text)) or bool(helper(normalize_name(text)))
            typed = _typed_fires(pack, rule["rule_id"], text)
            expected = control["kind"] == "positive"
            if legacy != typed or typed != expected:
                mismatches.append(
                    {
                        "rule_id": rule["rule_id"],
                        "control_id": control["control_id"],
                        "legacy": legacy,
                        "typed": typed,
                        "expected": expected,
                        "text": text,
                    }
                )
    assert mismatches == []


@pytest.mark.parametrize(
    ("text", "axis_id", "candidate"),
    [
        ("HIV发热咳嗽", "immunosuppressed_acute_infection", "肺炎"),
        ("光敏皮疹关节痛ANA阳性补体降低", "sle_organ_thrombosis_reproductive_risk", "系统性红斑狼疮"),
        ("眼红畏光角膜病变靶形皮疹水疱", "corneal_infection_with_target_rash", "角膜炎"),
        ("育龄女性偏头痛伴恶心头晕，旅行诱发", "migraine_reproductive_travel_trigger", "偏头痛"),
        ("头部外伤后持续头痛注意力不集中", "post_traumatic_headache_cognitive_vestibular", "脑震荡"),
        (
            "新生儿出生后呼吸急促，吃奶困难出汗，哭闹发绀，肺动脉高压",
            "infant_congenital_structural_heart_disease",
            "先天性心脏病",
        ),
        (
            "高危儿科下呼吸道感染：3岁发热咳嗽黄痰喘息，长期激素",
            "high_risk_pediatric_lower_respiratory_infection",
            "肺炎",
        ),
        ("棉签掏耳后耳痛耳堵耳鸣更疼", "acute_ear_pain_after_instrumentation", "中耳炎"),
        ("跌倒手肘着地后上臂剧痛肿胀活动受限", "upper_arm_trauma_fracture", "肱骨干骨折"),
    ],
)
def test_batch2_reaches_runtime_and_materializes_axis(
    text: str,
    axis_id: str,
    candidate: str,
) -> None:
    pack = parse_compiled_rule_pack(_payload())
    ordered, result = apply_diagnosis_candidate_rules(
        [{"disease": candidate, "score": 50, "source": "unit"}],
        case_state=_state(text),
        official_diseases=[candidate],
        rule_pack=pack,
    )
    assert ordered[0]["disease"] == candidate
    axes = materialize_diagnosis_rule_axes([], result)
    axis = next(item for item in axes if item["axis_id"] == axis_id)
    assert candidate in axis["candidate_official_names"]


def test_batch2_red_flag_axis_can_emit_empty_candidates() -> None:
    pack = parse_compiled_rule_pack(_payload())
    _, result = apply_diagnosis_candidate_rules(
        [{"disease": "肝硬化", "score": 50, "source": "unit"}],
        case_state=_state("呕血黑便伴肝硬化头晕心悸"),
        official_diseases=["肝硬化"],
        rule_pack=pack,
    )
    axes = materialize_diagnosis_rule_axes([], result)
    axis = next(item for item in axes if item["axis_id"] == "active_upper_gi_bleed")
    assert axis["candidate_official_names"] == []
    assert axis["priority"] == "red_flag"
