"""P0-4 first batch: ten diagnosis-axis rules preserve legacy trigger parity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pytest

from agent.knowledge.typed_rule_engine import RuleContext, apply_rules, parse_compiled_rule_pack
from agent.legacy_orchestrator import (
    apply_diagnosis_candidate_rules,
    diagnosis_rule_fact_codes,
    has_acute_lower_extremity_soft_tissue_infection_pattern,
    has_chronic_suppurative_middle_ear_pattern,
    has_congenital_syndactyly_pattern,
    has_high_energy_hindfoot_trauma_pattern,
    has_hyperlipidemia_with_xanthelasma_pattern,
    has_pediatric_congenital_glaucoma_pattern,
    has_perioral_dermatitis_pattern,
    has_seafood_acute_watery_diarrhea_pattern,
    has_traumatic_rib_fracture_pattern,
    has_water_aerosol_severe_pneumonia_pattern,
    materialize_diagnosis_rule_axes,
    normalize_name,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "agent" / "knowledge" / "clinical_pattern_rules.json"

HELPERS: Dict[str, Callable[[str], bool]] = {
    "acute_lower_extremity_soft_tissue_infection_axis": has_acute_lower_extremity_soft_tissue_infection_pattern,
    "hyperlipidemia_with_xanthelasma_axis": has_hyperlipidemia_with_xanthelasma_pattern,
    "water_aerosol_severe_pneumonia_axis": has_water_aerosol_severe_pneumonia_pattern,
    "pediatric_congenital_glaucoma_axis": has_pediatric_congenital_glaucoma_pattern,
    "high_energy_hindfoot_trauma_axis": has_high_energy_hindfoot_trauma_pattern,
    "traumatic_left_rib_fracture_axis": has_traumatic_rib_fracture_pattern,
    "topical_steroid_perioral_dermatitis_axis": has_perioral_dermatitis_pattern,
    "congenital_syndactyly_axis": has_congenital_syndactyly_pattern,
    "seafood_acute_watery_diarrhea_axis": has_seafood_acute_watery_diarrhea_pattern,
    "chronic_suppurative_middle_ear_axis": has_chronic_suppurative_middle_ear_pattern,
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


def _axis_rules() -> list[dict]:
    return [rule for rule in _payload()["rules"] if rule["rule_id"] in HELPERS]


def _typed_fires(pack, rule_id: str, text: str) -> bool:
    facts = diagnosis_rule_fact_codes(_state(text))
    result = apply_rules(pack, "diagnosis_candidates", RuleContext(fact_codes=facts))
    decisions = {decision.rule_id: decision for decision in result.decisions}
    return decisions[rule_id].outcome in {"applied", "matched_no_change"}


def test_batch_has_exactly_ten_axis_rules_with_three_positive_and_negative_controls() -> None:
    rules = _axis_rules()
    assert len(rules) == 10
    assert {rule["rule_id"] for rule in rules} == set(HELPERS)
    for rule in rules:
        assert len(rule["positive_controls"]) >= 3
        assert len(rule["negative_controls"]) >= 3


def test_batch_shadow_parity_all_controls() -> None:
    pack = parse_compiled_rule_pack(_payload())
    mismatches = []
    for rule in _axis_rules():
        rule_id = rule["rule_id"]
        helper = HELPERS[rule_id]
        for control in [*rule["positive_controls"], *rule["negative_controls"]]:
            text = " ".join(control["facts"])
            legacy = bool(helper(text)) or bool(helper(normalize_name(text)))
            typed = _typed_fires(pack, rule_id, text)
            expected = control["kind"] == "positive"
            if legacy != typed or typed != expected:
                mismatches.append(
                    {
                        "rule_id": rule_id,
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
        ("小腿红肿热痛伴发热寒战", "acute_lower_extremity_soft_tissue_infection", "蜂窝织炎"),
        ("42岁上眼睑黄色斑块，总胆固醇268", "hyperlipidemia_with_xanthelasma", "混合型高脂血症"),
        ("冷却塔暴露后高热咳嗽气短", "water_aerosol_severe_pneumonia_pathogen", "军团菌病"),
        ("婴儿眼压升高畏光流泪角膜混浊", "pediatric_congenital_glaucoma", "先天性青光眼"),
        ("高处坠落后足跟剧痛肿胀不能负重", "high_energy_hindfoot_trauma", "跟骨骨折"),
        ("跌倒撞到左肋后深呼吸胸痛", "traumatic_left_rib_fracture", "肋骨骨折"),
        ("面部用激素药膏后口周红斑丘疹灼热", "topical_steroid_associated_perioral_dermatitis", "口周皮炎"),
        ("孩子出生即手指长在一起", "congenital_syndactyly", "并指（趾）畸形"),
        ("生吃生蚝后突然腹痛呕吐水样腹泻", "seafood_acute_watery_diarrhea_pathogen", "副溶血性弧菌食物中毒"),
        ("反复三个月耳流脓伴听力下降", "chronic_suppurative_middle_ear", "化脓性中耳炎"),
    ],
)
def test_batch_reaches_runtime_and_materializes_axis(
    text: str,
    axis_id: str,
    candidate: str,
) -> None:
    pack = parse_compiled_rule_pack(_payload())
    candidates = [{"disease": candidate, "score": 50, "source": "unit"}]
    ordered, result = apply_diagnosis_candidate_rules(
        candidates,
        case_state=_state(text),
        official_diseases=[candidate],
        rule_pack=pack,
    )
    assert ordered[0]["disease"] == candidate
    axes = materialize_diagnosis_rule_axes([], result)
    axis = next(item for item in axes if item["axis_id"] == axis_id)
    assert candidate in axis["candidate_official_names"]
