"""T11/T12: clinical patterns may only migrate into closed, data-only opcodes.

The typed engine is the boundary that keeps knowledge declarative. A candidate
must never be able to smuggle a regex, a Python expression, a module name or an
unknown opcode through it, and an unsealed pack must never execute at all.
"""
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.knowledge.typed_rule_engine import (
    FACT_GROUP_OPCODE,
    FACT_GROUP_PARAMETER_FIELDS,
    PATTERN_FACT_CODES,
    CompiledRulePack,
    RuleContext,
    ShadowRuleReceipt,
    apply_rules,
    parse_compiled_rule_pack,
    shadow_rule_receipt,
)


def _content_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _fact_group_rule(
    *,
    rule_id: str = "acute_limb_soft_tissue_infection_pattern",
    all_groups: List[List[str]] | None = None,
    any_groups: List[List[str]] | None = None,
    excluded_groups: List[List[str]] | None = None,
    matched_fact_code: str = "acute_limb_soft_tissue_infection",
    priority: int = 10,
    active: bool = True,
) -> Dict[str, Any]:
    runtime: Dict[str, Any]
    if active:
        runtime = {
            "status": "active",
            "stage": "clinical_closure",
            "opcode": FACT_GROUP_OPCODE,
            "parameters": {
                "all_groups": (
                    all_groups
                    if all_groups is not None
                    else [["acute_limb_soft_tissue_infection"]]
                ),
                "any_groups": any_groups or [],
                "excluded_groups": excluded_groups or [],
                "matched_fact_code": matched_fact_code,
            },
        }
    else:
        runtime = {"status": "audit_only", "stage": "clinical_closure"}
    return {
        "rule_id": rule_id,
        "candidate_type": "clinical_closure_rule",
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["structured trigger for audit"],
        "required_evidence": ["objective finding"],
        "exclusions": ["documented exclusion"],
        "effect": {
            "add_exam_intent_ids": ["exam_intent_organ_involvement"],
            "deduplicate": "intent_id",
        },
        "positive_controls": [
            {
                "control_id": rule_id + "_positive",
                "kind": "positive",
                "facts": ["supported presentation"],
                "assertions": ["expected bounded behavior"],
            }
        ],
        "negative_controls": [
            {
                "control_id": rule_id + "_neighbor",
                "kind": "near_neighbor",
                "facts": ["nearby presentation"],
                "assertions": ["rule remains bounded"],
            }
        ],
        "source_refs": [{"path": "docs/not-present.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/not-present.py", "sha256": "d" * 64}],
        "priority": priority,
        "scope": {"phase": "closure", "application": "trigger_bound"},
        "runtime": runtime,
    }


def _pack(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    compiled = deepcopy(rules)
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": compiled,
        "rule_count": len(compiled),
        "rules_hash": _content_hash(compiled),
    }


# --- Step 1: the schema must refuse executable content ------------------------


@pytest.mark.parametrize(
    "poison",
    [
        "re.compile",
        "__import__",
        "eval",
        ".*发热.*",
        "agent.legacy_orchestrator",
        "has_acute_lower_extremity_soft_tissue_infection_pattern",
        "lambda x: True",
    ],
)
def test_regex_and_python_expressions_are_rejected(poison: str) -> None:
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([_fact_group_rule(all_groups=[[poison]])]))


def test_unknown_fact_code_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(
            _pack([_fact_group_rule(all_groups=[["not_a_registered_fact"]])])
        )


def test_unknown_opcode_is_rejected() -> None:
    rule = _fact_group_rule()
    rule["runtime"]["opcode"] = "exec_arbitrary_python"
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_parameter_fields_are_a_closed_whitelist() -> None:
    assert FACT_GROUP_PARAMETER_FIELDS == frozenset(
        {"all_groups", "any_groups", "excluded_groups", "matched_fact_code"}
    )
    rule = _fact_group_rule()
    rule["runtime"]["parameters"]["extra"] = ["x"]
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_matched_fact_code_must_be_registered() -> None:
    rule = _fact_group_rule(matched_fact_code="unregistered_effect_code")
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_pattern_fact_codes_cover_the_first_migration_batch() -> None:
    for code in (
        "acute_limb_soft_tissue_infection",
        "hyperlipidemia_with_xanthelasma",
        "severe_pneumonia_aerosol_exposure",
        "pediatric_congenital_glaucoma",
        "high_energy_hindfoot_trauma",
    ):
        assert code in PATTERN_FACT_CODES


def test_unsealed_pack_is_not_executable() -> None:
    """A hand-built pack object that never went through the parser must not run."""
    forged = CompiledRulePack.__new__(CompiledRulePack)
    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(forged, "clinical_closure", RuleContext())


# --- Step 2: match_fact_groups evaluates data only ----------------------------


def _matched(rule: Dict[str, Any], fact_codes: tuple[str, ...]) -> tuple[str, ...]:
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(
        pack,
        "clinical_closure",
        RuleContext(fact_codes=fact_codes),
    )
    return result.output_context.fact_codes


def test_all_groups_requires_every_group_to_fire() -> None:
    rule = _fact_group_rule(
        all_groups=[
            ["acute_limb_soft_tissue_infection"],
            ["severe_pneumonia_aerosol_exposure"],
        ],
        matched_fact_code="high_energy_hindfoot_trauma",
    )
    partial = _matched(rule, ("acute_limb_soft_tissue_infection",))
    assert "high_energy_hindfoot_trauma" not in partial

    complete = _matched(
        rule,
        ("acute_limb_soft_tissue_infection", "severe_pneumonia_aerosol_exposure"),
    )
    assert "high_energy_hindfoot_trauma" in complete


def test_any_groups_requires_one_group_to_fire() -> None:
    rule = _fact_group_rule(
        all_groups=[],
        any_groups=[
            ["hyperlipidemia_with_xanthelasma"],
            ["pediatric_congenital_glaucoma"],
        ],
        matched_fact_code="high_energy_hindfoot_trauma",
    )
    assert "high_energy_hindfoot_trauma" not in _matched(rule, ("fever",))
    assert "high_energy_hindfoot_trauma" in _matched(
        rule, ("pediatric_congenital_glaucoma",)
    )


def test_excluded_groups_block_the_match() -> None:
    rule = _fact_group_rule(
        all_groups=[["acute_limb_soft_tissue_infection"]],
        excluded_groups=[["drug_allergy"]],
        matched_fact_code="high_energy_hindfoot_trauma",
    )
    assert "high_energy_hindfoot_trauma" in _matched(
        rule, ("acute_limb_soft_tissue_infection",)
    )
    assert "high_energy_hindfoot_trauma" not in _matched(
        rule, ("acute_limb_soft_tissue_infection", "drug_allergy")
    )


def test_a_group_is_satisfied_by_any_member() -> None:
    rule = _fact_group_rule(
        all_groups=[["fever", "acute_limb_soft_tissue_infection"]],
        matched_fact_code="high_energy_hindfoot_trauma",
    )
    assert "high_energy_hindfoot_trauma" in _matched(rule, ("fever",))
    assert "high_energy_hindfoot_trauma" in _matched(
        rule, ("acute_limb_soft_tissue_infection",)
    )


def test_audit_only_rule_never_changes_the_context() -> None:
    # A pack must contain at least one active rule; the active peer is a no-op
    # for this fixture, so only the audit-only rule is under test.
    active = _fact_group_rule(
        rule_id="peer_active_pattern",
        all_groups=[["fever"]],
        matched_fact_code="fever",
        priority=5,
        active=True,
    )
    audit = _fact_group_rule(rule_id="audit_only_pattern", active=False, priority=20)
    pack = parse_compiled_rule_pack(_pack([active, audit]))
    result = apply_rules(
        pack,
        "clinical_closure",
        RuleContext(fact_codes=("acute_limb_soft_tissue_infection",)),
    )
    assert result.output_context.fact_codes == ("acute_limb_soft_tissue_infection",)
    assert all(d.outcome != "applied" for d in result.decisions)


def test_match_is_idempotent_and_does_not_duplicate_codes() -> None:
    rule = _fact_group_rule(
        all_groups=[["acute_limb_soft_tissue_infection"]],
        matched_fact_code="acute_limb_soft_tissue_infection",
    )
    codes = _matched(rule, ("acute_limb_soft_tissue_infection",))
    assert codes.count("acute_limb_soft_tissue_infection") == 1


def test_no_rule_with_empty_groups_can_fire_unconditionally() -> None:
    rule = _fact_group_rule(all_groups=[], any_groups=[], excluded_groups=[])
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


# --- Step 4: shadow receipts make parity auditable ---------------------------


def test_shadow_receipt_reports_equivalence() -> None:
    same = shadow_rule_receipt(rule_id="r1", legacy_result=True, typed_result=True)
    assert isinstance(same, ShadowRuleReceipt)
    assert same.equivalent is True
    assert same.legacy_result_hash == same.typed_result_hash

    differing = shadow_rule_receipt(rule_id="r1", legacy_result=True, typed_result=False)
    assert differing.equivalent is False
    assert differing.legacy_result_hash != differing.typed_result_hash


def test_shadow_receipt_hashes_are_stable() -> None:
    first = shadow_rule_receipt(rule_id="r1", legacy_result=True, typed_result=True)
    second = shadow_rule_receipt(rule_id="r1", legacy_result=True, typed_result=True)
    assert first == second


# --- T12 shadow parity: legacy pattern vs typed rule on fact codes ----------

from agent.legacy_orchestrator import (  # noqa: E402
    diagnosis_rule_fact_codes,
    has_acute_lower_extremity_soft_tissue_infection_pattern,
    has_high_energy_hindfoot_trauma_pattern,
    has_hyperlipidemia_with_xanthelasma_pattern,
    has_pediatric_congenital_glaucoma_pattern,
    has_water_aerosol_severe_pneumonia_pattern,
)


def _case_state_from_text(text: str) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }


def _typed_fires(rule: dict, fact_codes: tuple[str, ...]) -> bool:
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(pack, "clinical_closure", RuleContext(fact_codes=fact_codes))
    return result.output_context.fact_codes != fact_codes


def test_acute_limb_soft_tissue_infection_parity() -> None:
    rule = _fact_group_rule(
        rule_id="acute_limb_soft_tissue_infection_pattern",
        all_groups=[["limb_swelling"], ["skin_redness_heat"], ["fever"]],
        matched_fact_code="acute_limb_soft_tissue_infection",
    )
    positives = [
        "小腿红肿热痛，伴发热寒战",
        "足背肿胀皮温升高，高热39度",
        "下肢蜂窝织炎，皮肤发红发烫，高热",
    ]
    negatives = [
        "小腿肿胀但无发热",
        "发热但无下肢局部体征",
        "上肢红肿热痛伴发热",
    ]
    for text in positives:
        legacy = has_acute_lower_extremity_soft_tissue_infection_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is True, f"legacy should fire for: {text}"
        assert typed is True, f"typed should fire for: {text} (facts={facts})"
    for text in negatives:
        legacy = has_acute_lower_extremity_soft_tissue_infection_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is False, f"legacy should not fire for: {text}"
        assert typed is False, f"typed should not fire for: {text} (facts={facts})"


def test_hyperlipidemia_with_xanthelasma_parity() -> None:
    rule = _fact_group_rule(
        rule_id="hyperlipidemia_with_xanthelasma_pattern",
        all_groups=[["xanthelasma"], ["lipid_panel_abnormal", "lab_lipid"], ["adult", "lab_lipid"]],
        matched_fact_code="hyperlipidemia_with_xanthelasma",
        excluded_groups=[["noninfectious_eczema"]],
    )
    positives = [
        "42岁上眼睑黄色斑块，总胆固醇268，LDL172",
        "眼睑睑黄瘤，血脂偏高，50岁",
        "上眼睑发黄，高脂血症病史，60岁",
    ]
    negatives = [
        "2岁佝偻病方颅鸡胸，无眼睑斑块",
        "眼睑黄色斑块，但血脂正常且儿童",
        "否认眼睑斑块，仅有血脂异常",
    ]
    for text in positives:
        legacy = has_hyperlipidemia_with_xanthelasma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is True, f"legacy should fire for: {text}"
        assert typed is True, f"typed should fire for: {text} (facts={facts})"
    for text in negatives:
        legacy = has_hyperlipidemia_with_xanthelasma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is False, f"legacy should not fire for: {text}"
        assert typed is False, f"typed should not fire for: {text} (facts={facts})"


def test_water_aerosol_severe_pneumonia_parity() -> None:
    rule = _fact_group_rule(
        rule_id="water_aerosol_severe_pneumonia_pattern",
        all_groups=[["aerosol_water_exposure"], ["fever"], ["respiratory_symptom"], ["respiratory_failure"]],
        matched_fact_code="severe_pneumonia_aerosol_exposure",
    )
    positives = [
        "冷却塔暴露，高热寒战，咳嗽气短",
        "酒店热水淋浴，发热肌肉酸痛，咳嗽，呼吸困难",
        "集中空调冷却水暴露，高热，咳嗽，胸痛持续加重",
    ]
    negatives = [
        "高热咳嗽但无暴露史",
        "冷却塔暴露但无呼吸道症状",
        "社区获得性肺炎，高热咳嗽气短",
    ]
    for text in positives:
        legacy = has_water_aerosol_severe_pneumonia_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is True, f"legacy should fire for: {text}"
        assert typed is True, f"typed should fire for: {text} (facts={facts})"
    for text in negatives:
        legacy = has_water_aerosol_severe_pneumonia_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is False, f"legacy should not fire for: {text}"
        assert typed is False, f"typed should not fire for: {text} (facts={facts})"


def test_pediatric_congenital_glaucoma_parity() -> None:
    rule = _fact_group_rule(
        rule_id="pediatric_congenital_glaucoma_pattern",
        all_groups=[["pediatric_patient"], ["high_pressure"], ["infant_photophobia_tearing", "corneal_enlargement"]],
        matched_fact_code="pediatric_congenital_glaucoma",
    )
    positives = [
        "婴儿眼压升高，畏光流泪，角膜混浊",
        "幼儿眼压高，眼球增大，视力下降",
        "儿童眼压32，畏光，牛眼",
    ]
    negatives = [
        "成人眼压升高伴畏光",
        "儿童畏光流泪但眼压正常",
        "否认眼压升高，儿童结膜炎",
    ]
    for text in positives:
        legacy = has_pediatric_congenital_glaucoma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is True, f"legacy should fire for: {text}"
        assert typed is True, f"typed should fire for: {text} (facts={facts})"
    for text in negatives:
        legacy = has_pediatric_congenital_glaucoma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is False, f"legacy should not fire for: {text}"
        assert typed is False, f"typed should not fire for: {text} (facts={facts})"


def test_high_energy_hindfoot_trauma_parity() -> None:
    rule = _fact_group_rule(
        rule_id="high_energy_hindfoot_trauma_pattern",
        all_groups=[["high_energy_trauma"], ["hindfoot_deformity"]],
        matched_fact_code="high_energy_hindfoot_trauma",
    )
    positives = [
        "高处坠落，足跟剧痛肿胀，不能负重",
        "车祸，脚跟瘀斑，不敢踩地",
        "从高处掉下来，脚后跟肿胀，无法行走",
    ]
    negatives = [
        "足跟痛但无外伤史",
        "高处坠落但足部正常",
        "否认高处坠落，足跟扭伤",
    ]
    for text in positives:
        legacy = has_high_energy_hindfoot_trauma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is True, f"legacy should fire for: {text}"
        assert typed is True, f"typed should fire for: {text} (facts={facts})"
    for text in negatives:
        legacy = has_high_energy_hindfoot_trauma_pattern(text)
        facts = diagnosis_rule_fact_codes(_case_state_from_text(text))
        typed = _typed_fired(rule, facts)
        assert legacy is False, f"legacy should not fire for: {text}"
        assert typed is False, f"typed should not fire for: {text} (facts={facts})"


def _typed_fired(rule: dict, fact_codes: tuple[str, ...]) -> bool:
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(pack, "clinical_closure", RuleContext(fact_codes=fact_codes))
    return result.output_context.fact_codes != fact_codes
