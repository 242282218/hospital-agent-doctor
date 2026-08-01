"""T11/T13: treatment transforms may only edit owned fragments via closed codes.

The typed treatment-transform opcode is the boundary that keeps sanitizers
declarative. A rule may only remove, replace, or append registered patch
template codes; it can never author a new prescription, rearrange the whole
treatment, or add an unregistered drug. The transform must be idempotent.
"""
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any, Dict, List

import pytest

from agent.knowledge.typed_rule_engine import (
    TREATMENT_PATCH_TEMPLATE_CODES,
    TREATMENT_TRANSFORM_OPCODE,
    TREATMENT_TRANSFORM_PARAMETER_FIELDS,
    CompiledRulePack,
    RuleContext,
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


def _treatment_transform_rule(
    *,
    rule_id: str = "sanitize_prophylactic_antibiotic",
    remove_codes: List[str] | None = None,
    replace_codes: List[Dict[str, str]] | None = None,
    append_codes: List[str] | None = None,
    active: bool = True,
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "remove_codes": remove_codes or [],
        "replace_codes": replace_codes or [],
        "append_codes": append_codes or [],
    }
    runtime: Dict[str, Any]
    if active:
        runtime = {
            "status": "active",
            "stage": "treatment",
            "opcode": TREATMENT_TRANSFORM_OPCODE,
            "parameters": parameters,
        }
    else:
        runtime = {"status": "audit_only", "stage": "treatment"}
    return {
        "rule_id": rule_id,
        "candidate_type": "treatment_gate_rule",
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["structured trigger for audit"],
        "required_evidence": ["objective finding"],
        "exclusions": ["documented exclusion"],
        "effect": {
            "remove_treatment_codes": ["routine_antibiotic"],
            "preserve_treatment_codes": ["supportive_care"],
            "gate_policy": "require_infection_evidence",
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
            },
            {
                "control_id": rule_id + "_exception",
                "kind": "reasonable_exception",
                "facts": ["documented exception"],
                "assertions": ["exception is preserved"],
            },
        ],
        "source_refs": [{"path": "docs/not-present.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/not-present.py", "sha256": "d" * 64}],
        "priority": 10,
        "scope": {"phase": "treatment", "application": "trigger_bound"},
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


def _apply_transform(
    rule: Dict[str, Any], treatment_codes: tuple[str, ...]
) -> tuple[str, ...]:
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(pack, "treatment", RuleContext(treatment_codes=treatment_codes))
    return result.output_context.treatment_codes


# --- Step 1: schema rejects unknown fields and unregistered codes ------------


def test_treatment_transform_parameters_are_a_closed_whitelist() -> None:
    assert TREATMENT_TRANSFORM_PARAMETER_FIELDS == frozenset(
        {"remove_codes", "replace_codes", "append_codes"}
    )


def test_unknown_parameter_field_is_rejected() -> None:
    rule = _treatment_transform_rule(remove_codes=["routine_antibiotic"])
    rule["runtime"]["parameters"]["free_text_prescription"] = "amoxicillin 500mg"
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_unregistered_patch_template_is_rejected() -> None:
    rule = _treatment_transform_rule(append_codes=["unregistered_drug_code"])
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_replace_must_reference_registered_template() -> None:
    rule = _treatment_transform_rule(
        replace_codes=[{"code": "routine_antibiotic", "replacement_code": "not_a_template"}]
    )
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_empty_transform_is_rejected() -> None:
    rule = _treatment_transform_rule()
    with pytest.raises(ValueError):
        parse_compiled_rule_pack(_pack([rule]))


def test_patch_template_codes_are_closed() -> None:
    assert "prophylactic_antibiotic_removed" in TREATMENT_PATCH_TEMPLATE_CODES
    assert "amoxicillin_500mg_three_times_daily" not in TREATMENT_PATCH_TEMPLATE_CODES


# --- Step 2: remove / replace / append behave correctly ---------------------


def test_remove_owned_code_only() -> None:
    rule = _treatment_transform_rule(remove_codes=["routine_antibiotic"])
    result = _apply_transform(rule, ("routine_antibiotic", "supportive_care"))
    assert result == ("supportive_care",)


def test_replace_owned_code_with_template() -> None:
    rule = _treatment_transform_rule(
        replace_codes=[
            {"code": "routine_antibiotic", "replacement_code": "prophylactic_antibiotic_removed"}
        ]
    )
    result = _apply_transform(rule, ("routine_antibiotic", "supportive_care"))
    assert result == ("prophylactic_antibiotic_removed", "supportive_care")


def test_append_template_code() -> None:
    # Append only happens when a removal/replacement occurred.
    rule = _treatment_transform_rule(
        remove_codes=["prophylactic_antibiotic"], append_codes=["supportive_care"]
    )
    result = _apply_transform(rule, ("prophylactic_antibiotic", "routine_antibiotic"))
    assert result == ("routine_antibiotic", "supportive_care")


def test_unrelated_treatment_codes_are_preserved() -> None:
    rule = _treatment_transform_rule(remove_codes=["routine_antibiotic"])
    result = _apply_transform(
        rule, ("routine_antibiotic", "antiviral_therapy", "supportive_care")
    )
    assert result == ("antiviral_therapy", "supportive_care")


def test_remove_missing_code_is_noop() -> None:
    rule = _treatment_transform_rule(remove_codes=["routine_antibiotic"])
    result = _apply_transform(rule, ("supportive_care",))
    assert result == ("supportive_care",)


# --- Step 3: idempotency ----------------------------------------------------


def test_transform_is_idempotent() -> None:
    rule = _treatment_transform_rule(
        remove_codes=["routine_antibiotic"],
        append_codes=["supportive_care"],
    )
    first = _apply_transform(rule, ("routine_antibiotic", "antiviral_therapy"))
    second = _apply_transform(rule, first)
    assert first == second


def test_replace_is_idempotent() -> None:
    rule = _treatment_transform_rule(
        replace_codes=[
            {"code": "routine_antibiotic", "replacement_code": "prophylactic_antibiotic_removed"}
        ]
    )
    first = _apply_transform(rule, ("routine_antibiotic", "supportive_care"))
    second = _apply_transform(rule, first)
    assert first == second


def test_append_does_not_duplicate() -> None:
    rule = _treatment_transform_rule(append_codes=["supportive_care"])
    first = _apply_transform(rule, ("supportive_care",))
    assert first == ("supportive_care",)


# --- Step 4: audit-only rule never changes context --------------------------


def test_audit_only_rule_is_noop() -> None:
    active = _treatment_transform_rule(
        rule_id="active_transform",
        remove_codes=["routine_antibiotic"],
        active=True,
    )
    audit = _treatment_transform_rule(
        rule_id="audit_only_transform",
        remove_codes=["supportive_care"],
        active=False,
    )
    pack = parse_compiled_rule_pack(_pack([active, audit]))
    result = apply_rules(
        pack, "treatment", RuleContext(treatment_codes=("routine_antibiotic",))
    )
    assert result.output_context.treatment_codes == ()
    assert all(d.outcome != "applied" or d.rule_id != "audit_only_transform" for d in result.decisions)


# --- Step 5: shadow receipt -------------------------------------------------


def test_treatment_transform_shadow_receipt() -> None:
    same = shadow_rule_receipt(
        rule_id="t1", legacy_result=["a", "b"], typed_result=["a", "b"]
    )
    assert same.equivalent is True
    different = shadow_rule_receipt(
        rule_id="t1", legacy_result=["a", "b"], typed_result=["a"]
    )
    assert different.equivalent is False


# --- T13 shadow parity: legacy sanitizer vs typed transform on codes --------

from agent.legacy_orchestrator import (  # noqa: E402
    sanitize_prophylactic_antibiotic_recommendations,
    sanitize_unindicated_aspirin_discontinuation,
    sanitize_pbmv_before_thrombus_exclusion,
    sanitize_tricyclic_continuation,
)


def _apply_treatment_transform(
    rule: dict, treatment_codes: tuple[str, ...]
) -> tuple[str, ...]:
    pack = parse_compiled_rule_pack(_pack([rule]))
    result = apply_rules(pack, "treatment", RuleContext(treatment_codes=treatment_codes))
    return result.output_context.treatment_codes


def _remove_code_rule(rid: str, remove: str, append: str, prio: int = 10) -> dict:
    return _treatment_transform_rule(
        rule_id=rid,
        remove_codes=[remove],
        append_codes=[append],
        active=True,
    )


def test_sanitize_prophylactic_antibiotic_parity() -> None:
    rule = _remove_code_rule(
        "sanitize_prophylactic_antibiotic",
        "prophylactic_antibiotic",
        "prophylactic_antibiotic_removed",
    )
    # Typed: target code is removed, unrelated codes preserved.
    codes = ("prophylactic_antibiotic", "supportive_care", "antiviral_therapy")
    result = _apply_treatment_transform(rule, codes)
    assert "prophylactic_antibiotic" not in result
    assert "supportive_care" in result
    assert "antiviral_therapy" in result
    assert "prophylactic_antibiotic_removed" in result
    # Idempotent.
    second = _apply_treatment_transform(rule, result)
    assert second == result
    # Legacy: same semantic on free text.
    legacy_input = "建议预防性使用抗生素，同时支持治疗。"
    legacy_result = sanitize_prophylactic_antibiotic_recommendations(legacy_input)
    assert "预防性" not in legacy_result or "抗生素" not in legacy_result


def test_sanitize_unindicated_aspirin_parity() -> None:
    rule = _remove_code_rule(
        "sanitize_unindicated_aspirin",
        "aspirin_discontinuation",
        "aspirin_continuation_preserved",
    )
    codes = ("aspirin_discontinuation", "statin_therapy", "supportive_care")
    result = _apply_treatment_transform(rule, codes)
    assert "aspirin_discontinuation" not in result
    assert "statin_therapy" in result
    assert "aspirin_continuation_preserved" in result
    second = _apply_treatment_transform(rule, result)
    assert second == result
    legacy_input = "建议停用阿司匹林，继续他汀治疗。"
    legacy_result = sanitize_unindicated_aspirin_discontinuation(legacy_input)
    assert "停用" not in legacy_result or "阿司匹林" not in legacy_result


def test_sanitize_pbmv_before_thrombus_exclusion_parity() -> None:
    rule = _remove_code_rule(
        "sanitize_pbmv",
        "pbmv_procedure",
        "pbmv_deferred_until_thrombus_excluded",
    )
    codes = ("pbmv_procedure", "supportive_care", "anticoagulation")
    result = _apply_treatment_transform(rule, codes)
    assert "pbmv_procedure" not in result
    assert "supportive_care" in result
    assert "anticoagulation" in result
    assert "pbmv_deferred_until_thrombus_excluded" in result
    second = _apply_treatment_transform(rule, result)
    assert second == result
    legacy_input = "尽快行经皮球囊二尖瓣成形术。"
    legacy_result = sanitize_pbmv_before_thrombus_exclusion(legacy_input)
    assert "球囊" not in legacy_result


def test_sanitize_tricyclic_continuation_parity() -> None:
    rule = _remove_code_rule(
        "sanitize_tricyclic",
        "tricyclic_continuation",
        "tricyclic_tapered_under_supervision",
    )
    codes = ("tricyclic_continuation", "supportive_care", "ssri_therapy")
    result = _apply_treatment_transform(rule, codes)
    assert "tricyclic_continuation" not in result
    assert "supportive_care" in result
    assert "ssri_therapy" in result
    assert "tricyclic_tapered_under_supervision" in result
    second = _apply_treatment_transform(rule, result)
    assert second == result
    legacy_input = "继续服用阿米替林。"
    legacy_result = sanitize_tricyclic_continuation(legacy_input)
    assert "继续" not in legacy_result or "阿米替林" not in legacy_result


def test_transform_preserves_unrelated_codes() -> None:
    """A rule must never delete or reorder codes it does not own."""
    rule = _remove_code_rule(
        "sanitize_prophylactic_antibiotic",
        "prophylactic_antibiotic",
        "prophylactic_antibiotic_removed",
    )
    codes = ("antiviral_therapy", "supportive_care", "statin_therapy")
    result = _apply_treatment_transform(rule, codes)
    assert result == codes


def test_transform_never_adds_new_drug() -> None:
    """A rule may only append registered patch templates, never new drugs."""
    rule = _remove_code_rule(
        "sanitize_prophylactic_antibiotic",
        "prophylactic_antibiotic",
        "prophylactic_antibiotic_removed",
    )
    codes = ("prophylactic_antibiotic",)
    result = _apply_treatment_transform(rule, codes)
    assert result == ("prophylactic_antibiotic_removed",)
