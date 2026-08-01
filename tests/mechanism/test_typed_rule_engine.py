from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json

import pytest
import agent.knowledge.typed_rule_engine as typed_rule_engine

from agent.knowledge.typed_rule_engine import (
    CompiledRule,
    CompiledRulePack,
    RuleContext,
    RuleDecision,
    RuleDiagnosisCandidate,
    RuleResult,
    apply_rules,
    empty_compiled_rule_pack,
    parse_compiled_rule_pack,
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


_PHASES = {
    "diagnosis_differential_rule": "diagnosis",
    "clinical_closure_rule": "closure",
    "diagnosis_priority_rule": "diagnosis",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}
_STAGES = {
    "diagnosis_differential_rule": "diagnosis_candidates",
    "clinical_closure_rule": "clinical_closure",
    "diagnosis_priority_rule": "diagnosis_candidates",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}


def _effect(candidate_type: str) -> dict[str, object]:
    effects: dict[str, dict[str, object]] = {
        "diagnosis_differential_rule": {
            "add_diagnostic_axes": ["infectious_axis"],
            "ranking_policy": "evidence_ordered",
        },
        "clinical_closure_rule": {
            "add_exam_intent_ids": ["exam_intent_organ_involvement"],
            "deduplicate": "intent_id",
        },
        "diagnosis_priority_rule": {
            "priority_policy": "objective_evidence_first",
            "fallback_policy": "official_catalog_only",
        },
        "treatment_gate_rule": {
            "remove_treatment_codes": ["routine_antibiotic"],
            "preserve_treatment_codes": ["supportive_care"],
            "gate_policy": "require_infection_evidence",
        },
        "treatment_sequence_rule": {
            "ordered_patch_codes": ["stabilize_airway"],
            "sequence_policy": "acute_before_stable",
            "skip_patch_codes": ["elective_followup"],
        },
    }
    return deepcopy(effects[candidate_type])


def _active_runtime() -> dict[str, object]:
    return {
        "status": "active",
        "stage": "diagnosis_candidates",
        "opcode": "promote_supported_current_over_background",
        "parameters": {
            "target_roles": ["current_problem"],
            "target_support_levels": ["objective"],
            "background_roles": ["background_condition"],
            "background_relations": ["unrelated"],
            "excluded_relations": ["explains"],
            "preserve_urgencies": ["emergency"],
            "fallback_policy": "official_catalog_only",
        },
    }


DIFFERENTIAL_PARAMETERS = {
    "age_fact_codes": ["neonate", "infant"],
    "exposure_fact_codes": ["intrauterine_viral_exposure"],
    "manifestation_fact_codes": [
        "congenital_jaundice",
        "congenital_rash",
        "infant_hearing_abnormality",
        "thrombocytopenia",
        "congenital_neuroimaging_abnormality",
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "periventricular_calcifications",
        "microcephaly",
    ],
    "rubella_rank_fact_codes": [
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "rubella_igm_positive_in_infant",
        "rubella_pcr_positive_in_infant",
    ],
    "cmv_rank_fact_codes": [
        "periventricular_calcifications",
        "microcephaly",
        "cmv_igm_positive",
        "cmv_pcr_positive",
        "cmv_saliva_or_urine_pcr_positive_within_21_days",
    ],
    "rubella_confirmed_fact_codes": [
        "rubella_igm_positive_in_infant",
        "rubella_pcr_positive_in_infant",
    ],
    "cmv_confirmed_fact_codes": [
        "cmv_saliva_or_urine_pcr_positive_within_21_days"
    ],
    "rubella_axis_id": "congenital_rubella",
    "cmv_axis_id": "congenital_cmv",
}


def _differential_runtime() -> dict[str, object]:
    return {
        "status": "active",
        "stage": "diagnosis_candidates",
        "opcode": "expand_congenital_infection_axes",
        "parameters": deepcopy(DIFFERENTIAL_PARAMETERS),
    }


def _compiled_rule(
    *,
    candidate_type: str = "diagnosis_priority_rule",
    rule_id: str = "prefer_supported_current_problem",
    priority: int = 10,
    active: bool = True,
) -> dict[str, object]:
    negative_controls = [
        {
            "control_id": rule_id + "_neighbor",
            "kind": "near_neighbor",
            "facts": ["nearby presentation"],
            "assertions": ["rule remains bounded"],
        }
    ]
    if candidate_type in {"treatment_gate_rule", "treatment_sequence_rule"}:
        negative_controls.append(
            {
                "control_id": rule_id + "_exception",
                "kind": "reasonable_exception",
                "facts": ["documented exception"],
                "assertions": ["exception is preserved"],
            }
        )
    runtime = (
        _active_runtime()
        if active
        else {"status": "audit_only", "stage": _STAGES[candidate_type]}
    )
    return {
        "rule_id": rule_id,
        "candidate_type": candidate_type,
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["structured trigger for audit"],
        "required_evidence": ["objective finding"],
        "exclusions": ["documented exclusion"],
        "effect": _effect(candidate_type),
        "positive_controls": [
            {
                "control_id": rule_id + "_positive",
                "kind": "positive",
                "facts": ["supported presentation"],
                "assertions": ["expected bounded behavior"],
            }
        ],
        "negative_controls": negative_controls,
        "source_refs": [{"path": "docs/not-present.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/not-present.py", "sha256": "d" * 64}],
        "priority": priority,
        "scope": {"phase": _PHASES[candidate_type], "application": "trigger_bound"},
        "runtime": runtime,
    }


def _differential_rule(
    *,
    rule_id: str = "expand_congenital_infection_differential",
    priority: int = 10,
) -> dict[str, object]:
    rule = _compiled_rule(
        candidate_type="diagnosis_differential_rule",
        rule_id=rule_id,
        priority=priority,
        active=False,
    )
    rule["effect"] = {
        "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
        "ranking_policy": "evidence_ordered",
    }
    rule["runtime"] = _differential_runtime()
    return rule


def _compiled_pack(rules: list[dict[str, object]] | None = None) -> dict[str, object]:
    compiled_rules = deepcopy(rules if rules is not None else [_compiled_rule()])
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": compiled_rules,
        "rule_count": len(compiled_rules),
        "rules_hash": _content_hash(compiled_rules),
    }


def _rehash_pack(payload: dict[str, object]) -> None:
    rules = payload["rules"]
    payload["rule_count"] = len(rules)  # type: ignore[arg-type]
    payload["rules_hash"] = _content_hash(rules)


def _diagnosis_candidate(**overrides: object) -> RuleDiagnosisCandidate:
    values: dict[str, object] = {
        "official_name": "急性心肌梗死",
        "role": "current_problem",
        "support_level": "objective",
        "complaint_relation": "explains",
        "urgency": "emergency",
        "evidence_codes": ("ecg_st_elevation",),
        "is_official": True,
    }
    values.update(overrides)
    return RuleDiagnosisCandidate(**values)  # type: ignore[arg-type]


def test_rule_context_starts_empty() -> None:
    assert RuleContext() == RuleContext(
        diagnosis_candidates=(),
        preferred_diagnosis=None,
        diagnostic_axis_ids=(),
        exam_intent_ids=(),
        treatment_codes=(),
        fact_codes=(),
    )


def test_rule_data_models_are_frozen_typed_values() -> None:
    candidate = _diagnosis_candidate()
    context = RuleContext(
        diagnosis_candidates=(candidate,),
        preferred_diagnosis="急性心肌梗死",
        diagnostic_axis_ids=("acute_coronary_syndrome",),
        exam_intent_ids=("exam_intent_cardiac_injury",),
        treatment_codes=("activate_pci",),
        fact_codes=("ongoing_chest_pain",),
    )

    assert context.diagnosis_candidates == (candidate,)
    with pytest.raises(FrozenInstanceError):
        context.preferred_diagnosis = None  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_name", ""),
        ("role", "primary"),
        ("support_level", "likely"),
        ("complaint_relation", "maybe"),
        ("urgency", "critical"),
        ("evidence_codes", ("same", "same")),
        ("is_official", 1),
    ],
)
def test_diagnosis_candidate_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _diagnosis_candidate(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "diagnosis_candidates": (
                _diagnosis_candidate(),
                _diagnosis_candidate(role="differential"),
            )
        },
        {"diagnostic_axis_ids": ("duplicate", "duplicate")},
        {"exam_intent_ids": ("duplicate", "duplicate")},
        {"treatment_codes": ("duplicate", "duplicate")},
        {"fact_codes": ("duplicate", "duplicate")},
        {"diagnostic_axis_ids": ["mutable"]},
    ],
)
def test_rule_context_rejects_duplicate_or_mutable_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RuleContext(**overrides)  # type: ignore[arg-type]


def test_rule_result_reports_applied_rule_ids_once_in_decision_order() -> None:
    context = RuleContext()
    decisions = (
        RuleDecision("rule_a", "opcode", "applied", "changed"),
        RuleDecision("rule_b", "opcode", "not_matched", "trigger_missed"),
        RuleDecision("rule_a", "opcode", "applied", "changed_again"),
        RuleDecision("rule_c", "opcode", "applied", "changed"),
    )
    result = RuleResult(context, decisions, "0" * 64, "1" * 64)

    assert result.applied_rule_ids == ("rule_a", "rule_c")
    with pytest.raises(FrozenInstanceError):
        result.after_hash = "2" * 64  # type: ignore[misc]


def test_rule_decision_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        RuleDecision("rule_a", "opcode", "ignored", "reason")  # type: ignore[arg-type]


def test_empty_compiled_rule_pack_is_explicit_frozen_v2_value() -> None:
    pack = empty_compiled_rule_pack()

    assert isinstance(pack, CompiledRulePack)
    assert pack.schema_version == "compiled-knowledge-rules/v2"
    assert pack.rules == ()
    assert pack.rule_count == 0
    assert pack.rules_hash == _content_hash([])
    with pytest.raises(FrozenInstanceError):
        pack.rule_count = 1  # type: ignore[misc]


def test_compiled_models_cannot_be_constructed_outside_parser_or_factory() -> None:
    with pytest.raises(TypeError):
        CompiledRulePack()
    with pytest.raises(TypeError):
        CompiledRulePack(
            schema_version="compiled-knowledge-rules/v2",
            rules=[],  # type: ignore[arg-type]
            rule_count=0,
            rules_hash=_content_hash([]),
        )
    with pytest.raises(TypeError):
        CompiledRule(  # type: ignore[call-arg]
            rule_id="forged",
            candidate_type="diagnosis_priority_rule",
            candidate_hash="a" * 64,
            effect_hash="b" * 64,
            priority=0,
            phase="diagnosis",
            application="trigger_bound",
            runtime=None,
        )


@pytest.mark.parametrize(
    "stage",
    ["diagnosis_candidates", "clinical_closure", "treatment"],
)
def test_empty_compiled_rule_pack_is_a_hash_stable_noop(stage: str) -> None:
    context = RuleContext(
        diagnosis_candidates=(_diagnosis_candidate(),),
        preferred_diagnosis="急性心肌梗死",
        diagnostic_axis_ids=("acute_coronary_syndrome",),
        exam_intent_ids=("exam_intent_cardiac_injury",),
        treatment_codes=("activate_pci",),
        fact_codes=("ongoing_chest_pain",),
    )

    result = apply_rules(empty_compiled_rule_pack(), stage, context)  # type: ignore[arg-type]

    assert result.output_context == context
    assert result.decisions == ()
    assert result.before_hash == result.after_hash
    assert result.applied_rule_ids == ()


def test_parse_compiled_rule_pack_returns_an_immutable_typed_snapshot() -> None:
    payload = _compiled_pack()
    before = deepcopy(payload)

    pack = parse_compiled_rule_pack(payload)

    assert payload == before
    assert pack.schema_version == "compiled-knowledge-rules/v2"
    assert pack.rule_count == 1
    assert isinstance(pack.rules[0], CompiledRule)
    assert pack.rules[0].rule_id == "prefer_supported_current_problem"
    assert pack.rules[0].runtime.status == "active"
    assert pack.rules[0].runtime.parameters is not None
    assert pack.rules[0].runtime.parameters.target_roles == ("current_problem",)

    payload["rules"][0]["runtime"]["parameters"]["target_roles"][0] = "mutated"  # type: ignore[index]
    assert pack.rules[0].runtime.parameters.target_roles == ("current_problem",)
    with pytest.raises(FrozenInstanceError):
        pack.rules[0].priority = 999  # type: ignore[misc]


def test_parse_hash_is_invariant_to_mapping_insertion_order() -> None:
    payload = _compiled_pack()
    rule = payload["rules"][0]  # type: ignore[index]
    reordered_rule = dict(reversed(list(rule.items())))  # type: ignore[union-attr]
    reordered_payload = dict(reversed(list(payload.items())))
    reordered_payload["rules"] = [reordered_rule]

    assert _content_hash([reordered_rule]) == payload["rules_hash"]
    assert parse_compiled_rule_pack(reordered_payload) == parse_compiled_rule_pack(payload)


def test_parse_rejects_external_empty_pack_even_with_correct_hash() -> None:
    with pytest.raises(ValueError, match="non-empty|at least one"):
        parse_compiled_rule_pack(_compiled_pack([]))


def test_parse_rejects_tampered_rules_hash() -> None:
    payload = _compiled_pack()
    payload["rules_hash"] = "0" * 64

    with pytest.raises(ValueError, match="rules_hash"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": None}),
        lambda value: value.pop("rule_count"),
        lambda value: value.update({"schema_version": "compiled-knowledge-rules/v1"}),
        lambda value: value.update({"rule_count": True}),
        lambda value: value.update({"rule_count": 2}),
        lambda value: value.update({"rules_hash": "A" * 64}),
        lambda value: value.update({"rules": None}),
    ],
)
def test_parse_rejects_invalid_pack_envelope(mutate: object) -> None:
    payload = _compiled_pack()
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(ValueError):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_nan_as_value_error() -> None:
    payload = _compiled_pack()
    payload["rules"][0]["triggers"] = [float("nan")]  # type: ignore[index]

    with pytest.raises(ValueError):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rule: rule.update({"unexpected": "closed"}),
        lambda rule: rule.pop("effect_hash"),
    ],
)
def test_parse_rejects_unknown_or_missing_rule_fields(mutate: object) -> None:
    payload = _compiled_pack()
    mutate(payload["rules"][0])  # type: ignore[index,operator]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="rule fields"):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_non_mapping_rule_as_value_error() -> None:
    payload = _compiled_pack()
    payload["rules"] = [None]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="objects"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule_id", "Not-Snake"),
        ("candidate_hash", "A" * 64),
        ("effect_hash", "short"),
        ("priority", True),
        ("priority", -1),
        ("priority", 1001),
    ],
)
def test_parse_rejects_invalid_rule_identity_hash_or_priority(
    field: str,
    value: object,
) -> None:
    payload = _compiled_pack()
    payload["rules"][0][field] = value  # type: ignore[index]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match=field):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_duplicate_rule_ids() -> None:
    first = _compiled_rule(priority=10)
    duplicate = _compiled_rule(priority=10, active=False)
    payload = _compiled_pack([first, duplicate])

    with pytest.raises(ValueError, match="duplicate rule_id"):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_rules_not_pre_sorted_by_priority_and_id() -> None:
    payload = _compiled_pack(
        [
            _compiled_rule(rule_id="later_rule", priority=20),
            _compiled_rule(
                candidate_type="clinical_closure_rule",
                rule_id="earlier_rule",
                priority=10,
                active=False,
            ),
        ]
    )

    with pytest.raises(ValueError, match="sorted"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize("candidate_type", ["diagnosis_unknown_rule", [], {}])
def test_parse_rejects_unknown_or_unhashable_candidate_type(candidate_type: object) -> None:
    payload = _compiled_pack()
    payload["rules"][0]["candidate_type"] = candidate_type  # type: ignore[index]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="candidate_type"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "scope",
    [
        {"phase": "diagnosis"},
        {"phase": "treatment", "application": "trigger_bound"},
        {"phase": "diagnosis", "application": "global"},
        [],
    ],
)
def test_parse_rejects_invalid_or_type_mismatched_scope(scope: object) -> None:
    payload = _compiled_pack()
    payload["rules"][0]["scope"] = scope  # type: ignore[index]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="scope"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("triggers", []),
        ("required_evidence", "objective"),
        ("exclusions", [""]),
        ("triggers", [1]),
    ],
)
def test_parse_requires_nonempty_trimmed_string_lists(field: str, value: object) -> None:
    payload = _compiled_pack()
    payload["rules"][0][field] = value  # type: ignore[index]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match=field):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rule: rule["positive_controls"][0].update({"extra": None}),
        lambda rule: rule["positive_controls"][0].update({"kind": "near_neighbor"}),
        lambda rule: rule["negative_controls"][0].update({"kind": "positive"}),
        lambda rule: rule["positive_controls"][0].update({"facts": []}),
        lambda rule: rule["negative_controls"][0].update(
            {"control_id": rule["positive_controls"][0]["control_id"]}
        ),
        lambda rule: rule.update({"negative_controls": []}),
    ],
)
def test_parse_rejects_invalid_controls(mutate: object) -> None:
    payload = _compiled_pack()
    mutate(payload["rules"][0])  # type: ignore[index,operator]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="control|positive_controls|negative_controls"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize("missing_kind", ["near_neighbor", "reasonable_exception"])
def test_parse_requires_both_treatment_negative_control_kinds(missing_kind: str) -> None:
    active = _compiled_rule()
    treatment = _compiled_rule(
        candidate_type="treatment_gate_rule",
        rule_id="bounded_treatment_gate",
        priority=20,
        active=False,
    )
    treatment["negative_controls"] = [
        item
        for item in treatment["negative_controls"]  # type: ignore[union-attr]
        if item["kind"] != missing_kind
    ]
    payload = _compiled_pack([active, treatment])

    with pytest.raises(ValueError, match="near_neighbor|reasonable_exception"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.md",
        "C:/drive.md",
        "docs\\windows.md",
        "./docs/local.md",
        "docs/../escape.md",
    ],
)
def test_parse_rejects_non_project_relative_posix_ref_paths(path: str) -> None:
    payload = _compiled_pack()
    payload["rules"][0]["source_refs"][0]["path"] = path  # type: ignore[index]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="ref path"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rule: rule.update({"source_refs": []}),
        lambda rule: rule["source_refs"][0].update({"extra": None}),
        lambda rule: rule["test_refs"][0].update({"sha256": "A" * 64}),
        lambda rule: rule.update(
            {"source_refs": [rule["source_refs"][0], deepcopy(rule["source_refs"][0])]}
        ),
    ],
)
def test_parse_rejects_invalid_ref_structures_or_duplicates(mutate: object) -> None:
    payload = _compiled_pack()
    mutate(payload["rules"][0])  # type: ignore[index,operator]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="ref|source_refs|test_refs"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "candidate_type",
    [
        "diagnosis_differential_rule",
        "clinical_closure_rule",
        "diagnosis_priority_rule",
        "treatment_gate_rule",
        "treatment_sequence_rule",
    ],
)
def test_parse_accepts_every_type_specific_effect(candidate_type: str) -> None:
    target = _compiled_rule(
        candidate_type=candidate_type,
        rule_id="typed_effect_rule",
        priority=20,
        active=candidate_type == "diagnosis_priority_rule",
    )
    if candidate_type == "diagnosis_differential_rule":
        target["effect"]["ranking_policy"] = "preserve_dual_axis"  # type: ignore[index]
    if candidate_type == "treatment_gate_rule":
        target["effect"]["risk_modifier_codes"] = ["renal_impairment"]  # type: ignore[index]
    rules = [target] if candidate_type == "diagnosis_priority_rule" else [_compiled_rule(), target]

    pack = parse_compiled_rule_pack(_compiled_pack(rules))

    assert pack.rules[-1].candidate_type == candidate_type


def test_parse_does_not_match_or_reject_chinese_audit_text() -> None:
    target = _compiled_rule(
        candidate_type="diagnosis_differential_rule",
        rule_id="chinese_audit_text",
        priority=20,
        active=False,
    )
    target["triggers"] = ["诊断：感染；检查：影像；治疗：支持"]
    target["effect"]["add_diagnostic_axes"] = ["感染性病因轴"]  # type: ignore[index]
    payload = _compiled_pack([_compiled_rule(), target])

    assert parse_compiled_rule_pack(payload).rule_count == 2


@pytest.mark.parametrize(
    ("candidate_type", "mutate"),
    [
        ("diagnosis_differential_rule", lambda effect: effect.update({"extra": "closed"})),
        ("diagnosis_differential_rule", lambda effect: effect.pop("ranking_policy")),
        ("clinical_closure_rule", lambda effect: effect.update({"extra": "closed"})),
        ("diagnosis_priority_rule", lambda effect: effect.pop("fallback_policy")),
        ("diagnosis_priority_rule", lambda effect: effect.update({"extra": "closed"})),
        ("treatment_gate_rule", lambda effect: effect.pop("gate_policy")),
        ("treatment_gate_rule", lambda effect: effect.update({"extra": "closed"})),
        ("treatment_sequence_rule", lambda effect: effect.pop("skip_patch_codes")),
        ("treatment_sequence_rule", lambda effect: effect.update({"extra": "closed"})),
    ],
)
def test_parse_rejects_missing_or_unknown_type_specific_effect_fields(
    candidate_type: str,
    mutate: object,
) -> None:
    target = _compiled_rule(
        candidate_type=candidate_type,
        rule_id="invalid_effect_rule",
        priority=20,
        active=False,
    )
    mutate(target["effect"])  # type: ignore[operator]
    payload = _compiled_pack([_compiled_rule(), target])

    with pytest.raises(ValueError, match="effect"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    ("candidate_type", "mutate"),
    [
        ("diagnosis_differential_rule", lambda effect: effect.update({"add_diagnostic_axes": []})),
        (
            "diagnosis_differential_rule",
            lambda effect: effect.update({"add_diagnostic_axes": ["same", "same"]}),
        ),
        (
            "diagnosis_differential_rule",
            lambda effect: effect.update({"ranking_policy": "free_text"}),
        ),
        (
            "diagnosis_differential_rule",
            lambda effect: effect.update({"add_diagnostic_axes": ["x" * 65]}),
        ),
        (
            "clinical_closure_rule",
            lambda effect: effect.update({"add_exam_intent_ids": ["untyped_intent"]}),
        ),
        ("clinical_closure_rule", lambda effect: effect.update({"deduplicate": "label"})),
        (
            "diagnosis_priority_rule",
            lambda effect: effect.update({"priority_policy": "subjective_first"}),
        ),
        (
            "diagnosis_priority_rule",
            lambda effect: effect.update({"fallback_policy": True}),
        ),
        (
            "treatment_gate_rule",
            lambda effect: effect.update({"remove_treatment_codes": ["INVALID"]}),
        ),
        (
            "treatment_gate_rule",
            lambda effect: effect.update({"gate_policy": "always_remove"}),
        ),
        (
            "treatment_gate_rule",
            lambda effect: effect.update({"risk_modifier_codes": ["x", "x"]}),
        ),
        (
            "treatment_sequence_rule",
            lambda effect: effect.update({"sequence_policy": "free_order"}),
        ),
        (
            "treatment_sequence_rule",
            lambda effect: effect.update({"skip_patch_codes": "elective_followup"}),
        ),
    ],
)
def test_parse_rejects_invalid_type_specific_effect_values(
    candidate_type: str,
    mutate: object,
) -> None:
    target = _compiled_rule(
        candidate_type=candidate_type,
        rule_id="invalid_effect_value",
        priority=20,
        active=False,
    )
    mutate(target["effect"])  # type: ignore[operator]
    payload = _compiled_pack([_compiled_rule(), target])

    with pytest.raises(ValueError, match="effect"):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_pack_without_an_active_rule() -> None:
    payload = _compiled_pack([_compiled_rule(active=False)])

    with pytest.raises(ValueError, match="active runtime"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "runtime",
    [
        {"status": "audit_only", "stage": "diagnosis_candidates", "opcode": "unused"},
        {"status": "unknown", "stage": "diagnosis_candidates"},
        {"status": "audit_only", "stage": "treatment"},
        [],
    ],
)
def test_parse_rejects_invalid_audit_runtime_union(runtime: object) -> None:
    target = _compiled_rule(active=False, rule_id="invalid_audit_runtime", priority=20)
    target["runtime"] = runtime
    payload = _compiled_pack([_compiled_rule(), target])

    with pytest.raises(ValueError, match="runtime"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda runtime: runtime.pop("opcode"),
        lambda runtime: runtime.update({"extra": None}),
        lambda runtime: runtime.update({"stage": "treatment"}),
        lambda runtime: runtime.update({"opcode": "unknown_opcode"}),
        lambda runtime: runtime.update({"parameters": []}),
    ],
)
def test_parse_rejects_invalid_active_runtime_union(mutate: object) -> None:
    payload = _compiled_pack()
    mutate(payload["rules"][0]["runtime"])  # type: ignore[index,operator]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="runtime|opcode"):
        parse_compiled_rule_pack(payload)


def test_parse_rejects_active_runtime_for_non_priority_type() -> None:
    target = _compiled_rule(
        candidate_type="treatment_gate_rule",
        rule_id="active_treatment_gate",
        priority=20,
        active=False,
    )
    runtime = _active_runtime()
    runtime["stage"] = "treatment"
    target["runtime"] = runtime
    payload = _compiled_pack([_compiled_rule(), target])

    with pytest.raises(ValueError, match="runtime opcode is unsupported"):
        parse_compiled_rule_pack(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda parameters: parameters.update({"extra": []}),
        lambda parameters: parameters.pop("background_roles"),
        lambda parameters: parameters.update({"target_roles": []}),
        lambda parameters: parameters.update(
            {"target_roles": ["current_problem", "current_problem"]}
        ),
        lambda parameters: parameters.update({"target_support_levels": ["subjective"]}),
        lambda parameters: parameters.update({"background_roles": "background_condition"}),
        lambda parameters: parameters.update({"fallback_policy": "free_text"}),
    ],
)
def test_parse_rejects_invalid_active_runtime_parameters(mutate: object) -> None:
    payload = _compiled_pack()
    parameters = payload["rules"][0]["runtime"]["parameters"]  # type: ignore[index]
    mutate(parameters)  # type: ignore[operator]
    _rehash_pack(payload)

    with pytest.raises(ValueError, match="runtime parameter"):
        parse_compiled_rule_pack(payload)


def test_parse_accepts_exact_active_diagnosis_differential_runtime() -> None:
    pack = parse_compiled_rule_pack(_compiled_pack([_differential_rule()]))

    runtime = pack.rules[0].runtime
    parameters = runtime.parameters
    assert runtime.opcode == "expand_congenital_infection_axes"
    assert parameters is not None
    assert parameters.age_fact_codes == ("neonate", "infant")
    assert parameters.manifestation_fact_codes == (
        "congenital_jaundice",
        "congenital_rash",
        "infant_hearing_abnormality",
        "thrombocytopenia",
        "congenital_neuroimaging_abnormality",
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "periventricular_calcifications",
        "microcephaly",
    )
    assert parameters.rubella_axis_id == "congenital_rubella"
    assert parameters.cmv_axis_id == "congenital_cmv"


@pytest.mark.parametrize(
    ("candidate_type", "runtime"),
    [
        ("diagnosis_differential_rule", _active_runtime()),
        ("diagnosis_priority_rule", _differential_runtime()),
    ],
)
def test_parse_rejects_candidate_type_and_opcode_mismatch(
    candidate_type: str,
    runtime: dict[str, object],
) -> None:
    rule = (
        _differential_rule()
        if candidate_type == "diagnosis_differential_rule"
        else _compiled_rule()
    )
    rule["runtime"] = deepcopy(runtime)

    with pytest.raises(ValueError, match="runtime|opcode"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age_fact_codes", []),
        ("age_fact_codes", ["infant", "infant"]),
        ("age_fact_codes", ["Infant"]),
        ("cmv_rank_fact_codes", "microcephaly"),
        ("rubella_axis_id", "Congenital Rubella"),
    ],
)
def test_parse_rejects_invalid_differential_parameter_values(
    field: str,
    value: object,
) -> None:
    rule = _differential_rule()
    parameters = rule["runtime"]["parameters"]  # type: ignore[index]
    parameters[field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="runtime parameter"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age_fact_codes", ["adult"]),
        ("exposure_fact_codes", ["postnatal_viral_infection"]),
        ("manifestation_fact_codes", ["postnatal_viral_infection"]),
        ("rubella_rank_fact_codes", ["external_rubella_support"]),
        ("cmv_rank_fact_codes", ["external_cmv_support"]),
        ("rubella_confirmed_fact_codes", ["rubella_igm_positive"]),
        ("cmv_confirmed_fact_codes", ["cmv_igm_positive"]),
        ("rubella_axis_id", "external_rubella_axis"),
        ("cmv_axis_id", "external_cmv_axis"),
    ],
)
def test_parse_rejects_out_of_domain_differential_codes(
    field: str,
    value: object,
) -> None:
    rule = _differential_rule()
    parameters = rule["runtime"]["parameters"]  # type: ignore[index]
    parameters[field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="runtime differential parameter"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


@pytest.mark.parametrize(
    ("field", "removed_code"),
    [
        ("age_fact_codes", "neonate"),
        ("manifestation_fact_codes", "infant_hearing_abnormality"),
        ("rubella_rank_fact_codes", "congenital_cataract"),
        ("cmv_rank_fact_codes", "cmv_igm_positive"),
        ("rubella_confirmed_fact_codes", "rubella_igm_positive_in_infant"),
    ],
)
def test_parse_rejects_incomplete_canonical_differential_code_sets(
    field: str,
    removed_code: str,
) -> None:
    rule = _differential_rule()
    parameters = rule["runtime"]["parameters"]  # type: ignore[index]
    parameters[field].remove(removed_code)  # type: ignore[index]

    with pytest.raises(ValueError, match="canonical code set"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


@pytest.mark.parametrize(
    ("relationship", "message"),
    [
        ("missing_typical_manifestation", "typical manifestations"),
        ("confirmed_not_ranked", "confirmed.*rank"),
        ("trigger_overlap", "trigger groups"),
        ("confirmed_overlap", "confirmed.*overlap"),
    ],
)
def test_parse_rejects_invalid_differential_parameter_relationships(
    relationship: str,
    message: str,
) -> None:
    rule = _differential_rule()
    parameters = rule["runtime"]["parameters"]  # type: ignore[index]
    if relationship == "missing_typical_manifestation":
        parameters["manifestation_fact_codes"].remove("microcephaly")  # type: ignore[index]
    elif relationship == "confirmed_not_ranked":
        parameters["rubella_rank_fact_codes"].remove(  # type: ignore[index]
            "rubella_pcr_positive_in_infant"
        )
    elif relationship == "trigger_overlap":
        parameters["manifestation_fact_codes"].append("infant")  # type: ignore[index]
    else:
        parameters["rubella_confirmed_fact_codes"] = [  # type: ignore[index]
            "cmv_saliva_or_urine_pcr_positive_within_21_days"
        ]

    with pytest.raises(ValueError, match=message):
        parse_compiled_rule_pack(_compiled_pack([rule]))


@pytest.mark.parametrize("field_change", ["missing", "extra"])
def test_parse_rejects_non_exact_differential_parameter_fields(field_change: str) -> None:
    rule = _differential_rule()
    parameters = rule["runtime"]["parameters"]  # type: ignore[index]
    if field_change == "missing":
        parameters.pop("exposure_fact_codes")  # type: ignore[union-attr]
    else:
        parameters["extra_fact_codes"] = ["unexpected"]  # type: ignore[index]

    with pytest.raises(ValueError, match="runtime parameter fields"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


def test_parse_rejects_same_differential_axis_ids_and_unknown_opcode() -> None:
    same_axis = _differential_rule()
    parameters = same_axis["runtime"]["parameters"]  # type: ignore[index]
    parameters["cmv_axis_id"] = "congenital_rubella"  # type: ignore[index]
    unknown_opcode = _differential_rule(rule_id="unknown_differential_opcode", priority=20)
    unknown_opcode["runtime"]["opcode"] = "expand_unknown_axes"  # type: ignore[index]

    with pytest.raises(ValueError, match="axis"):
        parse_compiled_rule_pack(_compiled_pack([same_axis]))
    with pytest.raises(ValueError, match="opcode"):
        parse_compiled_rule_pack(_compiled_pack([unknown_opcode]))


@pytest.mark.parametrize(
    "effect",
    [
        {
            "add_diagnostic_axes": ["先天性风疹方向"],
            "ranking_policy": "evidence_ordered",
        },
        {
            "add_diagnostic_axes": ["巨细胞病毒方向", "先天性风疹方向"],
            "ranking_policy": "evidence_ordered",
        },
        {
            "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
            "ranking_policy": "preserve_dual_axis",
        },
    ],
)
def test_parse_rejects_active_differential_effect_runtime_mismatch(
    effect: dict[str, object],
) -> None:
    rule = _differential_rule()
    rule["effect"] = deepcopy(effect)

    with pytest.raises(ValueError, match="active differential effect"):
        parse_compiled_rule_pack(_compiled_pack([rule]))


def test_parse_allows_multiple_active_rules_with_same_stage_opcode() -> None:
    """Multiple active rules may share a stage+opcode; each is applied in order."""

    def rule_for(fact_code: str, match_code: str, rid: str, prio: int) -> dict[str, object]:
        return {
            "rule_id": rid,
            "candidate_type": "clinical_closure_rule",
            "candidate_hash": "a" * 64,
            "effect_hash": "b" * 64,
            "triggers": ["trigger"],
            "required_evidence": ["evidence"],
            "exclusions": ["exclusion"],
            "effect": {
                "add_exam_intent_ids": ["exam_intent_organ_involvement"],
                "deduplicate": "intent_id",
            },
            "positive_controls": [
                {"control_id": rid + "_pos", "kind": "positive", "facts": ["f"], "assertions": ["a"]}
            ],
            "negative_controls": [
                {"control_id": rid + "_neg", "kind": "near_neighbor", "facts": ["f"], "assertions": ["a"]}
            ],
            "source_refs": [{"path": "docs/x.md", "sha256": "c" * 64}],
            "test_refs": [{"path": "tests/x.py", "sha256": "d" * 64}],
            "priority": prio,
            "scope": {"phase": "closure", "application": "trigger_bound"},
            "runtime": {
                "status": "active",
                "stage": "clinical_closure",
                "opcode": "match_fact_groups",
                "parameters": {
                    "all_groups": [[fact_code]],
                    "any_groups": [],
                    "excluded_groups": [],
                    "matched_fact_code": match_code,
                },
            },
        }

    rules = [
        rule_for("fever", "acute_limb_soft_tissue_infection", "r1", 10),
        rule_for("xanthelasma", "hyperlipidemia_with_xanthelasma", "r2", 20),
    ]
    pack = parse_compiled_rule_pack(_compiled_pack(rules))
    assert pack.rule_count == 2
    result = apply_rules(pack, "clinical_closure", RuleContext(fact_codes=("fever", "xanthelasma")))
    assert "acute_limb_soft_tissue_infection" in result.output_context.fact_codes
    assert "hyperlipidemia_with_xanthelasma" in result.output_context.fact_codes


def test_apply_rules_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        apply_rules(empty_compiled_rule_pack(), "diagnosis", RuleContext())  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", ["clinical_closure", "treatment"])
def test_apply_rules_skips_audit_only_and_active_rules_for_other_stage(stage: str) -> None:
    audit_type = (
        "clinical_closure_rule" if stage == "clinical_closure" else "treatment_sequence_rule"
    )
    pack = parse_compiled_rule_pack(
        _compiled_pack(
            [
                _compiled_rule(),
                _compiled_rule(
                    candidate_type=audit_type,
                    rule_id="current_stage_audit_rule",
                    priority=20,
                    active=False,
                ),
            ]
        )
    )
    context = RuleContext(fact_codes=("unchanged",))

    result = apply_rules(pack, stage, context)  # type: ignore[arg-type]

    assert result.output_context == context
    assert result.decisions == ()
    assert result.before_hash == result.after_hash


def test_apply_rules_dispatches_current_stage_active_opcode_by_opcode_not_text() -> None:
    payload = _compiled_pack()
    payload["rules"][0]["rule_id"] = "misleading_noop_name"  # type: ignore[index]
    payload["rules"][0]["triggers"] = ["不要依据中文文本决定分派"]  # type: ignore[index]
    _rehash_pack(payload)
    pack = parse_compiled_rule_pack(payload)

    result = apply_rules(pack, "diagnosis_candidates", RuleContext())

    assert result.output_context == RuleContext()
    assert result.decisions == (
        RuleDecision(
            "misleading_noop_name",
            "promote_supported_current_over_background",
            "not_matched",
            "no_supported_current_problem",
        ),
    )


def test_apply_rules_rejects_reflection_forged_compiled_pack() -> None:
    pack = object.__new__(CompiledRulePack)
    object.__setattr__(pack, "schema_version", "compiled-knowledge-rules/v2")
    object.__setattr__(pack, "rules", [])
    object.__setattr__(pack, "rule_count", 0)
    object.__setattr__(pack, "rules_hash", _content_hash([]))

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(pack, "diagnosis_candidates", RuleContext())


def test_compiled_seal_registry_is_not_exposed_as_mutable_module_state() -> None:
    assert not hasattr(typed_rule_engine, "_VALIDATED_COMPILED_VALUES")


def test_apply_rules_rejects_reflection_mutated_parser_pack() -> None:
    pack = _priority_rule_pack()
    object.__setattr__(pack, "rule_count", 0)

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(pack, "diagnosis_candidates", RuleContext())


def test_apply_rules_rejects_reflection_mutated_nested_runtime() -> None:
    pack = _priority_rule_pack()
    object.__setattr__(pack.rules[0].runtime, "stage", "treatment")

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(pack, "diagnosis_candidates", RuleContext())


def test_apply_rules_rejects_reflection_repaired_outer_pack_with_mutated_nested_runtime() -> None:
    pack = _priority_rule_pack()
    object.__setattr__(pack.rules[0].runtime, "stage", "treatment")
    typed_rule_engine._register_compiled_value(pack)

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(pack, "diagnosis_candidates", RuleContext())


def test_apply_rules_rejects_reflection_mutated_differential_parameters() -> None:
    pack = _differential_rule_pack()
    parameters = pack.rules[0].runtime.parameters
    assert parameters is not None
    object.__setattr__(parameters, "age_fact_codes", ("adult",))

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(pack, "diagnosis_candidates", RuleContext())


def test_apply_rules_rejects_mutated_nested_graph_after_attempted_reregistration() -> None:
    pack = _differential_rule_pack()
    rule = pack.rules[0]
    runtime = rule.runtime
    parameters = runtime.parameters
    assert parameters is not None
    object.__setattr__(parameters, "age_fact_codes", ("adult",))

    for compiled_value in (parameters, runtime, rule, pack):
        typed_rule_engine._register_compiled_value(compiled_value)

    with pytest.raises(ValueError, match="validated CompiledRulePack"):
        apply_rules(
            pack,
            "diagnosis_candidates",
            RuleContext(
                diagnostic_axis_ids=("other_axis",),
                fact_codes=(
                    "adult",
                    "intrauterine_viral_exposure",
                    "congenital_jaundice",
                ),
            ),
        )


def _priority_rule_pack():
    return parse_compiled_rule_pack(_compiled_pack())


def _differential_rule_pack():
    return parse_compiled_rule_pack(_compiled_pack([_differential_rule()]))


def _congenital_facts(*extra: str) -> tuple[str, ...]:
    return (
        "infant",
        "intrauterine_viral_exposure",
        "infant_hearing_abnormality",
        *extra,
    )


def test_differential_opcode_defaults_to_dual_axes_and_preserves_other_context_fields() -> None:
    candidate = _diagnosis_candidate(
        official_name="先天性感染综合征",
        urgency="routine",
        evidence_codes=("congenital_jaundice",),
    )
    context = RuleContext(
        diagnosis_candidates=(candidate,),
        preferred_diagnosis="先天性感染综合征",
        diagnostic_axis_ids=(
            "other_first",
            "congenital_cmv",
            "other_second",
            "congenital_rubella",
            "other_third",
        ),
        exam_intent_ids=("exam_intent_organ_involvement",),
        treatment_codes=("supportive_care",),
        fact_codes=_congenital_facts(),
    )

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context == RuleContext(
        diagnosis_candidates=context.diagnosis_candidates,
        preferred_diagnosis=context.preferred_diagnosis,
        diagnostic_axis_ids=(
            "congenital_rubella",
            "congenital_cmv",
            "other_first",
            "other_second",
            "other_third",
        ),
        exam_intent_ids=context.exam_intent_ids,
        treatment_codes=context.treatment_codes,
        fact_codes=context.fact_codes,
    )
    assert result.decisions == (
        RuleDecision(
            "expand_congenital_infection_differential",
            "expand_congenital_infection_axes",
            "applied",
            "congenital_infection_axes_expanded",
        ),
    )
    assert result.before_hash != result.after_hash


@pytest.mark.parametrize(
    ("extra_facts", "expected_axes"),
    [
        (
            ("congenital_cataract", "patent_ductus_arteriosus"),
            ("congenital_rubella", "congenital_cmv"),
        ),
        (
            ("periventricular_calcifications", "microcephaly"),
            ("congenital_cmv", "congenital_rubella"),
        ),
        (
            (
                "rubella_pcr_positive_in_infant",
                "cmv_saliva_or_urine_pcr_positive_within_21_days",
                "microcephaly",
            ),
            ("congenital_cmv", "congenital_rubella"),
        ),
    ],
)
def test_differential_opcode_orders_dual_axes_by_stable_structured_score(
    extra_facts: tuple[str, ...],
    expected_axes: tuple[str, ...],
) -> None:
    context = RuleContext(
        diagnostic_axis_ids=("other_axis",),
        fact_codes=_congenital_facts(*extra_facts),
    )

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnostic_axis_ids == (*expected_axes, "other_axis")


@pytest.mark.parametrize(
    ("manifestation", "expected_first"),
    [
        ("congenital_cataract", "congenital_rubella"),
        ("patent_ductus_arteriosus", "congenital_rubella"),
        ("periventricular_calcifications", "congenital_cmv"),
        ("microcephaly", "congenital_cmv"),
    ],
)
def test_typical_rank_facts_also_satisfy_the_manifestation_gate(
    manifestation: str,
    expected_first: str,
) -> None:
    context = RuleContext(
        fact_codes=("infant", "intrauterine_viral_exposure", manifestation),
    )

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnostic_axis_ids == (
        expected_first,
        "congenital_cmv" if expected_first == "congenital_rubella" else "congenital_rubella",
    )
    assert result.decisions[0].outcome == "applied"


@pytest.mark.parametrize("support_code", ["cmv_igm_positive", "cmv_pcr_positive"])
def test_cmv_support_without_timed_specimen_pcr_keeps_both_axes(
    support_code: str,
) -> None:
    context = RuleContext(fact_codes=_congenital_facts(support_code))

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnostic_axis_ids == (
        "congenital_cmv",
        "congenital_rubella",
    )


@pytest.mark.parametrize(
    ("extra_facts", "expected_axis"),
    [
        (
            (
                "rubella_pcr_positive_in_infant",
                "periventricular_calcifications",
                "microcephaly",
            ),
            "congenital_rubella",
        ),
        (
            (
                "cmv_saliva_or_urine_pcr_positive_within_21_days",
                "congenital_cataract",
                "patent_ductus_arteriosus",
            ),
            "congenital_cmv",
        ),
    ],
)
def test_differential_opcode_keeps_only_the_single_confirmed_pathogen_axis(
    extra_facts: tuple[str, ...],
    expected_axis: str,
) -> None:
    context = RuleContext(
        diagnostic_axis_ids=("congenital_cmv", "other_axis", "congenital_rubella"),
        fact_codes=_congenital_facts(*extra_facts),
    )

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnostic_axis_ids == (expected_axis, "other_axis")


@pytest.mark.parametrize(
    ("fact_codes", "reason_code"),
    [
        (
            ("intrauterine_viral_exposure", "congenital_jaundice"),
            "congenital_infection_age_not_matched",
        ),
        (
            ("infant", "congenital_jaundice"),
            "congenital_infection_exposure_not_matched",
        ),
        (
            ("infant", "postnatal_viral_infection", "congenital_jaundice"),
            "congenital_infection_exposure_not_matched",
        ),
        (
            ("infant", "intrauterine_viral_exposure", "nonspecific_poor_feeding"),
            "congenital_infection_manifestation_not_matched",
        ),
    ],
    ids=["no_age", "isolated_jaundice", "postnatal_infection", "no_manifestation"],
)
def test_differential_opcode_requires_age_exposure_and_manifestation(
    fact_codes: tuple[str, ...],
    reason_code: str,
) -> None:
    context = RuleContext(
        diagnostic_axis_ids=("other_axis",),
        fact_codes=fact_codes,
    )

    result = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context is context
    assert result.before_hash == result.after_hash
    assert result.decisions[0].outcome == "not_matched"
    assert result.decisions[0].reason_code == reason_code


def test_differential_opcode_is_idempotent_after_owned_axes_are_rewritten() -> None:
    context = RuleContext(
        diagnostic_axis_ids=("congenital_cmv", "other_a", "congenital_rubella", "other_b"),
        fact_codes=_congenital_facts("congenital_cataract"),
    )

    first = apply_rules(_differential_rule_pack(), "diagnosis_candidates", context)
    second = apply_rules(
        _differential_rule_pack(),
        "diagnosis_candidates",
        first.output_context,
    )

    assert first.output_context.diagnostic_axis_ids == (
        "congenital_rubella",
        "congenital_cmv",
        "other_a",
        "other_b",
    )
    assert second.output_context is first.output_context
    assert second.before_hash == second.after_hash == first.after_hash
    assert second.decisions[0].outcome == "matched_no_change"
    assert second.decisions[0].reason_code == "congenital_infection_axes_already_ranked"


def _hearing_candidate(**overrides: object) -> RuleDiagnosisCandidate:
    values: dict[str, object] = {
        "official_name": "老年性听力损失",
        "role": "current_problem",
        "support_level": "objective",
        "complaint_relation": "unrelated",
        "urgency": "routine",
        "evidence_codes": ("audiometry_abnormal",),
        "is_official": True,
    }
    values.update(overrides)
    return _diagnosis_candidate(**values)


def _hypertension_candidate(**overrides: object) -> RuleDiagnosisCandidate:
    values: dict[str, object] = {
        "official_name": "原发性高血压",
        "role": "background_condition",
        "support_level": "objective",
        "complaint_relation": "unrelated",
        "urgency": "routine",
        "evidence_codes": ("history_hypertension",),
        "is_official": True,
    }
    values.update(overrides)
    return _diagnosis_candidate(**values)


@pytest.mark.parametrize(
    "fact_codes",
    [
        (),
        ("severe_blood_pressure_without_acute_target_organ_damage",),
        ("transient_blood_pressure_elevation_from_pain_or_anxiety",),
    ],
)
def test_priority_opcode_promotes_objective_current_problem_over_unrelated_background(
    fact_codes: tuple[str, ...],
) -> None:
    context = RuleContext(
        diagnosis_candidates=(_hypertension_candidate(), _hearing_candidate()),
        preferred_diagnosis="原发性高血压",
        fact_codes=fact_codes,
    )

    result = apply_rules(_priority_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnosis_candidates == (
        _hearing_candidate(),
        _hypertension_candidate(),
    )
    assert result.output_context.preferred_diagnosis == "老年性听力损失"
    assert result.decisions == (
        RuleDecision(
            "prefer_supported_current_problem",
            "promote_supported_current_over_background",
            "applied",
            "supported_current_problem_promoted",
        ),
    )
    assert result.before_hash != result.after_hash
    assert result.applied_rule_ids == ("prefer_supported_current_problem",)
    assert context.diagnosis_candidates[0].official_name == "原发性高血压"


@pytest.mark.parametrize(
    "context",
    [
        RuleContext(
            diagnosis_candidates=(
                _hypertension_candidate(complaint_relation="explains"),
                _hearing_candidate(),
            ),
            preferred_diagnosis="原发性高血压",
        ),
        RuleContext(
            diagnosis_candidates=(
                _hypertension_candidate(),
                _hearing_candidate(support_level="subjective", evidence_codes=()),
            ),
            preferred_diagnosis="原发性高血压",
        ),
        RuleContext(
            diagnosis_candidates=(
                _hypertension_candidate(),
                _hearing_candidate(support_level="none", evidence_codes=()),
            ),
            preferred_diagnosis="原发性高血压",
        ),
        RuleContext(
            diagnosis_candidates=(
                _hypertension_candidate(),
                _hearing_candidate(
                    support_level="none",
                    complaint_relation="unknown",
                    evidence_codes=("audiometry_result_uncertain",),
                ),
            ),
            preferred_diagnosis="原发性高血压",
        ),
        RuleContext(
            diagnosis_candidates=(
                _hypertension_candidate(),
                _hearing_candidate(is_official=False),
            ),
            preferred_diagnosis="原发性高血压",
        ),
    ],
)
def test_priority_opcode_does_not_promote_without_supported_independent_current_problem(
    context: RuleContext,
) -> None:
    result = apply_rules(_priority_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context == context
    assert result.before_hash == result.after_hash
    assert result.applied_rule_ids == ()
    assert len(result.decisions) == 1
    assert result.decisions[0].outcome in {"excluded", "not_matched"}


@pytest.mark.parametrize(
    "emergency_facts",
    [
        ("confirmed_hypertensive_emergency",),
        ("suspected_hypertensive_emergency_pending_exclusion",),
    ],
)
def test_priority_opcode_preserves_emergency_priority_and_independent_current_axis(
    emergency_facts: tuple[str, ...],
) -> None:
    emergency = _hypertension_candidate(
        official_name="高血压急症",
        role="current_problem",
        complaint_relation="explains",
        urgency="emergency",
        evidence_codes=("acute_target_organ_damage",),
    )
    context = RuleContext(
        diagnosis_candidates=(emergency, _hearing_candidate()),
        preferred_diagnosis="高血压急症",
        fact_codes=emergency_facts,
    )

    result = apply_rules(_priority_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context == context
    assert result.output_context.diagnosis_candidates == (emergency, _hearing_candidate())
    assert result.output_context.preferred_diagnosis == "高血压急症"
    assert result.decisions[0].outcome == "matched_no_change"
    assert result.decisions[0].reason_code == "emergency_priority_preserved"


def test_priority_opcode_is_idempotent_and_preserves_relative_order() -> None:
    first_current = _hearing_candidate(official_name="传导性听力损失")
    second_current = _hearing_candidate()
    first_background = _hypertension_candidate()
    second_background = _hypertension_candidate(
        official_name="高脂血症",
        evidence_codes=("history_hyperlipidemia",),
    )
    context = RuleContext(
        diagnosis_candidates=(
            first_background,
            first_current,
            second_background,
            second_current,
        ),
        preferred_diagnosis="原发性高血压",
    )

    first = apply_rules(_priority_rule_pack(), "diagnosis_candidates", context)
    second = apply_rules(
        _priority_rule_pack(),
        "diagnosis_candidates",
        first.output_context,
    )

    assert first.output_context.diagnosis_candidates == (
        first_current,
        second_current,
        first_background,
        second_background,
    )
    assert second.output_context == first.output_context
    assert second.before_hash == second.after_hash == first.after_hash
    assert second.decisions[0].outcome == "matched_no_change"


def test_priority_opcode_preserves_urgent_candidate_ahead_of_routine_target() -> None:
    urgent = _hypertension_candidate(
        official_name="高血压亚急症",
        role="current_problem",
        complaint_relation="explains",
        urgency="urgent",
        evidence_codes=("severe_hypertension",),
    )
    context = RuleContext(
        diagnosis_candidates=(urgent, _hypertension_candidate(), _hearing_candidate()),
        preferred_diagnosis="高血压亚急症",
    )

    result = apply_rules(_priority_rule_pack(), "diagnosis_candidates", context)

    assert result.output_context.diagnosis_candidates[0] == urgent
    assert result.output_context.preferred_diagnosis == "高血压亚急症"
