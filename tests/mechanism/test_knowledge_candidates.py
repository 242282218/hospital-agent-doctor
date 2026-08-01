from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from offline.artifacts import content_hash, file_hash, read_json
from offline.candidates import create_candidate, load_candidate, write_candidate
from offline.knowledge_rules import (
    KNOWLEDGE_CANDIDATE_TYPES,
    knowledge_rule_candidate,
    validate_knowledge_effect,
)
from scripts.knowledge.build_knowledge_candidates import (
    EXPECTED_RULES,
    build_knowledge_candidate_batch,
)


REQUIRED_EFFECT_FIELDS = {
    "schema_version",
    "rule_id",
    "triggers",
    "required_evidence",
    "exclusions",
    "effect",
    "positive_controls",
    "negative_controls",
    "source_refs",
    "test_refs",
    "priority",
    "scope",
    "review_requirements",
    "runtime",
}

RUNTIME_STAGE_BY_TYPE = {
    "diagnosis_differential_rule": "diagnosis_candidates",
    "clinical_closure_rule": "clinical_closure",
    "diagnosis_priority_rule": "diagnosis_candidates",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}

ACTIVE_PARAMETERS = {
    "target_roles": ["current_problem"],
    "target_support_levels": ["objective"],
    "background_roles": ["background_condition"],
    "background_relations": ["unrelated"],
    "excluded_relations": ["explains"],
    "preserve_urgencies": ["emergency"],
    "fallback_policy": "official_catalog_only",
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


def _differential_runtime() -> dict:
    return {
        "status": "active",
        "stage": "diagnosis_candidates",
        "opcode": "expand_congenital_infection_axes",
        "parameters": deepcopy(DIFFERENTIAL_PARAMETERS),
    }


def _runtime(candidate_type: str) -> dict:
    stage = RUNTIME_STAGE_BY_TYPE[candidate_type]
    if candidate_type == "diagnosis_differential_rule":
        return _differential_runtime()
    if candidate_type != "diagnosis_priority_rule":
        return {"status": "audit_only", "stage": stage}
    return {
        "status": "active",
        "stage": stage,
        "opcode": "promote_supported_current_over_background",
        "parameters": deepcopy(ACTIVE_PARAMETERS),
    }


def _write_ref(project_root: Path, relative_path: str, body: str) -> dict[str, str]:
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return {"path": relative_path, "sha256": file_hash(path)}


def _write_official_intent_registry(project_root: Path) -> None:
    path = project_root / "agent" / "knowledge" / "exam_intent_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "exam_intent_congenital_infection_organ_involvement",
                        "status": "verified",
                    }
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _typed_action(candidate_type: str) -> dict:
    return {
        "diagnosis_differential_rule": {
            "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
            "ranking_policy": "evidence_ordered",
        },
        "clinical_closure_rule": {
            "add_exam_intent_ids": ["exam_intent_congenital_infection_organ_involvement"],
            "deduplicate": "intent_id",
        },
        "diagnosis_priority_rule": {
            "priority_policy": "objective_evidence_first",
            "fallback_policy": "official_catalog_only",
        },
        "treatment_gate_rule": {
            "remove_treatment_codes": ["routine_antibiotic"],
            "preserve_treatment_codes": ["reassessment"],
            "gate_policy": "require_infection_evidence",
        },
        "treatment_sequence_rule": {
            "ordered_patch_codes": ["stabilize_hemodynamics", "decongest_if_overloaded"],
            "sequence_policy": "acute_before_stable",
            "skip_patch_codes": ["force_diuresis_when_euvolemic"],
        },
    }[candidate_type]


def _complete_effect(
    project_root: Path,
    *,
    rule_id: str = "safe_rule",
    treatment: bool = False,
    phase: str | None = None,
    candidate_type: str | None = None,
) -> dict:
    source_ref = _write_ref(project_root, "docs/source.md", "source\n")
    test_ref = _write_ref(project_root, "tests/test_source.py", "def test_source():\n    pass\n")
    if candidate_type == "clinical_closure_rule":
        _write_official_intent_registry(project_root)
    negative_controls = [
        {
            "control_id": "near-neighbor",
            "kind": "near_neighbor",
            "facts": ["相邻表现但缺少关键触发证据"],
            "assertions": ["规则不触发"],
        }
    ]
    exclusions = ["存在明确排除条件"]
    if treatment:
        negative_controls.append(
            {
                "control_id": "reasonable-exception",
                "kind": "reasonable_exception",
                "facts": ["存在需要保留处置的高风险例外"],
                "assertions": ["保留例外处置并升级复核"],
            }
        )
    resolved_candidate_type = candidate_type or (
        "treatment_gate_rule" if treatment else "diagnosis_differential_rule"
    )
    return {
        "schema_version": "clinical-knowledge-candidate/v2",
        "rule_id": rule_id,
        "triggers": ["存在明确临床触发组合"],
        "required_evidence": ["至少一项可核验客观证据"],
        "exclusions": exclusions,
        "effect": _typed_action(resolved_candidate_type),
        "positive_controls": [
            {
                "control_id": "positive-one",
                "kind": "positive",
                "facts": ["满足触发和证据要求"],
                "assertions": ["规则触发"],
            },
            {
                "control_id": "positive-two",
                "kind": "positive",
                "facts": ["第二组满足条件的近义事实"],
                "assertions": ["规则仍触发"],
            },
        ],
        "negative_controls": negative_controls,
        "source_refs": [source_ref],
        "test_refs": [test_ref],
        "priority": 50,
        "scope": {
            "phase": phase or ("treatment" if treatment else "diagnosis"),
            "application": "trigger_bound",
        },
        "review_requirements": ["人工复核触发、排除、负控与合理例外"],
        "runtime": _runtime(resolved_candidate_type),
    }


def test_candidate_type_whitelist_is_exact() -> None:
    assert KNOWLEDGE_CANDIDATE_TYPES == {
        "diagnosis_differential_rule",
        "clinical_closure_rule",
        "diagnosis_priority_rule",
        "treatment_gate_rule",
        "treatment_sequence_rule",
    }


@pytest.mark.parametrize(
    ("candidate_type", "expected_status"),
    [
        ("diagnosis_differential_rule", "active"),
        ("clinical_closure_rule", "audit_only"),
        ("diagnosis_priority_rule", "active"),
        ("treatment_gate_rule", "audit_only"),
        ("treatment_sequence_rule", "audit_only"),
    ],
)
def test_runtime_stage_mapping_and_supported_diagnosis_rules_can_be_active(
    tmp_path: Path,
    candidate_type: str,
    expected_status: str,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        phase="closure" if candidate_type == "clinical_closure_rule" else None,
        candidate_type=candidate_type,
        rule_id="runtime_contract_not_dispatched_by_rule_id",
    )

    validated = validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)

    assert validated["runtime"]["stage"] == RUNTIME_STAGE_BY_TYPE[candidate_type]
    assert validated["runtime"]["status"] == expected_status
    if expected_status == "active":
        assert validated["runtime"] == _runtime(candidate_type)
    else:
        assert set(validated["runtime"]) == {"status", "stage"}


@pytest.mark.parametrize(
    ("candidate_type", "runtime"),
    [
        ("diagnosis_differential_rule", {"status": "audit_only"}),
        (
            "diagnosis_differential_rule",
            {
                "status": "audit_only",
                "stage": "diagnosis_candidates",
                "parameters": {},
            },
        ),
        (
            "diagnosis_differential_rule",
            {"status": "pending", "stage": "diagnosis_candidates"},
        ),
        (
            "diagnosis_differential_rule",
            {"status": "audit_only", "stage": "unknown_stage"},
        ),
        (
            "diagnosis_differential_rule",
            {"status": "audit_only", "stage": []},
        ),
        (
            "diagnosis_differential_rule",
            {
                "status": "active",
                "stage": "diagnosis_candidates",
                "opcode": "promote_supported_current_over_background",
                "parameters": ACTIVE_PARAMETERS,
            },
        ),
        (
            "diagnosis_priority_rule",
            {
                "status": "active",
                "stage": "clinical_closure",
                "opcode": "promote_supported_current_over_background",
                "parameters": ACTIVE_PARAMETERS,
            },
        ),
        (
            "diagnosis_priority_rule",
            {
                "status": "active",
                "stage": "diagnosis_candidates",
                "opcode": "unknown_opcode",
                "parameters": ACTIVE_PARAMETERS,
            },
        ),
        (
            "diagnosis_priority_rule",
            {
                "status": "active",
                "stage": "diagnosis_candidates",
                "opcode": "promote_supported_current_over_background",
            },
        ),
        (
            "diagnosis_priority_rule",
            {
                "status": "active",
                "stage": "diagnosis_candidates",
                "opcode": "promote_supported_current_over_background",
                "parameters": ACTIVE_PARAMETERS,
                "unexpected": True,
            },
        ),
    ],
)
def test_runtime_union_fails_closed_for_unknown_missing_extra_or_mismatched_fields(
    tmp_path: Path,
    candidate_type: str,
    runtime: dict,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        phase="closure" if candidate_type == "clinical_closure_rule" else None,
        candidate_type=candidate_type,
    )
    effect["runtime"] = deepcopy(runtime)

    with pytest.raises(ValueError, match="runtime"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "mutate_parameters",
    [
        lambda value: value.pop("target_roles"),
        lambda value: value.update({"unexpected": ["current_problem"]}),
        lambda value: value.update({"target_roles": []}),
        lambda value: value.update(
            {"target_roles": ["current_problem", "current_problem"]}
        ),
        lambda value: value.update({"target_roles": ["background_condition"]}),
        lambda value: value.update({"fallback_policy": "free_text"}),
    ],
)
def test_active_runtime_parameters_are_exact_unique_nonempty_closed_enums(
    tmp_path: Path,
    mutate_parameters,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_priority_rule")
    mutate_parameters(effect["runtime"]["parameters"])

    with pytest.raises(ValueError, match="runtime"):
        validate_knowledge_effect("diagnosis_priority_rule", effect, project_root=tmp_path)


def test_offline_validator_accepts_exact_active_diagnosis_differential_runtime(
    tmp_path: Path,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()

    validated = validate_knowledge_effect(
        "diagnosis_differential_rule",
        effect,
        project_root=tmp_path,
    )

    assert validated["runtime"] == _differential_runtime()


@pytest.mark.parametrize(
    ("candidate_type", "runtime"),
    [
        (
            "diagnosis_differential_rule",
            {
                "status": "active",
                "stage": "diagnosis_candidates",
                "opcode": "promote_supported_current_over_background",
                "parameters": ACTIVE_PARAMETERS,
            },
        ),
        ("diagnosis_priority_rule", _differential_runtime()),
    ],
)
def test_offline_validator_rejects_candidate_type_and_opcode_mismatch(
    tmp_path: Path,
    candidate_type: str,
    runtime: dict,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type=candidate_type)
    effect["runtime"] = deepcopy(runtime)

    with pytest.raises(ValueError, match="runtime|opcode"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)


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
def test_offline_validator_rejects_invalid_differential_parameter_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    effect["runtime"]["parameters"][field] = value

    with pytest.raises(ValueError, match="runtime parameter"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


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
def test_offline_validator_rejects_out_of_domain_differential_codes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    effect["runtime"]["parameters"][field] = value

    with pytest.raises(ValueError, match="runtime differential parameter"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


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
def test_offline_validator_rejects_incomplete_canonical_differential_code_sets(
    tmp_path: Path,
    field: str,
    removed_code: str,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    effect["runtime"]["parameters"][field].remove(removed_code)

    with pytest.raises(ValueError, match="canonical code set"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("relationship", "message"),
    [
        ("missing_typical_manifestation", "typical manifestations"),
        ("confirmed_not_ranked", "confirmed.*rank"),
        ("trigger_overlap", "trigger groups"),
        ("confirmed_overlap", "confirmed.*overlap"),
    ],
)
def test_offline_validator_rejects_invalid_differential_parameter_relationships(
    tmp_path: Path,
    relationship: str,
    message: str,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    parameters = effect["runtime"]["parameters"]
    if relationship == "missing_typical_manifestation":
        parameters["manifestation_fact_codes"].remove("microcephaly")
    elif relationship == "confirmed_not_ranked":
        parameters["rubella_rank_fact_codes"].remove("rubella_pcr_positive_in_infant")
    elif relationship == "trigger_overlap":
        parameters["manifestation_fact_codes"].append("infant")
    else:
        parameters["rubella_confirmed_fact_codes"] = [
            "cmv_saliva_or_urine_pcr_positive_within_21_days"
        ]

    with pytest.raises(ValueError, match=message):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


@pytest.mark.parametrize("field_change", ["missing", "extra"])
def test_offline_validator_rejects_non_exact_differential_parameter_fields(
    tmp_path: Path,
    field_change: str,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    parameters = effect["runtime"]["parameters"]
    if field_change == "missing":
        parameters.pop("exposure_fact_codes")
    else:
        parameters["extra_fact_codes"] = ["unexpected"]

    with pytest.raises(ValueError, match="runtime parameter fields"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


def test_offline_validator_rejects_same_differential_axes_and_unknown_opcode(
    tmp_path: Path,
) -> None:
    same_axis = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    same_axis["runtime"] = _differential_runtime()
    same_axis["runtime"]["parameters"]["cmv_axis_id"] = "congenital_rubella"
    unknown_opcode = deepcopy(same_axis)
    unknown_opcode["runtime"] = _differential_runtime()
    unknown_opcode["runtime"]["opcode"] = "expand_unknown_axes"

    with pytest.raises(ValueError, match="axis"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            same_axis,
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="opcode"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            unknown_opcode,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "typed_effect",
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
def test_offline_validator_rejects_active_differential_effect_runtime_mismatch(
    tmp_path: Path,
    typed_effect: dict,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["runtime"] = _differential_runtime()
    effect["effect"] = deepcopy(typed_effect)

    with pytest.raises(ValueError, match="active differential effect"):
        validate_knowledge_effect(
            "diagnosis_differential_rule",
            effect,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("candidate_type", "phase", "stage"),
    [
        ("clinical_closure_rule", "closure", "clinical_closure"),
        ("treatment_gate_rule", "treatment", "treatment"),
        ("treatment_sequence_rule", "treatment", "treatment"),
    ],
)
def test_offline_validator_keeps_closure_and_treatment_active_runtime_fail_closed(
    tmp_path: Path,
    candidate_type: str,
    phase: str,
    stage: str,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        phase=phase,
        candidate_type=candidate_type,
    )
    effect["runtime"] = _differential_runtime()
    effect["runtime"]["stage"] = stage

    with pytest.raises(ValueError, match="active runtime"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "candidate_type",
    sorted(KNOWLEDGE_CANDIDATE_TYPES),
)
def test_five_whitelisted_types_accept_complete_effect(
    tmp_path: Path,
    candidate_type: str,
) -> None:
    treatment = candidate_type in {"treatment_gate_rule", "treatment_sequence_rule"}
    effect = _complete_effect(
        tmp_path,
        rule_id="rule_" + candidate_type,
        treatment=treatment,
        phase="closure" if candidate_type == "clinical_closure_rule" else None,
        candidate_type=candidate_type,
    )

    validated = validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)
    candidate = knowledge_rule_candidate(
        candidate_type=candidate_type,
        effect=effect,
        project_root=tmp_path,
    )

    assert set(validated) == REQUIRED_EFFECT_FIELDS
    assert candidate["candidate_id"] == effect["rule_id"]
    assert candidate["candidate_type"] == candidate_type
    assert candidate["status"] == "candidate"


def test_unknown_clinical_candidate_type_fails_closed(tmp_path: Path) -> None:
    effect = _complete_effect(tmp_path)

    with pytest.raises(ValueError, match="unknown clinical candidate type"):
        create_candidate(
            candidate_id="unknown-clinical",
            candidate_type="diagnosis_unknown_rule",
            proposed_effect=effect,
            evidence={"source": "offline_knowledge_design"},
            project_root=tmp_path,
        )


@pytest.mark.parametrize("candidate_type", [[], {}])
def test_candidate_type_rejects_unhashable_values_as_validation_errors(
    tmp_path: Path,
    candidate_type: object,
) -> None:
    effect = _complete_effect(tmp_path)

    with pytest.raises(ValueError, match="candidate type"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", [[], {}])
def test_control_kind_rejects_unhashable_values_as_validation_errors(
    tmp_path: Path,
    kind: object,
) -> None:
    effect = _complete_effect(tmp_path)
    effect["positive_controls"][0]["kind"] = kind

    with pytest.raises(ValueError, match="control kind"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "missing",
    [
        "triggers",
        "required_evidence",
        "exclusions",
        "effect",
        "positive_controls",
        "negative_controls",
        "source_refs",
        "test_refs",
    ],
)
def test_required_effect_fields_fail_closed(tmp_path: Path, missing: str) -> None:
    effect = _complete_effect(tmp_path)
    effect.pop(missing)

    with pytest.raises(ValueError, match="knowledge effect fields"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize("candidate_type", ["treatment_gate_rule", "treatment_sequence_rule"])
def test_treatment_rules_require_near_neighbor_and_reasonable_exception_controls(
    tmp_path: Path,
    candidate_type: str,
) -> None:
    without_exception = _complete_effect(
        tmp_path,
        treatment=False,
        candidate_type=candidate_type,
    )
    with pytest.raises(ValueError, match="reasonable exception"):
        validate_knowledge_effect(candidate_type, without_exception, project_root=tmp_path)

    without_near_neighbor = _complete_effect(
        tmp_path,
        treatment=True,
        candidate_type=candidate_type,
    )
    without_near_neighbor["negative_controls"] = [
        control
        for control in without_near_neighbor["negative_controls"]
        if control["kind"] == "reasonable_exception"
    ]
    with pytest.raises(ValueError, match="near-neighbor"):
        validate_knowledge_effect(candidate_type, without_near_neighbor, project_root=tmp_path)

    without_negative = _complete_effect(
        tmp_path,
        treatment=True,
        candidate_type=candidate_type,
    )
    without_negative["negative_controls"] = []
    with pytest.raises(ValueError, match="negative_controls"):
        validate_knowledge_effect(candidate_type, without_negative, project_root=tmp_path)


@pytest.mark.parametrize(
    ("location", "leaked_value"),
    [
        ("trigger", "Patient_00001 出现对应表现"),
        ("effect", {"expected": ["完整答案检查清单"]}),
        ("effect", {"reference": "完整标准治疗原文"}),
        ("effect", {"ground_truth": "完整标准答案"}),
        ("effect", {"evaluator_reasoning": "完整评分器推理片段"}),
    ],
)
def test_answer_like_leakage_is_quarantined(
    tmp_path: Path,
    location: str,
    leaked_value: object,
) -> None:
    effect = _complete_effect(tmp_path)
    if location == "trigger":
        effect["triggers"] = [leaked_value]
    else:
        effect["effect"] = leaked_value

    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )

    assert candidate["status"] == "quarantine"
    assert candidate["quarantine_reason"] == "leakage_marker"


@pytest.mark.parametrize(
    "answer_shape",
    [
        "Diagnosis: X; Examination: Y; Treatment: Z",
        "标准答案：诊断：X；检查：Y；治疗：Z",
    ],
)
@pytest.mark.parametrize(
    "location",
    ["triggers", "required_evidence", "control_facts", "control_assertions"],
)
def test_sectioned_answer_shape_is_quarantined(
    tmp_path: Path,
    answer_shape: str,
    location: str,
) -> None:
    effect = _complete_effect(tmp_path)
    if location in {"triggers", "required_evidence"}:
        effect[location] = [answer_shape]
    elif location == "control_facts":
        effect["positive_controls"][0]["facts"] = [answer_shape]
    else:
        effect["negative_controls"][0]["assertions"] = [answer_shape]

    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )

    assert candidate["status"] == "quarantine"
    assert candidate["quarantine_reason"] == "leakage_marker"


@pytest.mark.parametrize(
    "answer_sections",
    [
        ["Diagnosis: X", "Examination: Y", "Treatment: Z"],
        ["诊断：X", "检查：Y", "治疗：Z"],
    ],
)
def test_split_sectioned_answer_shape_is_quarantined(
    tmp_path: Path,
    answer_sections: list[str],
) -> None:
    effect = _complete_effect(tmp_path)
    effect["positive_controls"][0]["facts"] = answer_sections

    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )

    assert candidate["status"] == "quarantine"
    assert candidate["quarantine_reason"] == "leakage_marker"


@pytest.mark.parametrize(
    "clinical_text",
    [
        "Diagnosis depends on examination findings before treatment is selected.",
        "诊断需要结合进一步检查结果再决定治疗。",
    ],
)
def test_unsectioned_clinical_prose_is_not_quarantined(
    tmp_path: Path,
    clinical_text: str,
) -> None:
    effect = _complete_effect(tmp_path)
    effect["triggers"] = [clinical_text]

    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )

    assert candidate["status"] == "candidate"


@pytest.mark.parametrize(
    "bad_ref",
    [
        {"path": "docs/source.md", "sha256": "0" * 64},
        {"path": "D:/absolute/source.md", "sha256": "0" * 64},
        {"path": "C:relative/source.md", "sha256": "0" * 64},
        {"path": "//server/share/source.md", "sha256": "0" * 64},
        {"path": "../source.md", "sha256": "0" * 64},
        {"path": "docs\\source.md", "sha256": "0" * 64},
    ],
)
def test_source_and_test_refs_require_posix_relative_path_and_actual_sha256(
    tmp_path: Path,
    bad_ref: dict[str, str],
) -> None:
    effect = _complete_effect(tmp_path)
    effect["source_refs"] = [bad_ref]

    with pytest.raises(ValueError, match="ref"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize("path", ["D:/absolute/source.md", "C:relative/source.md"])
def test_windows_drive_refs_are_rejected_as_non_relative_posix(tmp_path: Path, path: str) -> None:
    effect = _complete_effect(tmp_path)
    effect["source_refs"] = [{"path": path, "sha256": "0" * 64}]

    with pytest.raises(ValueError, match="project-relative POSIX"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


def test_controls_reject_unknown_fields(tmp_path: Path) -> None:
    effect = _complete_effect(tmp_path)
    effect["positive_controls"][0]["reviewer_note"] = "not part of the typed schema"

    with pytest.raises(ValueError, match="control fields"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


def test_controls_reject_duplicate_id_across_positive_and_negative_groups(
    tmp_path: Path,
) -> None:
    effect = _complete_effect(tmp_path)
    effect["negative_controls"][0]["control_id"] = effect["positive_controls"][0][
        "control_id"
    ]

    with pytest.raises(ValueError, match="duplicate control_id"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


def test_closure_effect_does_not_borrow_the_host_project_intent_registry(
    tmp_path: Path,
) -> None:
    effect = _complete_effect(
        tmp_path,
        candidate_type="clinical_closure_rule",
        phase="closure",
    )
    registry = tmp_path / "agent" / "knowledge" / "exam_intent_map.json"
    registry.unlink()

    with pytest.raises(ValueError, match="verified official intents"):
        validate_knowledge_effect(
            "clinical_closure_rule",
            effect,
            project_root=tmp_path,
        )


def test_effect_rejects_boolean_nested_inside_clinical_action(tmp_path: Path) -> None:
    effect = _complete_effect(tmp_path)
    effect["effect"] = {"enabled": True}

    with pytest.raises(ValueError, match="effect values"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize(
    ("candidate_type", "typed_effect"),
    [
        (
            "diagnosis_differential_rule",
            {
                "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
                "ranking_policy": "evidence_ordered",
            },
        ),
        (
            "clinical_closure_rule",
            {
                "add_exam_intent_ids": [
                    "exam_intent_congenital_infection_organ_involvement"
                ],
                "deduplicate": "intent_id",
            },
        ),
        (
            "diagnosis_priority_rule",
            {
                "priority_policy": "objective_evidence_first",
                "fallback_policy": "official_catalog_only",
            },
        ),
        (
            "treatment_gate_rule",
            {
                "remove_treatment_codes": ["routine_antibiotic", "prophylactic_antibiotic"],
                "preserve_treatment_codes": ["etiology_specific_care", "reassessment"],
                "gate_policy": "require_infection_evidence",
            },
        ),
        (
            "treatment_sequence_rule",
            {
                "ordered_patch_codes": ["stabilize_hemodynamics", "decongest_if_overloaded"],
                "sequence_policy": "acute_before_stable",
                "skip_patch_codes": ["force_diuresis_when_euvolemic"],
            },
        ),
    ],
)
def test_five_candidate_types_accept_only_their_structured_effect_schema(
    tmp_path: Path,
    candidate_type: str,
    typed_effect: dict,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        phase="closure" if candidate_type == "clinical_closure_rule" else None,
        candidate_type=candidate_type,
    )
    effect["effect"] = typed_effect

    assert validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)["effect"] == typed_effect


def test_diagnosis_priority_human_effect_rejects_free_diagnosis_labels(
    tmp_path: Path,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_priority_rule")
    effect["effect"]["promote_diagnosis_labels"] = ["耳部方向"]

    with pytest.raises(ValueError, match="effect fields"):
        validate_knowledge_effect("diagnosis_priority_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize(
    ("candidate_type", "foreign_effect"),
    [
        ("diagnosis_differential_rule", {"ordered_patch_codes": ["stabilize_hemodynamics"]}),
        ("clinical_closure_rule", {"add_diagnostic_axes": ["完整诊断"]}),
        ("diagnosis_priority_rule", {"add_exam_intent_ids": ["exam_intent_physical_status"]}),
        ("treatment_gate_rule", {"sequence_policy": "acute_before_stable"}),
        ("treatment_sequence_rule", {"remove_treatment_codes": ["routine_antibiotic"]}),
    ],
)
def test_type_specific_effect_field_whitelists_reject_foreign_or_unknown_fields(
    tmp_path: Path,
    candidate_type: str,
    foreign_effect: dict,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        phase="closure" if candidate_type == "clinical_closure_rule" else None,
        candidate_type=candidate_type,
    )
    effect["effect"] = foreign_effect

    with pytest.raises(ValueError, match="effect fields"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "bad_string",
    [
        "诊断考虑肺炎。检查建议胸部CT、血常规、病原学检测。治疗给予抗菌药、补液并住院观察。",
        "A" * 65,
    ],
)
def test_recursive_effect_strings_reject_long_or_complete_answer_prose(
    tmp_path: Path,
    bad_string: str,
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["effect"] = {
        "add_diagnostic_axes": [bad_string],
        "ranking_policy": "evidence_ordered",
    }

    with pytest.raises(ValueError, match="effect string|answer-like"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "split_answer",
    [
        ["诊断肺炎", "检查胸部CT", "治疗抗菌药"],
        ["ＤＩＡＧＮＯＳＩＳ pneumonia", "ＥＸＡＭ chest_ct", "ＴＲＥＡＴＭＥＮＴ antibiotic"],
    ],
)
def test_recursive_effect_strings_reject_complete_answer_split_across_short_labels(
    tmp_path: Path,
    split_answer: list[str],
) -> None:
    effect = _complete_effect(tmp_path, candidate_type="diagnosis_differential_rule")
    effect["effect"] = {
        "add_diagnostic_axes": split_answer,
        "ranking_policy": "evidence_ordered",
    }

    with pytest.raises(ValueError, match="answer-like"):
        validate_knowledge_effect("diagnosis_differential_rule", effect, project_root=tmp_path)


@pytest.mark.parametrize(
    "leaked_key",
    [
        "ＥＸＰＥＣＴＥＤ",
        "Final_Diagnosis",
        "ＴＲＥＡＴＭＥＮＴ＿ＰＬＡＮ",
    ],
)
def test_nfkc_and_case_normalized_leakage_markers_are_quarantined(
    tmp_path: Path,
    leaked_key: str,
) -> None:
    effect = _complete_effect(tmp_path)
    effect["effect"] = {
        "add_diagnostic_axes": ["先天性风疹方向"],
        "ranking_policy": "evidence_ordered",
    }
    effect["triggers"] = [leaked_key + " 隐藏答案"]

    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )

    assert candidate["status"] == "quarantine"
    assert candidate["quarantine_reason"] == "leakage_marker"


@pytest.mark.parametrize(
    ("candidate_type", "phase"),
    [
        ("diagnosis_differential_rule", "treatment"),
        ("clinical_closure_rule", "diagnosis"),
        ("diagnosis_priority_rule", "closure"),
        ("treatment_gate_rule", "diagnosis"),
        ("treatment_sequence_rule", "closure"),
    ],
)
def test_candidate_type_requires_matching_scope_phase(
    tmp_path: Path,
    candidate_type: str,
    phase: str,
) -> None:
    effect = _complete_effect(
        tmp_path,
        treatment=candidate_type.startswith("treatment_"),
        candidate_type=candidate_type,
    )
    effect["scope"] = {"phase": phase, "application": "trigger_bound"}

    with pytest.raises(ValueError, match="scope phase"):
        validate_knowledge_effect(candidate_type, effect, project_root=tmp_path)


def test_batch_materializes_exact_six_pending_candidates_without_decisions_or_pointer_change(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    artifact_root = project_root / "outputs" / "offline" / "knowledge"
    source_paths = [
        project_root / "docs" / "design.md",
        project_root / "docs" / "offline-question.md",
    ]
    test_paths = [
        project_root / "tests" / "test_diagnosis.py",
        project_root / "tests" / "test_treatment.py",
    ]
    for index, path in enumerate(source_paths + test_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local evidence %d\n" % index, encoding="utf-8")
    _write_official_intent_registry(project_root)
    pointer = project_root / "releases" / "current.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text('{"release":"unchanged"}\n', encoding="utf-8")
    pointer_hash = file_hash(pointer)

    result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=source_paths,
        test_files=test_paths,
        artifact_root=artifact_root,
    )

    batch_dir = Path(result["batch_dir"])
    assert result["reused"] is False
    assert file_hash(pointer) == pointer_hash
    assert not (batch_dir / "decisions").exists()
    assert not list(batch_dir.rglob("*decision*"))
    assert {path.relative_to(batch_dir).as_posix() for path in batch_dir.rglob("*") if path.is_file()} == {
        "source_receipt.json",
        "review_checklist.json",
        *("candidates/%s.json" % rule_id for rule_id in EXPECTED_RULES),
    }
    receipt = read_json(batch_dir / "source_receipt.json")
    checklist = read_json(batch_dir / "review_checklist.json")
    assert receipt["batch_id"] == result["batch_id"]
    assert checklist["approval_status"] == "pending_user_review"
    assert set(checklist["rule_ids"]) == set(EXPECTED_RULES)
    assert {item["path"] for item in receipt["source_files"]} == {
        "docs/design.md",
        "docs/offline-question.md",
    }
    assert {item["path"] for item in receipt["test_files"]} == {
        "tests/test_diagnosis.py",
        "tests/test_treatment.py",
    }
    assert all(
        item["sha256"] == file_hash(project_root / item["path"])
        for item in receipt["source_files"] + receipt["test_files"]
    )
    assert len(receipt["candidate_hashes"]) == len(EXPECTED_RULES)

    found = {}
    effects_by_rule_id = {}
    control_counts = {}
    for rule_id, candidate_type in EXPECTED_RULES.items():
        candidate_path = batch_dir / "candidates" / (rule_id + ".json")
        candidate = load_candidate(candidate_path, project_root=project_root)
        found[rule_id] = candidate["candidate_type"]
        effect = candidate["proposed_effect"]
        effects_by_rule_id[rule_id] = effect
        assert candidate["status"] == "candidate"
        assert receipt["candidate_hashes"][rule_id] == candidate["candidate_hash"]
        assert set(effect) == REQUIRED_EFFECT_FIELDS
        assert effect["schema_version"] == "clinical-knowledge-candidate/v2"
        assert effect["runtime"]["stage"] == RUNTIME_STAGE_BY_TYPE[candidate_type]
        assert effect["scope"]["phase"] in {"diagnosis", "closure", "treatment"}
        assert effect["scope"]["application"] == "trigger_bound"
        assert len(effect["positive_controls"]) >= 2
        assert len(effect["negative_controls"]) >= 1
        control_counts[rule_id] = {
            kind: sum(
                control["kind"] == kind for control in effect["negative_controls"]
            )
            for kind in ("near_neighbor", "reasonable_exception")
        }
        if candidate_type.startswith("treatment_"):
            assert any(
                control["kind"] == "reasonable_exception"
                for control in effect["negative_controls"]
            )
    assert found == EXPECTED_RULES
    assert [
        rule_id
        for rule_id, effect in effects_by_rule_id.items()
        if effect["runtime"]["status"] == "active"
    ] == [
        "congenital_infection_differential",
        "symptom_over_background_condition",
    ]
    assert sum(
        effect["runtime"]["status"] == "audit_only"
        for effect in effects_by_rule_id.values()
    ) == 4
    assert all(
        set(effect["runtime"]) == {"status", "stage"}
        for rule_id, effect in effects_by_rule_id.items()
        if rule_id
        not in {
            "congenital_infection_differential",
            "symptom_over_background_condition",
        }
    )

    differential_effect = effects_by_rule_id["congenital_infection_differential"]
    assert differential_effect["runtime"] == _differential_runtime()
    differential_controls = {
        control["control_id"]: control
        for control in differential_effect["positive_controls"]
        + differential_effect["negative_controls"]
    }
    assert {
        control_id: control["kind"]
        for control_id, control in differential_controls.items()
    } == {
        "congenital_multi_system": "positive",
        "congenital_hearing_signal": "positive",
        "congenital_rubella_rank_signal": "positive",
        "congenital_cmv_rank_signal": "positive",
        "isolated_neonatal_jaundice": "near_neighbor",
        "postnatal_infection_pattern": "near_neighbor",
        "documented_single_pathogen": "reasonable_exception",
    }
    assert len(differential_effect["positive_controls"]) == 4
    assert len(differential_effect["negative_controls"]) == 3

    symptom_effect = effects_by_rule_id["symptom_over_background_condition"]
    assert symptom_effect["runtime"] == _runtime("diagnosis_priority_rule")
    assert symptom_effect["effect"] == {
        "priority_policy": "objective_evidence_first",
        "fallback_policy": "official_catalog_only",
    }
    assert "耳部方向" not in json.dumps(
        symptom_effect["effect"], ensure_ascii=False, sort_keys=True
    )
    assert all("时间演变" not in item for item in symptom_effect["required_evidence"])
    assert any(
        "异常检查" in item and "客观体征" in item
        for item in symptom_effect["required_evidence"]
    )
    trigger_text = " ".join(symptom_effect["triggers"])
    assert all(
        marker in trigger_text
        for marker in ("当前问题候选", "可追溯", "同系统", "客观证据", "无解释关系")
    )
    exclusion_text = " ".join(symptom_effect["exclusions"])
    assert all(
        marker in exclusion_text
        for marker in (
            "基础病能够解释",
            "已确诊急症",
            "疑似急症",
            "检查正常",
            "尚未检查",
            "结果不确定",
        )
    )
    controls = {
        control["control_id"]: control
        for control in symptom_effect["positive_controls"]
        + symptom_effect["negative_controls"]
    }
    assert {"hearing_over_hypertension", "focal_symptom_over_history"}.issubset(controls)
    assert controls["confirmed_hypertensive_emergency"]["kind"] == "reasonable_exception"
    assert controls["suspected_hypertensive_emergency"]["kind"] == "reasonable_exception"
    assert "新发或进展性急性靶器官损害" in " ".join(
        controls["confirmed_hypertensive_emergency"]["facts"]
    )
    assert "急性靶器官损害尚待排除" in " ".join(
        controls["suspected_hypertensive_emergency"]["facts"]
    )
    for control_id in (
        "confirmed_hypertensive_emergency",
        "suspected_hypertensive_emergency",
    ):
        assertions = " ".join(controls[control_id]["assertions"])
        assert "基础病急症优先" in assertions
        assert "独立耳部轴不得删除" in assertions
    for control_id in (
        "severe_elevation_without_acute_target_organ_damage",
        "transient_elevation_without_acute_target_organ_damage",
    ):
        assert controls[control_id]["kind"] == "positive"
        assert "无急性靶器官损害" in " ".join(controls[control_id]["facts"])
        assert "提升当前问题候选" in " ".join(controls[control_id]["assertions"])
    for control_id in (
        "subjective_symptom_normal_exam",
        "subjective_symptom_not_examined",
        "subjective_symptom_uncertain_result",
    ):
        assert controls[control_id]["kind"] == "near_neighbor"
    assert {
        rule_id: counts["near_neighbor"]
        for rule_id, counts in control_counts.items()
        if not 2 <= counts["near_neighbor"] <= 4
    } == {}
    assert {
        rule_id: counts["reasonable_exception"]
        for rule_id, counts in control_counts.items()
        if not 1 <= counts["reasonable_exception"] <= 3
    } == {}

    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in (batch_dir / "candidates").glob("*.json")
    )
    assert "Patient_" not in serialized
    assert '"expected"' not in serialized
    assert '"reference"' not in serialized
    assert '"ground_truth"' not in serialized
    assert '"evaluator_reasoning"' not in serialized


def test_load_candidate_rejects_typed_effect_that_was_valid_only_under_old_contract(
    tmp_path: Path,
) -> None:
    effect = _complete_effect(tmp_path)
    candidate = knowledge_rule_candidate(
        candidate_type="diagnosis_differential_rule",
        effect=effect,
        project_root=tmp_path,
    )
    candidate["proposed_effect"]["effect"] = {
        "add_diagnostic_axes": ["先天性风疹方向"],
        "ranking_policy": "evidence_ordered",
        "free_text": "诊断考虑肺炎。检查建议胸部CT。治疗给予抗菌药。",
    }
    candidate["effect_hash"] = content_hash(candidate["proposed_effect"])
    candidate["candidate_hash"] = content_hash(
        {
            key: value
            for key, value in candidate.items()
            if key not in {"candidate_hash", "effect_hash"}
        }
    )
    path = tmp_path / "candidate.json"
    write_candidate(path, candidate)

    with pytest.raises(ValueError, match="effect fields"):
        load_candidate(path, project_root=tmp_path)


def test_batch_schema_change_creates_new_pending_review_package_without_overwriting_old_batch(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "docs" / "source.md"
    test_file = project_root / "tests" / "test_rule.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source\n", encoding="utf-8")
    test_file.write_text("test\n", encoding="utf-8")
    _write_official_intent_registry(project_root)
    artifact_root = project_root / "outputs" / "offline" / "knowledge"
    old_batch_id = "legacy-batch"
    old_batch_dir = artifact_root / old_batch_id
    old_batch_dir.mkdir(parents=True)
    old_checklist = {
        "schema_version": "knowledge-review-checklist/v1",
        "batch_id": old_batch_id,
        "approval_status": "pending_user_review",
        "review_package_role": "superseded_not_final",
    }
    (old_batch_dir / "review_checklist.json").write_text(
        json.dumps(old_checklist, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    old_bytes = (old_batch_dir / "review_checklist.json").read_bytes()

    result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
        supersedes_batch_id=old_batch_id,
    )

    assert result["batch_id"] != old_batch_id
    assert (old_batch_dir / "review_checklist.json").read_bytes() == old_bytes
    checklist = read_json(Path(result["batch_dir"]) / "review_checklist.json")
    assert checklist["approval_status"] == "pending_user_review"
    assert checklist["review_package_role"] == "final_review_package_candidate"
    assert checklist["supersedes_batch_id"] == old_batch_id


def test_batch_reuses_identical_input_and_changes_id_when_source_bytes_change(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "docs" / "source.md"
    test_file = project_root / "tests" / "test_rule.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("version one\n", encoding="utf-8")
    test_file.write_text("def test_rule():\n    pass\n", encoding="utf-8")
    _write_official_intent_registry(project_root)
    artifact_root = project_root / "outputs" / "offline" / "knowledge"

    first = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )
    second = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )
    source.write_text("version two\n", encoding="utf-8")
    changed = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )

    assert second["reused"] is True
    assert second["batch_id"] == first["batch_id"]
    assert changed["batch_id"] != first["batch_id"]
    assert Path(first["batch_dir"]).exists()
    assert Path(changed["batch_dir"]).exists()


def test_batch_refuses_existing_batch_directory_without_complete_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "docs" / "source.md"
    test_file = project_root / "tests" / "test_rule.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source\n", encoding="utf-8")
    test_file.write_text("test\n", encoding="utf-8")
    _write_official_intent_registry(project_root)
    artifact_root = project_root / "outputs" / "offline" / "knowledge"
    first = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )
    batch_dir = Path(first["batch_dir"])
    for path in sorted(batch_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    batch_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises((FileExistsError, ValueError)):
        build_knowledge_candidate_batch(
            project_root=project_root,
            source_files=[source],
            test_files=[test_file],
            artifact_root=artifact_root,
        )
    assert not list(batch_dir.rglob("*.json"))


def test_batch_refuses_reuse_if_existing_batch_has_unexpected_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "docs" / "source.md"
    test_file = project_root / "tests" / "test_rule.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source\n", encoding="utf-8")
    test_file.write_text("test\n", encoding="utf-8")
    _write_official_intent_registry(project_root)
    artifact_root = project_root / "outputs" / "offline" / "knowledge"
    result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )
    unexpected = Path(result["batch_dir"]) / "compiled" / "knowledge_rules.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected files"):
        build_knowledge_candidate_batch(
            project_root=project_root,
            source_files=[source],
            test_files=[test_file],
            artifact_root=artifact_root,
        )


def test_batch_refuses_reuse_if_existing_content_was_changed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "docs" / "source.md"
    test_file = project_root / "tests" / "test_rule.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source\n", encoding="utf-8")
    test_file.write_text("test\n", encoding="utf-8")
    _write_official_intent_registry(project_root)
    artifact_root = project_root / "outputs" / "offline" / "knowledge"
    result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=[source],
        test_files=[test_file],
        artifact_root=artifact_root,
    )
    candidate_path = Path(result["batch_dir"]) / "candidates" / "congenital_infection_differential.json"
    tampered = deepcopy(read_json(candidate_path))
    tampered["status"] = "quarantine"
    candidate_path.write_text(str(tampered), encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError)):
        build_knowledge_candidate_batch(
            project_root=project_root,
            source_files=[source],
            test_files=[test_file],
            artifact_root=artifact_root,
        )
