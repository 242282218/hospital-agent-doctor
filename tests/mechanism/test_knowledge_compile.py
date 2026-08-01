from __future__ import annotations

import json
import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from agent.knowledge import typed_rule_engine
from offline.artifacts import canonical_json, content_hash, file_hash
from offline.candidates import load_candidate
import offline.knowledge_compile as knowledge_compile
from offline.knowledge_compile import compile_knowledge_rules
from scripts.knowledge.build_knowledge_candidates import (
    EXPECTED_RULES,
    build_knowledge_candidate_batch,
)


COMPILED_RULE_FIELDS = {
    "rule_id",
    "candidate_type",
    "candidate_hash",
    "effect_hash",
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
    "runtime",
}

CONTROL_SET_FIELDS = {
    "schema_version",
    "compiled_rules_hash",
    "catalog_hashes",
    "control_count",
    "controls",
    "control_set_hash",
}
STRUCTURED_CONTROL_FIELDS = {
    "rule_id",
    "control_id",
    "kind",
    "stage",
    "context",
    "expected_outcome",
}
RULE_CONTEXT_FIELDS = {
    "diagnosis_candidates",
    "preferred_diagnosis",
    "diagnostic_axis_ids",
    "exam_intent_ids",
    "treatment_codes",
    "fact_codes",
}
RULE_CANDIDATE_FIELDS = {
    "official_name",
    "role",
    "support_level",
    "complaint_relation",
    "urgency",
    "evidence_codes",
}
ACTIVE_RULE_IDS = (
    "congenital_infection_differential",
    "symptom_over_background_condition",
)


def test_active_rule_control_set_emits_all_typed_inventory() -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    build_control_set = getattr(
        builder_module,
        "build_active_rule_control_set",
    )

    result = build_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
        active_rule_ids=tuple(reversed(ACTIVE_RULE_IDS)),
    )

    assert set(result) == CONTROL_SET_FIELDS
    assert result["schema_version"] == "knowledge-rule-controls/v1"
    assert result["compiled_rules_hash"] == "1" * 64
    assert result["catalog_hashes"] == {
        "data/ref_data/diseases_catalog.json": "2" * 64,
    }
    assert result["control_count"] == 17
    canonical_controls = sorted(
        result["controls"],
        key=lambda item: (item["rule_id"], item["control_id"]),
    )
    assert result["control_set_hash"] == content_hash(
        {
            "schema_version": "knowledge-rule-controls/v1",
            "compiled_rules_hash": "1" * 64,
            "catalog_hashes": {
                "data/ref_data/diseases_catalog.json": "2" * 64,
            },
            "control_count": 17,
            "controls": canonical_controls,
        }
    )
    assert [(item["rule_id"], item["control_id"]) for item in result["controls"]] == sorted(
        (item["rule_id"], item["control_id"]) for item in result["controls"]
    )
    inventory = {
        rule_id: {
            item["control_id"]: item["kind"]
            for item in result["controls"]
            if item["rule_id"] == rule_id
        }
        for rule_id in ACTIVE_RULE_IDS
    }
    assert inventory == {
        "congenital_infection_differential": {
            "congenital_multi_system": "positive",
            "congenital_hearing_signal": "positive",
            "congenital_rubella_rank_signal": "positive",
            "congenital_cmv_rank_signal": "positive",
            "isolated_neonatal_jaundice": "near_neighbor",
            "postnatal_infection_pattern": "near_neighbor",
            "documented_single_pathogen": "reasonable_exception",
        },
        "symptom_over_background_condition": {
            "hearing_over_hypertension": "positive",
            "focal_symptom_over_history": "positive",
            "severe_elevation_without_acute_target_organ_damage": "positive",
            "transient_elevation_without_acute_target_organ_damage": "positive",
            "background_explains_current_problem": "near_neighbor",
            "subjective_symptom_normal_exam": "near_neighbor",
            "subjective_symptom_not_examined": "near_neighbor",
            "subjective_symptom_uncertain_result": "near_neighbor",
            "confirmed_hypertensive_emergency": "reasonable_exception",
            "suspected_hypertensive_emergency": "reasonable_exception",
        },
    }
    assert [item["kind"] for item in result["controls"]].count("positive") == 8
    assert [item["kind"] for item in result["controls"]].count("near_neighbor") == 6
    assert [item["kind"] for item in result["controls"]].count("reasonable_exception") == 3
    for control in result["controls"]:
        assert set(control) == STRUCTURED_CONTROL_FIELDS
        assert control["stage"] == "diagnosis_candidates"
        assert set(control["context"]) == RULE_CONTEXT_FIELDS
        assert set(control["expected_outcome"]) == {
            "outcome",
            "reason_code",
            "output_context",
        }
        assert set(control["expected_outcome"]["output_context"]) == RULE_CONTEXT_FIELDS
        for context in (
            control["context"],
            control["expected_outcome"]["output_context"],
        ):
            for candidate in context["diagnosis_candidates"]:
                assert set(candidate) == RULE_CANDIDATE_FIELDS
                assert "is_official" not in candidate

    congenital_controls = {
        item["control_id"]: item
        for item in result["controls"]
        if item["rule_id"] == "congenital_infection_differential"
    }
    for control in congenital_controls.values():
        initial = control["context"]
        expected = control["expected_outcome"]["output_context"]
        assert initial["diagnosis_candidates"]
        assert initial["preferred_diagnosis"] == "耳鸣"
        assert "other_diagnostic_axis" in initial["diagnostic_axis_ids"]
        assert "other_diagnostic_axis" in expected["diagnostic_axis_ids"]
        for field in (
            "diagnosis_candidates",
            "preferred_diagnosis",
            "exam_intent_ids",
            "treatment_codes",
            "fact_codes",
        ):
            assert expected[field] == initial[field]
    assert congenital_controls["congenital_multi_system"]["expected_outcome"][
        "output_context"
    ]["diagnostic_axis_ids"] == [
        "congenital_rubella",
        "congenital_cmv",
        "other_diagnostic_axis",
    ]
    assert congenital_controls["congenital_hearing_signal"]["expected_outcome"][
        "output_context"
    ]["diagnostic_axis_ids"] == [
        "congenital_rubella",
        "congenital_cmv",
        "other_diagnostic_axis",
    ]
    assert congenital_controls["congenital_rubella_rank_signal"]["expected_outcome"][
        "output_context"
    ]["diagnostic_axis_ids"][:2] == ["congenital_rubella", "congenital_cmv"]
    assert congenital_controls["congenital_cmv_rank_signal"]["expected_outcome"][
        "output_context"
    ]["diagnostic_axis_ids"][:2] == ["congenital_cmv", "congenital_rubella"]
    exception = congenital_controls["documented_single_pathogen"]
    assert exception["expected_outcome"] == {
        "outcome": "matched_no_change",
        "reason_code": "congenital_infection_axes_already_ranked",
        "output_context": exception["context"],
    }
    exception_facts = set(exception["context"]["fact_codes"])
    assert "cmv_saliva_or_urine_pcr_positive_within_21_days" in exception_facts
    assert exception_facts.isdisjoint(
        {
            "congenital_cataract",
            "patent_ductus_arteriosus",
            "rubella_igm_positive_in_infant",
            "rubella_pcr_positive_in_infant",
        }
    )
    assert exception["context"]["diagnostic_axis_ids"] == [
        "congenital_cmv",
        "other_diagnostic_axis",
    ]


def test_symptom_priority_control_set_remains_a_compatible_wrapper() -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )

    legacy = builder_module.build_symptom_priority_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
    )
    generic = builder_module.build_active_rule_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
        active_rule_ids=("symptom_over_background_condition",),
    )

    assert legacy == generic


@pytest.mark.parametrize(
    "active_rule_ids",
    [
        ("unknown_active_rule",),
        ("symptom_over_background_condition", "unknown_active_rule"),
    ],
)
def test_active_rule_control_set_fails_closed_for_unknown_active_rule_id(
    active_rule_ids: tuple[str, ...],
) -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )

    with pytest.raises(ValueError, match="unknown active rule_id"):
        builder_module.build_active_rule_control_set(
            "1" * 64,
            disease_catalog_hash="2" * 64,
            active_rule_ids=active_rule_ids,
        )


def test_symptom_priority_control_set_distinguishes_severe_urgent_from_emergency() -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    control_set = builder_module.build_symptom_priority_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
    )
    controls = {item["control_id"]: item for item in control_set["controls"]}
    severe = controls["severe_elevation_without_acute_target_organ_damage"]
    severe_background = next(
        candidate
        for candidate in severe["context"]["diagnosis_candidates"]
        if candidate["role"] == "background_condition"
    )
    severe_current = next(
        candidate
        for candidate in severe["context"]["diagnosis_candidates"]
        if candidate["role"] == "current_problem"
    )

    assert severe_background["urgency"] == "urgent"
    assert severe_current["support_level"] == "objective"
    assert severe_current["complaint_relation"] == "unrelated"
    assert severe_current["evidence_codes"]
    assert severe["expected_outcome"]["outcome"] == "applied"
    assert severe["expected_outcome"]["reason_code"] == (
        "supported_current_problem_promoted"
    )
    assert severe["expected_outcome"]["output_context"]["diagnosis_candidates"][0] == (
        severe_current
    )
    assert severe["expected_outcome"]["output_context"]["preferred_diagnosis"] == (
        severe_current["official_name"]
    )
    assert severe["expected_outcome"]["output_context"]["diagnosis_candidates"][1][
        "urgency"
    ] == "urgent"

    transient = controls["transient_elevation_without_acute_target_organ_damage"]
    transient_background = next(
        candidate
        for candidate in transient["context"]["diagnosis_candidates"]
        if candidate["role"] == "background_condition"
    )
    assert transient_background["urgency"] == "routine"

    for control_id in (
        "confirmed_hypertensive_emergency",
        "suspected_hypertensive_emergency",
    ):
        emergency = controls[control_id]
        emergency_background = next(
            candidate
            for candidate in emergency["context"]["diagnosis_candidates"]
            if candidate["role"] == "background_condition"
        )
        assert emergency_background["urgency"] == "emergency"
        assert emergency["expected_outcome"]["outcome"] == "matched_no_change"


def test_control_set_hash_binds_compiled_rules_catalog_and_count() -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    build_control_set = builder_module.build_symptom_priority_control_set
    first = build_control_set("1" * 64, disease_catalog_hash="2" * 64)
    compiled_changed = build_control_set("3" * 64, disease_catalog_hash="2" * 64)
    catalog_changed = build_control_set("1" * 64, disease_catalog_hash="4" * 64)

    assert compiled_changed["controls"] == first["controls"]
    assert catalog_changed["controls"] == first["controls"]
    assert compiled_changed["control_set_hash"] != first["control_set_hash"]
    assert catalog_changed["control_set_hash"] != first["control_set_hash"]
    assert knowledge_compile.knowledge_control_set_hash(
        schema_version=first["schema_version"],
        compiled_rules_hash=first["compiled_rules_hash"],
        catalog_hashes=first["catalog_hashes"],
        control_count=first["control_count"] + 1,
        controls=first["controls"],
    ) != first["control_set_hash"]
    assert knowledge_compile.knowledge_control_set_hash(
        schema_version=first["schema_version"],
        compiled_rules_hash=first["compiled_rules_hash"],
        catalog_hashes=first["catalog_hashes"],
        control_count=first["control_count"],
        controls=list(reversed(first["controls"])),
    ) == first["control_set_hash"]


@pytest.mark.parametrize("invalid_hash", [None, "A" * 64, "1" * 63])
def test_control_set_builder_rejects_invalid_catalog_hash(invalid_hash: object) -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )

    with pytest.raises(ValueError, match="disease_catalog_hash must be a lowercase sha256"):
        builder_module.build_symptom_priority_control_set(
            "1" * 64,
            disease_catalog_hash=invalid_hash,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("rule_id", []), ("control_id", 7)],
)
def test_control_set_hash_rejects_non_string_control_identifiers(
    field: str,
    invalid_value: object,
) -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    control_set = builder_module.build_symptom_priority_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
    )
    invalid_controls = deepcopy(control_set["controls"])
    invalid_controls[0][field] = invalid_value

    with pytest.raises(
        ValueError,
        match="control rule_id and control_id must be strings",
    ):
        knowledge_compile.knowledge_control_set_hash(
            schema_version=control_set["schema_version"],
            compiled_rules_hash=control_set["compiled_rules_hash"],
            catalog_hashes=control_set["catalog_hashes"],
            control_count=control_set["control_count"],
            controls=invalid_controls,
        )


def _non_json_controls(
    controls: list[dict[str, Any]],
    invalid_case: str,
) -> list[dict[str, Any]]:
    invalid = deepcopy(controls)
    if invalid_case == "mixed_key":
        invalid[0][7] = "non-string key"
    else:
        invalid[0]["context"]["fact_codes"].append(object())
    return invalid


@pytest.mark.parametrize("invalid_case", ["mixed_key", "nested_unserializable"])
def test_control_set_hash_rejects_non_json_serializable_controls(
    invalid_case: str,
) -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    control_set = builder_module.build_symptom_priority_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
    )

    with pytest.raises(
        ValueError,
        match="control set canonical core is not JSON serializable",
    ):
        knowledge_compile.knowledge_control_set_hash(
            schema_version=control_set["schema_version"],
            compiled_rules_hash=control_set["compiled_rules_hash"],
            catalog_hashes=control_set["catalog_hashes"],
            control_count=control_set["control_count"],
            controls=_non_json_controls(control_set["controls"], invalid_case),
        )


def test_control_set_hash_rejects_duplicate_control_pair() -> None:
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    control_set = builder_module.build_symptom_priority_control_set(
        "1" * 64,
        disease_catalog_hash="2" * 64,
    )
    duplicated = deepcopy(control_set["controls"])
    duplicated.append(deepcopy(duplicated[0]))

    with pytest.raises(ValueError, match="duplicate control rule_id/control_id pair"):
        knowledge_compile.knowledge_control_set_hash(
            schema_version=control_set["schema_version"],
            compiled_rules_hash=control_set["compiled_rules_hash"],
            catalog_hashes=control_set["catalog_hashes"],
            control_count=len(duplicated),
            controls=duplicated,
        )


@pytest.fixture()
def acceptance_inputs(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    project_root, candidates = candidate_batch
    catalog_path = project_root / "data" / "ref_data" / "diseases_catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "diseases": {
                    "心内科": ["原发性高血压"],
                    "耳鼻喉科": ["耳鸣"],
                    "骨科": ["跟骨骨折"],
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pack = compile_knowledge_rules(candidates, project_root=project_root)
    builder_module = importlib.import_module(
        "scripts.knowledge.build_knowledge_acceptance_controls"
    )
    active_rule_ids = tuple(
        sorted(
            rule["rule_id"]
            for rule in pack["rules"]
            if rule["runtime"]["status"] == "active"
        )
    )
    control_set = builder_module.build_active_rule_control_set(
        pack["rules_hash"],
        disease_catalog_hash=file_hash(catalog_path),
        active_rule_ids=active_rule_ids,
    )
    return project_root, catalog_path, pack, control_set


def test_acceptance_executes_all_active_controls_with_the_runtime_engine(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    pack_before = deepcopy(pack)
    controls_before = deepcopy(control_set)
    calls = 0
    real_apply_rules = typed_rule_engine.apply_rules

    def counted_apply_rules(*args: Any, **kwargs: Any) -> typed_rule_engine.RuleResult:
        nonlocal calls
        calls += 1
        return real_apply_rules(*args, **kwargs)

    monkeypatch.setattr(typed_rule_engine, "apply_rules", counted_apply_rules)

    report = knowledge_compile.build_knowledge_acceptance(
        pack,
        control_set,
        project_root=project_root,
        disease_catalog_path=catalog_path,
    )

    assert calls == 42
    assert pack == pack_before
    assert control_set == controls_before
    assert report["schema_version"] == "offline-knowledge-acceptance/v1"
    assert report["scope"] == {
        "active_rule_ids": [
            "congenital_infection_differential",
            "symptom_over_background_condition",
        ],
        "evaluated_active_rule_ids": [
            "congenital_infection_differential",
            "symptom_over_background_condition",
        ],
        "audit_only_rule_ids": [
            "congenital_infection_organ_closure",
            "hfref_phase_ordered_treatment",
            "negative_culture_antibiotic_stewardship",
            "qt_tricyclic_structural_heart_risk",
        ],
        "unevaluated_rule_ids": [
            "congenital_infection_organ_closure",
            "hfref_phase_ordered_treatment",
            "negative_culture_antibiotic_stewardship",
            "qt_tricyclic_structural_heart_risk",
        ],
    }
    assert report["input_hashes"] == {
        "compiled_pack_hash": content_hash(pack),
        "compiled_rules_hash": pack["rules_hash"],
        "control_set_hash": control_set["control_set_hash"],
        "catalog_hashes": {
            "data/ref_data/diseases_catalog.json": file_hash(catalog_path),
        },
    }
    assert report["metrics"] == {
        "positive_hits": 8,
        "misses": 0,
        "false_positives": 0,
        "exceptions_preserved": 3,
        "exception_failures": 0,
        "idempotency_failures": 0,
        "control_failures": 0,
        "p0_count": 0,
        "p0_applicable": False,
        "p0_status": "not_evaluated",
        "treatment_active_rule_count": 0,
    }
    assert report["status"] == "active_scope_passed"
    assert report["release_gate_passed"] is False
    assert report["acceptance_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "acceptance_hash"}
    )

    rules = {item["rule_id"]: item for item in report["rules"]}
    differential = rules["congenital_infection_differential"]
    symptom = rules["symptom_over_background_condition"]
    assert differential["acceptance_status"] == "active_scope_passed"
    assert differential["control_count"] == 7
    assert differential["hits"] == 4
    assert differential["exceptions_preserved"] == 1
    assert symptom["acceptance_status"] == "active_scope_passed"
    assert symptom["control_count"] == 10
    assert symptom["hits"] == 4
    assert symptom["exceptions_preserved"] == 2
    for active in (differential, symptom):
        assert active["p0_count"] == 0
        assert active["p0_applicable"] is False
        assert active["p0_status"] == "not_evaluated"
        assert all(
            control["passed"] and control["idempotent"]
            for control in active["controls"]
        )
        assert all(
            control["behavior_matches_oracle"]
            and control["kind_semantics_passed"]
            for control in active["controls"]
        )
        assert all(len(control["decisions"]) == 2 for control in active["controls"])
        assert all(
            len(control["replay"]["decisions"]) == 2
            for control in active["controls"]
        )
        assert all(control["replay"]["target_decision"] for control in active["controls"])
        assert all(
            control["replay"]["before_hash"]
            == control["replay"]["after_hash"]
            == control["after_hash"]
            for control in active["controls"]
        )
    actual = {
        control["control_id"]: control["target_decision"]
        for control in symptom["controls"]
    }
    assert actual["hearing_over_hypertension"] == {
        "outcome": "applied",
        "reason_code": "supported_current_problem_promoted",
    }
    assert actual["severe_elevation_without_acute_target_organ_damage"] == {
        "outcome": "applied",
        "reason_code": "supported_current_problem_promoted",
    }
    assert actual["transient_elevation_without_acute_target_organ_damage"] == {
        "outcome": "applied",
        "reason_code": "supported_current_problem_promoted",
    }
    assert actual["background_explains_current_problem"] == {
        "outcome": "excluded",
        "reason_code": "background_explains_current_problem",
    }
    assert actual["subjective_symptom_not_examined"] == {
        "outcome": "not_matched",
        "reason_code": "no_supported_current_problem",
    }
    assert actual["confirmed_hypertensive_emergency"] == {
        "outcome": "matched_no_change",
        "reason_code": "emergency_priority_preserved",
    }
    assert actual["suspected_hypertensive_emergency"] == {
        "outcome": "matched_no_change",
        "reason_code": "emergency_priority_preserved",
    }
    hearing = next(
        control
        for control in symptom["controls"]
        if control["control_id"] == "hearing_over_hypertension"
    )
    assert hearing["replay"]["target_decision"] == {
        "outcome": "matched_no_change",
        "reason_code": "supported_current_problem_already_preferred",
    }
    differential_actual = {
        control["control_id"]: control["target_decision"]
        for control in differential["controls"]
    }
    for control_id in (
        "congenital_multi_system",
        "congenital_hearing_signal",
        "congenital_rubella_rank_signal",
        "congenital_cmv_rank_signal",
    ):
        assert differential_actual[control_id] == {
            "outcome": "applied",
            "reason_code": "congenital_infection_axes_expanded",
        }
    for control_id in (
        "isolated_neonatal_jaundice",
        "postnatal_infection_pattern",
    ):
        assert differential_actual[control_id] == {
            "outcome": "not_matched",
            "reason_code": "congenital_infection_exposure_not_matched",
        }
    assert differential_actual["documented_single_pathogen"] == {
        "outcome": "matched_no_change",
        "reason_code": "congenital_infection_axes_already_ranked",
    }
    for rule_id in report["scope"]["audit_only_rule_ids"]:
        assert rules[rule_id]["acceptance_status"] == "audit_only_unverified"
        assert rules[rule_id]["control_count"] == 0
        assert rules[rule_id]["hits"] == 0
        assert rules[rule_id]["p0_count"] == 0
        assert rules[rule_id]["p0_applicable"] is False
        assert rules[rule_id]["p0_status"] == "not_evaluated"

    audit_treatment_rules = [
        rule
        for rule in rules.values()
        if rule["candidate_type"]
        in {"treatment_gate_rule", "treatment_sequence_rule"}
    ]
    assert len(audit_treatment_rules) == 3
    assert all(
        rule["acceptance_status"] == "audit_only_unverified"
        and rule["p0_applicable"] is False
        and rule["p0_status"] == "not_evaluated"
        for rule in audit_treatment_rules
    )


def _rehash_control_set(control_set: dict[str, Any]) -> None:
    controls = sorted(
        control_set["controls"],
        key=lambda item: (
            item.get("rule_id", "") if isinstance(item, dict) else "",
            item.get("control_id", "") if isinstance(item, dict) else "",
        ),
    )
    control_set["control_count"] = len(controls)
    control_set["control_set_hash"] = content_hash(
        {
            "schema_version": control_set["schema_version"],
            "compiled_rules_hash": control_set["compiled_rules_hash"],
            "catalog_hashes": control_set["catalog_hashes"],
            "control_count": control_set["control_count"],
            "controls": controls,
        }
    )


def test_acceptance_rejects_unrelated_disease_catalog_change(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    calls = 0

    def counted_apply_rules(*args: Any, **kwargs: Any) -> typed_rule_engine.RuleResult:
        nonlocal calls
        calls += 1
        raise AssertionError("runtime engine must not run before catalog validation")

    monkeypatch.setattr(typed_rule_engine, "apply_rules", counted_apply_rules)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["diseases"]["内分泌科"] = ["无关目录疾病"]
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="control set catalog_hashes mismatch"):
        knowledge_compile.build_knowledge_acceptance(
            pack,
            control_set,
            project_root=project_root,
            disease_catalog_path=catalog_path,
        )
    assert calls == 0


@pytest.mark.parametrize("ref_field", ["source_refs", "test_refs"])
def test_acceptance_revalidates_reference_file_bytes(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    ref_field: str,
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    calls = 0

    def counted_apply_rules(*args: Any, **kwargs: Any) -> typed_rule_engine.RuleResult:
        nonlocal calls
        calls += 1
        raise AssertionError("runtime engine must not run before ref validation")

    monkeypatch.setattr(typed_rule_engine, "apply_rules", counted_apply_rules)
    relative_path = _active_compiled_rule(pack)[ref_field][0]["path"]
    reference_path = project_root.joinpath(*relative_path.split("/"))
    reference_path.write_bytes(reference_path.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match=ref_field + " ref file hash mismatch"):
        knowledge_compile.build_knowledge_acceptance(
            pack,
            control_set,
            project_root=project_root,
            disease_catalog_path=catalog_path,
        )
    assert calls == 0


@pytest.mark.parametrize("invalid_case", ["mixed_key", "nested_unserializable"])
def test_acceptance_rejects_non_json_serializable_controls_before_engine(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    invalid = deepcopy(control_set)
    invalid["controls"] = _non_json_controls(invalid["controls"], invalid_case)
    calls = 0

    def counted_apply_rules(*args: Any, **kwargs: Any) -> typed_rule_engine.RuleResult:
        nonlocal calls
        calls += 1
        raise AssertionError("runtime engine must not run before control validation")

    monkeypatch.setattr(typed_rule_engine, "apply_rules", counted_apply_rules)

    with pytest.raises(
        ValueError,
        match="control set canonical core is not JSON serializable",
    ):
        knowledge_compile.build_knowledge_acceptance(
            pack,
            invalid,
            project_root=project_root,
            disease_catalog_path=catalog_path,
        )
    assert calls == 0


def test_acceptance_report_is_invariant_to_control_input_order(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    reversed_control_set = deepcopy(control_set)
    reversed_control_set["controls"].reverse()

    assert reversed_control_set["control_set_hash"] == control_set["control_set_hash"]
    forward = knowledge_compile.build_knowledge_acceptance(
        pack,
        control_set,
        project_root=project_root,
        disease_catalog_path=catalog_path,
    )
    reversed_report = knowledge_compile.build_knowledge_acceptance(
        pack,
        reversed_control_set,
        project_root=project_root,
        disease_catalog_path=catalog_path,
    )

    assert reversed_report["acceptance_hash"] == forward["acceptance_hash"]
    assert canonical_json(reversed_report).encode("utf-8") == canonical_json(forward).encode(
        "utf-8"
    )


def test_acceptance_derives_official_status_only_from_rebound_catalog(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["diseases"]["耳鼻喉科"].remove("耳鸣")
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    rebound_control_set = deepcopy(control_set)
    rebound_control_set["catalog_hashes"] = {
        "data/ref_data/diseases_catalog.json": file_hash(catalog_path),
    }
    _rehash_control_set(rebound_control_set)

    for control in rebound_control_set["controls"]:
        for context in (
            control["context"],
            control["expected_outcome"]["output_context"],
        ):
            assert all(
                "is_official" not in candidate
                for candidate in context["diagnosis_candidates"]
            )

    report = knowledge_compile.build_knowledge_acceptance(
        pack,
        rebound_control_set,
        project_root=project_root,
        disease_catalog_path=catalog_path,
    )

    active = next(
        rule
        for rule in report["rules"]
        if rule["rule_id"] == "symptom_over_background_condition"
    )
    assert report["status"] == "active_scope_failed"
    assert report["metrics"]["misses"] > 0
    assert active["acceptance_status"] == "active_scope_failed"
    assert active["misses"] > 0


def _active_compiled_rule(pack: dict[str, Any]) -> dict[str, Any]:
    return next(rule for rule in pack["rules"] if rule["runtime"]["status"] == "active")


def _typed_control_context(payload: dict[str, Any]) -> typed_rule_engine.RuleContext:
    return knowledge_compile._typed_context(
        payload,
        official_names=frozenset({"原发性高血压", "耳鸣", "跟骨骨折"}),
    )


def _rule_result(
    rule_id: str,
    output_context: typed_rule_engine.RuleContext,
    *,
    outcome: str | None,
    reason_code: str | None,
    before_hash: str,
    after_hash: str,
) -> typed_rule_engine.RuleResult:
    decisions = ()
    if outcome is not None and reason_code is not None:
        decisions = (
            typed_rule_engine.RuleDecision(
                rule_id=rule_id,
                opcode="prioritize_supported_current_problem",
                outcome=outcome,  # type: ignore[arg-type]
                reason_code=reason_code,
            ),
        )
    return typed_rule_engine.RuleResult(
        output_context=output_context,
        decisions=decisions,
        before_hash=before_hash,
        after_hash=after_hash,
    )


def _execute_with_results(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pack: dict[str, Any],
    control: dict[str, Any],
    results: list[typed_rule_engine.RuleResult],
) -> dict[str, Any]:
    first_target = next(
        (
            decision
            for decision in results[0].decisions
            if decision.rule_id == control["rule_id"]
        ),
        None,
    )
    if len(results) == 2 and first_target is not None and first_target.outcome == "applied":
        results = [*results, results[-1]]
    queued = iter(results)

    def fake_apply_rules(*args: Any, **kwargs: Any) -> typed_rule_engine.RuleResult:
        return next(queued)

    monkeypatch.setattr(typed_rule_engine, "apply_rules", fake_apply_rules)
    return knowledge_compile._execute_control(
        control,
        typed_pack=typed_rule_engine.parse_compiled_rule_pack(pack),
        official_names=frozenset({"原发性高血压", "耳鸣", "跟骨骨折"}),
    )


@pytest.mark.parametrize(
    ("kind", "authored_outcome", "expected_metric"),
    [
        pytest.param("positive", "not_matched", "misses", id="positive-oracle"),
        pytest.param("near_neighbor", "applied", "false_positives", id="near-neighbor-oracle"),
        pytest.param(
            "reasonable_exception",
            "applied",
            "exception_failures",
            id="exception-oracle",
        ),
    ],
)
def test_authored_oracle_cannot_self_certify_kind_semantics(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    authored_outcome: str,
    expected_metric: str,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(next(item for item in control_set["controls"] if item["kind"] == kind))
    authored_output = deepcopy(control["context"])
    if authored_outcome == "applied":
        authored_output["diagnosis_candidates"].reverse()
        authored_output["preferred_diagnosis"] = authored_output["diagnosis_candidates"][0][
            "official_name"
        ]
    authored_reason = "authored_" + authored_outcome
    control["expected_outcome"] = {
        "outcome": authored_outcome,
        "reason_code": authored_reason,
        "output_context": authored_output,
    }
    output_context = _typed_control_context(authored_output)
    before_hash = "1" * 64
    after_hash = "2" * 64 if authored_outcome == "applied" else before_hash
    replay_outcome = (
        "matched_no_change" if authored_outcome == "applied" else authored_outcome
    )
    replay_reason = (
        "supported_current_problem_already_preferred"
        if authored_outcome == "applied"
        else authored_reason
    )

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=authored_outcome,
                reason_code=authored_reason,
                before_hash=before_hash,
                after_hash=after_hash,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=replay_outcome,
                reason_code=replay_reason,
                before_hash=after_hash,
                after_hash=after_hash,
            ),
        ],
    )

    assert evaluated["behavior_passed"] is True
    assert evaluated["passed"] is False
    assert evaluated["behavior_matches_oracle"] is True
    assert evaluated["kind_semantics_passed"] is False
    assert evaluated["idempotent"] is True
    metrics = knowledge_compile._control_metrics([evaluated], treatment=False)
    assert metrics[expected_metric] == 1
    assert metrics["control_failures"] == 1


@pytest.mark.parametrize(
    ("control_id", "changed", "expected_metric"),
    [
        pytest.param(
            "hearing_over_hypertension",
            False,
            "misses",
            id="positive-applied-without-change",
        ),
        pytest.param(
            "subjective_symptom_not_examined",
            True,
            "false_positives",
            id="near-neighbor-non-applied-with-change",
        ),
        pytest.param(
            "confirmed_hypertensive_emergency",
            True,
            "exception_failures",
            id="exception-matched-no-change-with-change",
        ),
    ],
)
def test_kind_semantics_rejects_the_wrong_context_change_for_each_control_kind(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    control_id: str,
    changed: bool,
    expected_metric: str,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["control_id"] == control_id)
    )
    expected = control["expected_outcome"]
    output_context = _typed_control_context(expected["output_context"])
    after_hash = "2" * 64 if changed else "1" * 64
    replay_outcome = (
        "matched_no_change" if expected["outcome"] == "applied" else expected["outcome"]
    )
    replay_reason = (
        "supported_current_problem_already_preferred"
        if expected["outcome"] == "applied"
        else expected["reason_code"]
    )

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=expected["outcome"],
                reason_code=expected["reason_code"],
                before_hash="1" * 64,
                after_hash=after_hash,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=replay_outcome,
                reason_code=replay_reason,
                before_hash=after_hash,
                after_hash=after_hash,
            ),
        ],
    )

    assert evaluated["behavior_matches_oracle"] is True
    assert evaluated["kind_semantics_passed"] is False
    assert evaluated["idempotent"] is True
    assert evaluated["passed"] is False
    metrics = knowledge_compile._control_metrics([evaluated], treatment=False)
    assert metrics[expected_metric] == 1


def test_positive_metrics_require_the_actual_hit_to_match_the_authored_oracle(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    actual_output = _typed_control_context(control["expected_outcome"]["output_context"])
    control["expected_outcome"] = {
        "outcome": "not_matched",
        "reason_code": "authored_wrong_reason",
        "output_context": deepcopy(control["context"]),
    }

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                actual_output,
                outcome="applied",
                reason_code="supported_current_problem_promoted",
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                actual_output,
                outcome="matched_no_change",
                reason_code="supported_current_problem_already_preferred",
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
        ],
    )

    assert evaluated["behavior_matches_oracle"] is False
    assert evaluated["kind_semantics_passed"] is True
    assert evaluated["idempotent"] is True
    assert evaluated["passed"] is False
    metrics = knowledge_compile._control_metrics([evaluated], treatment=False)
    assert metrics["positive_hits"] == 0
    assert metrics["misses"] == 1


@pytest.mark.parametrize(
    ("replay_outcome", "replay_reason", "expected_target"),
    [
        pytest.param(
            "applied",
            "supported_current_problem_promoted",
            {
                "outcome": "applied",
                "reason_code": "supported_current_problem_promoted",
            },
            id="replay-still-applies",
        ),
        pytest.param(None, None, None, id="replay-target-missing"),
    ],
)
def test_replay_requires_the_target_rule_to_reach_the_applied_fixed_point(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    replay_outcome: str | None,
    replay_reason: str | None,
    expected_target: dict[str, str] | None,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    output_context = _typed_control_context(control["expected_outcome"]["output_context"])

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="applied",
                reason_code="supported_current_problem_promoted",
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=replay_outcome,
                reason_code=replay_reason,
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
        ],
    )

    assert evaluated["idempotent"] is False
    assert evaluated["passed"] is False
    assert evaluated["replay"]["target_decision"] == expected_target


def test_applied_replay_requires_a_stable_fixed_point_reason_code(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    output_context = _typed_control_context(control["expected_outcome"]["output_context"])

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="applied",
                reason_code="supported_current_problem_promoted",
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="matched_no_change",
                reason_code="wrong_fixed_point_reason",
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="matched_no_change",
                reason_code="different_fixed_point_reason",
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
        ],
    )

    assert evaluated["replay"]["target_decision"] == {
        "outcome": "matched_no_change",
        "reason_code": "wrong_fixed_point_reason",
    }
    assert evaluated["idempotent"] is False
    assert evaluated["passed"] is False


def test_applied_replay_accepts_an_opcode_specific_stable_fixed_point_reason(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    output_context = _typed_control_context(control["expected_outcome"]["output_context"])
    fixed_point_reason = "congenital_infection_axes_already_ranked"

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="applied",
                reason_code=control["expected_outcome"]["reason_code"],
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="matched_no_change",
                reason_code=fixed_point_reason,
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="matched_no_change",
                reason_code=fixed_point_reason,
                before_hash="2" * 64,
                after_hash="2" * 64,
            ),
        ],
    )

    assert evaluated["idempotent"] is True
    assert evaluated["passed"] is True


def test_applied_fixed_point_replay_rejects_changes_in_other_rule_decisions(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    output_context = _typed_control_context(control["expected_outcome"]["output_context"])
    target_fixed_point = typed_rule_engine.RuleDecision(
        control["rule_id"],
        "prioritize_supported_current_problem",
        "matched_no_change",
        "supported_current_problem_already_preferred",
    )

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="applied",
                reason_code=control["expected_outcome"]["reason_code"],
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            typed_rule_engine.RuleResult(
                output_context,
                (
                    target_fixed_point,
                    typed_rule_engine.RuleDecision(
                        "other_rule",
                        "other_opcode",
                        "applied",
                        "other_rule_applied",
                    ),
                ),
                "2" * 64,
                "2" * 64,
            ),
            typed_rule_engine.RuleResult(
                output_context,
                (
                    target_fixed_point,
                    typed_rule_engine.RuleDecision(
                        "other_rule",
                        "other_opcode",
                        "matched_no_change",
                        "other_rule_already_stable",
                    ),
                ),
                "2" * 64,
                "2" * 64,
            ),
        ],
    )

    assert evaluated["idempotent"] is False
    assert evaluated["passed"] is False


@pytest.mark.parametrize("instability", ["context", "hash"])
def test_replay_requires_stable_context_and_hash_with_the_correct_target_decision(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    instability: str,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(item for item in control_set["controls"] if item["kind"] == "positive")
    )
    output_context = _typed_control_context(control["expected_outcome"]["output_context"])
    replay_context = (
        _typed_control_context(control["context"])
        if instability == "context"
        else output_context
    )
    replay_after_hash = "3" * 64 if instability == "hash" else "2" * 64

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="applied",
                reason_code="supported_current_problem_promoted",
                before_hash="1" * 64,
                after_hash="2" * 64,
            ),
            _rule_result(
                control["rule_id"],
                replay_context,
                outcome="matched_no_change",
                reason_code="supported_current_problem_already_preferred",
                before_hash="2" * 64,
                after_hash=replay_after_hash,
            ),
        ],
    )

    assert evaluated["replay"]["target_decision"] == {
        "outcome": "matched_no_change",
        "reason_code": "supported_current_problem_already_preferred",
    }
    assert evaluated["idempotent"] is False
    assert evaluated["passed"] is False


@pytest.mark.parametrize(
    ("replay_outcome", "replay_reason"),
    [
        pytest.param(
            "excluded",
            "no_supported_current_problem",
            id="outcome-changed-only",
        ),
        pytest.param(
            "not_matched",
            "wrong_non_applied_reason",
            id="reason-changed-only",
        ),
    ],
)
def test_replay_preserves_each_non_applied_target_decision_field(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    replay_outcome: str,
    replay_reason: str,
) -> None:
    _, _, pack, control_set = acceptance_inputs
    control = deepcopy(
        next(
            item
            for item in control_set["controls"]
            if item["control_id"] == "subjective_symptom_not_examined"
        )
    )
    output_context = _typed_control_context(control["context"])

    evaluated = _execute_with_results(
        monkeypatch,
        pack=pack,
        control=control,
        results=[
            _rule_result(
                control["rule_id"],
                output_context,
                outcome="not_matched",
                reason_code="no_supported_current_problem",
                before_hash="1" * 64,
                after_hash="1" * 64,
            ),
            _rule_result(
                control["rule_id"],
                output_context,
                outcome=replay_outcome,
                reason_code=replay_reason,
                before_hash="1" * 64,
                after_hash="1" * 64,
            ),
        ],
    )

    assert evaluated["idempotent"] is False
    assert evaluated["passed"] is False
    assert evaluated["replay"]["target_decision"] == {
        "outcome": replay_outcome,
        "reason_code": replay_reason,
    }


@pytest.mark.parametrize(
    ("control_passed", "expected_count", "expected_status"),
    [(True, 0, "passed"), (False, 1, "failed")],
)
def test_active_treatment_p0_status_is_explicit(
    control_passed: bool,
    expected_count: int,
    expected_status: str,
) -> None:
    metrics = knowledge_compile._control_metrics(
        [
            {
                "kind": "near_neighbor",
                "behavior_matches_oracle": True,
                "kind_semantics_passed": control_passed,
                "idempotent": True,
                "passed": control_passed,
            }
        ],
        treatment=True,
    )

    assert metrics["p0_count"] == expected_count
    assert metrics["p0_applicable"] is True
    assert metrics["p0_status"] == expected_status


def test_treatment_p0_counts_an_idempotency_only_negative_control_failure() -> None:
    metrics = knowledge_compile._control_metrics(
        [
            {
                "kind": "near_neighbor",
                "behavior_matches_oracle": True,
                "kind_semantics_passed": True,
                "idempotent": False,
                "passed": False,
            }
        ],
        treatment=True,
    )

    assert metrics["false_positives"] == 0
    assert metrics["idempotency_failures"] == 1
    assert metrics["control_failures"] == 1
    assert metrics["p0_count"] == 1
    assert metrics["p0_applicable"] is True
    assert metrics["p0_status"] == "failed"


@pytest.mark.parametrize(
    ("kind", "treatment", "expected_applicable", "expected_status"),
    [
        pytest.param("positive", True, True, "passed", id="treatment-positive-failure"),
        pytest.param(
            "near_neighbor",
            False,
            False,
            "not_evaluated",
            id="non-treatment-negative-failure",
        ),
    ],
)
def test_p0_only_counts_failed_negative_controls_for_treatment_rules(
    kind: str,
    treatment: bool,
    expected_applicable: bool,
    expected_status: str,
) -> None:
    metrics = knowledge_compile._control_metrics(
        [
            {
                "kind": kind,
                "behavior_matches_oracle": True,
                "kind_semantics_passed": False,
                "idempotent": True,
                "passed": False,
            }
        ],
        treatment=treatment,
    )

    assert metrics["control_failures"] == 1
    assert metrics["p0_count"] == 0
    assert metrics["p0_applicable"] is expected_applicable
    assert metrics["p0_status"] == expected_status


@pytest.mark.parametrize(
    "invalid_case",
    [
        "legacy_natural_language",
        "missing_inventory",
        "extra_inventory",
        "duplicate_inventory",
        "wrong_kind",
        "extra_control_field",
        "candidate_self_reports_official",
        "non_mapping_control",
        "non_string_rule_id",
        "non_string_control_id",
        "missing_catalog_hashes",
        "extra_catalog_hash_path",
        "invalid_catalog_hash",
        "wrong_control_hash",
        "wrong_compiled_rules_binding",
        "wrong_control_count",
    ],
)
def test_acceptance_rejects_invalid_control_schema_inventory_and_hashes(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    invalid_case: str,
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    invalid = deepcopy(control_set)
    if invalid_case == "legacy_natural_language":
        rule = _active_compiled_rule(pack)
        invalid["controls"] = deepcopy(rule["positive_controls"] + rule["negative_controls"])
    elif invalid_case == "missing_inventory":
        invalid["controls"].pop()
    elif invalid_case == "extra_inventory":
        extra = deepcopy(invalid["controls"][0])
        extra["control_id"] = "unexpected_extra_control"
        invalid["controls"].append(extra)
    elif invalid_case == "duplicate_inventory":
        invalid["controls"].append(deepcopy(invalid["controls"][0]))
    elif invalid_case == "wrong_kind":
        invalid["controls"][0]["kind"] = (
            "positive"
            if invalid["controls"][0]["kind"] != "positive"
            else "near_neighbor"
        )
    elif invalid_case == "extra_control_field":
        invalid["controls"][0]["unexpected"] = True
    elif invalid_case == "candidate_self_reports_official":
        invalid["controls"][0]["context"]["diagnosis_candidates"][0]["is_official"] = True
    elif invalid_case == "non_mapping_control":
        invalid["controls"].append(7)
    elif invalid_case == "non_string_rule_id":
        invalid["controls"][0]["rule_id"] = []
    elif invalid_case == "non_string_control_id":
        invalid["controls"][0]["control_id"] = 7
    elif invalid_case == "missing_catalog_hashes":
        invalid.pop("catalog_hashes")
    elif invalid_case == "extra_catalog_hash_path":
        invalid["catalog_hashes"]["data/ref_data/other.json"] = "0" * 64
    elif invalid_case == "invalid_catalog_hash":
        invalid["catalog_hashes"]["data/ref_data/diseases_catalog.json"] = "A" * 64
    elif invalid_case == "wrong_control_hash":
        invalid["control_set_hash"] = "0" * 64
    elif invalid_case == "wrong_compiled_rules_binding":
        invalid["compiled_rules_hash"] = "0" * 64
    elif invalid_case == "wrong_control_count":
        invalid["control_count"] += 1
    if invalid_case not in {
        "wrong_control_hash",
        "wrong_compiled_rules_binding",
        "wrong_control_count",
        "missing_catalog_hashes",
        "non_string_rule_id",
        "non_string_control_id",
    }:
        _rehash_control_set(invalid)

    with pytest.raises(ValueError):
        knowledge_compile.build_knowledge_acceptance(
            pack,
            invalid,
            project_root=project_root,
            disease_catalog_path=catalog_path,
        )


def test_acceptance_uses_the_runtime_parser_to_reject_unhashed_pack_tampering(
    acceptance_inputs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    project_root, catalog_path, pack, control_set = acceptance_inputs
    tampered = deepcopy(pack)
    _active_compiled_rule(tampered)["runtime"]["parameters"]["target_roles"] = [
        "background_condition"
    ]

    with pytest.raises(ValueError, match="rules_hash mismatch"):
        knowledge_compile.build_knowledge_acceptance(
            tampered,
            control_set,
            project_root=project_root,
            disease_catalog_path=catalog_path,
        )


@pytest.fixture()
def candidate_batch(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    project_root = tmp_path / "project"
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

    result = build_knowledge_candidate_batch(
        project_root=project_root,
        source_files=source_paths,
        test_files=test_paths,
        artifact_root=project_root / "outputs" / "offline" / "knowledge",
    )
    batch_dir = Path(result["batch_dir"])
    candidates = [
        load_candidate(
            batch_dir / "candidates" / (rule_id + ".json"),
            project_root=project_root,
        )
        for rule_id in EXPECTED_RULES
    ]
    assert len(candidates) == 6
    return project_root, candidates


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


def _rehash(candidate: dict[str, Any]) -> None:
    candidate["effect_hash"] = content_hash(candidate["proposed_effect"])
    candidate["candidate_hash"] = content_hash(
        {
            key: value
            for key, value in candidate.items()
            if key not in {"candidate_hash", "effect_hash"}
        }
    )


def test_compile_emits_exact_schema_sorted_rules_and_omits_candidate_audit_fields(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["evidence"] = {
        "source": "offline_knowledge_design",
        "reviewer": "human-auditor",
        "source_excerpt": "full source narrative retained only for review",
    }
    _rehash(candidates[0])

    pack = compile_knowledge_rules(candidates, project_root=project_root)

    assert set(pack) == {"schema_version", "rules", "rule_count", "rules_hash"}
    assert pack["schema_version"] == "compiled-knowledge-rules/v2"
    assert pack["rule_count"] == 6
    assert pack["rules_hash"] == content_hash(pack["rules"])
    assert all(set(rule) == COMPILED_RULE_FIELDS for rule in pack["rules"])
    assert [rule["rule_id"] for rule in pack["rules"]] == [
        candidate["candidate_id"]
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item["proposed_effect"]["priority"],
                item["proposed_effect"]["rule_id"],
            ),
        )
    ]
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    assert all(
        rule["candidate_hash"] == candidates_by_id[rule["rule_id"]]["candidate_hash"]
        and rule["effect_hash"] == candidates_by_id[rule["rule_id"]]["effect_hash"]
        and rule["runtime"]
        == candidates_by_id[rule["rule_id"]]["proposed_effect"]["runtime"]
        for rule in pack["rules"]
    )
    assert sum(rule["runtime"]["status"] == "active" for rule in pack["rules"]) == 2
    active_rule = next(rule for rule in pack["rules"] if rule["runtime"]["status"] == "active")
    active_candidate = candidates_by_id[active_rule["rule_id"]]
    assert active_rule["runtime"] is not active_candidate["proposed_effect"]["runtime"]
    assert active_rule["runtime"]["parameters"] is not active_candidate["proposed_effect"]["runtime"]["parameters"]

    serialized = canonical_json(pack)
    assert "full source narrative retained only for review" not in serialized
    assert '"evidence"' not in serialized
    assert '"reviewer"' not in serialized
    assert '"review_requirements"' not in serialized
    for marker in (
        "patient_",
        "expected",
        "reference",
        "ground_truth",
        "evaluator_reasoning",
    ):
        assert marker not in serialized.casefold()


def test_compile_is_invariant_to_candidate_order(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch

    forward = compile_knowledge_rules(candidates, project_root=project_root)
    reversed_pack = compile_knowledge_rules(
        list(reversed(candidates)),
        project_root=project_root,
    )

    assert canonical_json(reversed_pack) == canonical_json(forward)


def test_compile_rejects_an_empty_candidate_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        compile_knowledge_rules([], project_root=tmp_path)


def test_compile_rejects_batch_with_no_active_runtime_rule(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    for active_candidate in candidates:
        runtime = active_candidate["proposed_effect"]["runtime"]
        if runtime["status"] != "active":
            continue
        active_candidate["proposed_effect"]["runtime"] = {
            "status": "audit_only",
            "stage": runtime["stage"],
        }
        _rehash(active_candidate)

    with pytest.raises(ValueError, match="active runtime"):
        compile_knowledge_rules(candidates, project_root=project_root)


def test_compile_rejects_v1_knowledge_effect(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["proposed_effect"]["schema_version"] = "clinical-knowledge-candidate/v1"
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="schema_version"):
        compile_knowledge_rules(candidates, project_root=project_root)


def test_compile_rejects_non_mapping_candidate_item(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate must be an object"):
        compile_knowledge_rules([None], project_root=tmp_path)  # type: ignore[list-item]


def test_compile_rebuilds_a_byte_equivalent_pack(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch

    first = compile_knowledge_rules(candidates, project_root=project_root)
    rebuilt = compile_knowledge_rules(deepcopy(candidates), project_root=project_root)

    assert canonical_json(rebuilt).encode("utf-8") == canonical_json(first).encode("utf-8")


def test_compile_does_not_modify_input(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    before = deepcopy(candidates)

    compile_knowledge_rules(candidates, project_root=project_root)

    assert candidates == before


def test_compile_rejects_duplicate_rule_id(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch

    with pytest.raises(ValueError, match="duplicate rule_id"):
        compile_knowledge_rules(
            [*candidates, deepcopy(candidates[0])],
            project_root=project_root,
        )


def test_compile_rejects_unknown_candidate_type(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["candidate_type"] = "diagnosis_unknown_rule"
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="unknown candidate_type"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize("candidate_type", [[], {}])
def test_compile_rejects_unhashable_candidate_type_as_validation_error(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    candidate_type: object,
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["candidate_type"] = candidate_type
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="candidate_type"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize("hash_field", ["candidate_hash", "effect_hash"])
def test_compile_rejects_hash_mismatch(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    hash_field: str,
) -> None:
    project_root, candidates = candidate_batch
    candidates[0][hash_field] = "0" * 64

    with pytest.raises(ValueError, match=hash_field + " mismatch"):
        compile_knowledge_rules(candidates, project_root=project_root)


def test_compile_rejects_candidate_id_that_differs_from_rule_id(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["candidate_id"] = "different_rule_id"
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="candidate_id must equal rule_id"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "candidate/v0"),
        ("status", "quarantine"),
    ],
)
def test_compile_rejects_non_candidate_envelope(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    field: str,
    value: str,
) -> None:
    project_root, candidates = candidate_batch
    candidates[0][field] = value
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="candidate/v1|status must be candidate"):
        compile_knowledge_rules(candidates, project_root=project_root)


def test_compile_rejects_unknown_candidate_field(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["unexpected_field"] = "must fail closed"
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="candidate fields"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize("removed_kind", ["near_neighbor", "reasonable_exception", "all"])
def test_compile_revalidates_treatment_controls(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    removed_kind: str,
) -> None:
    project_root, candidates = candidate_batch
    candidate = next(
        item for item in candidates if item["candidate_type"] == "treatment_gate_rule"
    )
    controls = candidate["proposed_effect"]["negative_controls"]
    candidate["proposed_effect"]["negative_controls"] = (
        []
        if removed_kind == "all"
        else [control for control in controls if control["kind"] != removed_kind]
    )
    _rehash(candidate)

    with pytest.raises(ValueError, match="negative_controls|near-neighbor|reasonable exception"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize(
    "marker",
    ["Patient_987", "expected", "reference", "ground_truth", "evaluator_reasoning"],
)
def test_compile_rejects_leakage_in_a_compilable_field(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    marker: str,
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["proposed_effect"]["triggers"] = ["leaked marker: " + marker]
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="leakage"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize(
    "answer_shape",
    [
        "Diagnosis: X; Examination: Y; Treatment: Z",
        "标准答案：诊断：X；检查：Y；治疗：Z",
    ],
)
def test_compile_rejects_sectioned_answer_shape(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    answer_shape: str,
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["proposed_effect"]["positive_controls"][0]["facts"] = [answer_shape]
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="leakage"):
        compile_knowledge_rules(candidates, project_root=project_root)


@pytest.mark.parametrize(
    "answer_sections",
    [
        ["Diagnosis: X", "Exam: Y", "Management: Z"],
        ["诊断：X", "检验：Y", "治疗：Z"],
    ],
)
def test_compile_rejects_split_sectioned_answer_shape(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
    answer_sections: list[str],
) -> None:
    project_root, candidates = candidate_batch
    candidates[0]["proposed_effect"]["required_evidence"] = answer_sections
    _rehash(candidates[0])

    with pytest.raises(ValueError, match="leakage"):
        compile_knowledge_rules(candidates, project_root=project_root)


def test_compile_revalidates_source_and_test_refs(
    candidate_batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    project_root, candidates = candidate_batch
    (project_root / "docs" / "design.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ref file hash mismatch"):
        compile_knowledge_rules(candidates, project_root=project_root)
