from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Sequence

from offline.knowledge_compile import knowledge_control_set_hash


_CONGENITAL_RULE_ID = "congenital_infection_differential"
_SYMPTOM_RULE_ID = "symptom_over_background_condition"
_STAGE = "diagnosis_candidates"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _candidate(
    official_name: str,
    *,
    role: str,
    support_level: str,
    complaint_relation: str,
    urgency: str = "routine",
    evidence_codes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "official_name": official_name,
        "role": role,
        "support_level": support_level,
        "complaint_relation": complaint_relation,
        "urgency": urgency,
        "evidence_codes": list(evidence_codes),
    }


def _context(
    candidates: Sequence[dict[str, Any]],
    *,
    preferred: str | None,
    fact_codes: Sequence[str] = (),
    diagnostic_axis_ids: Sequence[str] = (),
    exam_intent_ids: Sequence[str] = (),
    treatment_codes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "diagnosis_candidates": deepcopy(list(candidates)),
        "preferred_diagnosis": preferred,
        "diagnostic_axis_ids": list(diagnostic_axis_ids),
        "exam_intent_ids": list(exam_intent_ids),
        "treatment_codes": list(treatment_codes),
        "fact_codes": list(fact_codes),
    }


def _control(
    control_id: str,
    kind: str,
    context: dict[str, Any],
    *,
    rule_id: str = _SYMPTOM_RULE_ID,
    outcome: str,
    reason_code: str,
    output_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "control_id": control_id,
        "kind": kind,
        "stage": _STAGE,
        "context": deepcopy(context),
        "expected_outcome": {
            "outcome": outcome,
            "reason_code": reason_code,
            "output_context": deepcopy(output_context),
        },
    }


def _background(*, relation: str = "unrelated", urgency: str = "routine") -> dict[str, Any]:
    return _candidate(
        "原发性高血压",
        role="background_condition",
        support_level="objective",
        complaint_relation=relation,
        urgency=urgency,
        evidence_codes=("history_hypertension",),
    )


def _current(
    name: str = "耳鸣",
    *,
    support: str = "objective",
    evidence_codes: Sequence[str] = ("audiometry_abnormal",),
) -> dict[str, Any]:
    return _candidate(
        name,
        role="current_problem",
        support_level=support,
        complaint_relation="unrelated",
        evidence_codes=evidence_codes,
    )


def _positive_control(
    control_id: str,
    *,
    current: dict[str, Any] | None = None,
    fact_codes: Sequence[str] = (),
    background_urgency: str = "routine",
) -> dict[str, Any]:
    target = current or _current()
    background = _background(urgency=background_urgency)
    initial = _context([background, target], preferred="原发性高血压", fact_codes=fact_codes)
    expected = _context([target, background], preferred=target["official_name"], fact_codes=fact_codes)
    return _control(
        control_id,
        "positive",
        initial,
        outcome="applied",
        reason_code="supported_current_problem_promoted",
        output_context=expected,
    )


def _not_matched_control(
    control_id: str,
    current: dict[str, Any],
    *,
    fact_codes: Sequence[str],
) -> dict[str, Any]:
    context = _context(
        [_background(), current],
        preferred="原发性高血压",
        fact_codes=fact_codes,
    )
    return _control(
        control_id,
        "near_neighbor",
        context,
        outcome="not_matched",
        reason_code="no_supported_current_problem",
        output_context=context,
    )


def _emergency_control(
    control_id: str,
    evidence_code: str,
) -> dict[str, Any]:
    background = _background(urgency="emergency")
    background["evidence_codes"] = ["history_hypertension", evidence_code]
    context = _context(
        [background, _current()],
        preferred="原发性高血压",
        fact_codes=(evidence_code, "independent_audiometry_abnormal"),
    )
    return _control(
        control_id,
        "reasonable_exception",
        context,
        outcome="matched_no_change",
        reason_code="emergency_priority_preserved",
        output_context=context,
    )


def _symptom_controls() -> list[dict[str, Any]]:
    controls = [
        _positive_control("hearing_over_hypertension"),
        _positive_control(
            "focal_symptom_over_history",
            current=_current("跟骨骨折", evidence_codes=("hindfoot_imaging_abnormal",)),
        ),
        _positive_control(
            "severe_elevation_without_acute_target_organ_damage",
            fact_codes=("severe_blood_pressure_elevation", "no_acute_target_organ_damage"),
            background_urgency="urgent",
        ),
        _positive_control(
            "transient_elevation_without_acute_target_organ_damage",
            fact_codes=("transient_stress_elevation", "no_acute_target_organ_damage"),
        ),
        _background_explains_control(),
        _not_matched_control(
            "subjective_symptom_normal_exam",
            _current(support="subjective", evidence_codes=("audiometry_normal",)),
            fact_codes=("subjective_symptom", "same_system_exam_normal"),
        ),
        _not_matched_control(
            "subjective_symptom_not_examined",
            _current(support="none", evidence_codes=()),
            fact_codes=("subjective_symptom", "same_system_exam_not_completed"),
        ),
        _not_matched_control(
            "subjective_symptom_uncertain_result",
            _current(support="subjective", evidence_codes=("audiometry_uncertain",)),
            fact_codes=("subjective_symptom", "same_system_exam_uncertain"),
        ),
        _emergency_control("confirmed_hypertensive_emergency", "acute_target_organ_damage"),
        _emergency_control("suspected_hypertensive_emergency", "target_organ_damage_pending"),
    ]
    return controls


def _background_explains_control() -> dict[str, Any]:
    context = _context(
        [_background(relation="explains"), _current()],
        preferred="原发性高血压",
        fact_codes=("background_explains_current_problem",),
    )
    return _control(
        "background_explains_current_problem",
        "near_neighbor",
        context,
        outcome="excluded",
        reason_code="background_explains_current_problem",
        output_context=context,
    )


def _congenital_context(
    fact_codes: Sequence[str],
    *,
    diagnostic_axis_ids: Sequence[str] = ("other_diagnostic_axis",),
) -> dict[str, Any]:
    return _context(
        [_current(support="subjective", evidence_codes=("audiometry_normal",))],
        preferred="耳鸣",
        fact_codes=fact_codes,
        diagnostic_axis_ids=diagnostic_axis_ids,
        exam_intent_ids=("existing_exam_intent",),
        treatment_codes=("existing_treatment",),
    )


def _congenital_positive_control(
    control_id: str,
    fact_codes: Sequence[str],
    ordered_axes: Sequence[str],
) -> dict[str, Any]:
    context = _congenital_context(fact_codes)
    expected = deepcopy(context)
    expected["diagnostic_axis_ids"] = [*ordered_axes, "other_diagnostic_axis"]
    return _control(
        control_id,
        "positive",
        context,
        rule_id=_CONGENITAL_RULE_ID,
        outcome="applied",
        reason_code="congenital_infection_axes_expanded",
        output_context=expected,
    )


def _congenital_near_neighbor_control(
    control_id: str,
    fact_codes: Sequence[str],
) -> dict[str, Any]:
    context = _congenital_context(fact_codes)
    return _control(
        control_id,
        "near_neighbor",
        context,
        rule_id=_CONGENITAL_RULE_ID,
        outcome="not_matched",
        reason_code="congenital_infection_exposure_not_matched",
        output_context=context,
    )


def _congenital_controls() -> list[dict[str, Any]]:
    rubella_first = ("congenital_rubella", "congenital_cmv")
    cmv_first = ("congenital_cmv", "congenital_rubella")
    exception = _congenital_context(
        (
            "infant",
            "intrauterine_viral_exposure",
            "infant_hearing_abnormality",
            "cmv_saliva_or_urine_pcr_positive_within_21_days",
        ),
        diagnostic_axis_ids=("congenital_cmv", "other_diagnostic_axis"),
    )
    return [
        _congenital_positive_control(
            "congenital_multi_system",
            (
                "neonate",
                "intrauterine_viral_exposure",
                "congenital_jaundice",
                "congenital_rash",
            ),
            rubella_first,
        ),
        _congenital_positive_control(
            "congenital_hearing_signal",
            (
                "infant",
                "intrauterine_viral_exposure",
                "infant_hearing_abnormality",
            ),
            rubella_first,
        ),
        _congenital_positive_control(
            "congenital_rubella_rank_signal",
            ("infant", "intrauterine_viral_exposure", "congenital_cataract"),
            rubella_first,
        ),
        _congenital_positive_control(
            "congenital_cmv_rank_signal",
            (
                "infant",
                "intrauterine_viral_exposure",
                "periventricular_calcifications",
            ),
            cmv_first,
        ),
        _congenital_near_neighbor_control(
            "isolated_neonatal_jaundice",
            ("neonate", "congenital_jaundice"),
        ),
        _congenital_near_neighbor_control(
            "postnatal_infection_pattern",
            ("infant", "postnatal_viral_infection", "congenital_jaundice"),
        ),
        _control(
            "documented_single_pathogen",
            "reasonable_exception",
            exception,
            rule_id=_CONGENITAL_RULE_ID,
            outcome="matched_no_change",
            reason_code="congenital_infection_axes_already_ranked",
            output_context=exception,
        ),
    ]


_CONTROL_BUILDERS = {
    _CONGENITAL_RULE_ID: _congenital_controls,
    _SYMPTOM_RULE_ID: _symptom_controls,
}


def _validated_active_rule_ids(active_rule_ids: Sequence[str]) -> list[str]:
    if isinstance(active_rule_ids, (str, bytes)) or not isinstance(
        active_rule_ids, Sequence
    ):
        raise ValueError("active_rule_ids must be a non-empty sequence")
    rule_ids = list(active_rule_ids)
    if not rule_ids or any(not isinstance(rule_id, str) for rule_id in rule_ids):
        raise ValueError("active_rule_ids must be a non-empty sequence of strings")
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("active_rule_ids must not contain duplicates")
    unknown = sorted(set(rule_ids) - set(_CONTROL_BUILDERS))
    if unknown:
        raise ValueError("unknown active rule_id: %s" % ", ".join(unknown))
    return sorted(rule_ids)


def build_active_rule_control_set(
    compiled_rules_hash: str,
    *,
    disease_catalog_hash: str,
    active_rule_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(compiled_rules_hash, str) or not _SHA256_PATTERN.fullmatch(
        compiled_rules_hash
    ):
        raise ValueError("compiled_rules_hash must be a lowercase sha256")
    if not isinstance(disease_catalog_hash, str) or not _SHA256_PATTERN.fullmatch(
        disease_catalog_hash
    ):
        raise ValueError("disease_catalog_hash must be a lowercase sha256")
    rule_ids = _validated_active_rule_ids(active_rule_ids)
    controls = sorted(
        (
            control
            for rule_id in rule_ids
            for control in _CONTROL_BUILDERS[rule_id]()
        ),
        key=lambda item: (item["rule_id"], item["control_id"]),
    )
    core = {
        "schema_version": "knowledge-rule-controls/v1",
        "compiled_rules_hash": compiled_rules_hash,
        "catalog_hashes": {
            "data/ref_data/diseases_catalog.json": disease_catalog_hash,
        },
        "control_count": len(controls),
        "controls": controls,
    }
    return {
        **core,
        "control_set_hash": knowledge_control_set_hash(**core),
    }


def build_symptom_priority_control_set(
    compiled_rules_hash: str,
    *,
    disease_catalog_hash: str,
) -> dict[str, Any]:
    return build_active_rule_control_set(
        compiled_rules_hash,
        disease_catalog_hash=disease_catalog_hash,
        active_rule_ids=(_SYMPTOM_RULE_ID,),
    )
