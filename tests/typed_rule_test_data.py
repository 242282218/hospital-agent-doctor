from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json


def _content_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def active_diagnosis_priority_pack_payload() -> dict[str, object]:
    rule = {
        "rule_id": "prefer_supported_current_problem",
        "candidate_type": "diagnosis_priority_rule",
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["structured trigger for audit"],
        "required_evidence": ["objective finding"],
        "exclusions": ["documented exclusion"],
        "effect": {
            "priority_policy": "objective_evidence_first",
            "fallback_policy": "official_catalog_only",
        },
        "positive_controls": [
            {
                "control_id": "prefer_supported_current_problem_positive",
                "kind": "positive",
                "facts": ["supported presentation"],
                "assertions": ["expected bounded behavior"],
            }
        ],
        "negative_controls": [
            {
                "control_id": "prefer_supported_current_problem_neighbor",
                "kind": "near_neighbor",
                "facts": ["nearby presentation"],
                "assertions": ["rule remains bounded"],
            }
        ],
        "source_refs": [{"path": "docs/not-present.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/not-present.py", "sha256": "d" * 64}],
        "priority": 10,
        "scope": {"phase": "diagnosis", "application": "trigger_bound"},
        "runtime": {
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
        },
    }
    rules = [rule]
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": rules,
        "rule_count": len(rules),
        "rules_hash": _content_hash(rules),
    }


_CONGENITAL_DIFFERENTIAL_PARAMETERS = {
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


def active_congenital_differential_pack_payload() -> dict[str, object]:
    rule = {
        "rule_id": "congenital_infection_differential",
        "candidate_type": "diagnosis_differential_rule",
        "candidate_hash": "e" * 64,
        "effect_hash": "f" * 64,
        "triggers": ["structured congenital infection trigger"],
        "required_evidence": ["trusted congenital infection evidence"],
        "exclusions": ["isolated nonspecific symptom"],
        "effect": {
            "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
            "ranking_policy": "evidence_ordered",
        },
        "positive_controls": [
            {
                "control_id": "congenital_positive",
                "kind": "positive",
                "facts": ["congenital multisystem presentation"],
                "assertions": ["expand differential axes"],
            }
        ],
        "negative_controls": [
            {
                "control_id": "congenital_neighbor",
                "kind": "near_neighbor",
                "facts": ["isolated neonatal jaundice"],
                "assertions": ["do not expand axes"],
            }
        ],
        "source_refs": [{"path": "docs/not-present.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/not-present.py", "sha256": "d" * 64}],
        "priority": 10,
        "scope": {"phase": "diagnosis", "application": "trigger_bound"},
        "runtime": {
            "status": "active",
            "stage": "diagnosis_candidates",
            "opcode": "expand_congenital_infection_axes",
            "parameters": deepcopy(_CONGENITAL_DIFFERENTIAL_PARAMETERS),
        },
    }
    rules = [rule]
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": rules,
        "rule_count": len(rules),
        "rules_hash": _content_hash(rules),
    }
