from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal, TypeVar, cast


RuleStage = Literal["diagnosis_candidates", "clinical_closure", "treatment"]
RuleOutcome = Literal["not_matched", "excluded", "matched_no_change", "applied"]
CandidateType = Literal[
    "diagnosis_differential_rule",
    "diagnosis_axis_rule",
    "clinical_closure_rule",
    "diagnosis_priority_rule",
    "treatment_gate_rule",
    "treatment_sequence_rule",
]

_ROLES = frozenset({"current_problem", "background_condition", "differential"})
_SUPPORT_LEVELS = frozenset({"objective", "subjective", "none"})
_COMPLAINT_RELATIONS = frozenset({"explains", "unrelated", "unknown"})
_URGENCIES = frozenset({"routine", "urgent", "emergency"})
_OUTCOMES = frozenset({"not_matched", "excluded", "matched_no_change", "applied"})
_RULE_STAGES = frozenset({"diagnosis_candidates", "clinical_closure", "treatment"})
_PACK_FIELDS = frozenset({"schema_version", "rules", "rule_count", "rules_hash"})
_RULE_FIELDS = frozenset(
    {
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
)
_CANDIDATE_TYPES = frozenset(
    {
        "diagnosis_differential_rule",
        "diagnosis_axis_rule",
        "clinical_closure_rule",
        "diagnosis_priority_rule",
        "treatment_gate_rule",
        "treatment_sequence_rule",
    }
)
_PHASE_BY_TYPE = {
    "diagnosis_differential_rule": "diagnosis",
    "diagnosis_axis_rule": "diagnosis",
    "clinical_closure_rule": "closure",
    "diagnosis_priority_rule": "diagnosis",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}
_TREATMENT_TYPES = frozenset({"treatment_gate_rule", "treatment_sequence_rule"})
_STAGE_BY_TYPE = {
    "diagnosis_differential_rule": "diagnosis_candidates",
    "diagnosis_axis_rule": "diagnosis_candidates",
    "clinical_closure_rule": "clinical_closure",
    "diagnosis_priority_rule": "diagnosis_candidates",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}
FACT_GROUP_OPCODE = "match_fact_groups"
FACT_GROUP_PARAMETER_FIELDS = frozenset(
    {"all_groups", "any_groups", "excluded_groups", "matched_fact_code"}
)
DIAGNOSIS_AXIS_OPCODE = "emit_diagnosis_axis"
DIAGNOSIS_AXIS_PARAMETER_FIELDS = frozenset(
    {"all_groups", "any_groups", "excluded_groups"}
)
_AXIS_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,63}")
_CLINICAL_ROLES = frozenset({"current_problem", "background_condition", "differential"})
_AXIS_PRIORITIES = frozenset(
    {"routine", "high", "red_flag", "urgent", "emergency"}
)
_CLOSURE_REQUIREMENTS = frozenset(
    {
        "supported_official_diagnosis",
        "safe_escalation_or_supported_official_diagnosis",
        "urgent_hemostasis_and_resuscitation",
        "pediatric_glaucoma_specialist_closure",
        "objective_exam_support",
        "pathogen_confirmation",
        "imaging_confirmation",
    }
)
# Closed vocabulary for migrated clinical patterns. A candidate may only name a
# code that already exists here: no regex, no expression, no function name.
CLINICAL_PATTERN_FACT_CODES = frozenset(
    {
        "acute_limb_soft_tissue_infection",
        "hyperlipidemia_with_xanthelasma",
        "severe_pneumonia_aerosol_exposure",
        "pediatric_congenital_glaucoma",
        "high_energy_hindfoot_trauma",
        "limb_swelling",
        "limb_pain",
        "skin_redness_heat",
        "fever",
        "xanthelasma",
        "lipid_panel_abnormal",
        "lab_lipid",
        "adult",
        "aerosol_water_exposure",
        "respiratory_symptom",
        "respiratory_failure",
        "infant_photophobia_tearing",
        "corneal_enlargement",
        "high_energy_trauma",
        "hindfoot_deformity",
        "hindfoot_trauma_severity",
        "pediatric_patient",
        "high_pressure",
        "drug_allergy",
        "confirmed_resistance",
        "noninfectious_eczema",
        "isolated_vesicle_without_systemic_risk",
        "trauma_exposure",
        "rib_chest_wall_symptom",
        "topical_facial_steroid",
        "perioral_distribution",
        "inflammatory_papules",
        "congenital_syndactyly_fact",
        "high_risk_seafood_consumption",
        "watery_diarrhea",
        "acute_gastrointestinal_symptoms",
        "purulent_otorrhea",
        "hearing_loss_or_tinnitus",
        "chronic_course",
        # Batch-2 migrated helper booleans (closed fact codes, not function names).
        "active_upper_gi_bleed_pattern",
        "immunosuppressed_acute_infection_pattern",
        "sle_axis_pattern",
        "corneal_infection_target_rash_pattern",
        "migraine_reproductive_travel_pattern",
        "post_traumatic_cognitive_vestibular_pattern",
        "infant_congenital_structural_heart_pattern",
        "high_risk_pediatric_lower_respiratory_infection_pattern",
        "acute_ear_pain_after_instrumentation_pattern",
        "upper_arm_trauma_pattern",
        "palpitation_arrhythmia_pattern",
        "active_upper_gi_bleed_pattern",
        "elbow_overuse_pattern",
        "systemic_infection_or_inflammation_pattern",
        "hepato_splenic_cytopenia_pattern",
        "pulmonary_renal_vasculitis_pattern",
        "symptomatic_hypokalemia_malabsorption_pattern",
        "urinary_stone_infection_differential_pattern",
        "systemic_infection_hematologic_axis_pattern",
        "focal_ear_conductive_axis_pattern",
        "cryoglobulinemia_secondary_axis_pattern",
        "postmenopausal_urogenital_irritation_pattern",
        "chronic_alcohol_liver_injury_pattern",
        "pleuritic_pain_infection_embolism_pattern",
        "febrile_polyuria_dehydration_pattern",
        "postop_chylothorax_or_pleural_effusion_pattern",
        "immunosuppressed_progressive_respiratory_pattern",
        "seizure_intracranial_calcification_pattern",
        "acute_pressure_headache_intracranial_calcification_pattern",
        "decompensated_cirrhosis_pattern",
        "developmental_genetic_epilepsy_pattern",
        "post_spinal_surgery_positional_bilious_vomiting_pattern",
        "renovascular_hypertension_pattern",
        "anal_polyp_pattern",
        "rheumatoid_arthritis_ocular_pattern",
        "pediatric_leukocoria_red_flag_pattern",
        "decompensated_hfref_pattern",
        "acute_decompensated_heart_failure_pattern",
        "acute_pharyngitis_in_diabetic_child_pattern",
        "pml_imaging_pattern",
        "pediatric_upper_airway_danger_pattern",
        "suspected_asthma_control_pattern",
        "hypothalamic_pituitary_amenorrhea_pattern",
        "neck_mass_b_symptoms_pattern",
        "cholestatic_liver_disease_pattern",
        "congenital_infection_pattern",
        "leptospirosis_exposure_pattern",
        "diuretic_hypokalemia_pattern",
        "chest_wall_trauma_pattern",
        "methemoglobin_risk_pattern",
        "pediatric_airway_compression_pattern",
        "symptomatic_anemia_loss_pattern",
        "pediatric_progressive_night_blindness_pattern",
    }
)
# Public alias used by migration tests and offline compile.
PATTERN_FACT_CODES = CLINICAL_PATTERN_FACT_CODES

TREATMENT_TRANSFORM_OPCODE = "transform_treatment_codes"
TREATMENT_TRANSFORM_PARAMETER_FIELDS = frozenset(
    {"remove_codes", "replace_codes", "append_codes"}
)
# Replacements may only reference registered patch templates, never candidate
# free text, so a rule can never author a new prescription.
TREATMENT_PATCH_TEMPLATE_CODES = frozenset(
    {
        "prophylactic_antibiotic_removed",
        "aspirin_continuation_preserved",
        "pbmv_deferred_until_thrombus_excluded",
        "tricyclic_tapered_under_supervision",
        "supportive_care",
        "specialist_referral",
    }
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RULE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,63}")
_INTENT_ID_PATTERN = re.compile(r"exam_intent_[a-z0-9_]{2,63}")
_CONTROL_FIELDS = frozenset({"control_id", "kind", "facts", "assertions"})
_EFFECT_SCHEMAS: dict[str, dict[str, tuple[str, frozenset[str] | None]]] = {
    "diagnosis_differential_rule": {
        "add_diagnostic_axes": ("labels", None),
        "ranking_policy": ("enum", frozenset({"evidence_ordered", "preserve_dual_axis"})),
    },
    "diagnosis_axis_rule": {
        "axis_id": ("axis_id", None),
        "evidence": ("label_list", None),
        "missing_evidence": ("label_list", None),
        "candidate_official_names": ("label_list", None),
        "exam_intents": ("label_list", None),
        "treatment_risks": ("label_list", None),
        "clinical_role": ("enum", _CLINICAL_ROLES),
        "priority": ("enum", _AXIS_PRIORITIES),
        "closure_requirement": ("enum", _CLOSURE_REQUIREMENTS),
    },
    "clinical_closure_rule": {
        "add_exam_intent_ids": ("intent_ids", None),
        "deduplicate": ("enum", frozenset({"intent_id"})),
    },
    "diagnosis_priority_rule": {
        "priority_policy": ("enum", frozenset({"objective_evidence_first"})),
        "fallback_policy": ("enum", frozenset({"official_catalog_only"})),
    },
    "treatment_gate_rule": {
        "remove_treatment_codes": ("codes", None),
        "preserve_treatment_codes": ("codes", None),
        "gate_policy": (
            "enum",
            frozenset({"require_infection_evidence", "require_toxicity_evidence"}),
        ),
        "risk_modifier_codes": ("codes", None),
    },
    "treatment_sequence_rule": {
        "ordered_patch_codes": ("codes", None),
        "sequence_policy": ("enum", frozenset({"acute_before_stable"})),
        "skip_patch_codes": ("codes", None),
    },
}
_EFFECT_REQUIRED_FIELDS = {
    "diagnosis_differential_rule": frozenset({"add_diagnostic_axes", "ranking_policy"}),
    "diagnosis_axis_rule": frozenset(
        {
            "axis_id",
            "evidence",
            "missing_evidence",
            "candidate_official_names",
            "exam_intents",
            "treatment_risks",
            "clinical_role",
            "priority",
            "closure_requirement",
        }
    ),
    "clinical_closure_rule": frozenset({"add_exam_intent_ids", "deduplicate"}),
    "diagnosis_priority_rule": frozenset({"priority_policy", "fallback_policy"}),
    "treatment_gate_rule": frozenset(
        {"remove_treatment_codes", "preserve_treatment_codes", "gate_policy"}
    ),
    "treatment_sequence_rule": frozenset(
        {"ordered_patch_codes", "sequence_policy", "skip_patch_codes"}
    ),
}
_RUNTIME_PARAMETER_ENUMS = {
    "target_roles": frozenset({"current_problem"}),
    "target_support_levels": frozenset({"objective"}),
    "background_roles": frozenset({"background_condition"}),
    "background_relations": frozenset({"unrelated"}),
    "excluded_relations": frozenset({"explains"}),
    "preserve_urgencies": frozenset({"emergency"}),
}
_RUNTIME_PARAMETER_FIELDS = frozenset({*_RUNTIME_PARAMETER_ENUMS, "fallback_policy"})
_DIFFERENTIAL_CODE_LIST_FIELDS = frozenset(
    {
        "age_fact_codes",
        "exposure_fact_codes",
        "manifestation_fact_codes",
        "rubella_rank_fact_codes",
        "cmv_rank_fact_codes",
        "rubella_confirmed_fact_codes",
        "cmv_confirmed_fact_codes",
    }
)
_DIFFERENTIAL_PARAMETER_FIELDS = frozenset(
    {*_DIFFERENTIAL_CODE_LIST_FIELDS, "rubella_axis_id", "cmv_axis_id"}
)
_DIFFERENTIAL_CODE_DOMAINS = {
    "age_fact_codes": frozenset({"neonate", "infant"}),
    "exposure_fact_codes": frozenset({"intrauterine_viral_exposure"}),
    "manifestation_fact_codes": frozenset(
        {
            "congenital_jaundice",
            "congenital_rash",
            "infant_hearing_abnormality",
            "thrombocytopenia",
            "congenital_neuroimaging_abnormality",
            "congenital_cataract",
            "patent_ductus_arteriosus",
            "periventricular_calcifications",
            "microcephaly",
        }
    ),
    "rubella_rank_fact_codes": frozenset(
        {
            "congenital_cataract",
            "patent_ductus_arteriosus",
            "rubella_igm_positive_in_infant",
            "rubella_pcr_positive_in_infant",
        }
    ),
    "cmv_rank_fact_codes": frozenset(
        {
            "periventricular_calcifications",
            "microcephaly",
            "cmv_igm_positive",
            "cmv_pcr_positive",
            "cmv_saliva_or_urine_pcr_positive_within_21_days",
        }
    ),
    "rubella_confirmed_fact_codes": frozenset(
        {"rubella_igm_positive_in_infant", "rubella_pcr_positive_in_infant"}
    ),
    "cmv_confirmed_fact_codes": frozenset(
        {"cmv_saliva_or_urine_pcr_positive_within_21_days"}
    ),
}
_TYPICAL_CONGENITAL_MANIFESTATIONS = frozenset(
    {
        "congenital_cataract",
        "patent_ductus_arteriosus",
        "periventricular_calcifications",
        "microcephaly",
    }
)
_COMPILED_CONSTRUCTION_TOKEN = object()
_OFFICIAL_DISEASES_CACHE: frozenset[str] | None = None


def _official_disease_names() -> frozenset[str]:
    global _OFFICIAL_DISEASES_CACHE
    if _OFFICIAL_DISEASES_CACHE is not None:
        return _OFFICIAL_DISEASES_CACHE
    catalog_path = Path(__file__).resolve().parents[2] / "data" / "ref_data" / "diseases_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    diseases = payload.get("diseases") if isinstance(payload, Mapping) else None
    if not isinstance(diseases, Mapping):
        raise ValueError("official disease catalog is invalid")
    names = frozenset(
        str(name).strip()
        for values in diseases.values()
        if isinstance(values, list)
        for name in values
        if str(name).strip()
    )
    if not names:
        raise ValueError("official disease catalog is empty")
    _OFFICIAL_DISEASES_CACHE = names
    return names


class _ParserConstructed:
    def __init__(self, *, _token: object) -> None:
        if _token is not _COMPILED_CONSTRUCTION_TOKEN:
            raise TypeError("compiled values must be created by the parser or empty factory")


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _enum(value: object, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field} has an unsupported value")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple")
    items = tuple(_nonempty_string(item, field=field) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field} contains duplicate codes")
    return items


@dataclass(frozen=True)
class RuleDiagnosisCandidate:
    official_name: str
    role: Literal["current_problem", "background_condition", "differential"]
    support_level: Literal["objective", "subjective", "none"]
    complaint_relation: Literal["explains", "unrelated", "unknown"]
    urgency: Literal["routine", "urgent", "emergency"]
    evidence_codes: tuple[str, ...]
    is_official: bool

    def __post_init__(self) -> None:
        _nonempty_string(self.official_name, field="official_name")
        _enum(self.role, field="role", allowed=_ROLES)
        _enum(self.support_level, field="support_level", allowed=_SUPPORT_LEVELS)
        _enum(self.complaint_relation, field="complaint_relation", allowed=_COMPLAINT_RELATIONS)
        _enum(self.urgency, field="urgency", allowed=_URGENCIES)
        _string_tuple(self.evidence_codes, field="evidence_codes")
        if type(self.is_official) is not bool:
            raise ValueError("is_official must be bool")


@dataclass(frozen=True)
class RuleDiagnosisAxis:
    """Closed diagnosis-axis template emitted by emit_diagnosis_axis."""

    axis_id: str
    evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    candidate_official_names: tuple[str, ...]
    exam_intents: tuple[str, ...]
    treatment_risks: tuple[str, ...]
    clinical_role: str = "current_problem"
    priority: str = "routine"
    closure_requirement: str = "supported_official_diagnosis"

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, str) or not _AXIS_ID_PATTERN.fullmatch(self.axis_id):
            raise ValueError("axis_id must be snake_case")
        for field in (
            "evidence",
            "missing_evidence",
            "exam_intents",
            "treatment_risks",
        ):
            _string_tuple(getattr(self, field), field=field)
        # Red-flag axes may emit empty candidates and force safe escalation instead.
        if self.candidate_official_names:
            _string_tuple(self.candidate_official_names, field="candidate_official_names")
        elif self.priority != "red_flag":
            raise ValueError("candidate_official_names must be non-empty")
        _enum(self.clinical_role, field="clinical_role", allowed=_CLINICAL_ROLES)
        _enum(self.priority, field="priority", allowed=_AXIS_PRIORITIES)
        _enum(
            self.closure_requirement,
            field="closure_requirement",
            allowed=_CLOSURE_REQUIREMENTS,
        )

    def as_legacy_axis(self) -> dict[str, object]:
        names = list(self.candidate_official_names)
        return {
            "axis_id": self.axis_id,
            "source": "rule",
            "status": "suspected",
            "evidence": list(self.evidence),
            "missing_evidence": list(self.missing_evidence),
            "candidate_official_names": names,
            "rule_candidate_official_names": names,
            "exam_intents": list(self.exam_intents),
            "treatment_risks": list(self.treatment_risks),
            "clinical_role": self.clinical_role,
            "priority": self.priority,
            "closure_requirement": self.closure_requirement,
        }


@dataclass(frozen=True)
class RuleContext:
    diagnosis_candidates: tuple[RuleDiagnosisCandidate, ...] = ()
    preferred_diagnosis: str | None = None
    diagnostic_axis_ids: tuple[str, ...] = ()
    diagnosis_axes: tuple[RuleDiagnosisAxis, ...] = ()
    exam_intent_ids: tuple[str, ...] = ()
    treatment_codes: tuple[str, ...] = ()
    fact_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.diagnosis_candidates, tuple) or not all(
            isinstance(candidate, RuleDiagnosisCandidate)
            for candidate in self.diagnosis_candidates
        ):
            raise ValueError("diagnosis_candidates must be a typed tuple")
        names = tuple(candidate.official_name for candidate in self.diagnosis_candidates)
        if len(set(names)) != len(names):
            raise ValueError("diagnosis_candidates contains duplicate official_name")
        if self.preferred_diagnosis is not None:
            _nonempty_string(self.preferred_diagnosis, field="preferred_diagnosis")
        if not isinstance(self.diagnosis_axes, tuple) or not all(
            isinstance(axis, RuleDiagnosisAxis) for axis in self.diagnosis_axes
        ):
            raise ValueError("diagnosis_axes must be a typed tuple")
        axis_ids = tuple(axis.axis_id for axis in self.diagnosis_axes)
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("diagnosis_axes contains duplicate axis_id")
        for field in (
            "diagnostic_axis_ids",
            "exam_intent_ids",
            "treatment_codes",
            "fact_codes",
        ):
            _string_tuple(getattr(self, field), field=field)


@dataclass(frozen=True)
class RuleDecision:
    rule_id: str
    opcode: str
    outcome: RuleOutcome
    reason_code: str

    def __post_init__(self) -> None:
        _nonempty_string(self.rule_id, field="rule_id")
        _nonempty_string(self.opcode, field="opcode")
        _enum(self.outcome, field="outcome", allowed=_OUTCOMES)
        _nonempty_string(self.reason_code, field="reason_code")


@dataclass(frozen=True)
class RuleResult:
    output_context: RuleContext
    decisions: tuple[RuleDecision, ...]
    before_hash: str
    after_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_context, RuleContext):
            raise ValueError("output_context must be RuleContext")
        if not isinstance(self.decisions, tuple) or not all(
            isinstance(decision, RuleDecision) for decision in self.decisions
        ):
            raise ValueError("decisions must be a typed tuple")
        _nonempty_string(self.before_hash, field="before_hash")
        _nonempty_string(self.after_hash, field="after_hash")

    @property
    def applied_rule_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        applied: list[str] = []
        for decision in self.decisions:
            if decision.outcome == "applied" and decision.rule_id not in seen:
                seen.add(decision.rule_id)
                applied.append(decision.rule_id)
        return cast(tuple[str, ...], tuple(applied))


@dataclass(frozen=True, init=False)
class CompiledRuleParameters(_ParserConstructed):
    target_roles: tuple[str, ...]
    target_support_levels: tuple[str, ...]
    background_roles: tuple[str, ...]
    background_relations: tuple[str, ...]
    excluded_relations: tuple[str, ...]
    preserve_urgencies: tuple[str, ...]
    fallback_policy: str


@dataclass(frozen=True, init=False)
class CompiledDifferentialRuleParameters(_ParserConstructed):
    age_fact_codes: tuple[str, ...]
    exposure_fact_codes: tuple[str, ...]
    manifestation_fact_codes: tuple[str, ...]
    rubella_rank_fact_codes: tuple[str, ...]
    cmv_rank_fact_codes: tuple[str, ...]
    rubella_confirmed_fact_codes: tuple[str, ...]
    cmv_confirmed_fact_codes: tuple[str, ...]
    rubella_axis_id: str
    cmv_axis_id: str


@dataclass(frozen=True, init=False)
class CompiledRuleRuntime(_ParserConstructed):
    status: Literal["audit_only", "active"]
    stage: RuleStage
    opcode: str | None = None
    parameters: CompiledRuleParameters | CompiledDifferentialRuleParameters | None = None


@dataclass(frozen=True, init=False)
class CompiledRule(_ParserConstructed):
    rule_id: str
    candidate_type: CandidateType
    candidate_hash: str
    effect_hash: str
    priority: int
    phase: Literal["diagnosis", "closure", "treatment"]
    application: Literal["trigger_bound"]
    runtime: CompiledRuleRuntime


@dataclass(frozen=True, init=False)
class CompiledRulePack(_ParserConstructed):
    schema_version: str
    rules: tuple[CompiledRule, ...]
    rule_count: int
    rules_hash: str


_CompiledValue = TypeVar("_CompiledValue")


def _compiled_state(value: object) -> object:
    if isinstance(value, _ParserConstructed):
        return (
            type(value).__name__,
            tuple(
                (field, _compiled_state(field_value))
                for field, field_value in sorted(vars(value).items())
            ),
        )
    if isinstance(value, tuple):
        return tuple(_compiled_state(item) for item in value)
    return value


def _compiled_seal(value: object) -> str:
    return sha256(repr(_compiled_state(value)).encode("utf-8")).hexdigest()


def _compiled_registry_accessors() -> tuple[
    Callable[[object], None],
    Callable[[object], bool],
]:
    validated: dict[int, tuple[object, str]] = {}

    def register(value: object) -> None:
        registered = validated.get(id(value))
        if registered is not None and registered[0] is value:
            return
        validated[id(value)] = (value, _compiled_seal(value))

    def is_validated(value: object) -> bool:
        registered = validated.get(id(value))
        return bool(
            registered is not None
            and registered[0] is value
            and registered[1] == _compiled_seal(value)
        )

    return register, is_validated


_register_compiled_value, _is_validated_compiled_value = (
    _compiled_registry_accessors()
)
del _compiled_registry_accessors


def _compiled_value(model: type[_CompiledValue], **fields: object) -> _CompiledValue:
    value = model(_token=_COMPILED_CONSTRUCTION_TOKEN)
    for field, field_value in fields.items():
        object.__setattr__(value, field, field_value)
    _register_compiled_value(value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _context_payload(context: RuleContext) -> dict[str, object]:
    return {
        "diagnosis_candidates": [
            {
                "official_name": candidate.official_name,
                "role": candidate.role,
                "support_level": candidate.support_level,
                "complaint_relation": candidate.complaint_relation,
                "urgency": candidate.urgency,
                "evidence_codes": list(candidate.evidence_codes),
                "is_official": candidate.is_official,
            }
            for candidate in context.diagnosis_candidates
        ],
        "preferred_diagnosis": context.preferred_diagnosis,
        "diagnostic_axis_ids": list(context.diagnostic_axis_ids),
        "diagnosis_axes": [
            {
                "axis_id": axis.axis_id,
                "evidence": list(axis.evidence),
                "missing_evidence": list(axis.missing_evidence),
                "candidate_official_names": list(axis.candidate_official_names),
                "exam_intents": list(axis.exam_intents),
                "treatment_risks": list(axis.treatment_risks),
                "clinical_role": axis.clinical_role,
                "priority": axis.priority,
                "closure_requirement": axis.closure_requirement,
            }
            for axis in context.diagnosis_axes
        ],
        "exam_intent_ids": list(context.exam_intent_ids),
        "treatment_codes": list(context.treatment_codes),
        "fact_codes": list(context.fact_codes),
    }


def empty_compiled_rule_pack() -> CompiledRulePack:
    return _compiled_value(
        CompiledRulePack,
        schema_version="compiled-knowledge-rules/v2",
        rules=(),
        rule_count=0,
        rules_hash=_content_hash([]),
    )


def _compiled_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty string list")
    if not all(isinstance(item, str) and item and item.strip() == item for item in value):
        raise ValueError(f"{field} must be a non-empty trimmed string list")
    return value


def _validate_rule_identity(rule: Mapping[str, Any]) -> tuple[str, str, int]:
    rule_id = rule.get("rule_id")
    if not isinstance(rule_id, str) or not _RULE_ID_PATTERN.fullmatch(rule_id):
        raise ValueError("rule_id must be snake_case")
    candidate_type = rule.get("candidate_type")
    if not isinstance(candidate_type, str) or candidate_type not in _CANDIDATE_TYPES:
        raise ValueError("candidate_type is unsupported")
    for field in ("candidate_hash", "effect_hash"):
        value = rule.get(field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"{field} must be a lowercase sha256")
    priority = rule.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 1000:
        raise ValueError("priority must be an integer from 0 to 1000")
    return rule_id, candidate_type, priority


def _validate_scope(value: object, *, candidate_type: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"phase", "application"}:
        raise ValueError("scope fields must be phase and application")
    if value.get("phase") != _PHASE_BY_TYPE[candidate_type]:
        raise ValueError("scope phase does not match candidate_type")
    if value.get("application") != "trigger_bound":
        raise ValueError("scope application must be trigger_bound")


def _validate_control(control: object, *, field: str, kinds: frozenset[str]) -> tuple[str, str]:
    if not isinstance(control, Mapping) or set(control) != _CONTROL_FIELDS:
        raise ValueError(f"{field} control fields must exactly match the schema")
    control_id = control.get("control_id")
    if not isinstance(control_id, str) or not _RULE_ID_PATTERN.fullmatch(
        control_id.replace("-", "_")
    ):
        raise ValueError("control_id is invalid")
    kind = control.get("kind")
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError(f"{field} control kind is invalid")
    _compiled_string_list(control.get("facts"), field=f"{field}.facts")
    _compiled_string_list(control.get("assertions"), field=f"{field}.assertions")
    return control_id, kind


def _validate_controls(rule: Mapping[str, Any], *, candidate_type: str) -> None:
    positive = rule.get("positive_controls")
    negative = rule.get("negative_controls")
    if not isinstance(positive, list) or not positive:
        raise ValueError("positive_controls must be non-empty")
    if not isinstance(negative, list) or not negative:
        raise ValueError("negative_controls must be non-empty")
    validated = [
        *(_validate_control(item, field="positive_controls", kinds=frozenset({"positive"})) for item in positive),
        *(
            _validate_control(
                item,
                field="negative_controls",
                kinds=frozenset({"near_neighbor", "reasonable_exception"}),
            )
            for item in negative
        ),
    ]
    control_ids = [control_id for control_id, _ in validated]
    if len(set(control_ids)) != len(control_ids):
        raise ValueError("duplicate control_id")
    negative_kinds = {kind for _, kind in validated if kind != "positive"}
    if candidate_type in _TREATMENT_TYPES and "near_neighbor" not in negative_kinds:
        raise ValueError("treatment rule requires near_neighbor negative control")
    if candidate_type in _TREATMENT_TYPES and "reasonable_exception" not in negative_kinds:
        raise ValueError("treatment rule requires reasonable_exception negative control")


def _validate_ref(ref: object, *, field: str) -> tuple[str, str]:
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{field} ref fields must be path and sha256")
    path = ref.get("path")
    invalid_path = (
        not isinstance(path, str)
        or not path
        or path.strip() != path
        or path.startswith("/")
        or "\\" in path
        or bool(re.match(r"^[A-Za-z]:", path))
        or any(part in {"", ".", ".."} for part in path.split("/"))
    )
    if invalid_path:
        raise ValueError(f"{field} ref path must be project-relative POSIX")
    value = ref.get("sha256")
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} ref sha256 is invalid")
    return path, value


def _validate_refs(value: object, *, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} refs must be non-empty")
    refs = [_validate_ref(item, field=field) for item in value]
    paths = [path for path, _ in refs]
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate {field} ref path")


def _effect_string(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 64:
        raise ValueError(f"effect string is invalid: {field}")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"effect structured value is invalid: {field}")
    return value


def _effect_list(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"effect field must be a non-empty list: {field}")
    items = [
        _effect_string(item, field=f"{field}[{index}]", pattern=pattern)
        for index, item in enumerate(value)
    ]
    if len(set(items)) != len(items):
        raise ValueError(f"effect field contains duplicate values: {field}")


def _validate_effect(value: object, *, candidate_type: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("effect must be a non-empty object")
    schema = _EFFECT_SCHEMAS[candidate_type]
    fields = set(value)
    required = _EFFECT_REQUIRED_FIELDS[candidate_type]
    if not required.issubset(fields) or not fields.issubset(schema):
        raise ValueError(f"{candidate_type} effect fields do not match the typed whitelist")
    for field, raw_value in value.items():
        kind, allowed = schema[field]
        if kind == "labels":
            _effect_list(raw_value, field=field)
        elif kind == "label_list":
            # Chinese clinical labels; empty lists allowed (missing_evidence / risks).
            if not isinstance(raw_value, list):
                raise ValueError(f"effect field must be a list: {field}")
            seen: set[str] = set()
            for index, item in enumerate(raw_value):
                if not isinstance(item, str) or not item or item.strip() != item or len(item) > 128:
                    raise ValueError(f"effect label is invalid: {field}[{index}]")
                if item in seen:
                    raise ValueError(f"effect field contains duplicate values: {field}")
                seen.add(item)
        elif kind == "codes":
            _effect_list(raw_value, field=field, pattern=_CODE_PATTERN)
        elif kind == "intent_ids":
            _effect_list(raw_value, field=field, pattern=_INTENT_ID_PATTERN)
        elif kind == "axis_id":
            if not isinstance(raw_value, str) or not _AXIS_ID_PATTERN.fullmatch(raw_value):
                raise ValueError(f"effect axis_id is invalid: {field}")
        elif kind == "enum":
            enum_value = _effect_string(raw_value, field=field, pattern=_CODE_PATTERN)
            if allowed is None or enum_value not in allowed:
                raise ValueError(f"effect enum is unsupported: {field}")
        else:
            raise ValueError(f"unsupported effect kind: {kind}")


def _runtime_enum_list(value: object, *, field: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"runtime parameter must be a non-empty enum list: {field}")
    if not all(isinstance(item, str) and item in allowed for item in value):
        raise ValueError(f"runtime parameter has an unsupported enum: {field}")
    if len(set(value)) != len(value):
        raise ValueError(f"runtime parameter contains duplicate values: {field}")
    return tuple(value)


def _validated_parameters(value: object) -> CompiledRuleParameters:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_PARAMETER_FIELDS:
        raise ValueError("runtime parameter fields must exactly match the opcode schema")
    enum_values = {
        field: _runtime_enum_list(value.get(field), field=field, allowed=allowed)
        for field, allowed in _RUNTIME_PARAMETER_ENUMS.items()
    }
    fallback_policy = value.get("fallback_policy")
    if fallback_policy != "official_catalog_only":
        raise ValueError("runtime parameter has an unsupported enum: fallback_policy")
    return _compiled_value(
        CompiledRuleParameters,
        target_roles=enum_values["target_roles"],
        target_support_levels=enum_values["target_support_levels"],
        background_roles=enum_values["background_roles"],
        background_relations=enum_values["background_relations"],
        excluded_relations=enum_values["excluded_relations"],
        preserve_urgencies=enum_values["preserve_urgencies"],
        fallback_policy=fallback_policy,
    )


def _runtime_code(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _CODE_PATTERN.fullmatch(value):
        raise ValueError(f"runtime parameter must be a structured code: {field}")
    return value


def _runtime_code_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"runtime parameter must be a non-empty code list: {field}")
    codes = tuple(_runtime_code(item, field=field) for item in value)
    if len(set(codes)) != len(codes):
        raise ValueError(f"runtime parameter contains duplicate codes: {field}")
    return codes


def _validate_differential_code_lists(
    code_lists: Mapping[str, tuple[str, ...]],
) -> None:
    age = set(code_lists["age_fact_codes"])
    exposure = set(code_lists["exposure_fact_codes"])
    manifestation = set(code_lists["manifestation_fact_codes"])
    if age & exposure or age & manifestation or exposure & manifestation:
        raise ValueError("runtime differential trigger groups must not overlap")
    rubella_confirmed = set(code_lists["rubella_confirmed_fact_codes"])
    cmv_confirmed = set(code_lists["cmv_confirmed_fact_codes"])
    if rubella_confirmed & cmv_confirmed:
        raise ValueError("runtime differential confirmed groups must not overlap")
    for field, codes in code_lists.items():
        if not set(codes).issubset(_DIFFERENTIAL_CODE_DOMAINS[field]):
            raise ValueError(f"runtime differential parameter is unsupported: {field}")
    if not _TYPICAL_CONGENITAL_MANIFESTATIONS.issubset(manifestation):
        raise ValueError("runtime differential typical manifestations are required")
    if not rubella_confirmed.issubset(code_lists["rubella_rank_fact_codes"]):
        raise ValueError("runtime differential rubella confirmed facts must be rank facts")
    if not cmv_confirmed.issubset(code_lists["cmv_rank_fact_codes"]):
        raise ValueError("runtime differential cmv confirmed facts must be rank facts")
    for field, codes in code_lists.items():
        if set(codes) != _DIFFERENTIAL_CODE_DOMAINS[field]:
            raise ValueError(
                f"runtime differential parameter must match canonical code set: {field}"
            )


def _validate_active_differential_effect(value: object) -> None:
    expected = {
        "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
        "ranking_policy": "evidence_ordered",
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("active differential effect does not match its runtime")


def _validated_differential_parameters(
    value: object,
) -> CompiledDifferentialRuleParameters:
    if not isinstance(value, Mapping) or set(value) != _DIFFERENTIAL_PARAMETER_FIELDS:
        raise ValueError("runtime parameter fields must exactly match the opcode schema")
    code_lists = {
        field: _runtime_code_list(value.get(field), field=field)
        for field in _DIFFERENTIAL_CODE_LIST_FIELDS
    }
    _validate_differential_code_lists(code_lists)
    rubella_axis_id = _runtime_code(value.get("rubella_axis_id"), field="rubella_axis_id")
    cmv_axis_id = _runtime_code(value.get("cmv_axis_id"), field="cmv_axis_id")
    if rubella_axis_id != "congenital_rubella":
        raise ValueError("runtime differential parameter rubella_axis_id is unsupported")
    if cmv_axis_id != "congenital_cmv":
        raise ValueError("runtime differential parameter cmv_axis_id is unsupported")
    return _compiled_value(
        CompiledDifferentialRuleParameters,
        age_fact_codes=code_lists["age_fact_codes"],
        exposure_fact_codes=code_lists["exposure_fact_codes"],
        manifestation_fact_codes=code_lists["manifestation_fact_codes"],
        rubella_rank_fact_codes=code_lists["rubella_rank_fact_codes"],
        cmv_rank_fact_codes=code_lists["cmv_rank_fact_codes"],
        rubella_confirmed_fact_codes=code_lists["rubella_confirmed_fact_codes"],
        cmv_confirmed_fact_codes=code_lists["cmv_confirmed_fact_codes"],
        rubella_axis_id=rubella_axis_id,
        cmv_axis_id=cmv_axis_id,
    )


def _fact_code_groups(
    value: object,
    *,
    field: str,
) -> tuple[tuple[str, ...], ...]:
    """Groups of registered fact codes only: data, never expressions."""
    if not isinstance(value, list):
        raise ValueError("%s must be a list of code groups" % field)
    groups: list[tuple[str, ...]] = []
    for group in value:
        if not isinstance(group, list) or not group:
            raise ValueError("%s entries must be non-empty code lists" % field)
        codes: list[str] = []
        for code in group:
            if not isinstance(code, str):
                raise ValueError("%s codes must be strings" % field)
            if code not in CLINICAL_PATTERN_FACT_CODES:
                raise ValueError("%s contains an unregistered fact code: %r" % (field, code))
            codes.append(code)
        if len(set(codes)) != len(codes):
            raise ValueError("%s group contains duplicate codes" % field)
        groups.append(tuple(codes))
    return tuple(groups)


@dataclass(frozen=True, init=False)
class CompiledFactGroupParameters(_ParserConstructed):
    all_groups: tuple[tuple[str, ...], ...]
    any_groups: tuple[tuple[str, ...], ...]
    excluded_groups: tuple[tuple[str, ...], ...]
    matched_fact_code: str
    # Copied from validated clinical_closure_rule effect so runtime can apply it.
    add_exam_intent_ids: tuple[str, ...]
    deduplicate: str | None


@dataclass(frozen=True, init=False)
class CompiledTreatmentTransformParameters(_ParserConstructed):
    remove_codes: tuple[str, ...]
    replace_codes: tuple[tuple[str, str], ...]
    append_codes: tuple[str, ...]


def _validated_fact_group_effect(
    effect: object,
) -> tuple[tuple[str, ...], str | None]:
    """Lift clinical_closure effect fields into runtime-usable typed values."""
    if not isinstance(effect, Mapping):
        raise ValueError("clinical_closure effect must be an object")
    raw_intents = effect.get("add_exam_intent_ids")
    if not isinstance(raw_intents, list) or not raw_intents:
        raise ValueError("add_exam_intent_ids must be a non-empty list")
    intents: list[str] = []
    for item in raw_intents:
        if not isinstance(item, str) or not _INTENT_ID_PATTERN.fullmatch(item):
            raise ValueError("add_exam_intent_ids entries must be exam_intent_* ids")
        intents.append(item)
    if len(set(intents)) != len(intents):
        raise ValueError("add_exam_intent_ids contains duplicates")
    deduplicate = effect.get("deduplicate")
    if deduplicate is not None:
        if not isinstance(deduplicate, str) or deduplicate != "intent_id":
            raise ValueError("deduplicate must be intent_id when present")
    return tuple(intents), deduplicate if isinstance(deduplicate, str) else None


def _validated_fact_group_parameters(
    value: object,
    *,
    effect: object,
) -> CompiledFactGroupParameters:
    if not isinstance(value, Mapping) or set(value) != FACT_GROUP_PARAMETER_FIELDS:
        raise ValueError("fact group parameters must exactly match the typed schema")
    all_groups = _fact_code_groups(value.get("all_groups"), field="all_groups")
    any_groups = _fact_code_groups(value.get("any_groups"), field="any_groups")
    excluded_groups = _fact_code_groups(value.get("excluded_groups"), field="excluded_groups")
    if not all_groups and not any_groups:
        raise ValueError("fact group rule needs at least one all_groups or any_groups entry")
    matched = value.get("matched_fact_code")
    if not isinstance(matched, str) or matched not in CLINICAL_PATTERN_FACT_CODES:
        raise ValueError("matched_fact_code must be a registered fact code")
    add_exam_intent_ids, deduplicate = _validated_fact_group_effect(effect)
    return _compiled_value(
        CompiledFactGroupParameters,
        all_groups=all_groups,
        any_groups=any_groups,
        excluded_groups=excluded_groups,
        matched_fact_code=matched,
        add_exam_intent_ids=add_exam_intent_ids,
        deduplicate=deduplicate,
    )


def _validated_treatment_transform_parameters(
    value: object,
) -> CompiledTreatmentTransformParameters:
    if not isinstance(value, Mapping) or set(value) != TREATMENT_TRANSFORM_PARAMETER_FIELDS:
        raise ValueError("treatment transform parameters must exactly match the typed schema")
    remove = value.get("remove_codes")
    append = value.get("append_codes")
    replace = value.get("replace_codes")
    if not isinstance(remove, list) or not isinstance(append, list) or not isinstance(replace, list):
        raise ValueError("treatment transform parameters must be lists")
    removed = tuple(_treatment_code(code, field="remove_codes") for code in remove)
    appended = tuple(
        _patch_template_code(code, field="append_codes") for code in append
    )
    pairs: list[tuple[str, str]] = []
    for item in replace:
        if not isinstance(item, Mapping) or set(item) != {"code", "replacement_code"}:
            raise ValueError("replace_codes entries must be code/replacement_code objects")
        pairs.append(
            (
                _treatment_code(item.get("code"), field="replace_codes"),
                _patch_template_code(item.get("replacement_code"), field="replace_codes"),
            )
        )
    if not removed and not pairs and not appended:
        raise ValueError("treatment transform needs at least one operation")
    return _compiled_value(
        CompiledTreatmentTransformParameters,
        remove_codes=removed,
        replace_codes=tuple(pairs),
        append_codes=appended,
    )


def _treatment_code(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _CODE_PATTERN.fullmatch(value):
        raise ValueError("%s must contain snake_case treatment codes" % field)
    return value


def _patch_template_code(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in TREATMENT_PATCH_TEMPLATE_CODES:
        raise ValueError("%s must reference a registered patch template code" % field)
    return value



@dataclass(frozen=True, init=False)
class CompiledDiagnosisAxisParameters(_ParserConstructed):
    all_groups: tuple[tuple[str, ...], ...]
    any_groups: tuple[tuple[str, ...], ...]
    excluded_groups: tuple[tuple[str, ...], ...]
    axis: RuleDiagnosisAxis


def _label_tuple(value: object, *, field: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list of labels" % field)
    if not value and not allow_empty:
        raise ValueError("%s must be a non-empty label list" % field)
    labels: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item.strip() != item:
            raise ValueError("%s entries must be non-empty trimmed strings" % field)
        labels.append(item)
    if len(set(labels)) != len(labels):
        raise ValueError("%s contains duplicate labels" % field)
    return tuple(labels)


def _validated_diagnosis_axis_parameters(
    value: object,
    *,
    effect: object,
) -> CompiledDiagnosisAxisParameters:
    if not isinstance(value, Mapping) or set(value) != DIAGNOSIS_AXIS_PARAMETER_FIELDS:
        raise ValueError("diagnosis axis parameters must exactly match the typed schema")
    if not isinstance(effect, Mapping):
        raise ValueError("diagnosis_axis_rule effect must be an object")
    all_groups = _fact_code_groups(value.get("all_groups"), field="all_groups")
    any_groups = _fact_code_groups(value.get("any_groups"), field="any_groups")
    excluded_groups = _fact_code_groups(value.get("excluded_groups"), field="excluded_groups")
    if not all_groups and not any_groups:
        raise ValueError("diagnosis axis rule needs at least one all_groups or any_groups entry")
    axis_id = effect.get("axis_id")
    if not isinstance(axis_id, str) or not _AXIS_ID_PATTERN.fullmatch(axis_id):
        raise ValueError("axis_id must be snake_case")
    priority = str(effect.get("priority") or "routine")
    raw_candidates = effect.get("candidate_official_names")
    if raw_candidates is None:
        raise ValueError("candidate_official_names is required")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate_official_names must be a list")
    if raw_candidates:
        candidate_official_names = _label_tuple(
            raw_candidates, field="candidate_official_names"
        )
    elif priority == "red_flag":
        candidate_official_names = ()
    else:
        raise ValueError("candidate_official_names must be a non-empty label list")
    unknown_candidates = sorted(set(candidate_official_names) - _official_disease_names())
    if unknown_candidates:
        raise ValueError(
            "candidate_official_names contains non-catalog diseases: %s"
            % unknown_candidates
        )
    axis = RuleDiagnosisAxis(
        axis_id=axis_id,
        evidence=_label_tuple(effect.get("evidence"), field="evidence"),
        missing_evidence=_label_tuple(
            effect.get("missing_evidence"), field="missing_evidence", allow_empty=True
        ),
        candidate_official_names=candidate_official_names,
        exam_intents=_label_tuple(effect.get("exam_intents"), field="exam_intents", allow_empty=True),
        treatment_risks=_label_tuple(
            effect.get("treatment_risks"), field="treatment_risks", allow_empty=True
        ),
        clinical_role=str(effect.get("clinical_role") or "current_problem"),
        priority=priority,
        closure_requirement=str(
            effect.get("closure_requirement") or "supported_official_diagnosis"
        ),
    )
    return _compiled_value(
        CompiledDiagnosisAxisParameters,
        all_groups=all_groups,
        any_groups=any_groups,
        excluded_groups=excluded_groups,
        axis=axis,
    )


def _apply_diagnosis_axis_emit(
    rule: CompiledRule,
    context: RuleContext,
) -> tuple[RuleContext, RuleDecision]:
    """Fact-group match then emit a closed diagnosis axis template."""
    parameters = rule.runtime.parameters
    if not isinstance(parameters, CompiledDiagnosisAxisParameters):
        raise RuntimeError("diagnosis axis rule is missing typed parameters")
    present = frozenset(context.fact_codes)

    def fires(group: tuple[str, ...]) -> bool:
        return bool(present.intersection(group))

    if any(fires(group) for group in parameters.excluded_groups):
        return context, RuleDecision(
            rule.rule_id,
            DIAGNOSIS_AXIS_OPCODE,
            "excluded",
            "excluded_group_present",
        )
    if parameters.all_groups and not all(fires(group) for group in parameters.all_groups):
        return context, RuleDecision(
            rule.rule_id,
            DIAGNOSIS_AXIS_OPCODE,
            "not_matched",
            "all_groups_incomplete",
        )
    if parameters.any_groups and not any(fires(group) for group in parameters.any_groups):
        return context, RuleDecision(
            rule.rule_id,
            DIAGNOSIS_AXIS_OPCODE,
            "not_matched",
            "any_groups_missed",
        )
    axis = parameters.axis
    existing_ids = {item.axis_id for item in context.diagnosis_axes}
    if axis.axis_id in existing_ids or axis.axis_id in context.diagnostic_axis_ids:
        return context, RuleDecision(
            rule.rule_id,
            DIAGNOSIS_AXIS_OPCODE,
            "matched_no_change",
            "axis_already_present",
        )
    updated = dataclasses.replace(
        context,
        diagnosis_axes=tuple([*context.diagnosis_axes, axis]),
        diagnostic_axis_ids=tuple([*context.diagnostic_axis_ids, axis.axis_id]),
    )
    return updated, RuleDecision(
        rule.rule_id,
        DIAGNOSIS_AXIS_OPCODE,
        "applied",
        "diagnosis_axis_emitted",
    )



def _validated_runtime(
    value: object,
    *,
    candidate_type: str,
    effect: object,
) -> CompiledRuleRuntime:
    if not isinstance(value, Mapping):
        raise ValueError("runtime must be an object")
    stage = value.get("stage")
    if not isinstance(stage, str) or stage != _STAGE_BY_TYPE[candidate_type]:
        raise ValueError("runtime stage does not match candidate_type")
    status = value.get("status")
    if status == "audit_only":
        if set(value) != {"status", "stage"}:
            raise ValueError("audit-only runtime fields must be status and stage")
        return _compiled_value(
            CompiledRuleRuntime,
            status="audit_only",
            stage=cast(RuleStage, stage),
            opcode=None,
            parameters=None,
        )
    if status != "active":
        raise ValueError("runtime status is unsupported")
    if set(value) != {"status", "stage", "opcode", "parameters"}:
        raise ValueError("active runtime fields must exactly match the typed schema")
    opcode = value.get("opcode")
    if candidate_type == "diagnosis_priority_rule":
        if opcode != "promote_supported_current_over_background":
            raise ValueError("runtime opcode is unsupported")
        parameters: object = (
            _validated_parameters(value.get("parameters"))
        )
    elif candidate_type == "diagnosis_differential_rule":
        if opcode != "expand_congenital_infection_axes":
            raise ValueError("runtime opcode is unsupported")
        _validate_active_differential_effect(effect)
        parameters = _validated_differential_parameters(value.get("parameters"))
    elif candidate_type == "diagnosis_axis_rule":
        if opcode != DIAGNOSIS_AXIS_OPCODE:
            raise ValueError("runtime opcode is unsupported")
        parameters = _validated_diagnosis_axis_parameters(
            value.get("parameters"),
            effect=effect,
        )
    elif candidate_type == "clinical_closure_rule":
        if opcode != FACT_GROUP_OPCODE:
            raise ValueError("runtime opcode is unsupported")
        parameters = _validated_fact_group_parameters(
            value.get("parameters"),
            effect=effect,
        )
    elif candidate_type in _TREATMENT_TYPES:
        if opcode != TREATMENT_TRANSFORM_OPCODE:
            raise ValueError("runtime opcode is unsupported")
        parameters = _validated_treatment_transform_parameters(value.get("parameters"))
    else:
        raise ValueError("active runtime is unsupported for candidate_type")
    if not isinstance(opcode, str):
        raise ValueError("runtime opcode is unsupported")
    return _compiled_value(
        CompiledRuleRuntime,
        status="active",
        stage=cast(RuleStage, stage),
        opcode=opcode,
        parameters=parameters,
    )


def _validate_rule_structure(rule: Mapping[str, Any]) -> tuple[str, str, int]:
    if set(rule) != _RULE_FIELDS:
        raise ValueError("compiled rule fields must exactly match schema v2")
    rule_id, candidate_type, priority = _validate_rule_identity(rule)
    for field in ("triggers", "required_evidence", "exclusions"):
        _compiled_string_list(rule.get(field), field=field)
    _validate_scope(rule.get("scope"), candidate_type=candidate_type)
    _validate_controls(rule, candidate_type=candidate_type)
    _validate_refs(rule.get("source_refs"), field="source_refs")
    _validate_refs(rule.get("test_refs"), field="test_refs")
    _validate_effect(rule.get("effect"), candidate_type=candidate_type)
    _validated_runtime(
        rule.get("runtime"),
        candidate_type=candidate_type,
        effect=rule.get("effect"),
    )
    return rule_id, candidate_type, priority


def _typed_rule(rule: Mapping[str, Any]) -> CompiledRule:
    scope = rule.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("rule scope and runtime must be objects")
    candidate_type = rule.get("candidate_type")
    if not isinstance(candidate_type, str):
        raise ValueError("candidate_type is unsupported")
    return _compiled_value(
        CompiledRule,
        rule_id=rule.get("rule_id"),
        candidate_type=cast(CandidateType, candidate_type),
        candidate_hash=rule.get("candidate_hash"),
        effect_hash=rule.get("effect_hash"),
        priority=rule.get("priority"),
        phase=scope.get("phase"),
        application=scope.get("application"),
        runtime=_validated_runtime(
            rule.get("runtime"),
            candidate_type=candidate_type,
            effect=rule.get("effect"),
        ),
    )


def _parse_compiled_rule_pack(payload: object) -> CompiledRulePack:
    if not isinstance(payload, Mapping) or set(payload) != _PACK_FIELDS:
        raise ValueError("compiled rule pack fields must exactly match schema v2")
    if payload.get("schema_version") != "compiled-knowledge-rules/v2":
        raise ValueError("unsupported compiled rule pack schema_version")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("compiled rule pack rules must be a non-empty list")
    rule_count = payload.get("rule_count")
    if isinstance(rule_count, bool) or not isinstance(rule_count, int) or rule_count != len(rules):
        raise ValueError("rule_count must be an integer equal to rules length")
    rules_hash = payload.get("rules_hash")
    if not isinstance(rules_hash, str) or not _SHA256_PATTERN.fullmatch(rules_hash):
        raise ValueError("rules_hash must be a lowercase sha256")
    if _content_hash(rules) != rules_hash:
        raise ValueError("rules_hash mismatch")
    if not all(isinstance(rule, Mapping) for rule in rules):
        raise ValueError("compiled rules must be objects")
    identities = [_validate_rule_structure(rule) for rule in rules]
    rule_ids = [rule_id for rule_id, _, _ in identities]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("duplicate rule_id")
    order = [(priority, rule_id) for rule_id, _, priority in identities]
    if order != sorted(order):
        raise ValueError("compiled rules must already be sorted by priority and rule_id")
    typed_rules = tuple(_typed_rule(rule) for rule in rules)
    if not any(rule.runtime.status == "active" for rule in typed_rules):
        raise ValueError("at least one active runtime rule is required")
    return _compiled_value(
        CompiledRulePack,
        schema_version="compiled-knowledge-rules/v2",
        rules=typed_rules,
        rule_count=rule_count,
        rules_hash=rules_hash,
    )


def parse_compiled_rule_pack(payload: object) -> CompiledRulePack:
    try:
        return _parse_compiled_rule_pack(payload)
    except ValueError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError) as exc:
        raise ValueError("invalid compiled rule pack") from exc


def _assert_validated_compiled_pack(pack: object) -> CompiledRulePack:
    if not isinstance(pack, CompiledRulePack) or not _is_validated_compiled_value(pack):
        raise ValueError("pack must be a validated CompiledRulePack")
    if (
        pack.schema_version != "compiled-knowledge-rules/v2"
        or not isinstance(pack.rules, tuple)
        or pack.rule_count != len(pack.rules)
        or not isinstance(pack.rules_hash, str)
        or not _SHA256_PATTERN.fullmatch(pack.rules_hash)
    ):
        raise ValueError("pack must be a validated CompiledRulePack")
    for rule in pack.rules:
        if not isinstance(rule, CompiledRule) or not _is_validated_compiled_value(rule):
            raise ValueError("pack must be a validated CompiledRulePack")
        if not _is_validated_compiled_value(rule.runtime):
            raise ValueError("pack must be a validated CompiledRulePack")
        if rule.runtime.parameters is not None and not _is_validated_compiled_value(
            rule.runtime.parameters
        ):
            raise ValueError("pack must be a validated CompiledRulePack")
    return pack


def _rule_result(
    context: RuleContext,
    decisions: tuple[RuleDecision, ...],
    before_hash: str,
) -> RuleResult:
    return RuleResult(
        context,
        decisions,
        before_hash,
        _content_hash(_context_payload(context)),
    )


def _apply_supported_current_priority(
    rule: CompiledRule,
    context: RuleContext,
) -> tuple[RuleContext, RuleDecision]:
    runtime = rule.runtime
    parameters = runtime.parameters
    if not isinstance(parameters, CompiledRuleParameters):
        raise RuntimeError("active priority opcode is missing parameters")
    emergency = next(
        (
            candidate
            for candidate in context.diagnosis_candidates
            if candidate.urgency in parameters.preserve_urgencies
        ),
        None,
    )
    if emergency is not None:
        return context, RuleDecision(
            rule.rule_id,
            cast(str, runtime.opcode),
            "matched_no_change",
            "emergency_priority_preserved",
        )
    backgrounds = tuple(
        candidate
        for candidate in context.diagnosis_candidates
        if candidate.role in parameters.background_roles
    )
    if any(candidate.complaint_relation in parameters.excluded_relations for candidate in backgrounds):
        return context, RuleDecision(
            rule.rule_id,
            cast(str, runtime.opcode),
            "excluded",
            "background_explains_current_problem",
        )
    targets = tuple(
        candidate
        for candidate in context.diagnosis_candidates
        if candidate.role in parameters.target_roles
        and candidate.support_level in parameters.target_support_levels
        and candidate.complaint_relation != "unknown"
        and candidate.evidence_codes
        and (candidate.is_official or parameters.fallback_policy != "official_catalog_only")
    )
    if not targets:
        return context, RuleDecision(
            rule.rule_id,
            cast(str, runtime.opcode),
            "not_matched",
            "no_supported_current_problem",
        )
    unrelated_background = tuple(
        candidate
        for candidate in backgrounds
        if candidate.complaint_relation in parameters.background_relations
    )
    if not unrelated_background:
        return context, RuleDecision(
            rule.rule_id,
            cast(str, runtime.opcode),
            "not_matched",
            "no_unrelated_background_condition",
        )
    target_ids = {id(candidate) for candidate in targets}
    ordered = targets + tuple(
        candidate
        for candidate in context.diagnosis_candidates
        if id(candidate) not in target_ids
    )
    preferred = targets[0].official_name
    if ordered == context.diagnosis_candidates and context.preferred_diagnosis == preferred:
        return context, RuleDecision(
            rule.rule_id,
            cast(str, runtime.opcode),
            "matched_no_change",
            "supported_current_problem_already_preferred",
        )
    output = RuleContext(
        diagnosis_candidates=ordered,
        preferred_diagnosis=preferred,
        diagnostic_axis_ids=context.diagnostic_axis_ids,
        diagnosis_axes=context.diagnosis_axes,
        exam_intent_ids=context.exam_intent_ids,
        treatment_codes=context.treatment_codes,
        fact_codes=context.fact_codes,
    )
    return output, RuleDecision(
        rule.rule_id,
        cast(str, runtime.opcode),
        "applied",
        "supported_current_problem_promoted",
    )


def _congenital_trigger_miss(
    parameters: CompiledDifferentialRuleParameters,
    fact_codes: frozenset[str],
) -> str | None:
    required_groups = (
        (parameters.age_fact_codes, "congenital_infection_age_not_matched"),
        (parameters.exposure_fact_codes, "congenital_infection_exposure_not_matched"),
        (
            parameters.manifestation_fact_codes,
            "congenital_infection_manifestation_not_matched",
        ),
    )
    return next(
        (reason for codes, reason in required_groups if fact_codes.isdisjoint(codes)),
        None,
    )


def _ordered_congenital_axes(
    parameters: CompiledDifferentialRuleParameters,
    fact_codes: frozenset[str],
) -> tuple[str, ...]:
    rubella_confirmed = not fact_codes.isdisjoint(parameters.rubella_confirmed_fact_codes)
    cmv_confirmed = not fact_codes.isdisjoint(parameters.cmv_confirmed_fact_codes)
    if rubella_confirmed != cmv_confirmed:
        return (
            (parameters.rubella_axis_id,)
            if rubella_confirmed
            else (parameters.cmv_axis_id,)
        )
    rubella_score = sum(code in fact_codes for code in parameters.rubella_rank_fact_codes)
    cmv_score = sum(code in fact_codes for code in parameters.cmv_rank_fact_codes)
    if cmv_score > rubella_score:
        return parameters.cmv_axis_id, parameters.rubella_axis_id
    return parameters.rubella_axis_id, parameters.cmv_axis_id


def _apply_congenital_infection_differential(
    rule: CompiledRule,
    context: RuleContext,
) -> tuple[RuleContext, RuleDecision]:
    runtime = rule.runtime
    parameters = runtime.parameters
    if not isinstance(parameters, CompiledDifferentialRuleParameters):
        raise RuntimeError("active differential opcode is missing parameters")
    opcode = cast(str, runtime.opcode)
    fact_codes = frozenset(context.fact_codes)
    miss_reason = _congenital_trigger_miss(parameters, fact_codes)
    if miss_reason is not None:
        return context, RuleDecision(
            rule.rule_id,
            opcode,
            "not_matched",
            miss_reason,
        )
    ordered_axes = _ordered_congenital_axes(parameters, fact_codes)
    owned_axes = {parameters.rubella_axis_id, parameters.cmv_axis_id}
    other_axes = tuple(
        axis_id for axis_id in context.diagnostic_axis_ids if axis_id not in owned_axes
    )
    diagnostic_axis_ids = ordered_axes + other_axes
    if diagnostic_axis_ids == context.diagnostic_axis_ids:
        return context, RuleDecision(
            rule.rule_id,
            opcode,
            "matched_no_change",
            "congenital_infection_axes_already_ranked",
        )
    return replace(
        context,
        diagnostic_axis_ids=diagnostic_axis_ids,
    ), RuleDecision(
        rule.rule_id,
        opcode,
        "applied",
        "congenital_infection_axes_expanded",
    )


def apply_rules(pack: CompiledRulePack, stage: RuleStage, context: RuleContext) -> RuleResult:
    if not isinstance(stage, str) or stage not in _RULE_STAGES:
        raise ValueError("stage has an unsupported value")
    pack = _assert_validated_compiled_pack(pack)
    if not isinstance(context, RuleContext):
        raise ValueError("context must be RuleContext")
    context_hash = _content_hash(_context_payload(context))
    output_context = context
    decisions: list[RuleDecision] = []
    for rule in pack.rules:
        runtime = rule.runtime
        if runtime.status != "active" or runtime.stage != stage:
            continue
        if runtime.opcode == "promote_supported_current_over_background":
            output_context, decision = _apply_supported_current_priority(rule, output_context)
            decisions.append(decision)
            continue
        if runtime.opcode == "expand_congenital_infection_axes":
            output_context, decision = _apply_congenital_infection_differential(
                rule,
                output_context,
            )
            decisions.append(decision)
            continue
        if runtime.opcode == FACT_GROUP_OPCODE:
            output_context, decision = _apply_fact_group_match(rule, output_context)
            decisions.append(decision)
            continue
        if runtime.opcode == DIAGNOSIS_AXIS_OPCODE:
            output_context, decision = _apply_diagnosis_axis_emit(rule, output_context)
            decisions.append(decision)
            continue
        if runtime.opcode == TREATMENT_TRANSFORM_OPCODE:
            output_context, decision = _apply_treatment_transform(rule, output_context)
            decisions.append(decision)
            continue
        raise RuntimeError(f"compiled rule opcode {runtime.opcode} is not implemented")
    return _rule_result(output_context, tuple(decisions), context_hash)


def _merge_exam_intent_ids(
    existing: tuple[str, ...],
    additions: tuple[str, ...],
    *,
    deduplicate: str | None,
) -> tuple[str, ...]:
    merged = list(existing)
    seen = set(existing) if deduplicate == "intent_id" else set()
    for intent_id in additions:
        if deduplicate == "intent_id" and intent_id in seen:
            continue
        merged.append(intent_id)
        seen.add(intent_id)
    return tuple(merged)


def _apply_fact_group_match(
    rule: CompiledRule,
    context: RuleContext,
) -> tuple[RuleContext, RuleDecision]:
    """Pure set logic over registered fact codes; no text, regex or expression."""
    parameters = rule.runtime.parameters
    if not isinstance(parameters, CompiledFactGroupParameters):
        raise RuntimeError("fact group rule is missing typed parameters")
    present = frozenset(context.fact_codes)

    def fires(group: tuple[str, ...]) -> bool:
        return bool(present.intersection(group))

    if any(fires(group) for group in parameters.excluded_groups):
        return context, RuleDecision(
            rule.rule_id,
            FACT_GROUP_OPCODE,
            "excluded",
            "excluded_group_present",
        )
    if parameters.all_groups and not all(fires(group) for group in parameters.all_groups):
        return context, RuleDecision(
            rule.rule_id,
            FACT_GROUP_OPCODE,
            "not_matched",
            "all_groups_incomplete",
        )
    if parameters.any_groups and not any(fires(group) for group in parameters.any_groups):
        return context, RuleDecision(
            rule.rule_id,
            FACT_GROUP_OPCODE,
            "not_matched",
            "any_groups_missed",
        )
    intent_ids = _merge_exam_intent_ids(
        context.exam_intent_ids,
        parameters.add_exam_intent_ids,
        deduplicate=parameters.deduplicate,
    )
    if parameters.matched_fact_code in present:
        if intent_ids == context.exam_intent_ids:
            return context, RuleDecision(
                rule.rule_id,
                FACT_GROUP_OPCODE,
                "matched_no_change",
                "fact_code_already_present",
            )
        updated = dataclasses.replace(context, exam_intent_ids=intent_ids)
        return updated, RuleDecision(
            rule.rule_id,
            FACT_GROUP_OPCODE,
            "applied",
            "exam_intent_ids_appended",
        )
    updated = dataclasses.replace(
        context,
        fact_codes=tuple([*context.fact_codes, parameters.matched_fact_code]),
        exam_intent_ids=intent_ids,
    )
    return updated, RuleDecision(
        rule.rule_id,
        FACT_GROUP_OPCODE,
        "applied",
        "fact_code_appended",
    )


def _apply_treatment_transform(
    rule: CompiledRule,
    context: RuleContext,
) -> tuple[RuleContext, RuleDecision]:
    """Closed treatment-code transform: remove, replace, append by code only.

    Replacements reference registered patch templates, never candidate free
    text, so a rule can never author a new prescription. The transform is
    idempotent: codes already removed stay removed, already replaced stay
    replaced, already appended are not duplicated.
    """
    parameters = rule.runtime.parameters
    if not isinstance(parameters, CompiledTreatmentTransformParameters):
        raise RuntimeError("treatment transform rule is missing typed parameters")
    opcode = cast(str, rule.runtime.opcode)
    codes = list(context.treatment_codes)
    remove_set = set(parameters.remove_codes)
    replace_map = {code: replacement for code, replacement in parameters.replace_codes}
    append_set = set(parameters.append_codes)

    new_codes: list[str] = []
    changed = False
    for code in codes:
        if code in remove_set:
            changed = True
            continue
        new_codes.append(replace_map.get(code, code))
        if code in replace_map:
            changed = True
    # Only append when an actual removal/replacement occurred; this keeps the
    # transform a no-op when the target code is absent.
    if changed:
        for code in parameters.append_codes:
            if code not in new_codes:
                new_codes.append(code)

    if tuple(new_codes) == context.treatment_codes:
        return context, RuleDecision(
            rule.rule_id,
            opcode,
            "matched_no_change",
            "treatment_codes_already_transformed",
        )
    updated = dataclasses.replace(context, treatment_codes=tuple(new_codes))
    return updated, RuleDecision(
        rule.rule_id,
        opcode,
        "applied",
        "treatment_codes_transformed",
    )


@dataclass(frozen=True)
class ShadowRuleReceipt:
    """Auditable legacy-vs-typed comparison for one rule."""

    rule_id: str
    legacy_result_hash: str
    typed_result_hash: str
    equivalent: bool


def shadow_rule_receipt(
    *,
    rule_id: str,
    legacy_result: object,
    typed_result: object,
) -> ShadowRuleReceipt:
    legacy_hash = _content_hash(legacy_result)
    typed_hash = _content_hash(typed_result)
    return ShadowRuleReceipt(
        rule_id=str(rule_id),
        legacy_result_hash=legacy_hash,
        typed_result_hash=typed_hash,
        equivalent=legacy_hash == typed_hash,
    )
