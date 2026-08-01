from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence

from offline.artifacts import file_hash


KNOWLEDGE_CANDIDATE_TYPES = frozenset(
    {
        "diagnosis_differential_rule",
        "clinical_closure_rule",
        "diagnosis_priority_rule",
        "treatment_gate_rule",
        "treatment_sequence_rule",
    }
)
KNOWLEDGE_EFFECT_FIELDS = frozenset(
    {
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
)
TREATMENT_CANDIDATE_TYPES = frozenset({"treatment_gate_rule", "treatment_sequence_rule"})
_CANDIDATE_PHASES = {
    "diagnosis_differential_rule": "diagnosis",
    "clinical_closure_rule": "closure",
    "diagnosis_priority_rule": "diagnosis",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}
_RUNTIME_STAGE_BY_CANDIDATE_TYPE = {
    "diagnosis_differential_rule": "diagnosis_candidates",
    "clinical_closure_rule": "clinical_closure",
    "diagnosis_priority_rule": "diagnosis_candidates",
    "treatment_gate_rule": "treatment",
    "treatment_sequence_rule": "treatment",
}
_RUNTIME_STAGES = frozenset({"diagnosis_candidates", "clinical_closure", "treatment"})
_ACTIVE_RUNTIME_OPCODE = "promote_supported_current_over_background"
_DIFFERENTIAL_RUNTIME_OPCODE = "expand_congenital_infection_axes"
_ACTIVE_RUNTIME_PARAMETER_ENUMS = {
    "target_roles": frozenset({"current_problem"}),
    "target_support_levels": frozenset({"objective"}),
    "background_roles": frozenset({"background_condition"}),
    "background_relations": frozenset({"unrelated"}),
    "excluded_relations": frozenset({"explains"}),
    "preserve_urgencies": frozenset({"emergency"}),
}
_ACTIVE_RUNTIME_PARAMETER_FIELDS = frozenset(
    {*_ACTIVE_RUNTIME_PARAMETER_ENUMS, "fallback_policy"}
)
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

_RULE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTROL_KINDS = frozenset({"positive", "near_neighbor", "reasonable_exception"})
_EFFECT_STRING_MAX_LENGTH = 64
_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,63}")
_INTENT_ID_PATTERN = re.compile(r"exam_intent_[a-z0-9_]{2,63}")
_ANSWER_SECTION_PATTERNS = (
    ("诊断", "检查", "治疗"),
    ("diagnosis", "examination", "treatment"),
    ("diagnosis", "exam", "treatment"),
)
_EFFECT_SCHEMAS: Dict[str, Dict[str, tuple[str, Any]]] = {
    "diagnosis_differential_rule": {
        "add_diagnostic_axes": ("labels", None),
        "ranking_policy": ("enum", frozenset({"evidence_ordered", "preserve_dual_axis"})),
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
        "gate_policy": ("enum", frozenset({"require_infection_evidence", "require_toxicity_evidence"})),
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
    "clinical_closure_rule": frozenset({"add_exam_intent_ids", "deduplicate"}),
    "diagnosis_priority_rule": frozenset({"priority_policy", "fallback_policy"}),
    "treatment_gate_rule": frozenset(
        {"remove_treatment_codes", "preserve_treatment_codes", "gate_policy"}
    ),
    "treatment_sequence_rule": frozenset(
        {"ordered_patch_codes", "sequence_policy", "skip_patch_codes"}
    ),
}


def is_unknown_clinical_candidate_type(candidate_type: Any) -> bool:
    return bool(
        isinstance(candidate_type, str)
        and candidate_type not in KNOWLEDGE_CANDIDATE_TYPES
        and candidate_type.startswith(("diagnosis_", "clinical_", "treatment_"))
    )


def _nonempty_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty string list" % field)
    if not all(isinstance(item, str) and item.strip() == item and item for item in value):
        raise ValueError("%s must be a non-empty string list" % field)
    return list(value)


def _validate_control(control: Any, *, field: str) -> Dict[str, Any]:
    if not isinstance(control, Mapping):
        raise ValueError("%s entries must be objects" % field)
    required = {"control_id", "kind", "facts", "assertions"}
    if set(control) != required:
        raise ValueError("%s control fields must exactly match the typed schema" % field)
    control_id = control.get("control_id")
    kind = control.get("kind")
    if not isinstance(control_id, str) or not _RULE_ID_PATTERN.fullmatch(control_id.replace("-", "_")):
        raise ValueError("invalid control_id")
    if not isinstance(kind, str) or kind not in _CONTROL_KINDS:
        raise ValueError("invalid control kind")
    _nonempty_strings(control.get("facts"), field=field + ".facts")
    _nonempty_strings(control.get("assertions"), field=field + ".assertions")
    return deepcopy(dict(control))


def _validate_controls(value: Any, *, field: str, allowed_kinds: set[str]) -> list[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be non-empty" % field)
    controls = [_validate_control(item, field=field) for item in value]
    if any(item["kind"] not in allowed_kinds for item in controls):
        raise ValueError("invalid %s kind" % field)
    control_ids = [item["control_id"] for item in controls]
    if len(set(control_ids)) != len(control_ids):
        raise ValueError("duplicate control_id")
    return controls


def _validate_runtime_enum_list(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str],
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("runtime parameter must be a non-empty enum list: %s" % field)
    if not all(isinstance(item, str) and item in allowed for item in value):
        raise ValueError("runtime parameter has unsupported enum: %s" % field)
    if len(set(value)) != len(value):
        raise ValueError("runtime parameter contains duplicate values: %s" % field)
    return list(value)


def _validate_runtime_code(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _CODE_PATTERN.fullmatch(value):
        raise ValueError("runtime parameter must be a structured code: %s" % field)
    return value


def _validate_runtime_code_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("runtime parameter must be a non-empty code list: %s" % field)
    codes = [_validate_runtime_code(item, field=field) for item in value]
    if len(set(codes)) != len(codes):
        raise ValueError("runtime parameter contains duplicate codes: %s" % field)
    return codes


def _validate_priority_runtime_parameters(parameters: Any) -> None:
    if not isinstance(parameters, Mapping) or set(parameters) != _ACTIVE_RUNTIME_PARAMETER_FIELDS:
        raise ValueError("runtime parameter fields must exactly match the opcode schema")
    for field, allowed in _ACTIVE_RUNTIME_PARAMETER_ENUMS.items():
        _validate_runtime_enum_list(parameters.get(field), field=field, allowed=allowed)
    if parameters.get("fallback_policy") != "official_catalog_only":
        raise ValueError("runtime parameter has unsupported enum: fallback_policy")


def _validate_differential_code_lists(code_lists: Mapping[str, list[str]]) -> None:
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
            raise ValueError("runtime differential parameter is unsupported: %s" % field)
    if not _TYPICAL_CONGENITAL_MANIFESTATIONS.issubset(manifestation):
        raise ValueError("runtime differential typical manifestations are required")
    if not rubella_confirmed.issubset(code_lists["rubella_rank_fact_codes"]):
        raise ValueError("runtime differential rubella confirmed facts must be rank facts")
    if not cmv_confirmed.issubset(code_lists["cmv_rank_fact_codes"]):
        raise ValueError("runtime differential cmv confirmed facts must be rank facts")
    for field, codes in code_lists.items():
        if set(codes) != _DIFFERENTIAL_CODE_DOMAINS[field]:
            raise ValueError(
                "runtime differential parameter must match canonical code set: %s"
                % field
            )


def _validate_active_differential_effect(value: Any) -> None:
    expected = {
        "add_diagnostic_axes": ["先天性风疹方向", "巨细胞病毒方向"],
        "ranking_policy": "evidence_ordered",
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("active differential effect does not match its runtime")


def _validate_differential_runtime_parameters(parameters: Any) -> None:
    if not isinstance(parameters, Mapping) or set(parameters) != _DIFFERENTIAL_PARAMETER_FIELDS:
        raise ValueError("runtime parameter fields must exactly match the opcode schema")
    code_lists = {
        field: _validate_runtime_code_list(parameters.get(field), field=field)
        for field in _DIFFERENTIAL_CODE_LIST_FIELDS
    }
    _validate_differential_code_lists(code_lists)
    rubella_axis_id = _validate_runtime_code(
        parameters.get("rubella_axis_id"),
        field="rubella_axis_id",
    )
    cmv_axis_id = _validate_runtime_code(
        parameters.get("cmv_axis_id"),
        field="cmv_axis_id",
    )
    if rubella_axis_id != "congenital_rubella":
        raise ValueError("runtime differential parameter rubella_axis_id is unsupported")
    if cmv_axis_id != "congenital_cmv":
        raise ValueError("runtime differential parameter cmv_axis_id is unsupported")


def _validate_runtime(
    candidate_type: str,
    runtime: Any,
    *,
    effect: Any,
) -> Dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be an object")
    status = runtime.get("status")
    expected_stage = _RUNTIME_STAGE_BY_CANDIDATE_TYPE[candidate_type]
    stage = runtime.get("stage")
    if not isinstance(stage, str) or stage not in _RUNTIME_STAGES or stage != expected_stage:
        raise ValueError("runtime stage does not match candidate type")

    if status == "audit_only":
        if set(runtime) != {"status", "stage"}:
            raise ValueError("audit-only runtime fields must be status and stage")
        return deepcopy(dict(runtime))

    if status != "active":
        raise ValueError("unsupported runtime status")
    if set(runtime) != {"status", "stage", "opcode", "parameters"}:
        raise ValueError("active runtime fields must exactly match the typed schema")
    opcode = runtime.get("opcode")
    if candidate_type == "diagnosis_priority_rule":
        if opcode != _ACTIVE_RUNTIME_OPCODE:
            raise ValueError("unsupported runtime opcode")
        _validate_priority_runtime_parameters(runtime.get("parameters"))
    elif candidate_type == "diagnosis_differential_rule":
        if opcode != _DIFFERENTIAL_RUNTIME_OPCODE:
            raise ValueError("unsupported runtime opcode")
        _validate_active_differential_effect(effect)
        _validate_differential_runtime_parameters(runtime.get("parameters"))
    else:
        raise ValueError("active runtime is unsupported for candidate type")
    return deepcopy(dict(runtime))


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _answer_like_prose(value: str) -> bool:
    normalized = _normalized_text(value)
    return any(all(marker in normalized for marker in group) for group in _ANSWER_SECTION_PATTERNS)


def _effect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend(_effect_strings(key))
            strings.extend(_effect_strings(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_effect_strings(item))
        return strings
    return []


def _validate_no_answer_like_effect(value: Any) -> None:
    combined = " ".join(_effect_strings(value))
    if combined and _answer_like_prose(combined):
        raise ValueError("answer-like diagnosis/examination/treatment prose is prohibited: effect")


def _validate_effect_string(value: Any, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("effect string must be non-empty and trimmed: %s" % field)
    if len(value) > _EFFECT_STRING_MAX_LENGTH:
        raise ValueError("effect string exceeds maximum length: %s" % field)
    if _answer_like_prose(value):
        raise ValueError("answer-like diagnosis/examination/treatment prose is prohibited: %s" % field)
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError("effect string has invalid structured code: %s" % field)
    return value


def _validate_effect_list(
    value: Any,
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("effect field must be a non-empty list: %s" % field)
    items = [
        _validate_effect_string(item, field="%s[%d]" % (field, index), pattern=pattern)
        for index, item in enumerate(value)
    ]
    if len(set(items)) != len(items):
        raise ValueError("effect field contains duplicate values: %s" % field)
    return items


def _official_intent_ids(project_root: Path) -> frozenset[str]:
    path = Path(project_root) / "agent" / "knowledge" / "exam_intent_map.json"
    if not path.is_file():
        return frozenset()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data.get("rules") if isinstance(data, Mapping) else None
    if not isinstance(rules, list):
        raise ValueError("official exam intent registry is invalid")
    return frozenset(
        rule.get("id")
        for rule in rules
        if isinstance(rule, Mapping)
        and rule.get("status") == "verified"
        and isinstance(rule.get("id"), str)
    )


def _validate_typed_effect(
    candidate_type: str,
    proposed_action: Any,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    if not isinstance(proposed_action, Mapping) or not proposed_action:
        raise ValueError("effect must be a non-empty object")
    schema = _EFFECT_SCHEMAS[candidate_type]
    fields = set(proposed_action)
    required = _EFFECT_REQUIRED_FIELDS[candidate_type]
    if not required.issubset(fields) or not fields.issubset(schema):
        raise ValueError("%s effect fields do not match the typed whitelist" % candidate_type)
    _validate_no_answer_like_effect(proposed_action)

    validated: Dict[str, Any] = {}
    for field, value in proposed_action.items():
        kind, constraint = schema[field]
        if kind == "labels":
            validated[field] = _validate_effect_list(value, field=field)
        elif kind == "label":
            validated[field] = _validate_effect_string(value, field=field)
        elif kind == "codes":
            validated[field] = _validate_effect_list(value, field=field, pattern=_CODE_PATTERN)
        elif kind == "intent_ids":
            intent_ids = _validate_effect_list(value, field=field, pattern=_INTENT_ID_PATTERN)
            official_ids = _official_intent_ids(project_root)
            if not official_ids or any(intent_id not in official_ids for intent_id in intent_ids):
                raise ValueError("effect exam intent ids must be verified official intents")
            validated[field] = intent_ids
        elif kind == "enum":
            enum_value = _validate_effect_string(value, field=field, pattern=_CODE_PATTERN)
            if enum_value not in constraint:
                raise ValueError("effect field has unsupported deterministic primitive: %s" % field)
            validated[field] = enum_value
        else:
            raise ValueError("unsupported typed effect field: %s" % field)
    return validated


def _validate_ref(value: Any, *, project_root: Path, field: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("%s ref fields must be path and sha256" % field)
    relative_path = value.get("path")
    expected_hash = value.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or re.match(r"^[A-Za-z]:", relative_path)
    ):
        raise ValueError("%s ref path must be project-relative POSIX" % field)
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts or "." in posix_path.parts:
        raise ValueError("%s ref path must be project-relative POSIX" % field)
    if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
        raise ValueError("%s ref sha256 is invalid" % field)

    project_root = Path(project_root).resolve()
    path = (project_root / Path(*posix_path.parts)).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("%s ref escapes project root" % field) from exc
    if not path.is_file():
        raise ValueError("%s ref file does not exist" % field)
    if file_hash(path) != expected_hash:
        raise ValueError("%s ref file hash mismatch" % field)
    return {"path": relative_path, "sha256": expected_hash}


def _validate_refs(value: Any, *, project_root: Path, field: str) -> list[Dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s refs must be non-empty" % field)
    refs = [_validate_ref(item, project_root=project_root, field=field) for item in value]
    paths = [item["path"] for item in refs]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate %s ref path" % field)
    return refs


def validate_knowledge_effect(
    candidate_type: str,
    effect: Mapping[str, Any],
    *,
    project_root: Path,
) -> Dict[str, Any]:
    if not isinstance(candidate_type, str) or candidate_type not in KNOWLEDGE_CANDIDATE_TYPES:
        raise ValueError("unknown clinical candidate type: %s" % candidate_type)
    if not isinstance(effect, Mapping) or set(effect) != KNOWLEDGE_EFFECT_FIELDS:
        raise ValueError("knowledge effect fields must match clinical-knowledge-candidate/v2")
    if effect.get("schema_version") != "clinical-knowledge-candidate/v2":
        raise ValueError("unsupported knowledge effect schema_version")
    rule_id = effect.get("rule_id")
    if not isinstance(rule_id, str) or not _RULE_ID_PATTERN.fullmatch(rule_id):
        raise ValueError("invalid knowledge rule_id")

    _nonempty_strings(effect.get("triggers"), field="triggers")
    _nonempty_strings(effect.get("required_evidence"), field="required_evidence")
    _nonempty_strings(effect.get("exclusions"), field="exclusions")
    proposed_action = effect.get("effect")
    try:
        _validate_typed_effect(candidate_type, proposed_action, project_root=project_root)
    except ValueError as exc:
        if isinstance(proposed_action, Mapping) and any(
            isinstance(value, bool) for value in proposed_action.values()
        ):
            raise ValueError("effect values must be typed strings or string lists") from exc
        raise
    positive_controls = _validate_controls(
        effect.get("positive_controls"),
        field="positive_controls",
        allowed_kinds={"positive"},
    )
    negative_controls = _validate_controls(
        effect.get("negative_controls"),
        field="negative_controls",
        allowed_kinds={"near_neighbor", "reasonable_exception"},
    )
    control_ids = [
        item["control_id"] for item in positive_controls + negative_controls
    ]
    if len(set(control_ids)) != len(control_ids):
        raise ValueError("duplicate control_id")
    if candidate_type in TREATMENT_CANDIDATE_TYPES and not any(
        item["kind"] == "near_neighbor" for item in negative_controls
    ):
        raise ValueError("treatment rule requires a near-neighbor negative control")
    if candidate_type in TREATMENT_CANDIDATE_TYPES and not any(
        item["kind"] == "reasonable_exception" for item in negative_controls
    ):
        raise ValueError("treatment rule requires a reasonable exception control")
    _validate_refs(effect.get("source_refs"), project_root=project_root, field="source_refs")
    _validate_refs(effect.get("test_refs"), project_root=project_root, field="test_refs")

    priority = effect.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 1000:
        raise ValueError("priority must be an integer from 0 to 1000")
    scope = effect.get("scope")
    if not isinstance(scope, Mapping) or set(scope) != {"phase", "application"}:
        raise ValueError("scope fields must be phase and application")
    if scope.get("phase") != _CANDIDATE_PHASES[candidate_type]:
        raise ValueError("candidate type scope phase mismatch")
    if scope.get("application") != "trigger_bound":
        raise ValueError("scope application must be trigger_bound")
    _nonempty_strings(effect.get("review_requirements"), field="review_requirements")
    _validate_runtime(
        candidate_type,
        effect.get("runtime"),
        effect=proposed_action,
    )
    return deepcopy(dict(effect))


def knowledge_rule_candidate(
    *,
    candidate_type: str,
    effect: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    from offline.candidates import create_candidate, leakage_reason

    if leakage_reason(effect, {"source": "offline_knowledge_design"}):
        return create_candidate(
            candidate_id=str(effect.get("rule_id") or "quarantined_knowledge_candidate"),
            candidate_type=candidate_type,
            proposed_effect=effect,
            evidence={"source": "offline_knowledge_design"},
            project_root=project_root,
        )

    validated = validate_knowledge_effect(candidate_type, effect, project_root=project_root)
    return create_candidate(
        candidate_id=validated["rule_id"],
        candidate_type=candidate_type,
        proposed_effect=validated,
        evidence={"source": "offline_knowledge_design"},
        project_root=project_root,
    )
