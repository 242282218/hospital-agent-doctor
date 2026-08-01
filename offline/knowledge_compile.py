from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Dict, Mapping, Sequence

from agent.knowledge import typed_rule_engine
from offline.artifacts import content_hash, file_hash
from offline.candidates import leakage_reason
from offline.knowledge_rules import KNOWLEDGE_CANDIDATE_TYPES, validate_knowledge_effect


_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_type",
        "proposed_effect",
        "evidence",
        "status",
        "candidate_hash",
        "effect_hash",
    }
)
_COMPILED_EFFECT_FIELDS = (
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
)
_CONTROL_SET_FIELDS = frozenset(
    {
        "schema_version",
        "compiled_rules_hash",
        "catalog_hashes",
        "control_count",
        "controls",
        "control_set_hash",
    }
)
_CONTROL_FIELDS = frozenset(
    {"rule_id", "control_id", "kind", "stage", "context", "expected_outcome"}
)
_CONTEXT_FIELDS = frozenset(
    {
        "diagnosis_candidates",
        "preferred_diagnosis",
        "diagnostic_axis_ids",
        "exam_intent_ids",
        "treatment_codes",
        "fact_codes",
    }
)
_DIAGNOSIS_CANDIDATE_FIELDS = frozenset(
    {
        "official_name",
        "role",
        "support_level",
        "complaint_relation",
        "urgency",
        "evidence_codes",
    }
)
_EXPECTED_OUTCOME_FIELDS = frozenset({"outcome", "reason_code", "output_context"})
_CONTROL_KINDS = frozenset({"positive", "near_neighbor", "reasonable_exception"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_controls(
    controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    canonical = []
    seen_pairs: set[tuple[str, str]] = set()
    for control in controls:
        if not isinstance(control, Mapping):
            raise ValueError("structured controls must be objects")
        item = deepcopy(dict(control))
        if not isinstance(item.get("rule_id"), str) or not isinstance(
            item.get("control_id"), str
        ):
            raise ValueError("control rule_id and control_id must be strings")
        pair = (item["rule_id"], item["control_id"])
        if pair in seen_pairs:
            raise ValueError("duplicate control rule_id/control_id pair")
        seen_pairs.add(pair)
        canonical.append(item)
    return sorted(
        canonical,
        key=lambda control: (control["rule_id"], control["control_id"]),
    )


def knowledge_control_set_hash(
    *,
    schema_version: str,
    compiled_rules_hash: str,
    catalog_hashes: Mapping[str, str],
    control_count: int,
    controls: Sequence[Mapping[str, Any]],
) -> str:
    canonical_controls = _canonical_controls(controls)
    core = {
        "schema_version": schema_version,
        "compiled_rules_hash": compiled_rules_hash,
        "catalog_hashes": deepcopy(dict(catalog_hashes)),
        "control_count": control_count,
        "controls": canonical_controls,
    }
    try:
        return content_hash(core)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "control set canonical core is not JSON serializable"
        ) from exc


def _validate_candidate(
    candidate: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    data = deepcopy(dict(candidate))
    if data.get("schema_version") != "candidate/v1":
        raise ValueError("candidate schema_version must be candidate/v1")
    if data.get("status") != "candidate":
        raise ValueError("candidate status must be candidate")
    if set(data) != _CANDIDATE_FIELDS:
        raise ValueError("candidate fields must exactly match candidate/v1")

    candidate_type = data.get("candidate_type")
    if not isinstance(candidate_type, str) or candidate_type not in KNOWLEDGE_CANDIDATE_TYPES:
        raise ValueError("unknown candidate_type: %s" % candidate_type)
    effect = data.get("proposed_effect")
    evidence = data.get("evidence")
    if not isinstance(effect, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("candidate effect and evidence must be objects")

    if content_hash(effect) != data.get("effect_hash"):
        raise ValueError("effect_hash mismatch")
    hash_body = {
        key: value
        for key, value in data.items()
        if key not in {"candidate_hash", "effect_hash"}
    }
    if content_hash(hash_body) != data.get("candidate_hash"):
        raise ValueError("candidate_hash mismatch")
    if leakage_reason(effect, evidence):
        raise ValueError("candidate contains leakage")

    validated_effect = validate_knowledge_effect(
        candidate_type,
        effect,
        project_root=project_root,
    )
    if data.get("candidate_id") != validated_effect["rule_id"]:
        raise ValueError("candidate_id must equal rule_id")
    return data, validated_effect


def _compile_rule(
    candidate: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> Dict[str, Any]:
    rule = {
        "rule_id": effect["rule_id"],
        "candidate_type": candidate["candidate_type"],
        "candidate_hash": candidate["candidate_hash"],
        "effect_hash": candidate["effect_hash"],
    }
    rule.update({field: deepcopy(effect[field]) for field in _COMPILED_EFFECT_FIELDS})
    return rule


def compile_knowledge_rules(
    candidates: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    root = Path(project_root).resolve()
    rules = []
    rule_ids = set()
    for candidate in candidates:
        validated_candidate, validated_effect = _validate_candidate(
            candidate,
            project_root=root,
        )
        rule_id = validated_effect["rule_id"]
        if rule_id in rule_ids:
            raise ValueError("duplicate rule_id: %s" % rule_id)
        rule_ids.add(rule_id)
        rules.append(_compile_rule(validated_candidate, validated_effect))

    if not any(rule["runtime"]["status"] == "active" for rule in rules):
        raise ValueError("at least one active runtime rule is required")
    rules.sort(key=lambda rule: (rule["priority"], rule["rule_id"]))
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": rules,
        "rule_count": len(rules),
        "rules_hash": content_hash(rules),
    }


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a string list" % field)
    if not all(isinstance(item, str) and item and item.strip() == item for item in value):
        raise ValueError("%s must be a trimmed string list" % field)
    if len(set(value)) != len(value):
        raise ValueError("%s contains duplicate values" % field)
    return tuple(value)


def _catalog_names(project_root: Path, catalog_path: Path) -> tuple[frozenset[str], str]:
    expected = (project_root / "data" / "ref_data" / "diseases_catalog.json").resolve()
    actual = Path(catalog_path).resolve()
    if actual != expected:
        raise ValueError("disease catalog path must be data/ref_data/diseases_catalog.json")
    if not actual.is_file():
        raise ValueError("disease catalog file does not exist")
    payload = json.loads(actual.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"diseases"}:
        raise ValueError("disease catalog fields must exactly match the schema")
    diseases = payload.get("diseases")
    if not isinstance(diseases, Mapping) or not diseases:
        raise ValueError("disease catalog diseases must be a non-empty object")
    names: set[str] = set()
    for department, entries in diseases.items():
        if not isinstance(department, str) or not department.strip():
            raise ValueError("disease catalog department is invalid")
        names.update(_string_list(entries, field="disease catalog entries"))
    return frozenset(names), file_hash(actual)


def _validated_ref(ref: object, *, field: str, project_root: Path) -> dict[str, str]:
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError("%s ref fields must be path and sha256" % field)
    relative = ref.get("path")
    expected_hash = ref.get("sha256")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("%s ref path must be project-relative POSIX" % field)
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("%s ref path must be project-relative POSIX" % field)
    if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
        raise ValueError("%s ref sha256 is invalid" % field)
    path = (project_root / Path(*posix_path.parts)).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("%s ref escapes project root" % field) from exc
    if not path.is_file() or file_hash(path) != expected_hash:
        raise ValueError("%s ref file hash mismatch" % field)
    return {"path": relative, "sha256": expected_hash}


def _validated_refs(rule: Mapping[str, Any], *, project_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for field in ("source_refs", "test_refs"):
        raw_refs = rule.get(field)
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("%s refs must be non-empty" % field)
        refs = [_validated_ref(ref, field=field, project_root=project_root) for ref in raw_refs]
        if len({ref["path"] for ref in refs}) != len(refs):
            raise ValueError("duplicate %s ref path" % field)
        result[field] = sorted(refs, key=lambda ref: ref["path"])
    return result


def _typed_candidate(
    value: object,
    *,
    official_names: frozenset[str],
) -> typed_rule_engine.RuleDiagnosisCandidate:
    if not isinstance(value, Mapping) or set(value) != _DIAGNOSIS_CANDIDATE_FIELDS:
        raise ValueError("diagnosis candidate fields must exactly match the control schema")
    name = value.get("official_name")
    if not isinstance(name, str):
        raise ValueError("diagnosis candidate official_name must be a string")
    return typed_rule_engine.RuleDiagnosisCandidate(
        official_name=name,
        role=value.get("role"),
        support_level=value.get("support_level"),
        complaint_relation=value.get("complaint_relation"),
        urgency=value.get("urgency"),
        evidence_codes=_string_list(value.get("evidence_codes"), field="evidence_codes"),
        is_official=name in official_names,
    )


def _typed_context(
    value: object,
    *,
    official_names: frozenset[str],
) -> typed_rule_engine.RuleContext:
    if not isinstance(value, Mapping) or set(value) != _CONTEXT_FIELDS:
        raise ValueError("rule context fields must exactly match the control schema")
    candidates = value.get("diagnosis_candidates")
    if not isinstance(candidates, list):
        raise ValueError("diagnosis_candidates must be a list")
    preferred = value.get("preferred_diagnosis")
    if preferred is not None and not isinstance(preferred, str):
        raise ValueError("preferred_diagnosis must be a string or null")
    return typed_rule_engine.RuleContext(
        diagnosis_candidates=tuple(
            _typed_candidate(candidate, official_names=official_names)
            for candidate in candidates
        ),
        preferred_diagnosis=preferred,
        diagnostic_axis_ids=_string_list(value.get("diagnostic_axis_ids"), field="diagnostic_axis_ids"),
        exam_intent_ids=_string_list(value.get("exam_intent_ids"), field="exam_intent_ids"),
        treatment_codes=_string_list(value.get("treatment_codes"), field="treatment_codes"),
        fact_codes=_string_list(value.get("fact_codes"), field="fact_codes"),
    )


def _context_payload(context: typed_rule_engine.RuleContext) -> dict[str, Any]:
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
        "exam_intent_ids": list(context.exam_intent_ids),
        "treatment_codes": list(context.treatment_codes),
        "fact_codes": list(context.fact_codes),
    }


def _decision_payload(decision: typed_rule_engine.RuleDecision) -> dict[str, str]:
    return {
        "rule_id": decision.rule_id,
        "opcode": decision.opcode,
        "outcome": decision.outcome,
        "reason_code": decision.reason_code,
    }


def _validate_expected(value: object, *, official_names: frozenset[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_OUTCOME_FIELDS:
        raise ValueError("expected_outcome fields must exactly match the control schema")
    if value.get("outcome") not in {"not_matched", "excluded", "matched_no_change", "applied"}:
        raise ValueError("expected_outcome outcome is unsupported")
    reason = value.get("reason_code")
    if not isinstance(reason, str) or not reason or reason.strip() != reason:
        raise ValueError("expected_outcome reason_code is invalid")
    _typed_context(value.get("output_context"), official_names=official_names)


def _compiled_control_inventory(rule: Mapping[str, Any]) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for field in ("positive_controls", "negative_controls"):
        for control in rule[field]:
            control_id = control["control_id"]
            if control_id in inventory:
                raise ValueError("compiled rule has duplicate control_id")
            inventory[control_id] = control["kind"]
    return inventory


def _validate_control_set(
    control_set: object,
    *,
    raw_rules: Mapping[str, Mapping[str, Any]],
    active_rule_ids: Sequence[str],
    official_names: frozenset[str],
    rules_hash: str,
    catalog_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(control_set, Mapping) or set(control_set) != _CONTROL_SET_FIELDS:
        raise ValueError("control set fields must exactly match knowledge-rule-controls/v1")
    if control_set.get("schema_version") != "knowledge-rule-controls/v1":
        raise ValueError("unsupported control set schema_version")
    if control_set.get("compiled_rules_hash") != rules_hash:
        raise ValueError("control set compiled_rules_hash mismatch")
    bound_catalog_hashes = control_set.get("catalog_hashes")
    if (
        not isinstance(bound_catalog_hashes, Mapping)
        or set(bound_catalog_hashes) != {"data/ref_data/diseases_catalog.json"}
    ):
        raise ValueError("control set catalog_hashes must bind only the disease catalog")
    bound_catalog_hash = bound_catalog_hashes.get(
        "data/ref_data/diseases_catalog.json"
    )
    if (
        not isinstance(bound_catalog_hash, str)
        or not _SHA256_PATTERN.fullmatch(bound_catalog_hash)
    ):
        raise ValueError("control set disease catalog hash must be a lowercase sha256")
    if dict(bound_catalog_hashes) != dict(catalog_hashes):
        raise ValueError("control set catalog_hashes mismatch")
    controls = control_set.get("controls")
    count = control_set.get("control_count")
    if not isinstance(controls, list) or isinstance(count, bool) or count != len(controls):
        raise ValueError("control_count must equal controls length")
    canonical = _canonical_controls(controls)
    expected_hash = knowledge_control_set_hash(
        schema_version=control_set["schema_version"],
        compiled_rules_hash=control_set["compiled_rules_hash"],
        catalog_hashes=bound_catalog_hashes,
        control_count=count,
        controls=canonical,
    )
    if control_set.get("control_set_hash") != expected_hash:
        raise ValueError("control_set_hash mismatch")
    _validate_control_inventory(canonical, raw_rules=raw_rules, active_rule_ids=active_rule_ids, official_names=official_names)
    return canonical


def _validate_control_inventory(
    controls: Sequence[Mapping[str, Any]],
    *,
    raw_rules: Mapping[str, Mapping[str, Any]],
    active_rule_ids: Sequence[str],
    official_names: frozenset[str],
) -> None:
    actual: dict[str, dict[str, str]] = {rule_id: {} for rule_id in active_rule_ids}
    for control in controls:
        if set(control) != _CONTROL_FIELDS:
            raise ValueError("structured control fields must exactly match the schema")
        rule_id = control.get("rule_id")
        control_id = control.get("control_id")
        kind = control.get("kind")
        if rule_id not in actual or not isinstance(control_id, str) or kind not in _CONTROL_KINDS:
            raise ValueError("control inventory contains an unknown rule, id, or kind")
        if control_id in actual[rule_id]:
            raise ValueError("duplicate structured control_id")
        if control.get("stage") != raw_rules[rule_id]["runtime"]["stage"]:
            raise ValueError("control stage does not match active rule stage")
        _typed_context(control.get("context"), official_names=official_names)
        _validate_expected(control.get("expected_outcome"), official_names=official_names)
        actual[rule_id][control_id] = kind
    expected = {rule_id: _compiled_control_inventory(raw_rules[rule_id]) for rule_id in active_rule_ids}
    if actual != expected:
        raise ValueError("structured control inventory does not match active compiled rules")


def _target_decision(
    decisions: Sequence[typed_rule_engine.RuleDecision],
    rule_id: str,
) -> typed_rule_engine.RuleDecision | None:
    matched = [decision for decision in decisions if decision.rule_id == rule_id]
    if len(matched) > 1:
        raise ValueError("runtime emitted duplicate decisions for a rule")
    return matched[0] if matched else None


def _execute_control(
    control: Mapping[str, Any],
    *,
    typed_pack: typed_rule_engine.CompiledRulePack,
    official_names: frozenset[str],
) -> dict[str, Any]:
    context = _typed_context(control["context"], official_names=official_names)
    expected = control["expected_outcome"]
    expected_context = _typed_context(expected["output_context"], official_names=official_names)
    result = typed_rule_engine.apply_rules(typed_pack, control["stage"], context)
    replay = typed_rule_engine.apply_rules(typed_pack, control["stage"], result.output_context)
    target = _target_decision(result.decisions, control["rule_id"])
    replay_target = _target_decision(replay.decisions, control["rule_id"])
    fixed_point_replay = None
    fixed_point_target = None
    if target is not None and target.outcome == "applied":
        fixed_point_replay = typed_rule_engine.apply_rules(
            typed_pack,
            control["stage"],
            replay.output_context,
        )
        fixed_point_target = _target_decision(
            fixed_point_replay.decisions,
            control["rule_id"],
        )
    target_payload = None if target is None else {
        "outcome": target.outcome,
        "reason_code": target.reason_code,
    }
    replay_target_payload = None if replay_target is None else {
        "outcome": replay_target.outcome,
        "reason_code": replay_target.reason_code,
    }
    fixed_point_target_payload = None if fixed_point_target is None else {
        "outcome": fixed_point_target.outcome,
        "reason_code": fixed_point_target.reason_code,
    }
    behavior_matches_oracle = target_payload == {
        "outcome": expected["outcome"],
        "reason_code": expected["reason_code"],
    } and result.output_context == expected_context
    changed = result.before_hash != result.after_hash
    if control["kind"] == "positive":
        kind_semantics_passed = target is not None and target.outcome == "applied" and changed
    elif control["kind"] == "near_neighbor":
        kind_semantics_passed = not (
            (target is not None and target.outcome == "applied") or changed
        )
    else:
        kind_semantics_passed = (
            target is not None
            and target.outcome == "matched_no_change"
            and not changed
            and behavior_matches_oracle
        )
    if target is None or replay_target is None:
        replay_decision_stable = False
    elif target.outcome == "applied":
        replay_decision_stable = (
            replay_target.outcome == "matched_no_change"
            and replay_target_payload == fixed_point_target_payload
        )
    else:
        replay_decision_stable = replay_target_payload == target_payload
    fixed_point_stable = fixed_point_replay is None or (
        fixed_point_replay.output_context == replay.output_context
        and fixed_point_replay.before_hash
        == fixed_point_replay.after_hash
        == replay.after_hash
        and fixed_point_replay.decisions == replay.decisions
        and fixed_point_replay.applied_rule_ids == replay.applied_rule_ids
    )
    idempotent = (
        replay.output_context == result.output_context
        and replay.before_hash == replay.after_hash == result.after_hash
        and replay_decision_stable
        and fixed_point_stable
    )
    return {
        "rule_id": control["rule_id"],
        "control_id": control["control_id"],
        "kind": control["kind"],
        "stage": control["stage"],
        "input_context": _context_payload(context),
        "expected_outcome": deepcopy(expected),
        "output_context": _context_payload(result.output_context),
        "before_hash": result.before_hash,
        "after_hash": result.after_hash,
        "decisions": [_decision_payload(decision) for decision in result.decisions],
        "target_decision": target_payload,
        "behavior_passed": behavior_matches_oracle,
        "behavior_matches_oracle": behavior_matches_oracle,
        "kind_semantics_passed": kind_semantics_passed,
        "idempotent": idempotent,
        "passed": behavior_matches_oracle and kind_semantics_passed and idempotent,
        "replay": {
            "output_context": _context_payload(replay.output_context),
            "before_hash": replay.before_hash,
            "after_hash": replay.after_hash,
            "decisions": [_decision_payload(decision) for decision in replay.decisions],
            "target_decision": replay_target_payload,
        },
    }


def _control_metrics(controls: Sequence[Mapping[str, Any]], *, treatment: bool) -> dict[str, Any]:
    failed = [control for control in controls if not control["passed"]]
    p0_count = sum(treatment and control["kind"] != "positive" for control in failed)
    return {
        "positive_hits": sum(control["kind"] == "positive" and control["kind_semantics_passed"] and control["behavior_matches_oracle"] for control in controls),
        "misses": sum(control["kind"] == "positive" and not (control["kind_semantics_passed"] and control["behavior_matches_oracle"]) for control in controls),
        "false_positives": sum(control["kind"] == "near_neighbor" and not control["kind_semantics_passed"] for control in controls),
        "exceptions_preserved": sum(control["kind"] == "reasonable_exception" and control["kind_semantics_passed"] for control in controls),
        "exception_failures": sum(control["kind"] == "reasonable_exception" and not control["kind_semantics_passed"] for control in controls),
        "idempotency_failures": sum(not control["idempotent"] for control in controls),
        "control_failures": len(failed),
        "p0_count": p0_count,
        "p0_applicable": treatment,
        "p0_status": ("failed" if p0_count else "passed") if treatment else "not_evaluated",
    }


def _rule_metadata(
    rule: Mapping[str, Any],
    *,
    refs: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    return {
        "rule_id": rule["rule_id"],
        "candidate_type": rule["candidate_type"],
        "candidate_hash": rule["candidate_hash"],
        "effect_hash": rule["effect_hash"],
        "runtime": deepcopy(rule["runtime"]),
        "source_refs": refs["source_refs"],
        "test_refs": refs["test_refs"],
        "source_refs_hash": content_hash(refs["source_refs"]),
        "test_refs_hash": content_hash(refs["test_refs"]),
    }


def _active_rule_report(
    rule: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    *,
    typed_pack: typed_rule_engine.CompiledRulePack,
    official_names: frozenset[str],
    refs: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    evaluated = [
        _execute_control(control, typed_pack=typed_pack, official_names=official_names)
        for control in controls
        if control["rule_id"] == rule["rule_id"]
    ]
    treatment = rule["candidate_type"] in {"treatment_gate_rule", "treatment_sequence_rule"}
    metrics = _control_metrics(evaluated, treatment=treatment)
    return {
        **_rule_metadata(rule, refs=refs),
        "acceptance_status": "active_scope_passed" if metrics["control_failures"] == 0 else "active_scope_failed",
        "control_count": len(evaluated),
        "hits": metrics["positive_hits"],
        **{key: value for key, value in metrics.items() if key != "positive_hits"},
        "controls": evaluated,
    }


def _audit_rule_report(
    rule: Mapping[str, Any],
    *,
    refs: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    return {
        **_rule_metadata(rule, refs=refs),
        "acceptance_status": "audit_only_unverified",
        "control_count": 0,
        "hits": 0,
        "misses": 0,
        "false_positives": 0,
        "exceptions_preserved": 0,
        "exception_failures": 0,
        "idempotency_failures": 0,
        "control_failures": 0,
        "p0_count": 0,
        "p0_applicable": False,
        "p0_status": "not_evaluated",
        "controls": [],
    }


def _aggregate_metrics(rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = [rule for rule in rules if rule["acceptance_status"] != "audit_only_unverified"]
    keys = (
        "hits",
        "misses",
        "false_positives",
        "exceptions_preserved",
        "exception_failures",
        "idempotency_failures",
        "control_failures",
        "p0_count",
    )
    totals = {key: sum(rule[key] for rule in active) for key in keys}
    totals["positive_hits"] = totals.pop("hits")
    totals["treatment_active_rule_count"] = sum(
        rule["candidate_type"] in {"treatment_gate_rule", "treatment_sequence_rule"}
        for rule in active
    )
    totals["p0_applicable"] = totals["treatment_active_rule_count"] > 0
    totals["p0_status"] = (
        "failed" if totals["p0_count"] else "passed"
    ) if totals["p0_applicable"] else "not_evaluated"
    return totals


def build_knowledge_acceptance(
    pack: Mapping[str, Any],
    control_set: Mapping[str, Any],
    *,
    project_root: Path,
    disease_catalog_path: Path,
) -> Dict[str, Any]:
    typed_pack = typed_rule_engine.parse_compiled_rule_pack(pack)
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    official_names, catalog_hash = _catalog_names(root, disease_catalog_path)
    catalog_hashes = {"data/ref_data/diseases_catalog.json": catalog_hash}
    raw_rules = {rule["rule_id"]: rule for rule in pack["rules"]}
    active_ids = sorted(
        rule_id for rule_id, rule in raw_rules.items() if rule["runtime"]["status"] == "active"
    )
    audit_ids = sorted(set(raw_rules) - set(active_ids))
    controls = _validate_control_set(
        control_set,
        raw_rules=raw_rules,
        active_rule_ids=active_ids,
        official_names=official_names,
        rules_hash=pack["rules_hash"],
        catalog_hashes=catalog_hashes,
    )
    refs = {rule_id: _validated_refs(rule, project_root=root) for rule_id, rule in raw_rules.items()}
    reports = []
    for rule_id in sorted(raw_rules):
        rule = raw_rules[rule_id]
        if rule_id in active_ids:
            reports.append(_active_rule_report(rule, controls, typed_pack=typed_pack, official_names=official_names, refs=refs[rule_id]))
        else:
            reports.append(_audit_rule_report(rule, refs=refs[rule_id]))
    metrics = _aggregate_metrics(reports)
    status = "active_scope_passed" if metrics["control_failures"] == 0 else "active_scope_failed"
    report = {
        "schema_version": "offline-knowledge-acceptance/v1",
        "scope": {
            "active_rule_ids": active_ids,
            "evaluated_active_rule_ids": active_ids,
            "audit_only_rule_ids": audit_ids,
            "unevaluated_rule_ids": audit_ids,
        },
        "input_hashes": {
            "compiled_pack_hash": content_hash(pack),
            "compiled_rules_hash": pack["rules_hash"],
            "control_set_hash": control_set["control_set_hash"],
            "catalog_hashes": catalog_hashes,
        },
        "rules": reports,
        "metrics": metrics,
        "status": status,
        "release_gate_passed": status == "active_scope_passed" and not audit_ids,
    }
    report["acceptance_hash"] = content_hash(report)
    return report
