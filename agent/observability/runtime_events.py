"""Privacy-bounded runtime event constructors (no file I/O)."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional, Sequence


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(body.encode("utf-8")).hexdigest()


def safe_action_command_event(
    command: Any,
    *,
    action_sequence: int,
) -> Dict[str, Any]:
    action_type = str(getattr(command, "action_type", "") or "")
    payload = dict(getattr(command, "payload", {}) or {})
    event: Dict[str, Any] = {
        "type": "action_command",
        "command_id": str(getattr(command, "command_id", "") or ""),
        "action_sequence": int(action_sequence),
        "action_type": action_type,
        "payload_hash": canonical_hash(payload),
        "revision_role": "metadata_only",
        "blackboard_revision": int(getattr(command, "blackboard_revision", 0) or 0),
    }
    if action_type == "order_examination":
        items = payload.get("items") or payload.get("examinations") or []
        event["exam_items"] = [str(x) for x in list(items) if str(x).strip()]
    return event


_OBSERVATION_STATUSES = frozenset(
    {"ok", "sent", "succeeded", "invalid", "unavailable", "error", "not_sent", "unknown"}
)


def _observation_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in _OBSERVATION_STATUSES else "unknown"


def _observation_status_counts(value: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(value, Mapping):
        return counts
    for raw_status, raw_count in value.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            status = _observation_status(raw_status)
            counts[status] = counts.get(status, 0) + count
    return counts


def safe_action_observation_event(
    envelope: Any,
    *,
    action_sequence: int,
    command_id: str = "",
) -> Dict[str, Any]:
    raw = getattr(envelope, "raw_result", None)
    if raw is None and isinstance(envelope, Mapping):
        raw = envelope.get("raw_result")
    status = _observation_status(
        getattr(envelope, "status", "") or getattr(envelope, "dispatch_status", "")
    )
    observation_status = _observation_status(
        getattr(envelope, "observation_status", "")
        or getattr(envelope, "outcome", "")
        or status
    )
    item_status_counts: Dict[str, int] = {}
    if isinstance(raw, Mapping):
        results = raw.get("results")
        if isinstance(results, Mapping):
            for value in results.values():
                raw_status = value.get("status") if isinstance(value, Mapping) else None
                item_status = _observation_status(raw_status)
                item_status_counts[item_status] = item_status_counts.get(item_status, 0) + 1
        invalid = raw.get("invalid_items") or []
        if invalid:
            item_status_counts["invalid"] = item_status_counts.get("invalid", 0) + len(list(invalid))
    return {
        "type": "action_observation",
        "command_id": command_id or str(getattr(envelope, "command_id", "") or ""),
        "action_sequence": int(action_sequence),
        "dispatch_status": status,
        "observation_status": observation_status,
        "item_status_counts": item_status_counts,
        "result_hash": canonical_hash(raw),
    }


def safe_final_submission_event(
    *,
    payload_hash: str,
    legacy_verifier_hash: str,
    five_dimension_gate_hash: str,
    issue_codes: Sequence[str],
    patch_count: int,
    authorization_result: str,
    command_id: str = "",
    action_sequence: int = 0,
) -> Dict[str, Any]:
    """Record final-submission bindings without retaining clinical text."""
    return {
        "type": "final_submission",
        "payload_hash": str(payload_hash or ""),
        "legacy_verifier_hash": str(legacy_verifier_hash or ""),
        "five_dimension_gate_hash": str(five_dimension_gate_hash or ""),
        "issue_codes": [str(code) for code in issue_codes if str(code).strip()],
        "patch_count": int(patch_count),
        "authorization_result": str(authorization_result or "unknown"),
        "command_id": str(command_id or ""),
        "action_sequence": int(action_sequence),
    }


def safe_budget_event(
    case_budget: Optional[Mapping[str, Any]],
    llm_audit: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    prompt_name_counts: Dict[str, int] = {}
    success = 0
    provider_error = 0
    parse_error = 0
    repair = 0
    fallback = 0
    cap_rejected = 0
    audit_rows = [row for row in llm_audit or [] if isinstance(row, Mapping)]
    for row in audit_rows:
        name = str(row.get("prompt_name") or "unknown")
        prompt_name_counts[name] = prompt_name_counts.get(name, 0) + 1
        outcome = str(row.get("outcome") or "unknown")
        role = str(row.get("role") or "main")
        if outcome == "success":
            success += 1
        if outcome == "provider_error":
            provider_error += 1
        if outcome == "parse_error":
            parse_error += 1
        if role == "repair":
            repair += 1
        if bool(row.get("fallback")):
            fallback += 1
        if outcome == "cap_rejected":
            cap_rejected += 1
    budget = dict(case_budget or {})
    return {
        "type": "llm_budget_summary",
        "cap": int(budget.get("cap") or budget.get("hard_cap") or 0),
        "attempt": len(audit_rows),
        "success": success,
        "provider_error": provider_error,
        "parse_error": parse_error,
        "repair": repair,
        "fallback": fallback,
        "cap_rejected": cap_rejected,
        "prompt_name_counts": prompt_name_counts,
    }


def _optional_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def safe_llm_attempt_event(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the fixed, non-clinical LLM-attempt trace contract."""
    return {
        "type": "llm_attempt",
        "call_id": str(row.get("call_id") or ""),
        "prompt_name": str(row.get("prompt_name") or "unknown"),
        "role": str(row.get("role") or "main"),
        "attempt_index": int(row.get("attempt_index") or 1),
        "retry": bool(row.get("retry")),
        "transient": bool(row.get("transient")),
        "provider": str(row.get("provider") or "unknown"),
        "model": str(row.get("model") or "unknown"),
        "outcome": str(row.get("outcome") or "unknown"),
        "usage_source": str(row.get("usage_source") or "unknown"),
        "prompt_tokens": _optional_nonnegative_int(row.get("prompt_tokens")),
        "completion_tokens": _optional_nonnegative_int(row.get("completion_tokens")),
        "total_tokens": _optional_nonnegative_int(row.get("total_tokens")),
        "prompt_chars": int(row.get("prompt_chars") or 0),
        "response_chars": _optional_nonnegative_int(row.get("response_chars")),
        "latency_ms": _optional_nonnegative_int(row.get("latency_ms")),
        "final_accepted": bool(row.get("final_accepted")),
    }


def safe_verifier_event(
    verifier_result: Mapping[str, Any],
    treatment_plan: str,
) -> Dict[str, Any]:
    issues = list(verifier_result.get("issues") or [])
    codes = []
    for issue in issues:
        if isinstance(issue, Mapping):
            code = str(issue.get("code") or issue.get("issue_code") or "").strip()
            if code:
                codes.append(code)
        else:
            text = str(issue or "").strip()
            if text:
                codes.append(text[:64])
    return {
        "type": "verifier_summary",
        "passed": bool(verifier_result.get("passed")),
        "issue_codes": codes,
        "patch_applied": bool(
            verifier_result.get("patch_applied")
            or verifier_result.get("patched")
            or verifier_result.get("treatment_plan")
        ),
        "treatment_hash": canonical_hash(str(treatment_plan or "")),
    }


def safe_runtime_decision_event(
    *,
    action: str,
    reason_code: str = "",
    question: str = "",
    ordered_exam_count: int = 0,
) -> Dict[str, Any]:
    return {
        "type": "runtime_decision",
        "action": str(action or ""),
        "reason_code": str(reason_code or "")[:128],
        "question_hash": canonical_hash(str(question or "")) if question else "",
        "ordered_exam_count": int(ordered_exam_count),
    }


def safe_exam_plan_event(plan: Mapping[str, Any]) -> Dict[str, Any]:
    exams = [str(x) for x in list(plan.get("examinations") or []) if str(x).strip()]
    reason_codes = [str(x) for x in list(plan.get("reason_codes") or []) if str(x).strip()]
    accepted = list(plan.get("accepted") or [])
    source_counts: Dict[str, int] = {}
    semantic_keys = []
    for row in accepted:
        if not isinstance(row, Mapping):
            continue
        source = str(row.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        key = str(row.get("semantic_key") or "").strip()
        if key:
            semantic_keys.append(key)
    value = plan.get("value") if isinstance(plan.get("value"), Mapping) else {}
    return {
        "type": "exam_plan",
        "examinations": exams,
        "reason_codes": reason_codes,
        "source_counts": source_counts,
        "open_gap_ids": [str(x) for x in list(plan.get("open_gap_ids") or [])],
        "semantic_keys": [str(x) for x in list(value.get("semantic_keys") or semantic_keys)],
        "value": {
            "gap_ids": [str(x) for x in list(value.get("gap_ids") or [])],
            "intent_ids": [str(x) for x in list(value.get("intent_ids") or [])],
            "candidate_hash_before": str(value.get("candidate_hash_before") or "unknown"),
            "candidate_hash_after": str(value.get("candidate_hash_after") or "unknown"),
            "treatment_changed": value.get("treatment_changed", "unknown"),
            "urgency_changed": value.get("urgency_changed", "unknown"),
            "cost": value.get("cost") if value.get("cost") is not None else None,
            "duration_ms": value.get("duration_ms") if value.get("duration_ms") is not None else None,
        },
    }


def safe_diagnosis_state_event(
    *,
    axis_ids: Sequence[str],
    candidate_names: Sequence[str],
    consistency_issue_codes: Sequence[str] = (),
) -> Dict[str, Any]:
    names = [str(x) for x in candidate_names if str(x).strip()]
    return {
        "type": "diagnosis_state",
        "axis_ids": [str(x) for x in axis_ids if str(x).strip()],
        "candidate_names": names,
        "candidate_count": len(names),
        "consistency_issue_codes": [str(x) for x in consistency_issue_codes if str(x).strip()],
    }


def safe_case_end_event(
    *,
    finished: bool,
    diagnosis_names: Sequence[str],
    action_count: int,
    llm_attempts: int,
) -> Dict[str, Any]:
    return {
        "type": "case_end",
        "finished": bool(finished),
        "diagnosis_names": [str(x) for x in diagnosis_names if str(x).strip()],
        "action_count": int(action_count),
        "llm_attempts": int(llm_attempts),
    }


def safe_case_error_event(
    *,
    error_type: str,
    stage: str,
    action_count: int,
) -> Dict[str, Any]:
    return {
        "type": "case_error",
        "error_type": str(error_type or "Exception"),
        "stage": str(stage or "unknown"),
        "action_count": int(action_count),
    }


class SequencedEventSink:
    """Owns global event sequence for one case run."""

    def __init__(self, *, append, case_run_id: str) -> None:
        self._append = append
        self._case_run_id = case_run_id
        self._event_sequence = 0

    def __call__(self, event: Mapping[str, Any]) -> None:
        self._event_sequence += 1
        body = {
            "schema_version": "clinical-runtime-event/v1",
            "case_run_id": self._case_run_id,
            **_project_runtime_event(event),
            "sequence": self._event_sequence,
        }
        self._append(body)


def emit_runtime_event(event_sink, event: Mapping[str, Any]) -> None:
    if event_sink is not None:
        event_sink(event)


def _safe_hash(value: Any) -> str:
    text = str(value or "")
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower()):
        return text
    return canonical_hash(value)


def _safe_prefixed_sha256(value: Any) -> str:
    text = str(value or "")
    if text.startswith("sha256:") and len(text) == 71:
        digest = text.removeprefix("sha256:")
        if digest == digest.lower() and all(char in "0123456789abcdef" for char in digest):
            return text
    return _safe_hash(value)


def _project_candidate_hash(value: Any) -> str:
    if value == "unknown":
        return "unknown"
    return _safe_prefixed_sha256(value)


def _project_runtime_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Project trace events at the persistence boundary to prevent text leaks."""
    raw = dict(event)
    event_type = str(raw.get("type") or "")
    projector = _EVENT_PROJECTORS.get(event_type)
    if projector is None:
        return {
            "type": "unrecognized_runtime_event",
            "event_type_hash": canonical_hash(event_type),
            "event_hash": canonical_hash(raw),
            "field_count": len(raw),
        }
    return projector(raw)


def _project_case_start(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "case_start",
        "mode": str(event.get("mode") or "unknown"),
        "authority": str(event.get("authority") or "unknown"),
        "clinical_engine": str(event.get("clinical_engine") or "unknown"),
        "catalog_leaf_count": int(event.get("catalog_leaf_count") or 0),
        "release_pack_hash": _safe_hash(event.get("release_pack_hash")),
        "typed_rule_count": int(event.get("typed_rule_count") or 0),
        "runtime_identity_status": str(event.get("runtime_identity_status") or "legacy_unverified"),
        "runtime_identity_hash": _safe_hash(event.get("runtime_identity_hash")),
        "runtime_code_hash": _safe_hash(event.get("runtime_code_hash")),
        "prompt_pack_hash": _safe_hash(event.get("prompt_pack_hash")),
        "authority_policy_hash": _safe_hash(event.get("authority_policy_hash")),
    }


def _project_action_command(event: Mapping[str, Any]) -> Dict[str, Any]:
    exam_items = list(event.get("exam_items") or [])
    return {
        "type": "action_command",
        "command_id": str(event.get("command_id") or ""),
        "action_sequence": int(event.get("action_sequence") or 0),
        "action_type": str(event.get("action_type") or ""),
        "payload_hash": _safe_hash(
            event.get("payload_hash") or event.get("payload") or {}
        ),
        "revision_role": str(event.get("revision_role") or ""),
        "blackboard_revision": int(event.get("blackboard_revision") or 0),
        "exam_items_hash": canonical_hash(exam_items),
        "exam_item_count": len(exam_items),
    }


def _project_action_observation(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "action_observation",
        "command_id": str(event.get("command_id") or ""),
        "action_sequence": int(event.get("action_sequence") or 0),
        "dispatch_status": _observation_status(event.get("dispatch_status")),
        "observation_status": _observation_status(event.get("observation_status")),
        "item_status_counts": _observation_status_counts(event.get("item_status_counts")),
        "result_hash": _safe_hash(event.get("result_hash") or event.get("raw_result") or {}),
    }


def _project_final_submission(event: Mapping[str, Any]) -> Dict[str, Any]:
    issue_codes = list(event.get("issue_codes") or [])
    return {
        "type": "final_submission",
        "payload_hash": _safe_hash(event.get("payload_hash")),
        "legacy_verifier_hash": _safe_hash(event.get("legacy_verifier_hash")),
        "five_dimension_gate_hash": _safe_hash(event.get("five_dimension_gate_hash")),
        "issue_codes_hash": canonical_hash(issue_codes),
        "issue_count": len(issue_codes),
        "patch_count": int(event.get("patch_count") or 0),
        "authorization_result": str(event.get("authorization_result") or "unknown"),
        "command_id": str(event.get("command_id") or ""),
        "action_sequence": int(event.get("action_sequence") or 0),
    }


def _project_llm_budget_summary(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "llm_budget_summary",
        "cap": int(event.get("cap") or 0),
        "attempt": int(event.get("attempt") or 0),
        "success": int(event.get("success") or 0),
        "provider_error": int(event.get("provider_error") or 0),
        "parse_error": int(event.get("parse_error") or 0),
        "repair": int(event.get("repair") or 0),
        "fallback": int(event.get("fallback") or 0),
        "cap_rejected": int(event.get("cap_rejected") or 0),
        "prompt_name_counts": dict(event.get("prompt_name_counts") or {}),
    }


def _project_llm_attempt(event: Mapping[str, Any]) -> Dict[str, Any]:
    return safe_llm_attempt_event(event)


def _project_verifier_summary(event: Mapping[str, Any]) -> Dict[str, Any]:
    issue_codes = list(event.get("issue_codes") or [])
    return {
        "type": "verifier_summary",
        "passed": bool(event.get("passed")),
        "issue_codes_hash": canonical_hash(issue_codes),
        "issue_count": len(issue_codes),
        "patch_applied": bool(event.get("patch_applied")),
        "treatment_hash": _safe_hash(event.get("treatment_hash")),
    }


def _project_runtime_decision(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "runtime_decision",
        "action": str(event.get("action") or ""),
        "reason_hash": canonical_hash(event.get("reason_code") or event.get("reason") or ""),
        "question_hash": _safe_hash(event.get("question_hash")) if event.get("question_hash") else "",
        "ordered_exam_count": int(event.get("ordered_exam_count") or 0),
    }


def _project_exam_plan(event: Mapping[str, Any]) -> Dict[str, Any]:
    examinations = list(event.get("examinations") or [])
    reason_codes = list(event.get("reason_codes") or [])
    open_gap_ids = list(event.get("open_gap_ids") or [])
    semantic_keys = list(event.get("semantic_keys") or [])
    value = event.get("value") if isinstance(event.get("value"), Mapping) else {}
    gap_ids = list(value.get("gap_ids") or [])
    intent_ids = list(value.get("intent_ids") or [])
    treatment_changed = value.get("treatment_changed", "unknown")
    urgency_changed = value.get("urgency_changed", "unknown")
    return {
        "type": "exam_plan",
        "examinations_hash": canonical_hash(examinations),
        "examination_count": len(examinations),
        "reason_codes_hash": canonical_hash(reason_codes),
        "reason_count": len(reason_codes),
        "source_counts": dict(event.get("source_counts") or {}),
        "open_gap_ids_hash": canonical_hash(open_gap_ids),
        "open_gap_count": len(open_gap_ids),
        "semantic_keys_hash": canonical_hash(semantic_keys),
        "semantic_key_count": len(semantic_keys),
        "value_gap_ids_hash": canonical_hash(gap_ids),
        "value_gap_count": len(gap_ids),
        "value_intent_ids_hash": canonical_hash(intent_ids),
        "value_intent_count": len(intent_ids),
        "candidate_hash_before": _project_candidate_hash(
            value.get("candidate_hash_before")
        ),
        "candidate_hash_after": _project_candidate_hash(
            value.get("candidate_hash_after")
        ),
        "treatment_changed": (
            treatment_changed if treatment_changed in {True, False, "unknown"} else "unknown"
        ),
        "urgency_changed": (
            urgency_changed if urgency_changed in {True, False, "unknown"} else "unknown"
        ),
        "cost": value.get("cost") if isinstance(value.get("cost"), (int, float)) else None,
        "duration_ms": (
            value.get("duration_ms")
            if isinstance(value.get("duration_ms"), (int, float))
            else None
        ),
    }


def _project_diagnosis_state(event: Mapping[str, Any]) -> Dict[str, Any]:
    axis_ids = list(event.get("axis_ids") or [])
    candidate_names = list(event.get("candidate_names") or [])
    issue_codes = list(event.get("consistency_issue_codes") or [])
    return {
        "type": "diagnosis_state",
        "axis_ids_hash": canonical_hash(axis_ids),
        "axis_count": len(axis_ids),
        "candidate_names_hash": canonical_hash(candidate_names),
        "candidate_count": int(event.get("candidate_count") or len(candidate_names)),
        "consistency_issue_codes_hash": canonical_hash(issue_codes),
        "consistency_issue_count": len(issue_codes),
    }


def _project_case_end(event: Mapping[str, Any]) -> Dict[str, Any]:
    diagnosis_names = list(event.get("diagnosis_names") or [])
    return {
        "type": "case_end",
        "finished": bool(event.get("finished")),
        "diagnosis_names_hash": canonical_hash(diagnosis_names),
        "diagnosis_count": len(diagnosis_names),
        "action_count": int(event.get("action_count") or 0),
        "llm_attempts": int(event.get("llm_attempts") or 0),
    }


def _project_case_error(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "case_error",
        "error_type": str(event.get("error_type") or "Exception"),
        "stage": str(event.get("stage") or "unknown"),
        "action_count": int(event.get("action_count") or 0),
    }


_EVENT_PROJECTORS = {
    "case_start": _project_case_start,
    "action_command": _project_action_command,
    "action_observation": _project_action_observation,
    "final_submission": _project_final_submission,
    "llm_budget_summary": _project_llm_budget_summary,
    "llm_attempt": _project_llm_attempt,
    "verifier_summary": _project_verifier_summary,
    "runtime_decision": _project_runtime_decision,
    "exam_plan": _project_exam_plan,
    "diagnosis_state": _project_diagnosis_state,
    "case_end": _project_case_end,
    "case_error": _project_case_error,
}
