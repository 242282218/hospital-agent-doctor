"""Legacy candidate/axis/evidence projection into HypothesisItem (shadow only)."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from agent.clinical.model.hypothesis import HypothesisItem


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean(value)
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            text = _clean(item)
            if text and text not in out:
                out.append(text)
        return out
    text = _clean(value)
    return [text] if text else []


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()
    return "%s%s" % (prefix, digest[:16])


def _confidence_from_candidate(candidate: Mapping[str, Any]) -> str:
    try:
        score = float(candidate.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    priority = _clean(candidate.get("priority")).lower()
    if score >= 100 or priority in {"high", "red_flag", "urgent"}:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _role_from_candidate(candidate: Mapping[str, Any]) -> str:
    role = _clean(candidate.get("role")).lower()
    if role in {"current_problem", "primary", "selected"}:
        return "current_problem" if role != "selected" else "selected"
    if role in {"background", "background_condition", "history"}:
        return "background"
    if role:
        return role
    return "differential"


def _axes_for_disease(
    disease: str, diagnosis_axes: Sequence[Mapping[str, Any]]
) -> List[Mapping[str, Any]]:
    target = _clean(disease)
    matched: List[Mapping[str, Any]] = []
    for axis in diagnosis_axes or ():
        if not isinstance(axis, Mapping):
            continue
        names = _as_text_list(
            axis.get("candidate_official_names") or axis.get("rule_candidate_official_names")
        )
        if target and target in names:
            matched.append(axis)
    return matched


def _support_ids_for_candidate(
    candidate: Mapping[str, Any],
    axes: Sequence[Mapping[str, Any]],
    case_state: Mapping[str, Any],
) -> Tuple[str, ...]:
    ids: List[str] = []
    for evidence in _as_text_list(candidate.get("matched_evidence")):
        ids.append(_stable_id("patient:", evidence))
    for axis in axes:
        axis_id = _clean(axis.get("axis_id")) or "axis"
        for index, evidence in enumerate(_as_text_list(axis.get("evidence")), start=1):
            ids.append("axis:%s:support:%d" % (axis_id, index))
            _ = evidence
    # Objective exam positives as exam:<name>:<hash>
    results = case_state.get("examination_results")
    if isinstance(results, Mapping):
        for name, payload in results.items():
            exam_name = _clean(name)
            if not exam_name or not isinstance(payload, Mapping):
                continue
            status = _clean(payload.get("status")).lower()
            if status in {"abnormal", "positive"}:
                ids.append(
                    "exam:%s:%s"
                    % (
                        exam_name,
                        sha256(str(payload.get("result") or status).encode("utf-8")).hexdigest()[
                            :12
                        ],
                    )
                )
    # de-dupe preserve order
    return tuple(dict.fromkeys(ids))


def _oppose_ids_for_axes(
    axes: Sequence[Mapping[str, Any]], case_state: Mapping[str, Any]
) -> Tuple[str, ...]:
    ids: List[str] = []
    for axis in axes:
        axis_id = _clean(axis.get("axis_id")) or "axis"
        for index, evidence in enumerate(
            _as_text_list(axis.get("opposing_evidence") or axis.get("negative_evidence")),
            start=1,
        ):
            ids.append("axis:%s:oppose:%d" % (axis_id, index))
            _ = evidence
    results = case_state.get("examination_results")
    if isinstance(results, Mapping):
        for name, payload in results.items():
            exam_name = _clean(name)
            if not exam_name or not isinstance(payload, Mapping):
                continue
            status = _clean(payload.get("status")).lower()
            result_text = str(payload.get("result") or "")
            if status in {"normal", "negative"} or any(
                marker in result_text for marker in ("阴性", "正常", "未见", "未检出")
            ):
                ids.append(
                    "exam:%s:%s"
                    % (
                        exam_name,
                        sha256(str(payload.get("result") or status).encode("utf-8")).hexdigest()[
                            :12
                        ],
                    )
                )
    return tuple(dict.fromkeys(ids))


def _gap_ids_for_axes(axes: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    ids: List[str] = []
    for axis in axes:
        axis_id = _clean(axis.get("axis_id")) or "axis"
        missing = _as_text_list(axis.get("missing_evidence"))
        if missing:
            ids.append("gap:%s" % axis_id)
        for gap in _as_text_list(axis.get("open_gap_ids")):
            ids.append(gap if gap.startswith("gap:") else "gap:%s" % gap)
    return tuple(dict.fromkeys(ids))


def _exam_intents_for(
    axes: Sequence[Mapping[str, Any]], case_state: Mapping[str, Any]
) -> Tuple[str, ...]:
    intents: List[str] = []
    for axis in axes:
        for intent in _as_text_list(axis.get("exam_intents")):
            if intent not in intents:
                intents.append(intent)
    for intent in _as_text_list(case_state.get("typed_exam_intent_ids")):
        if intent not in intents:
            intents.append(intent)
    return tuple(intents)


def _risk_tags_for(axes: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    risks: List[str] = []
    for axis in axes:
        for risk in _as_text_list(axis.get("treatment_risks")):
            if risk not in risks:
                risks.append(risk)
    return tuple(risks)


def build_legacy_hypotheses(
    *,
    disease_candidates: Sequence[Mapping[str, Any]],
    diagnosis_axes: Sequence[Mapping[str, Any]],
    case_state: Mapping[str, Any],
    selected_diagnosis: str = "",
) -> Tuple[HypothesisItem, ...]:
    """Project legacy candidates into stable HypothesisItem list (shadow only)."""
    selected = _clean(selected_diagnosis)
    items: List[HypothesisItem] = []
    seen = set()

    def add_from_candidate(candidate: Mapping[str, Any]) -> None:
        name = _clean(candidate.get("disease") or candidate.get("official_disease_name"))
        if not name or name in seen:
            return
        seen.add(name)
        axes = _axes_for_disease(name, diagnosis_axes)
        axis_ids = sorted(
            _clean(axis.get("axis_id")) for axis in axes if _clean(axis.get("axis_id"))
        )
        hypothesis_id = _stable_id("hyp-", name, ",".join(axis_ids))
        items.append(
            HypothesisItem(
                hypothesis_id=hypothesis_id,
                official_disease_name=name,
                role=_role_from_candidate(candidate),
                confidence=_confidence_from_candidate(candidate),
                supporting_evidence_ids=_support_ids_for_candidate(
                    candidate, axes, case_state
                ),
                opposing_evidence_ids=_oppose_ids_for_axes(axes, case_state),
                open_gap_ids=_gap_ids_for_axes(axes),
                required_exam_intents=_exam_intents_for(axes, case_state),
                treatment_risk_tags=_risk_tags_for(axes),
                status="selected" if selected and name == selected else "active",
            )
        )

    for candidate in disease_candidates or ():
        if isinstance(candidate, Mapping):
            add_from_candidate(candidate)

    # Ensure selected diagnosis appears even if selector omitted it from candidates.
    if selected and selected not in seen:
        add_from_candidate(
            {
                "disease": selected,
                "role": "selected",
                "score": 100,
                "matched_evidence": [],
            }
        )
        # mark selected
        items[-1] = HypothesisItem(
            hypothesis_id=items[-1].hypothesis_id,
            official_disease_name=items[-1].official_disease_name,
            role=items[-1].role,
            confidence=items[-1].confidence,
            supporting_evidence_ids=items[-1].supporting_evidence_ids,
            opposing_evidence_ids=items[-1].opposing_evidence_ids,
            open_gap_ids=items[-1].open_gap_ids,
            required_exam_intents=items[-1].required_exam_intents,
            treatment_risk_tags=items[-1].treatment_risk_tags,
            status="selected",
        )
    return tuple(items)


def verify_selected_hypothesis_traceability(
    *,
    selected_diagnosis: str,
    hypotheses: Sequence[HypothesisItem],
    reasoning: str = "",
) -> Tuple[bool, Tuple[str, ...]]:
    """Shadow-only traceability check. Never blocks final submission."""
    selected = _clean(selected_diagnosis)
    issues: List[str] = []
    if not selected:
        return True, ()

    matched = [
        item
        for item in hypotheses
        if _clean(item.official_disease_name) == selected
        or _clean(item.status) == "selected"
    ]
    if not matched:
        issues.append("selected_hypothesis_missing")
        return False, tuple(issues)

    item = matched[0]
    supports = tuple(item.supporting_evidence_ids or ())
    opposes = tuple(item.opposing_evidence_ids or ())
    if not supports:
        issues.append("selected_support_missing")
    if opposes and not supports:
        issues.append("selected_only_opposed")

    reason = _clean(reasoning)
    if reason and selected:
        # Reuse simple explicit rejection markers already common in fixtures.
        negative_patterns = (
            "不支持" + selected,
            "排除" + selected,
            "否定" + selected,
            "不是" + selected,
            "不考虑" + selected,
        )
        if any(pattern in reason for pattern in negative_patterns):
            issues.append("reasoning_rejects_selected")

    if issues:
        return False, tuple(dict.fromkeys(issues))
    return True, ()
