from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PRIORITY_RANK = {"routine": 0, "high": 1, "red_flag": 2}
ROLE_RANK = {
    "background_condition": 0,
    "background_history": 0,
    "secondary": 1,
    "current_problem": 2,
}


@dataclass(frozen=True)
class CandidatePoolDecision:
    candidates: tuple[dict[str, Any], ...]
    passed: bool
    safe_escalation_required: bool
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class SelectedDiagnosisDecision:
    diagnosis: str
    passed: bool
    reselected: bool
    safe_escalation_required: bool
    issue_codes: tuple[str, ...]


def axis_sources(axis: Mapping[str, Any]) -> set[str]:
    return {item for item in str(axis.get("source") or "").split("+") if item}


def axis_is_dominant(axis: Mapping[str, Any]) -> bool:
    sources = axis_sources(axis)
    evidence = {str(item).strip() for item in axis.get("evidence", []) if str(item).strip()}
    return (
        str(axis.get("clinical_role") or "current_problem") == "current_problem"
        and str(axis.get("status") or "suspected") in {"confirmed", "suspected"}
        and ("rule" in sources or axis.get("validated") is True)
        and len(evidence) >= 2
    )


def axis_requires_closure(axis: Mapping[str, Any]) -> bool:
    priority = str(axis.get("priority") or "routine")
    closure = str(axis.get("closure_requirement") or "").strip()
    return priority == "red_flag" or (priority == "high" and bool(closure))


def supported_axis_candidates(axis: Mapping[str, Any]) -> tuple[str, ...]:
    sources = axis_sources(axis)
    names: list[str] = []
    if "rule" in sources:
        names.extend(axis.get("rule_candidate_official_names") or axis.get("candidate_official_names") or [])
    if "llm" in sources and axis.get("validated") is True:
        names.extend(axis.get("promotable_candidate_official_names") or [])
    return tuple(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))


def official_catalog_names(disease_catalog: Mapping[str, Sequence[str]] | None) -> set[str]:
    names: set[str] = set()
    if not disease_catalog:
        return names
    for diseases in disease_catalog.values():
        for name in diseases or []:
            cleaned = str(name).strip()
            if cleaned:
                names.add(cleaned)
    return names


def candidate_role(item: Mapping[str, Any]) -> str:
    return str(item.get("role") or "current_problem").strip() or "current_problem"


def candidate_priority(item: Mapping[str, Any]) -> str:
    return str(item.get("priority") or "routine").strip() or "routine"


def candidate_sort_key(item: Mapping[str, Any]) -> tuple[int, int, int, float, str]:
    priority = PRIORITY_RANK.get(candidate_priority(item), 0)
    role = ROLE_RANK.get(candidate_role(item), 1)
    covered = 1 if item.get("axis_id") else 0
    try:
        score = float(item.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    return (-priority, -role, -covered, -score, str(item.get("disease") or ""))


def flatten_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in candidates if isinstance(item, Mapping)]


def dominant_axes(diagnosis_axes: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    axes = [dict(item) for item in (diagnosis_axes or []) if isinstance(item, Mapping)]
    return [axis for axis in axes if axis_is_dominant(axis)]


def enforce_candidate_pool_consistency(
    candidates: Sequence[Mapping[str, Any]],
    *,
    diagnosis_axes: Sequence[Mapping[str, Any]] | None,
    disease_catalog: Mapping[str, Sequence[str]] | None,
    limit: int = 8,
) -> CandidatePoolDecision:
    working = flatten_candidates(candidates)
    official = official_catalog_names(disease_catalog)
    existing = {str(item.get("disease") or "").strip() for item in working}
    issues: list[str] = []
    safe_escalation_required = False
    dominant = dominant_axes(diagnosis_axes)

    for axis in dominant:
        supported = [name for name in supported_axis_candidates(axis) if name in official]
        priority = str(axis.get("priority") or "routine")
        if axis_requires_closure(axis) and not supported:
            safe_escalation_required = True
            issue = (
                "red_flag_axis_underspecified"
                if str(axis.get("priority") or "routine") == "red_flag"
                else "dominant_axis_underspecified"
            )
            if issue not in issues:
                issues.append(issue)
            continue
        for name in supported:
            if name in existing:
                continue
            working.append(
                {
                    "disease": name,
                    "score": 100,
                    "source": "diagnosis_consistency_gate",
                    "role": "current_problem",
                    "matched_evidence": list(axis.get("evidence") or []),
                    "evidence_polarity": "positive",
                    "priority": priority,
                    "axis_id": axis.get("axis_id"),
                }
            )
            existing.add(name)

    working.sort(key=candidate_sort_key)
    protected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for axis in dominant:
        for name in supported_axis_candidates(axis):
            if name not in existing or name in seen:
                continue
            match = next((item for item in working if str(item.get("disease") or "") == name), None)
            if match is None:
                continue
            protected.append(match)
            seen.add(name)

    ordered: list[dict[str, Any]] = list(protected)
    for item in working:
        name = str(item.get("disease") or "").strip()
        if not name or name in seen:
            continue
        ordered.append(item)
        seen.add(name)

    truncated = ordered[: max(int(limit), 0)] if limit else ordered
    # Ensure dominant candidates survive limit truncation.
    if limit and len(truncated) < len(ordered):
        keep_names = {str(item.get("disease") or "") for item in protected}
        for item in ordered:
            name = str(item.get("disease") or "")
            if name in keep_names and name not in {str(x.get("disease") or "") for x in truncated}:
                if len(truncated) < limit:
                    truncated.append(item)
                else:
                    # Replace lowest priority non-protected tail when needed.
                    for index in range(len(truncated) - 1, -1, -1):
                        tail_name = str(truncated[index].get("disease") or "")
                        if tail_name not in keep_names:
                            truncated[index] = item
                            break
        truncated.sort(key=candidate_sort_key)

    passed = not safe_escalation_required and not issues
    return CandidatePoolDecision(
        candidates=tuple(truncated),
        passed=passed,
        safe_escalation_required=safe_escalation_required,
        issue_codes=tuple(issues),
    )


def diagnosis_covers_axis(diagnosis: str, axis: Mapping[str, Any]) -> bool:
    diagnosis_name = str(diagnosis or "").strip()
    if not diagnosis_name:
        return False
    supported = set(supported_axis_candidates(axis))
    return diagnosis_name in supported


def select_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    diagnosis_axes: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    dominant = dominant_axes(diagnosis_axes)
    preferred_names: set[str] = set()
    for axis in dominant:
        preferred_names.update(supported_axis_candidates(axis))

    ranked = sorted(
        (
            (
                (
                    1 if str(item.get("disease") or "") in preferred_names else 0,
                    PRIORITY_RANK.get(candidate_priority(item), 0),
                    ROLE_RANK.get(candidate_role(item), 1),
                    float(item.get("score") or 0),
                ),
                item,
            )
            for item in candidates
        ),
        key=lambda pair: (-pair[0][0], -pair[0][1], -pair[0][2], -pair[0][3], str(pair[1].get("disease") or "")),
    )
    return dict(ranked[0][1]) if ranked else None


def enforce_selected_diagnosis_consistency(
    diagnosis: str,
    *,
    candidates: Sequence[Mapping[str, Any]],
    diagnosis_axes: Sequence[Mapping[str, Any]] | None,
) -> SelectedDiagnosisDecision:
    working = flatten_candidates(candidates)
    diagnosis_name = str(diagnosis or "").strip()
    dominant = dominant_axes(diagnosis_axes)
    issues: list[str] = []
    safe_escalation_required = False

    for axis in dominant:
        supported = supported_axis_candidates(axis)
        if str(axis.get("priority") or "") == "red_flag" and not supported:
            safe_escalation_required = True
            if "red_flag_axis_underspecified" not in issues:
                issues.append("red_flag_axis_underspecified")

    if not working:
        return SelectedDiagnosisDecision(
            diagnosis=diagnosis_name,
            passed=not safe_escalation_required and not issues,
            reselected=False,
            safe_escalation_required=safe_escalation_required,
            issue_codes=tuple(issues),
        )

    current = next((item for item in working if str(item.get("disease") or "") == diagnosis_name), None)
    covers_dominant = any(diagnosis_covers_axis(diagnosis_name, axis) for axis in dominant) if dominant else True
    if current is not None and covers_dominant and candidate_role(current) == "current_problem":
        return SelectedDiagnosisDecision(
            diagnosis=diagnosis_name,
            passed=not safe_escalation_required and not issues,
            reselected=False,
            safe_escalation_required=safe_escalation_required,
            issue_codes=tuple(issues),
        )

    preferred = select_best_candidate(working, diagnosis_axes=diagnosis_axes)
    if preferred is None:
        return SelectedDiagnosisDecision(
            diagnosis=diagnosis_name,
            passed=False,
            reselected=False,
            safe_escalation_required=safe_escalation_required,
            issue_codes=tuple(issues + ["selected_diagnosis_axis_mismatch"]),
        )

    preferred_name = str(preferred.get("disease") or "").strip()
    reselected = preferred_name != diagnosis_name
    preferred_covers = (
        any(diagnosis_covers_axis(preferred_name, axis) for axis in dominant)
        if dominant
        else True
    )
    if reselected:
        if current is not None and candidate_role(current) in {
            "background_condition",
            "background_history",
            "secondary",
        }:
            issues.append("background_overrides_current_problem")
        else:
            issues.append("selected_diagnosis_axis_mismatch")
    elif dominant and not preferred_covers:
        issues.append("selected_diagnosis_axis_mismatch")
    elif current is not None and candidate_role(current) in {
        "background_condition",
        "background_history",
        "secondary",
    } and dominant and preferred_covers and preferred_name == diagnosis_name:
        # Selected name matches a supported axis disease, but was tagged as background.
        issues.append("background_overrides_current_problem")

    return SelectedDiagnosisDecision(
        diagnosis=preferred_name or diagnosis_name,
        passed=not safe_escalation_required and not issues,
        reselected=reselected,
        safe_escalation_required=safe_escalation_required,
        issue_codes=tuple(dict.fromkeys(issues)),
    )
