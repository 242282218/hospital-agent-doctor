"""Pure policy: whether treatment review LLM should run, and allowed scope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TreatmentReviewDecision:
    should_review: bool
    reason_codes: Tuple[str, ...]
    allowed_scope: str  # "none" | "issue_scoped" | "safety_only"


def _is_actionable_issue(issue: Mapping[str, Any]) -> bool:
    if not isinstance(issue, Mapping):
        return False
    if issue.get("patchable") is True:
        return True
    severity = str(issue.get("severity") or "").strip().lower()
    return severity == "must_fix"


def _issue_codes(issues: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    codes = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or issue.get("issue_code") or "").strip()
        if code and code not in codes:
            codes.append(code)
    return tuple(codes)


def decide_treatment_review_policy(
    *,
    verifier_issues: Sequence[Mapping[str, Any]],
    safety_issues: Sequence[Mapping[str, Any]],
    diagnosis_conflicted: bool,
    diagnosis_count: int,
    high_risk_treatment: bool,
) -> TreatmentReviewDecision:
    """Decide whether to call treatment_review LLM.

    Priority:
    1. Filter to patchable or severity=must_fix issues.
    2. If diagnosis_conflicted: only safety_only when actionable safety remains.
    3. Else issue_scoped when actionable issues, multi-diagnosis, or high-risk.
    4. Else none / verifier_clean_low_risk.
    """
    actionable_verifier = [i for i in verifier_issues if _is_actionable_issue(i)]
    actionable_safety = [i for i in safety_issues if _is_actionable_issue(i)]
    reasons: list[str] = []

    if diagnosis_conflicted:
        if actionable_safety:
            reasons.append("diagnosis_conflicted")
            reasons.append("actionable_safety_issue")
            reasons.extend(_issue_codes(actionable_safety))
            return TreatmentReviewDecision(
                should_review=True,
                reason_codes=tuple(dict.fromkeys(reasons)),
                allowed_scope="safety_only",
            )
        reasons.append("diagnosis_conflicted")
        reasons.append("no_actionable_safety_append")
        return TreatmentReviewDecision(
            should_review=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
            allowed_scope="none",
        )

    if actionable_verifier:
        reasons.append("actionable_verifier_issue")
        reasons.extend(_issue_codes(actionable_verifier))
    if actionable_safety:
        reasons.append("actionable_safety_issue")
        reasons.extend(_issue_codes(actionable_safety))
    if int(diagnosis_count or 0) > 1:
        reasons.append("multi_diagnosis")
    if high_risk_treatment:
        reasons.append("high_risk_treatment")

    if reasons:
        return TreatmentReviewDecision(
            should_review=True,
            reason_codes=tuple(dict.fromkeys(reasons)),
            allowed_scope="issue_scoped",
        )

    return TreatmentReviewDecision(
        should_review=False,
        reason_codes=("verifier_clean_low_risk",),
        allowed_scope="none",
    )
