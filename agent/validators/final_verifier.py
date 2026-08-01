from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from agent.clinical.model import ClinicalBlackboard, SkillOperation, SkillProposal, TreatmentState


def _normalize(value: object) -> str:
    """Whitespace-insensitive key for standard-name comparison."""
    return "".join(str(value or "").split())


@dataclass(frozen=True)
class VerifierIssue:
    severity: str
    field: str
    code: str
    problem: str
    evidence_ids: Tuple[str, ...] = ()
    allowed_fix_scope: Tuple[str, ...] = ()
    patchable: bool = False


@dataclass(frozen=True)
class VerifierReport:
    passed: bool
    issues: Tuple[VerifierIssue, ...]
    p0_count: int
    unsupported_personalization_count: int


class FinalVerifier:
    """Deterministic verifier: reports concrete issues, never free-rewrites treatment text.

    Catalog parity is mandatory. The legacy online verifier already blocks
    `invalid_diagnosis_name`, `invalid_exam_name` and `duplicate_exam_name`; this
    verifier must raise the same P0 issues, otherwise wiring it into the online
    path would silently weaken standard-name enforcement (25% of the score).
    Pass the official catalogs to enable those checks.
    """

    def __init__(
        self,
        *,
        official_diseases: Iterable[str] = (),
        examination_leaves: Iterable[str] = (),
    ) -> None:
        self._official_diseases = frozenset(_normalize(name) for name in official_diseases if _normalize(name))
        self._examination_leaves = frozenset(
            _normalize(name) for name in examination_leaves if _normalize(name)
        )

    def verify(self, snapshot: ClinicalBlackboard) -> VerifierReport:
        issues = []
        issues.extend(self._catalog_issues(snapshot))
        selected = [h for h in snapshot.hypothesis_set if h.status == "selected"]
        if not selected:
            issues.append(
                VerifierIssue(
                    severity="must_fix",
                    field="diagnosis",
                    code="missing_selected_diagnosis",
                    problem="no selected diagnosis",
                )
            )
        else:
            hyp = selected[0]
            if not hyp.supporting_evidence_ids:
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="diagnosis",
                        code="diagnosis_without_evidence",
                        problem="selected diagnosis lacks evidence refs",
                    )
                )
            if not hyp.official_disease_name.strip():
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="diagnosis",
                        code="empty_diagnosis",
                        problem="selected diagnosis is empty",
                    )
                )

        draft = snapshot.treatment_state.draft_text.strip()
        if not draft:
            issues.append(
                VerifierIssue(
                    severity="must_fix",
                    field="treatment_plan",
                    code="empty_treatment",
                    problem="treatment draft is empty",
                )
            )
        else:
            if "Patient_" in draft:
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="treatment_plan",
                        code="patient_id_leak",
                        problem="treatment draft contains patient id",
                    )
                )
            if "无证据" in draft or "凭经验个体化" in draft:
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="personalization",
                        code="unsupported_personalization",
                        problem="treatment claims personalization without evidence",
                        patchable=True,
                        allowed_fix_scope=("draft_text",),
                    )
                )
            banned = ("立即大剂量激素", "无需随访")
            for phrase in banned:
                if phrase in draft:
                    issues.append(
                        VerifierIssue(
                            severity="must_fix",
                            field="treatment_plan",
                            code="unsafe_phrase",
                            problem="unsafe treatment phrase: %s" % phrase,
                            patchable=False,
                        )
                    )

        p0 = sum(1 for issue in issues if issue.severity == "must_fix")
        unsupported = sum(1 for issue in issues if issue.code == "unsupported_personalization")
        return VerifierReport(
            passed=p0 == 0,
            issues=tuple(issues),
            p0_count=p0,
            unsupported_personalization_count=unsupported,
        )

    def _catalog_issues(self, snapshot: ClinicalBlackboard) -> List[VerifierIssue]:
        """Standard-name and duplicate checks, at parity with the legacy verifier.

        Checks are skipped only when the corresponding catalog was not supplied,
        so an unconfigured verifier cannot fabricate a pass over an unknown name.
        """
        issues: List[VerifierIssue] = []
        if self._official_diseases:
            for hyp in snapshot.hypothesis_set:
                if hyp.status != "selected":
                    continue
                name = _normalize(hyp.official_disease_name)
                if name and name not in self._official_diseases:
                    issues.append(
                        VerifierIssue(
                            severity="must_fix",
                            field="diagnosis",
                            code="invalid_diagnosis_name",
                            problem=hyp.official_disease_name.strip(),
                            patchable=False,
                        )
                    )

        seen: set[str] = set()
        for exam in snapshot.examination_state:
            leaf = _normalize(exam.catalog_leaf_name)
            if not leaf:
                continue
            if leaf in seen:
                # A2 §3: a Blackboard that still holds a duplicate examination
                # must fail-closed. The unified submission pipeline dedups BEFORE
                # building the final payload; any duplicate that survives here is
                # a contract violation, so it is a blocking must_fix (not a soft
                # should_fix annotation) and forces report.passed=False.
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="examinations",
                        code="duplicate_exam_name",
                        problem=exam.catalog_leaf_name.strip(),
                        patchable=True,
                    )
                )
            seen.add(leaf)
            if self._examination_leaves and leaf not in self._examination_leaves:
                issues.append(
                    VerifierIssue(
                        severity="must_fix",
                        field="examinations",
                        code="invalid_exam_name",
                        problem=exam.catalog_leaf_name.strip(),
                        patchable=False,
                    )
                )
        return issues

    def propose_issue_replacement(
        self, snapshot: ClinicalBlackboard, proposal_id: str = "final-verifier"
    ) -> SkillProposal:
        report = self.verify(snapshot)
        operations = (
            SkillOperation(
                "replace_verifier_issues",
                {
                    "issues": [
                        {
                            "severity": issue.severity,
                            "field": issue.field,
                            "code": issue.code,
                            "problem": issue.problem,
                            "evidence_ids": list(issue.evidence_ids),
                            "allowed_fix_scope": list(issue.allowed_fix_scope),
                            "patchable": issue.patchable,
                        }
                        for issue in report.issues
                    ]
                },
            ),
        )
        return SkillProposal(
            proposal_id=proposal_id,
            skill_name="FinalVerifier",
            input_revision=snapshot.revision,
            purpose="replace_verifier_issues",
            operations=operations,
        )


class TreatmentDraftSanitizer:
    """Deterministic deletions and safety trims; never free-form rewrite."""

    UNSAFE_PHRASES = ("立即大剂量激素", "无需随访", "凭经验个体化")

    def sanitize(self, draft_text: str, state: TreatmentState) -> str:
        text = str(draft_text or "")
        for phrase in self.UNSAFE_PHRASES:
            text = text.replace(phrase, "")
        text = text.replace("Patient_", "患者")
        return " ".join(text.split()).strip()

    def propose(
        self, snapshot: ClinicalBlackboard, proposal_id: str = "treatment-sanitizer"
    ) -> SkillProposal:
        cleaned = self.sanitize(
            snapshot.treatment_state.draft_text, snapshot.treatment_state
        )
        return SkillProposal(
            proposal_id=proposal_id,
            skill_name="TreatmentDraftSanitizer",
            input_revision=snapshot.revision,
            purpose="sanitize_treatment_draft",
            operations=(
                SkillOperation(
                    "update_treatment_draft",
                    {
                        "urgency_and_disposition": snapshot.treatment_state.urgency_and_disposition,
                        "treatment_items": snapshot.treatment_state.treatment_items,
                        "draft_text": cleaned,
                    },
                ),
            ),
            confidence="high",
        )
