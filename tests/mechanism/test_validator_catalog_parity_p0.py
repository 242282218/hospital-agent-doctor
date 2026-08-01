"""P0: agent/validators/FinalVerifier must reach catalog parity with the legacy verifier.

The legacy online verifier (`agent.legacy_orchestrator.final_verifier`) already blocks
`invalid_diagnosis_name`, `invalid_exam_name` and `duplicate_exam_name`. The newer
structured verifier in `agent/validators` had none of them, so wiring it into the
online path would have silently dropped standard-name enforcement, which is 25% of
the competition score.

These tests pin the parity so the structured verifier can never become a downgrade.
"""

from __future__ import annotations

from agent.clinical.model import (
    ClinicalBlackboard,
    ExamIntent,
    HypothesisItem,
    TreatmentState,
)
from agent.validators.final_verifier import FinalVerifier

OFFICIAL_DISEASES = ("蜂窝织炎", "社区获得性肺炎")
EXAM_LEAVES = ("体格检查", "全血细胞计数（CBC）", "细菌培养及鉴定")

GOOD_PLAN = "针对蜂窝织炎给予静脉抗生素治疗，监测体温和炎症指标，门诊随访复查。"


def _snapshot(*, diagnosis: str, exams: tuple, plan: str = GOOD_PLAN) -> ClinicalBlackboard:
    return ClinicalBlackboard(
        hypothesis_set=(
            HypothesisItem(
                hypothesis_id="hyp-1",
                official_disease_name=diagnosis,
                supporting_evidence_ids=("ev-1",),
                status="selected",
            ),
        ),
        examination_state=tuple(
            ExamIntent(
                exam_intent_id="exam-%d" % index,
                catalog_leaf_name=name,
                status="resulted",
            )
            for index, name in enumerate(exams)
        ),
        treatment_state=TreatmentState(draft_text=plan),
    )


def _codes(report) -> list:
    return [issue.code for issue in report.issues]


def test_clean_case_passes_with_catalogs() -> None:
    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(
        _snapshot(diagnosis="蜂窝织炎", exams=("体格检查", "细菌培养及鉴定"))
    )
    assert report.passed is True, _codes(report)
    assert _codes(report) == []


def test_diagnosis_outside_catalog_is_must_fix() -> None:
    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(_snapshot(diagnosis="幻觉病名", exams=("体格检查",)))
    issue = next(i for i in report.issues if i.code == "invalid_diagnosis_name")
    assert report.passed is False
    assert issue.severity == "must_fix"
    # A hallucinated official name must never be auto-patched into a submission.
    assert issue.patchable is False
    assert issue.field == "diagnosis"


def test_exam_outside_catalog_is_must_fix() -> None:
    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(
        _snapshot(diagnosis="蜂窝织炎", exams=("皮肤检查", "生命体征"))
    )
    bad = [i for i in report.issues if i.code == "invalid_exam_name"]
    assert report.passed is False
    assert {i.problem for i in bad} == {"皮肤检查", "生命体征"}
    assert all(i.severity == "must_fix" and i.patchable is False for i in bad)


def test_duplicate_exam_is_reported() -> None:
    # A2 §3 unlocks the contract: a Blackboard that still holds a duplicate
    # examination must fail-closed. Previously the duplicate was `should_fix`
    # and did not fail the report; the unified submission pipeline cannot
    # tolerate that soft annotation, so it must now be a blocking must_fix.
    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(
        _snapshot(diagnosis="蜂窝织炎", exams=("体格检查", "体格检查"))
    )
    dup = next(i for i in report.issues if i.code == "duplicate_exam_name")
    assert dup.problem == "体格检查"
    assert dup.severity == "must_fix"
    assert dup.patchable is True
    assert report.passed is False


def test_catalog_codes_match_legacy_verifier_codes() -> None:
    """The three catalog codes must be spelled exactly as the legacy verifier spells them."""
    import inspect

    from agent import legacy_orchestrator

    legacy_source = inspect.getsource(legacy_orchestrator.final_verifier)
    for code in ("invalid_diagnosis_name", "invalid_exam_name", "duplicate_exam_name"):
        assert code in legacy_source, "legacy verifier no longer emits %s" % code

    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(
        _snapshot(diagnosis="幻觉病名", exams=("体格检查", "体格检查", "皮肤检查"))
    )
    assert set(_codes(report)) >= {
        "invalid_diagnosis_name",
        "invalid_exam_name",
        "duplicate_exam_name",
    }


def test_without_catalogs_the_checks_are_skipped_not_faked() -> None:
    """No catalog means no claim: name checks are skipped, never reported as pass.

    Existing callers construct FinalVerifier() with no arguments; that must stay
    working, but it must also not pretend a hallucinated name was validated.
    """
    verifier = FinalVerifier()
    report = verifier.verify(_snapshot(diagnosis="幻觉病名", exams=("皮肤检查",)))
    assert "invalid_diagnosis_name" not in _codes(report)
    assert "invalid_exam_name" not in _codes(report)


def test_duplicate_exam_blocks_submission_in_blackboard_verifier() -> None:
    """A2 §3 contract: when the Blackboard still holds a duplicate examination,
    `duplicate_exam_name` must be a blocking must_fix (not a soft should_fix),
    so `report.passed` is False and the verifier cannot silently endorse the
    duplicated plan.

    This pins the single-source behavior the A2 pipeline relies on: deterministic
    dedup happens BEFORE the final payload is built; if any duplicate survives,
    the verifier must fail-closed, not just annotate a weak finding.
    """
    verifier = FinalVerifier(
        official_diseases=OFFICIAL_DISEASES,
        examination_leaves=EXAM_LEAVES,
    )
    report = verifier.verify(
        _snapshot(diagnosis="蜂窝织炎", exams=("体格检查", "体格检查"))
    )
    dup = next(i for i in report.issues if i.code == "duplicate_exam_name")
    assert dup.severity == "must_fix", (
        "duplicate_exam_name must be must_fix in the Blackboard verifier; the "
        "A2 unified pipeline will not tolerate stale duplicates."
    )
    assert report.passed is False, (
        "duplicate examinations still present must fail the report outright"
    )
