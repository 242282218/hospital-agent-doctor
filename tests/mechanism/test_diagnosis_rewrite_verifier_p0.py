"""P0: diagnosis-change rebuild must pass through the REAL final_verifier.

Codex Round-2 counter-example: after reselecting Hashimoto (from migraine), the
rebuild path in finalize_treatment_with_verified_fallback runs apply_treatment_safety
then SYNTHESIZES receipt={"passed": True} without ever calling final_verifier on the
rebuilt text. The same synthetic-passed bug exists for the safe_escalation and
conservative_fallback paths.

Requirement: any selector/verifier/fallback that changes the diagnosis MUST discard
the old-axis treatment, rebuild, then run safety + the REAL final_verifier on the
SAME rebuilt text. receipt may only copy a verifier result for that exact text.
"""
from __future__ import annotations

from agent.legacy_orchestrator import (
    FinalVerificationError,
    finalize_treatment_with_verified_fallback,
)


def test_diagnosis_change_rebuild_must_run_real_final_verifier() -> None:
    """Rebuild after diagnosis change must converge via the REAL final_verifier
    (converge_verified_treatment), not synthesize passed=True. Proof: we monkeypatch
    converge_verified_treatment to return a sentinel; the receipt must carry that
    sentinel, proving the real verifier result is copied — not a hand-written flat
    success."""
    import agent.legacy_orchestrator as lo
    sentinel = {"passed": True, "issues": [], "__SENTINEL__": True}
    real_converge = lo.converge_verified_treatment
    lo.converge_verified_treatment = lambda **_: dict(sentinel)
    try:
        _, plan, _, receipt = lo.finalize_treatment_with_verified_fallback(
            diagnosis="偏头痛",
            treatment_plan="针对偏头痛急性发作：优先曲坦类（舒马曲坦）缓解，监测用药天数。",
            reasoning="偏头痛急性期特异性治疗。",
            verifier_result={
                "passed": True,
                "patched_treatment": "针对偏头痛急性发作：优先曲坦类（舒马曲坦）缓解，监测用药天数。",
            },
            examinations=[],
            official_diseases=["偏头痛", "桥本甲状腺炎"],
            examination_catalog={"神经内科": ["偏头痛"]},
            exam_plan_trace=[],
            case_features={
                "candidate_diagnoses": ["偏头痛", "桥本甲状腺炎"],
                "selected_diagnosis_consistency": {
                    "before": "偏头痛",
                    "after": "桥本甲状腺炎",
                    "reselected": True,
                },
            },
            safety_profiles=[],
        )
    finally:
        lo.converge_verified_treatment = real_converge
    assert "曲坦" not in plan, "triptan must be discarded after Hashimoto reselect"
    assert receipt.get("degraded") == "diagnosis_changed_rebuild"
    # The receipt must carry the sentinel from the REAL converge call — proving the
    # rebuilt text was checked by the real verifier, not a synthetic passed=True.
    assert receipt.get("__SENTINEL__") is True, (
        "diagnosis-changed rebuild must copy the REAL verifier result; got %r" % receipt
    )


def test_red_flag_no_synthetic_passed_true() -> None:
    """A red-flag escalation path must never synthesize passed=True when the
    escalation plan cannot validate. It must raise FinalVerificationError so the
    per-case isolation layer converts it to a structured incomplete receipt."""
    raised = False
    try:
        _, _, _, receipt = finalize_treatment_with_verified_fallback(
            diagnosis="下消化道出血",
            treatment_plan="",
            reasoning="",
            verifier_result=None,
            examinations=[],
            official_diseases=["下消化道出血"],
            examination_catalog={},
            exam_plan_trace=[],
            case_features={
                "red_flags": ["大量呕血", "失血性休克"],
                "candidate_diagnoses": ["下消化道出血"],
                "positive_findings": ["大量呕血"],
            },
            safety_profiles=[],
        )
    except FinalVerificationError:
        raised = True
    except Exception as exc:
        raise AssertionError("expected FinalVerificationError, got %r" % exc)
    if not raised:
        # If it returned, it must NOT be a synthetic safe_escalation faking passed.
        assert receipt.get("degraded") != "safe_escalation" or receipt.get("unresolved_axis_id"), (
            "safe escalation must name the axis and not fake passed"
        )


def test_active_upper_gi_bleed_rejects_generic_conservative_text() -> None:
    """Fixed counter-example: active upper GI bleed must NOT be allowed to pass with
    a generic 'insufficient info, observe' conservative text (passed=true)."""
    # A dominant red_flag axis (active bleed) with no valid escalation plan and a
    # generic conservative fallback must fail verification, not pass.
    raised = False
    try:
        _, plan, _, receipt = finalize_treatment_with_verified_fallback(
            diagnosis="活动性上消化道出血",
            treatment_plan="信息不足，对症支持，建议观察。",
            reasoning="信息有限。",
            verifier_result=None,
            examinations=[],
            official_diseases=["活动性上消化道出血"],
            examination_catalog={},
            exam_plan_trace=[],
            case_features={
                "red_flags": ["呕血", "黑便"],
                "candidate_diagnoses": ["活动性上消化道出血"],
                "positive_findings": ["呕血", "黑便"],
                "diagnosis_axes": [
                    {
                        "axis_id": "active_upper_gi_bleed",
                        "priority": "red_flag",
                        "status": "confirmed",
                        "source": "rule",
                        "evidence": ["呕血", "黑便"],
                        "clinical_role": "current_problem",
                    }
                ],
            },
            safety_profiles=[],
        )
    except FinalVerificationError:
        raised = True
    if not raised:
        # Generic conservative text must not pass as a valid escalation.
        assert receipt.get("passed") is not True or "急诊" in plan or "转诊" in plan, (
            "generic conservative text must not pass for active bleed; got %r" % receipt
        )
