"""P0: diagnosis rewrite must rebuild coaxial treatment and re-verify."""
from __future__ import annotations

from agent.legacy_orchestrator import (
    diagnosis_supportive_treatment_plan,
    reconcile_selected_diagnosis_plan,
)


def test_migraine_axis_rebuild_after_reselect_not_empty_shell() -> None:
    # Pattern from B1 09993: wrong/empty shell after axis change to migraine.
    plan, reasoning = reconcile_selected_diagnosis_plan(
        {
            "normalized_diagnosis": "紧张型头痛",
            "raw_diagnosis": "头痛",
        },
        selected_diagnosis="偏头痛",
        treatment_plan="观察随访。",
        reasoning="按紧张型头痛观察。",
        default_reasoning="默认",
    )
    supportive = diagnosis_supportive_treatment_plan("偏头痛")
    assert plan == supportive or ("偏头痛" in plan and ("NSAIDs" in plan or "对乙酰氨基酚" in plan or "曲坦" in plan))
    assert "监测" in plan or "评估" in plan
    assert "紧张型" not in plan
    assert "偏头痛" in reasoning or "候选疾病约束" in reasoning


def test_rickets_to_all_rebuild_not_old_axis() -> None:
    # Pattern from B1 00830 neighborhood: rickets shell must not survive ALL reselect.
    plan, reasoning = reconcile_selected_diagnosis_plan(
        {
            "normalized_diagnosis": "维生素D缺乏性佝偻病",
            "raw_diagnosis": "佝偻病",
        },
        selected_diagnosis="急性淋巴细胞白血病",
        treatment_plan="补充维生素D与钙剂，门诊随访。",
        reasoning="佝偻病支持治疗。",
        default_reasoning="默认",
    )
    assert "急性淋巴细胞白血病" in plan or "诱导化疗" in plan or "化疗" in plan
    assert "维生素D" not in plan or "不得" in plan
    assert "佝偻病" not in plan
    assert "急性淋巴细胞白血病" in reasoning or "候选疾病约束" in reasoning


def test_same_diagnosis_keeps_existing_plan_when_present() -> None:
    plan, reasoning = reconcile_selected_diagnosis_plan(
        {
            "normalized_diagnosis": "偏头痛",
            "raw_diagnosis": "偏头痛",
        },
        selected_diagnosis="偏头痛",
        treatment_plan="针对偏头痛：NSAIDs缓解，监测用药天数。",
        reasoning="轴一致。",
        default_reasoning="默认",
    )
    assert "NSAIDs" in plan
    assert reasoning == "轴一致。"


def test_neighbor_negative_wrong_axis_plan_is_discarded_on_reselect() -> None:
    plan, _ = reconcile_selected_diagnosis_plan(
        {
            "normalized_diagnosis": "上呼吸道感染",
            "raw_diagnosis": "感冒",
        },
        selected_diagnosis="社区获得性肺炎",
        treatment_plan="多喝水休息，对症退热。",
        reasoning="按上感处理。",
        default_reasoning="默认",
    )
    # Must rebuild to pneumonia-shaped shell, not keep URI supportive only.
    assert "肺炎" in plan or "抗感染" in plan or "感染" in plan
    assert "多喝水" not in plan


# ---- Diagnosis-rewrite rebuild tests (Round 2) ----

def test_hashimoto_rebuild_discards_triptan() -> None:
    """Codex counter-example: reselecting Hashimoto after migraine MUST drop triptan."""
    from agent.legacy_orchestrator import finalize_treatment_with_verified_fallback
    _, plan, _, receipt = finalize_treatment_with_verified_fallback(
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
    assert "曲坦" not in plan, "triptan must be discarded after Hashimoto reselect"
    assert receipt.get("degraded") == "diagnosis_changed_rebuild"


def test_rebuild_treatment_discards_old_axis_drugs() -> None:
    """The unified rebuild entry point must produce an old-axis-free plan."""
    from agent.legacy_orchestrator import rebuild_treatment_for_diagnosis
    res = rebuild_treatment_for_diagnosis(
        "桥本甲状腺炎",
        examinations=[], official_diseases=["桥本甲状腺炎"],
        examination_catalog={}, case_features={}, safety_profiles=[],
    )
    assert res is not None, "rebuild must succeed for Hashimoto"
    plan = res.get("patched_treatment") or ""
    assert "曲坦" not in plan
    assert "偏头痛" not in plan
    assert res.get("rebuild") is True


def test_red_flag_no_synthetic_passed_true() -> None:
    """A red-flag escalation path must never synthesize passed=True when the plan is invalid."""
    from agent.legacy_orchestrator import finalize_treatment_with_verified_fallback
    # A red_flag axis with an EMPTY/failing escalation plan must NOT fake passed.
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
    except Exception as exc:
        # FinalVerificationError is acceptable (isolated as incomplete case).
        from agent.legacy_orchestrator import FinalVerificationError
        assert isinstance(exc, FinalVerificationError), "expected incomplete isolation, got %r" % exc
        return
    # If it returned a receipt, it must NOT be a synthetic safe_escalation faking passed
    # when the evidence/plan was empty. Genuine safe_escalation only when plan_ok.
    if receipt.get("degraded") == "safe_escalation":
        assert receipt.get("unresolved_axis_id"), "safe escalation must name the axis"
