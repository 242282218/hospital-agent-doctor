from __future__ import annotations

from agent.legacy_orchestrator import (
    build_treatment_review_evidence_catalog,
    decide_treatment_review,
)


def decide(
    original: str,
    review: dict,
    *,
    diagnosis: str,
    case_state: dict,
    diagnosis_axes: list[dict],
) -> dict:
    catalog = build_treatment_review_evidence_catalog(
        case_state=case_state,
        diagnosis=diagnosis,
        diagnosis_axes=diagnosis_axes,
        verifier_issues=[],
    )
    return decide_treatment_review(
        review,
        original_treatment_plan=original,
        case_state=case_state,
        diagnosis=diagnosis,
        diagnosis_axes=diagnosis_axes,
        verifier_issues=[],
        evidence_catalog=catalog,
    )


def _00717_leukocoria_axes() -> list[dict]:
    return [{
        "axis_id": "pediatric_leukocoria_retinoblastoma_until_excluded",
        "source": "rule",
        "status": "confirmed",
        "priority": "red_flag",
        "clinical_role": "current_problem",
        "evidence": ["新生儿白瞳", "红光反射消失且无法追视"],
        "candidate_official_names": ["先天性白内障", "视网膜母细胞瘤"],
        "rule_candidate_official_names": ["先天性白内障", "视网膜母细胞瘤"],
    }]


def test_patient_00717_conflicted_final_diagnosis_cannot_replace_eye_emergency_plan() -> None:
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    decision = decide(
        original,
        {"edits": [{
            "edit_id": "replace_eye_emergency",
            "operation": "replace",
            "target": original,
            "replacement": "按白色糠疹保湿、防晒并观察。",
            "evidence_refs": ["diagnosis:final"],
        }]},
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert decision["treatment_plan"] == original
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_patient_00717_freetext_conflicted_cannot_swap_emergency_for_moisturizer() -> None:
    """Free-text review under conflicted final must not drop emergency eye disposition."""
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    decision = decide(
        original,
        {
            "treatment_plan": "按白色糠疹保湿、防晒并观察。",
            "evidence_refs": ["diagnosis:final"],
        },
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert decision["treatment_plan"] == original
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_freetext_conflicted_append_only_preserves_original_and_accepts() -> None:
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    revised = original + "\n监测生命体征"
    decision = decide(
        original,
        {
            "treatment_plan": revised,
            "evidence_refs": ["diagnosis:final"],
        },
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "accepted"
    assert original in decision["treatment_plan"]
    assert "监测生命体征" in decision["treatment_plan"]
    assert "diagnosis_axis_conflict" not in decision["reason_codes"]


def test_freetext_conflicted_cancel_append_is_rejected() -> None:
    """Keeping original as substring then canceling referral must not pass as append."""
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    revised = original + "\n取消以上全部转诊，仅保湿观察。"
    decision = decide(
        original,
        {
            "treatment_plan": revised,
            "evidence_refs": ["diagnosis:final"],
        },
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert decision["treatment_plan"] == original
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_freetext_conflicted_non_safety_append_is_rejected() -> None:
    """Conflicted free-text may not append diagnosis-specific non-safety treatment."""
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    revised = original + "\n按白色糠疹每日保湿防晒。"
    decision = decide(
        original,
        {
            "treatment_plan": revised,
            "evidence_refs": ["diagnosis:final"],
        },
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert decision["treatment_plan"] == original
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_freetext_conflicted_soft_cancel_with_referral_token_is_rejected() -> None:
    """Cancel-out that still mentions 转诊/急诊 surface tokens must fail safety-only."""
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    cases = [
        # Pure soft cancel: must hit diagnosis_axis_conflict (not only sensitive gates).
        (original + "\n请勿继续转诊。", True),
        (original + "\n可暂缓转诊。", True),
        (original + "\n请勿继续转诊，改用糠疹外用激素。", True),
        # May be rejected earlier by sensitive medication gate; original must still be kept.
        (original + "\n勿继续急诊路径，改为保湿。", False),
    ]
    for revised, require_conflict_code in cases:
        decision = decide(
            original,
            {
                "treatment_plan": revised,
                "evidence_refs": ["diagnosis:final"],
            },
            diagnosis="白色糠疹",
            case_state={
                "chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]
            },
            diagnosis_axes=axes,
        )
        assert decision["status"] == "rejected", revised
        assert decision["treatment_plan"] == original, revised
        if require_conflict_code:
            assert "diagnosis_axis_conflict" in decision["reason_codes"], (
                revised,
                decision["reason_codes"],
            )


def test_freetext_conflicted_mixed_monitor_and_diagnosis_care_is_rejected() -> None:
    """Safety token plus diagnosis-specific care is not safety-only append."""
    original = "立即转诊儿童眼科，评估先天性白内障并排除视网膜母细胞瘤。"
    axes = _00717_leukocoria_axes()
    revised = original + "\n并监测生命体征；同时按白色糠疹保湿。"
    decision = decide(
        original,
        {
            "treatment_plan": revised,
            "evidence_refs": ["diagnosis:final"],
        },
        diagnosis="白色糠疹",
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert decision["treatment_plan"] == original
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_comorbid_01602_review_cannot_remove_all_heart_failure_treatment() -> None:
    original = "根据灌注和血压谨慎滴定心衰药物，并在容量超负荷时利尿去充血。"
    axes = [{
        "axis_id": "reduced_ejection_fraction_heart_failure",
        "source": "rule",
        "status": "confirmed",
        "clinical_role": "current_problem",
        "evidence": ["LVEF 35%", "端坐呼吸和下肢水肿"],
        "candidate_official_names": ["心力衰竭"],
    }]
    decision = decide(
        original,
        {"edits": [{
            "edit_id": "remove_hf_path",
            "operation": "replace",
            "target": original,
            "replacement": "立即暂停全部利尿剂和β受体阻滞剂，仅观察低血压。",
            "evidence_refs": ["diagnosis:final"],
        }]},
        diagnosis="低血压",
        case_state={"chat_history": [{"from": "patient", "text": "LVEF 35%，端坐呼吸并下肢水肿。"}]},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "rejected"
    assert "diagnosis_axis_conflict" in decision["reason_codes"]


def test_hfref_case_features_remove_non_dihydropyridine_ccb_recommendation() -> None:
    from agent.legacy_orchestrator import sanitize_contraindicated_recommendations

    plan = "房颤心率快时给予地尔硫卓控制心室率；同时根据容量状态利尿去充血。"
    result = sanitize_contraindicated_recommendations(
        plan,
        {"case_text": "超声示LVEF 35%，端坐呼吸和双下肢水肿。"},
    )
    assert "地尔硫卓" not in result
    assert "利尿去充血" in result


def test_remove_anti_tb_dominant_claims_removes_complete_entry_and_is_idempotent() -> None:
    from agent.legacy_orchestrator import remove_anti_tb_dominant_claims

    plan = (
        "1. **抗结核治疗**：立即启动标准四联方案（异烟肼、利福平、吡嗪酰胺、乙胺丁醇），疗程通常至少6个月。\n"
        "2. 监测氧合和肝肾功能，并尽快完成病原学复核。"
    )
    result = remove_anti_tb_dominant_claims(plan)
    assert "抗结核" not in result
    assert "异烟肼" not in result
    assert "疗程通常至少" not in result
    assert "****" not in result
    assert "（）" not in result
    assert "监测氧合" in result
    assert remove_anti_tb_dominant_claims(result) == result


def test_comorbid_01869_missing_evidence_can_append_exams_but_not_delay_treatment() -> None:
    original = "由风湿科和眼科联合评估，并按活动性巩膜炎路径及时抗炎治疗。"
    axes = [{
        "axis_id": "rheumatoid_arthritis_with_ocular_involvement",
        "source": "rule",
        "status": "suspected",
        "evidence": ["RF和Anti-CCP强阳性", "眼痛充血畏光"],
        "missing_evidence": ["裂隙灯", "ESR和CRP"],
    }]
    decision = decide(
        original,
        {"edits": [
            {
                "edit_id": "delay_treatment",
                "operation": "replace",
                "target": "并按活动性巩膜炎路径及时抗炎治疗。",
                "replacement": "待裂隙灯结果后再决定是否抗炎治疗。",
                "evidence_refs": ["axis:rheumatoid_arthritis_with_ocular_involvement:missing:1"],
            },
            {
                "edit_id": "add_exam_closure",
                "operation": "append",
                "target": "",
                "replacement": "完善裂隙灯、ESR和CRP评估。",
                "evidence_refs": [
                    "axis:rheumatoid_arthritis_with_ocular_involvement:missing:1",
                    "axis:rheumatoid_arthritis_with_ocular_involvement:missing:2",
                ],
            },
        ]},
        diagnosis="类风湿关节炎",
        case_state={},
        diagnosis_axes=axes,
    )
    assert decision["status"] == "partial"
    assert decision["accepted_edit_ids"] == ["add_exam_closure"]
    assert "missing_evidence_cannot_delay_treatment" in decision["reason_codes"]
