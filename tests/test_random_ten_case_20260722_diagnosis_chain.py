from __future__ import annotations

from agent.diagnosis_consistency import (
    enforce_candidate_pool_consistency,
    enforce_selected_diagnosis_consistency,
)
from agent.legacy_orchestrator import (
    build_name_map,
    load_knowledge_registry,
    merge_axis_disease_candidates,
    select_diagnosis_axes,
    select_disease_candidates,
    validate_axis_consult,
)


def case_state(patient_text: str, examination_results: dict | None = None) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": patient_text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": examination_results or {},
        "exam_decision_trace": [],
    }


def candidate_names(items: list[dict]) -> set[str]:
    return {str(item.get("disease") or "") for item in items}


def test_patient_00717_neonatal_leukocoria_opens_congenital_cataract_axis() -> None:
    case = case_state(
        "新生儿从出生起双眼白色反光，红光反射消失，无法注视和追视。"
    )
    axes = select_diagnosis_axes({
        "demographics": [{"label": "新生儿", "evidence": "新生儿"}],
        "symptom_clusters": [
            {"label": "先天白瞳", "evidence": "从出生起双眼白色反光"},
            {"label": "视觉追踪异常", "evidence": "红光反射消失且无法追视"},
        ],
    })
    axis = next(item for item in axes if item["axis_id"] == "pediatric_leukocoria_retinoblastoma_until_excluded")
    assert "先天性白内障" in axis["candidate_official_names"]
    assert "视网膜母细胞瘤" in axis["candidate_official_names"]


def test_patient_00717_white_color_fragment_cannot_recall_pityriasis_alba() -> None:
    case = case_state("新生儿从出生起双眼白色反光，红光反射消失，无法追视。")
    candidates = select_disease_candidates(case, {"眼科": ["先天性白内障"], "皮肤科": ["白色糠疹"]})
    assert "白色糠疹" not in candidate_names(candidates)


def test_comorbid_01602_hfref_axis_outranks_historical_hypotension() -> None:
    text = (
        "患者突发气短、端坐呼吸、双下肢水肿。超声心动图示LVEF 35%，"
        "LVEDD 60mm，PASP 52mmHg。最近曾因低血压调整利尿剂和β受体阻滞剂。"
    )
    axes = select_diagnosis_axes({
        "symptom_clusters": [
            {"label": "心衰淤血症状", "evidence": "端坐呼吸和双下肢水肿"},
        ],
        "organ_risk": [
            {"label": "射血分数降低", "evidence": "LVEF 35%，LVEDD 60mm，PASP 52mmHg"},
        ],
        "medication_history": [
            {"label": "历史低血压调药", "evidence": "曾因低血压调整药物"},
        ],
    })
    axis = next(item for item in axes if item["axis_id"] == "heart_failure_decompensation")
    assert axis["clinical_role"] == "current_problem"
    assert "心力衰竭" in axis["candidate_official_names"]
    assert "do_not_delay_diuresis_for_labs_only" in axis["treatment_risks"]


def test_comorbid_01871_pml_axis_keeps_aids_as_background() -> None:
    axes = select_diagnosis_axes({
        "background_conditions": [
            {"label": "HIV免疫抑制", "evidence": "HIV感染且抗病毒药漏服"},
        ],
        "symptom_clusters": [
            {"label": "亚急性多灶神经缺损", "evidence": "六周进展性偏瘫、构音障碍和共济失调"},
        ],
        "exam_evidence": [
            {"label": "非占位性白质病灶", "evidence": "MRI示多发不对称皮质下白质病灶，无强化、无占位效应"},
        ],
    })
    axis = next(item for item in axes if item["axis_id"] == "immunosuppressed_subacute_multifocal_white_matter_disease")
    assert axis["candidate_official_names"] == ["进行性多灶性白质脑病"]
    assert axis["clinical_role"] == "current_problem"


def test_patient_03763_airway_red_flags_create_red_flag_axis() -> None:
    axes = select_diagnosis_axes({
        "demographics": [{"label": "幼儿", "evidence": "2岁幼儿"}],
        "symptom_clusters": [
            {"label": "高热", "evidence": "体温39.6℃"},
            {"label": "深部咽喉气道红旗", "evidence": "流涎、拒食、吞咽困难、声音闷"},
        ],
    })
    axis = next(item for item in axes if item["axis_id"] == "pediatric_deep_pharyngeal_airway_danger")
    assert axis["priority"] == "red_flag"
    assert axis["closure_requirement"] == "safe_escalation_or_supported_official_diagnosis"


def test_hfref_and_pml_verified_aliases_reach_official_catalog() -> None:
    aliases = load_knowledge_registry()["alias_map"]
    official = build_name_map(["心力衰竭", "进行性多灶性白质脑病"])
    from agent.legacy_orchestrator import alias_to_official

    assert alias_to_official("HFrEF", aliases, official) == "心力衰竭"
    assert alias_to_official("慢性心力衰竭急性加重", aliases, official) == "心力衰竭"
    assert alias_to_official("PML", aliases, official) == "进行性多灶性白质脑病"


def test_candidate_pool_adds_supported_dominant_rule_axis_candidate() -> None:
    axes = [{
        "axis_id": "reduced_ejection_fraction_heart_failure",
        "source": "rule",
        "status": "confirmed",
        "clinical_role": "current_problem",
        "priority": "high",
        "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
        "rule_candidate_official_names": ["心力衰竭"],
        "candidate_official_names": ["心力衰竭"],
    }]
    result = enforce_candidate_pool_consistency(
        [{"disease": "低血压", "score": 90, "role": "background_condition"}],
        diagnosis_axes=axes,
        disease_catalog={"心内科": ["心力衰竭", "低血压"]},
        limit=8,
    )
    assert result.passed is True
    assert "心力衰竭" in candidate_names(list(result.candidates))


def test_selected_diagnosis_rejects_background_over_current_problem() -> None:
    axes = [{
        "axis_id": "immunosuppressed_subacute_multifocal_white_matter_disease",
        "source": "rule",
        "status": "confirmed",
        "clinical_role": "current_problem",
        "priority": "high",
        "evidence": ["亚急性多灶神经缺损", "不对称白质病灶"],
        "rule_candidate_official_names": ["进行性多灶性白质脑病"],
    }]
    decision = enforce_selected_diagnosis_consistency(
        "获得性免疫缺陷综合征（AIDS）",
        candidates=[
            {"disease": "获得性免疫缺陷综合征（AIDS）", "role": "background_condition", "score": 100},
            {"disease": "进行性多灶性白质脑病", "role": "current_problem", "score": 65},
        ],
        diagnosis_axes=axes,
    )
    assert decision.diagnosis == "进行性多灶性白质脑病"
    assert decision.reselected is True
    assert "background_overrides_current_problem" in decision.issue_codes


def test_red_flag_axis_without_official_candidate_requires_safe_escalation() -> None:
    axes = [{
        "axis_id": "pediatric_deep_pharyngeal_airway_danger",
        "source": "rule",
        "status": "suspected",
        "clinical_role": "current_problem",
        "priority": "red_flag",
        "closure_requirement": "safe_escalation_or_supported_official_diagnosis",
        "evidence": ["幼儿高热", "流涎和声音闷"],
        "rule_candidate_official_names": [],
    }]
    result = enforce_candidate_pool_consistency(
        [{"disease": "上呼吸道感染", "score": 80}],
        diagnosis_axes=axes,
        disease_catalog={"呼吸科": ["上呼吸道感染"]},
        limit=8,
    )
    assert result.safe_escalation_required is True
    assert "red_flag_axis_underspecified" in result.issue_codes


def test_case_state_patient_text_opens_pml_and_airway_rule_axes() -> None:
    pml_case = case_state(
        "HIV感染且漏服抗逆转录病毒药，使用激素和免疫抑制剂。六周进展性偏瘫、构音障碍和共济失调。"
        "MRI示多发不对称皮质下白质病灶，无强化、无占位效应。"
    )
    pml_axes = select_diagnosis_axes({}, case_state=pml_case)
    pml_axis = next(item for item in pml_axes if item["axis_id"] == "immunosuppressed_subacute_multifocal_white_matter_disease")
    assert pml_axis["candidate_official_names"] == ["进行性多灶性白质脑病"]

    airway_case = case_state(
        "2岁幼儿昨晚突然发烧，喉咙剧痛不肯吃东西，还流口水。声音变闷，下巴下面有点肿。"
    )
    airway_axes = select_diagnosis_axes({}, case_state=airway_case)
    airway_axis = next(item for item in airway_axes if item["axis_id"] == "pediatric_deep_pharyngeal_airway_danger")
    assert airway_axis["priority"] == "red_flag"


    scenarios = [
        (
            "肾动脉狭窄，血压220/130mmHg并剧烈头痛、视物模糊。",
            "renovascular_hypertension_with_target_organ_risk",
            "肾血管性高血压",
        ),
        (
            "长期面部外用激素药膏后，口周和鼻翼出现红斑丘疹、灼热紧绷。",
            "topical_steroid_associated_perioral_dermatitis",
            "口周皮炎",
        ),
        (
            "反复鲜红便血，指检见肛管内柔软有蒂结节。",
            "pedunculated_anal_lesion_with_hematochezia",
            "肛门息肉",
        ),
        (
            "RF阳性、Anti-CCP阳性，晨僵和多关节肿痛伴眼痛畏光。",
            "rheumatoid_arthritis_with_ocular_involvement",
            "类风湿关节炎",
        ),
    ]
    for text, axis_id, diagnosis in scenarios:
        axes = select_diagnosis_axes({"symptom_clusters": [{"label": "病例证据", "evidence": text}]})
        axis = next(item for item in axes if item["axis_id"] == axis_id)
        assert axis["clinical_role"] == "current_problem"
        assert axis["priority"] == "high"
        assert axis["candidate_official_names"] == [diagnosis]


def test_background_allergic_history_and_unrelated_vibrio_are_demoted() -> None:
    from agent.legacy_orchestrator import prune_unsupported_disease_candidates

    candidates = prune_unsupported_disease_candidates(
        [
            {"disease": "花粉症", "score": 80, "source": "catalog_match"},
            {"disease": "慢性鼻炎", "score": 75, "source": "catalog_match"},
            {"disease": "副溶血性弧菌食物中毒", "score": 90, "source": "catalog_match"},
        ],
        case_state("有过敏性鼻炎史。长期面部外用激素后口周红斑丘疹灼热。"),
    )
    by_name = {item["disease"]: item for item in candidates}
    assert by_name["花粉症"]["role"] == "background_history"
    assert by_name["慢性鼻炎"]["role"] == "background_history"
    assert "副溶血性弧菌食物中毒" not in by_name


def test_alias_normalizes_rectal_polyp_to_official_anal_polyp() -> None:
    from agent.legacy_orchestrator import alias_to_official

    aliases = load_knowledge_registry()["alias_map"]
    official = build_name_map(["肛门息肉"])
    assert alias_to_official("直肠息肉", aliases, official) == "肛门息肉"


def test_selector_does_not_fall_back_to_unrelated_first_candidate() -> None:
    from agent.legacy_orchestrator import select_allowed_candidate_diagnosis

    selected = select_allowed_candidate_diagnosis(
        {"normalized_diagnosis": ""},
        [
            {"disease": "副溶血性弧菌食物中毒"},
            {"disease": "化脓性扁桃体炎"},
        ],
        default_diagnosis="化脓性扁桃体炎",
    )
    assert selected == "化脓性扁桃体炎"


    from agent.legacy_orchestrator import decide_treatment_review, build_treatment_review_evidence_catalog

    original = "继续观察。"
    axes = [{
        "axis_id": "pediatric_leukocoria_retinoblastoma_until_excluded",
        "source": "rule",
        "status": "confirmed",
        "priority": "red_flag",
        "clinical_role": "current_problem",
        "evidence": ["新生儿白瞳", "红光反射消失且无法追视"],
        "candidate_official_names": ["先天性白内障", "视网膜母细胞瘤"],
    }]
    catalog = build_treatment_review_evidence_catalog(
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis="白色糠疹",
        diagnosis_axes=axes,
        verifier_issues=[],
    )
    decision = decide_treatment_review(
        {"edits": [{
            "edit_id": "replace_eye_emergency",
            "operation": "replace",
            "target": original,
            "replacement": "按白色糠疹观察。",
            "evidence_refs": ["diagnosis:final"],
        }]},
        original_treatment_plan=original,
        case_state={"chat_history": [{"from": "patient", "text": "新生儿白瞳，红光反射消失且无法追视。"}]},
        diagnosis="白色糠疹",
        diagnosis_axes=axes,
        verifier_issues=[],
        evidence_catalog=catalog,
    )
    for key in ["status", "accepted_edit_ids", "rejected_edits", "reason_codes", "before_hash", "after_hash"]:
        assert key in decision


def test_final_verifier_preserves_selected_candidate_metadata() -> None:
    from agent.legacy_orchestrator import final_verifier

    result = final_verifier(
        diagnosis="心力衰竭",
        examinations=[],
        treatment_plan="容量管理并监测电解质。",
        official_diseases=["心力衰竭", "低血压"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features={
            "diagnosis_axes": [{
                "axis_id": "reduced_ejection_fraction_heart_failure",
                "source": "rule",
                "status": "confirmed",
                "clinical_role": "current_problem",
                "priority": "high",
                "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
                "rule_candidate_official_names": ["心力衰竭"],
            }],
            "candidate_diagnoses": ["心力衰竭", "低血压"],
            "diagnosis_candidate_records": [
                {
                    "disease": "心力衰竭",
                    "role": "current_problem",
                    "priority": "high",
                    "axis_id": "reduced_ejection_fraction_heart_failure",
                    "source": "diagnosis_axis",
                    "matched_evidence": ["LVEF 35%", "端坐呼吸和水肿"],
                },
                {
                    "disease": "低血压",
                    "role": "background_history",
                    "priority": "routine",
                    "source": "catalog_match",
                },
            ],
        },
        safety_profiles=[],
    )
    assert not any(issue.get("code") == "diagnosis_conflicts_with_high_risk_axis" for issue in result["issues"])


def test_validate_single_axis_preserves_semantic_metadata() -> None:
    from agent.legacy_orchestrator import validate_single_axis

    result = validate_single_axis(
        {
            "axis_id": "hfref",
            "status": "confirmed",
            "clinical_role": "current_problem",
            "priority": "high",
            "closure_requirement": "supported_official_diagnosis",
            "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
            "candidate_official_names": ["心力衰竭"],
        },
        "LVEF 35%，端坐呼吸和水肿。",
        build_name_map(["心力衰竭"]),
        supported_official_names={"心力衰竭"},
    )
    assert result is not None
    assert result["clinical_role"] == "current_problem"
    assert result["priority"] == "high"
    assert result["closure_requirement"] == "supported_official_diagnosis"


def test_case_state_audit_keys_exist_on_consistency_summaries() -> None:
    expected = {
        "diagnosis_candidate_consistency",
        "selected_diagnosis_consistency",
        "treatment_review_decision",
        "initial_final_verifier",
        "final_verifier",
    }
    # Ensure the pure decision objects already expose the fields later persisted into case_state.
    from agent.diagnosis_consistency import enforce_candidate_pool_consistency, enforce_selected_diagnosis_consistency

    pool = enforce_candidate_pool_consistency(
        [{"disease": "心力衰竭", "score": 10}],
        diagnosis_axes=[{
            "axis_id": "reduced_ejection_fraction_heart_failure",
            "source": "rule",
            "status": "confirmed",
            "clinical_role": "current_problem",
            "priority": "high",
            "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
            "rule_candidate_official_names": ["心力衰竭"],
        }],
        disease_catalog={"心内科": ["心力衰竭"]},
        limit=8,
    )
    selected = enforce_selected_diagnosis_consistency(
        "心力衰竭",
        candidates=[{"disease": "心力衰竭", "role": "current_problem", "score": 10}],
        diagnosis_axes=[{
            "axis_id": "reduced_ejection_fraction_heart_failure",
            "source": "rule",
            "status": "confirmed",
            "clinical_role": "current_problem",
            "priority": "high",
            "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
            "rule_candidate_official_names": ["心力衰竭"],
        }],
    )
    assert pool.passed is True
    assert selected.passed is True
    assert expected  # document required case_state audit surface

