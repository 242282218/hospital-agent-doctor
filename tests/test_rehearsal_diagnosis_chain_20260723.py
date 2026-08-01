from __future__ import annotations

from agent.diagnosis_consistency import (
    enforce_candidate_pool_consistency,
    enforce_selected_diagnosis_consistency,
)
from agent.legacy_orchestrator import (
    build_safe_escalation_plan,
    dominant_axis_for_alignment,
    exam_applicable_to_case,
    finalize_treatment_with_verified_fallback,
    has_acute_lower_extremity_soft_tissue_infection_pattern,
    merge_axis_disease_candidates,
    preferred_safe_escalation_diagnosis,
    prune_unsupported_disease_candidates,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
    validate_safe_escalation_plan,
)


def case_state(patient_text: str) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": patient_text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }


def test_lower_extremity_soft_tissue_infection_opens_cellulitis_axis() -> None:
    text = "高龄患者左小腿红肿热痛两天，伴寒战发热，对青霉素过敏。"
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text)
    axes = select_diagnosis_axes(
        {"symptom_clusters": [{"label": "病例", "evidence": text}]},
        case_state=case_state(text),
    )
    axis = next(
        item for item in axes if item["axis_id"] == "acute_lower_extremity_soft_tissue_infection"
    )
    assert axis["priority"] == "high"
    assert "蜂窝织炎" in axis["candidate_official_names"]


def test_patient_00435_colloquial_leg_infection_opens_cellulitis_axis() -> None:
    text = (
        "大概两天前左小腿突然开始不对劲。先是脚踝附近发红发烫，然后慢慢往上肿，摸着很痛。"
        "浑身发冷又发热。我有高血压、膝关节炎和慢性腿肿，还有足癣。对青霉素过敏。"
    )
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text)
    axes = select_diagnosis_axes({}, case_state=case_state(text))
    axis = next(
        item for item in axes if item["axis_id"] == "acute_lower_extremity_soft_tissue_infection"
    )
    assert "蜂窝织炎" in axis["candidate_official_names"]


def test_patient_00435_v3_wording_cold_shiver_and_fatigue_opens_cellulitis() -> None:
    text = (
        "两天前我觉得左小腿怪怪的。先是脚踝附近发红发烫，后来肿得厉害，往上延伸。"
        "走路和腿垂着时更痛，还觉得冷飕飕的，浑身没劲。脚趾之前也痒裂了。"
        "我有高血压、慢性静脉功能不全和膝关节炎。对青霉素过敏。"
    )
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text)
    axes = select_diagnosis_axes({}, case_state=case_state(text))
    axis = next(
        item for item in axes if item["axis_id"] == "acute_lower_extremity_soft_tissue_infection"
    )
    assert "蜂窝织炎" in axis["candidate_official_names"]
    candidates = select_disease_candidates(
        case_state(text),
        {"普外科": ["蜂窝织炎"], "骨科": ["关节炎", "膝关节炎"]},
        limit=12,
    )
    names = {item["disease"] for item in candidates}
    assert "蜂窝织炎" in names


def test_patient_01871_hiv_progressive_neurology_opens_pml_without_mri_text() -> None:
    text = (
        "大概6周前开始觉得特别累，还有点低烧。之后左边身体越来越没力气，拿东西笨拙，"
        "说话也找不着词。最近10天走路不稳，眼睛也看不清了。"
        "长期在吃抗逆转录病毒药、降压药和他汀，还有激素和另一种免疫抑制剂，但经常忘吃。"
        "有高血压和高脂血症，还有HIV感染。"
    )
    axes = select_diagnosis_axes({}, case_state=case_state(text))
    axis = next(
        item
        for item in axes
        if item["axis_id"] == "immunosuppressed_subacute_multifocal_white_matter_disease"
    )
    assert axis["candidate_official_names"] == ["进行性多灶性白质脑病"]


def test_soft_tissue_infection_demotes_arthritis_and_promotes_cellulitis() -> None:
    text = "左小腿红肿热痛、皮温升高，伴全身寒战发热。"
    candidates = prune_unsupported_disease_candidates(
        [
            {"disease": "关节炎", "score": 90, "source": "catalog_match"},
            {"disease": "蜂窝织炎", "score": 20, "source": "catalog_match"},
        ],
        case_state(text),
    )
    by_name = {item["disease"]: item for item in candidates}
    assert by_name["蜂窝织炎"]["score"] >= 110
    assert by_name["关节炎"]["role"] == "background_history"
    assert by_name["关节炎"]["score"] <= 25


def test_safe_escalation_aligns_diagnosis_to_dominant_axis() -> None:
    case_features = {
        "diagnosis_axes": [
            {
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "source": "rule",
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "high",
                "closure_requirement": "supported_official_diagnosis",
                "evidence": ["左小腿红肿热痛", "寒战发热"],
                "rule_candidate_official_names": ["蜂窝织炎"],
                "candidate_official_names": ["蜂窝织炎"],
            }
        ],
        "diagnosis_candidate_records": [
            {"disease": "关节炎", "role": "background_history", "score": 90},
            {
                "disease": "蜂窝织炎",
                "role": "current_problem",
                "score": 110,
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
            },
        ],
        "candidate_diagnoses": ["关节炎", "蜂窝织炎"],
    }
    diagnosis, plan, reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="关节炎",
        treatment_plan="抬高患肢。",
        reasoning="原诊断为关节炎",
        verifier_result={"passed": False, "issues": [{"problem": "diagnosis mismatch"}], "patched_treatment": ""},
        examinations=["细菌培养及鉴定"],
        official_diseases=["关节炎", "蜂窝织炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    assert diagnosis == "蜂窝织炎"
    assert receipt.get("degraded") in {
        "safe_escalation",
        "safe_escalation_unverified",
        "axis_aligned_repair",
        "conservative_fallback",
    }
    assert receipt.get("aligned_diagnosis", diagnosis) == "蜂窝织炎"
    # An unverified escalation must never claim verifier proof it does not have.
    if receipt.get("degraded") == "safe_escalation_unverified":
        assert receipt.get("passed") is False
        assert receipt.get("verification_status") == "axis_closure_only"
    if receipt.get("degraded") in {"safe_escalation", "safe_escalation_unverified"}:
        assert "急诊" in plan or "住院" in plan
        assert "acute_lower_extremity_soft_tissue_infection" in reasoning
        assert validate_safe_escalation_plan(
            plan,
            axis_id="acute_lower_extremity_soft_tissue_infection",
            evidence=["左小腿红肿热痛", "寒战发热"],
        )


def test_high_priority_axis_candidates_outrank_lexical_catalog_matches() -> None:
    merged = merge_axis_disease_candidates(
        [{"disease": "低血压", "score": 90, "source": "catalog_match", "role": "current_problem"}],
        diagnosis_axes=[
            {
                "axis_id": "reduced_ejection_fraction_heart_failure",
                "source": "rule",
                "status": "confirmed",
                "clinical_role": "current_problem",
                "priority": "high",
                "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
                "rule_candidate_official_names": ["心力衰竭"],
                "candidate_official_names": ["心力衰竭"],
            }
        ],
        disease_catalog={"心内科": ["心力衰竭", "低血压"]},
        limit=8,
    )
    by_name = {item["disease"]: item for item in merged}
    assert by_name["心力衰竭"]["score"] > by_name["低血压"]["score"]
    decision = enforce_selected_diagnosis_consistency(
        "低血压",
        candidates=merged,
        diagnosis_axes=[
            {
                "axis_id": "reduced_ejection_fraction_heart_failure",
                "source": "rule",
                "status": "confirmed",
                "clinical_role": "current_problem",
                "priority": "high",
                "evidence": ["LVEF 35%", "端坐呼吸和水肿"],
                "rule_candidate_official_names": ["心力衰竭"],
            }
        ],
    )
    assert decision.diagnosis == "心力衰竭"


def test_pediatric_male_only_exam_blocked() -> None:
    assert exam_applicable_to_case(
        "前列腺超声",
        "2岁幼儿昨晚突然发烧，喉咙剧痛不肯吃东西，还流口水。声音变闷，下巴下面有点肿。",
    ) is False
    assert exam_applicable_to_case(
        "前列腺超声",
        "45岁男性尿频尿急伴会阴痛。",
    ) is True


def test_pml_axis_and_candidate_survive_background_aids_noise() -> None:
    text = (
        "HIV感染且漏服抗逆转录病毒药，使用激素和免疫抑制剂。"
        "六周进展性偏瘫、构音障碍和共济失调。"
        "MRI示多发不对称皮质下白质病灶，无强化、无占位效应。"
        "既往混合型高脂血症。"
    )
    axes = select_diagnosis_axes({}, case_state=case_state(text))
    axis = next(
        item
        for item in axes
        if item["axis_id"] == "immunosuppressed_subacute_multifocal_white_matter_disease"
    )
    assert "进行性多灶性白质脑病" in axis["candidate_official_names"]
    candidates = select_disease_candidates(
        case_state(text),
        {
            "神经科": ["进行性多灶性白质脑病", "偏头痛"],
            "感染科": ["获得性免疫缺陷综合征（AIDS）"],
            "内分泌科": ["混合型高脂血症"],
        },
        limit=12,
    )
    names = {item["disease"] for item in candidates}
    assert "进行性多灶性白质脑病" in names
    pool = enforce_candidate_pool_consistency(
        candidates,
        diagnosis_axes=axes,
        disease_catalog={
            "神经科": ["进行性多灶性白质脑病", "偏头痛"],
            "感染科": ["获得性免疫缺陷综合征（AIDS）"],
            "内分泌科": ["混合型高脂血症"],
        },
        limit=8,
    )
    assert "进行性多灶性白质脑病" in {item["disease"] for item in pool.candidates}


def test_airway_red_flag_plan_has_safe_escalation_closure() -> None:
    plan, reasoning = build_safe_escalation_plan(
        axis_id="pediatric_deep_pharyngeal_airway_danger",
        closure_requirement="safe_escalation_or_supported_official_diagnosis",
        evidence=["幼儿高热", "流涎和声音闷"],
        existing_treatment="先观察。",
    )
    assert "急诊" in plan or "住院" in plan
    assert "专科" in plan
    assert "pediatric_deep_pharyngeal_airway_danger" in reasoning


def test_correct_cellulitis_diagnosis_is_not_marked_conflicted_for_review() -> None:
    from agent.legacy_orchestrator import treatment_review_diagnosis_consistency

    axes = [{
        "axis_id": "acute_lower_extremity_soft_tissue_infection",
        "source": "rule",
        "status": "suspected",
        "clinical_role": "current_problem",
        "priority": "high",
        "evidence": ["左小腿红肿热痛", "寒战发热"],
        "rule_candidate_official_names": ["蜂窝织炎"],
        "candidate_official_names": ["蜂窝织炎"],
    }]
    assert treatment_review_diagnosis_consistency("蜂窝织炎", axes) == "consistent"
    assert treatment_review_diagnosis_consistency("关节炎", axes) == "conflicted"


def test_finalize_never_raises_when_conservative_plan_exists() -> None:
    case_features = {
        "diagnosis_axes": [{
            "axis_id": "acute_lower_extremity_soft_tissue_infection",
            "source": "rule",
            "status": "suspected",
            "clinical_role": "current_problem",
            "priority": "high",
            "closure_requirement": "supported_official_diagnosis",
            "evidence": ["左小腿红肿热痛", "寒战发热"],
            "rule_candidate_official_names": ["蜂窝织炎"],
            "candidate_official_names": ["蜂窝织炎"],
        }],
        "diagnosis_candidate_records": [
            {
                "disease": "蜂窝织炎",
                "role": "current_problem",
                "score": 110,
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
            }
        ],
        "candidate_diagnoses": ["蜂窝织炎"],
    }
    diagnosis, plan, _reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="蜂窝织炎",
        treatment_plan="抬高患肢。",
        reasoning="x",
        verifier_result={
            "passed": False,
            "issues": [{
                "code": "diagnosis_conflicts_with_high_risk_axis",
                "problem": "最终诊断未覆盖已验证的当前主问题或红旗轴。",
                "edit": "蜂窝织炎",
                "patchable": False,
            }],
            "patched_treatment": "",
            "selected_diagnosis": "蜂窝织炎",
        },
        examinations=["全血细胞计数（CBC）"],
        official_diseases=["蜂窝织炎", "关节炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    assert diagnosis == "蜂窝织炎"
    assert plan
    # passed may only be True when a real verifier proved this exact text; an
    # unproven red-flag plan is still submitted but must declare it is unverified.
    if receipt.get("passed") is True:
        assert receipt.get("verification_status") != "axis_closure_only"
    else:
        assert receipt.get("degraded", "").endswith("_unverified")
        assert receipt.get("issues")
    assert receipt.get("degraded") in {
        "safe_escalation",
        "safe_escalation_unverified",
        "axis_aligned_repair",
        "reverified",
        "conservative_fallback",
        "conservative_fallback_unverified",
    }


def _dual_axis_empty_llm_over_soft_tissue_features() -> dict:
    """Canary 00435: empty high LLM axis listed before rule soft-tissue axis."""
    return {
        "diagnosis_axes": [
            {
                "axis_id": "acute_lower_extremity_inflammation",
                "source": "llm",
                "validated": True,
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "high",
                "closure_requirement": "supported_official_diagnosis",
                "evidence": ["左小腿红肿热痛", "寒战发热"],
                "rule_candidate_official_names": [],
                "candidate_official_names": [],
                "promotable_candidate_official_names": [],
            },
            {
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "source": "rule",
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "high",
                "closure_requirement": "supported_official_diagnosis",
                "evidence": ["下肢或小腿局灶红肿热痛", "发热寒战等全身感染征象"],
                "rule_candidate_official_names": ["蜂窝织炎"],
                "candidate_official_names": ["蜂窝织炎"],
            },
        ],
        "diagnosis_candidate_records": [
            {"disease": "关节炎", "role": "current_problem", "score": 90, "source": "catalog_match"},
            {
                "disease": "蜂窝织炎",
                "role": "current_problem",
                "score": 110,
                "source": "diagnosis_axis",
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
            },
        ],
        "candidate_diagnoses": ["关节炎", "蜂窝织炎"],
    }


def test_v3_soft_tissue_required_differential_and_candidates_include_cellulitis() -> None:
    text = (
        "两天前我觉得左小腿怪怪的。先是脚踝附近发红发烫，后来肿得厉害，往上延伸。"
        "走路和腿垂着时更痛，还觉得冷飕飕的，浑身没劲。脚趾之前也痒裂了。"
        "我有高血压、慢性静脉功能不全和膝关节炎。对青霉素过敏。"
    )
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text)
    assert "蜂窝织炎" in required_differential_from_case(case_state(text))
    candidates = select_disease_candidates(
        case_state(text),
        {"普外科": ["蜂窝织炎"], "骨科": ["关节炎", "膝关节炎"]},
        limit=12,
    )
    by_name = {item["disease"]: item for item in candidates}
    assert "蜂窝织炎" in by_name
    if "关节炎" in by_name:
        assert by_name["关节炎"].get("role") == "background_history" or int(
            by_name["关节炎"].get("score") or 0
        ) <= 25
    if "膝关节炎" in by_name:
        # History mention must not outrank soft-tissue infection current problem.
        assert int(by_name.get("蜂窝织炎", {}).get("score") or 0) >= int(
            by_name["膝关节炎"].get("score") or 0
        )


def test_dominant_axis_prefers_named_soft_tissue_over_empty_llm() -> None:
    features = _dual_axis_empty_llm_over_soft_tissue_features()
    dominant = dominant_axis_for_alignment(features)
    assert dominant is not None
    assert dominant["axis_id"] == "acute_lower_extremity_soft_tissue_infection"
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="关节炎",
        case_features=features,
        escalation_axis=dominant,
        official_diseases=["关节炎", "蜂窝织炎"],
    )
    assert preferred == "蜂窝织炎"


def test_finalize_empty_llm_high_axis_does_not_keep_arthritis() -> None:
    case_features = _dual_axis_empty_llm_over_soft_tissue_features()
    diagnosis, plan, reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="关节炎",
        treatment_plan="抬高患肢。",
        reasoning="原诊断为关节炎",
        verifier_result={
            "passed": False,
            "issues": [{"problem": "diagnosis mismatch"}],
            "patched_treatment": "",
        },
        examinations=["细菌培养及鉴定"],
        official_diseases=["关节炎", "蜂窝织炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    assert diagnosis == "蜂窝织炎"
    assert receipt.get("aligned_diagnosis", diagnosis) == "蜂窝织炎"
    assert plan
    if receipt.get("degraded") == "safe_escalation":
        assert "急诊" in plan or "住院" in plan
        assert "acute_lower_extremity_soft_tissue_infection" in reasoning or "蜂窝织炎" in (
            receipt.get("aligned_diagnosis") or diagnosis
        )


def test_soft_tissue_label_does_not_override_higher_red_flag_gi_bleed() -> None:
    """Named red_flag GI axis must outrank soft-tissue cellulitis hard-bind."""
    case_features = {
        "diagnosis_axes": [
            {
                "axis_id": "active_upper_gi_bleed",
                "source": "rule",
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "red_flag",
                "closure_requirement": "safe_escalation_or_supported_official_diagnosis",
                "evidence": ["呕血黑便", "肝硬化门脉高压背景"],
                "rule_candidate_official_names": ["上消化道出血"],
                "candidate_official_names": ["上消化道出血"],
            },
            {
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "source": "rule",
                "status": "suspected",
                "clinical_role": "current_problem",
                "priority": "high",
                "closure_requirement": "supported_official_diagnosis",
                "evidence": ["下肢或小腿局灶红肿热痛", "发热寒战等全身感染征象"],
                "rule_candidate_official_names": ["蜂窝织炎"],
                "candidate_official_names": ["蜂窝织炎"],
            },
        ],
        "diagnosis_candidate_records": [
            {
                "disease": "上消化道出血",
                "role": "current_problem",
                "score": 120,
                "source": "diagnosis_axis",
                "axis_id": "active_upper_gi_bleed",
                "priority": "red_flag",
            },
            {
                "disease": "蜂窝织炎",
                "role": "current_problem",
                "score": 110,
                "source": "diagnosis_axis",
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "priority": "high",
            },
        ],
        "candidate_diagnoses": ["上消化道出血", "蜂窝织炎"],
    }
    dominant = dominant_axis_for_alignment(case_features)
    assert dominant is not None
    assert dominant["axis_id"] == "active_upper_gi_bleed"
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis="蜂窝织炎",
        case_features=case_features,
        escalation_axis=dominant,
        official_diseases=["上消化道出血", "蜂窝织炎", "关节炎"],
    )
    assert preferred == "上消化道出血"
    diagnosis, plan, _reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="蜂窝织炎",
        treatment_plan="抬高患肢。",
        reasoning="误绑蜂窝织炎",
        verifier_result={"passed": False, "issues": [], "patched_treatment": ""},
        examinations=["血常规"],
        official_diseases=["上消化道出血", "蜂窝织炎", "关节炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )
    assert diagnosis == "上消化道出血"
    assert receipt.get("aligned_diagnosis", diagnosis) == "上消化道出血"
    assert diagnosis != "蜂窝织炎"
    assert plan


def test_department_fallback_pool_injects_axis_names_over_arthritis() -> None:
    """Department-only catalog can omit 蜂窝织炎; axis names must still realign."""
    department_diseases = ["关节炎", "膝关节炎"]
    axes = [
        {
            "axis_id": "acute_lower_extremity_soft_tissue_infection",
            "source": "rule",
            "status": "suspected",
            "clinical_role": "current_problem",
            "priority": "high",
            "closure_requirement": "supported_official_diagnosis",
            "evidence": ["左小腿红肿热痛", "寒战发热"],
            "rule_candidate_official_names": ["蜂窝织炎"],
            "candidate_official_names": ["蜂窝织炎"],
        }
    ]
    axis_records = [
        {
            "disease": name,
            "role": "current_problem",
            "source": "diagnosis_axis",
            "axis_id": "acute_lower_extremity_soft_tissue_infection",
            "priority": "high",
        }
        for axis in axes
        for name in axis["candidate_official_names"]
    ]
    fallback_candidates = (
        [{"disease": item} for item in department_diseases]
        + [{"disease": "关节炎"}]
        + axis_records
    )
    decision = enforce_selected_diagnosis_consistency(
        "关节炎",
        candidates=fallback_candidates,
        diagnosis_axes=axes,
    )
    # Either consistency reselects or preferred_safe_escalation hard-binds.
    preferred = preferred_safe_escalation_diagnosis(
        diagnosis=decision.diagnosis,
        case_features={
            "diagnosis_axes": axes,
            "diagnosis_candidate_records": fallback_candidates,
            "candidate_diagnoses": [item["disease"] for item in fallback_candidates],
        },
        escalation_axis=dominant_axis_for_alignment({"diagnosis_axes": axes}),
        official_diseases=["关节炎", "膝关节炎", "蜂窝织炎"],
    )
    assert preferred == "蜂窝织炎"
    diagnosis, _plan, _reasoning, receipt = finalize_treatment_with_verified_fallback(
        diagnosis="关节炎",
        treatment_plan="抬高患肢。",
        reasoning="科室路径误选关节炎",
        verifier_result={"passed": False, "issues": [], "patched_treatment": ""},
        examinations=["全血细胞计数（CBC）"],
        official_diseases=["关节炎", "膝关节炎", "蜂窝织炎"],
        examination_catalog={},
        exam_plan_trace=[],
        case_features={
            "diagnosis_axes": axes,
            "diagnosis_candidate_records": fallback_candidates,
            "candidate_diagnoses": [item["disease"] for item in fallback_candidates],
        },
        safety_profiles=[],
    )
    assert diagnosis == "蜂窝织炎"
    assert receipt.get("aligned_diagnosis", diagnosis) == "蜂窝织炎"


def test_soft_tissue_neighbor_joint_history_does_not_open_cellulitis_alone() -> None:
    """Near-negative: isolated chronic knee OA history is not soft-tissue infection."""
    text = "长期膝关节炎病史，无新发红肿热痛，无发热寒战。"
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text) is False
