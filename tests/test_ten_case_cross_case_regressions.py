"""Cross-case regressions from the 2026-07-16 ten-case evaluation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from agent.legacy_orchestrator import (
    MyDoctorAgent,
    alias_to_official,
    build_name_map,
    extract_intake_facts,
    final_verifier,
    flatten_disease_catalog,
    flatten_examination_catalog,
    exams_for_intent,
    explicit_name_scope,
    has_consumed_high_risk_seafood,
    inject_axis_differentials,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    match_standard_name,
    merge_diagnosis_axes,
    merge_axis_disease_candidates,
    normalize_candidates_from_diagnostic_context,
    open_coverage_gaps,
    result_value_without_reference,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
    select_next_clinical_action,
    targeted_result_pair_status,
    urine_culture_evidence_status,
    validate_axis_consult,
)


def state(*patient_texts: str) -> dict:
    return {
        "chat_history": [
            {"from": "patient", "text": text}
            for text in patient_texts
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }


def axes(case: dict) -> list[dict]:
    return select_diagnosis_axes(extract_intake_facts(case))


def candidate_names(case: dict) -> list[str]:
    case["diagnosis_axes"] = axes(case)
    return [
        item["disease"]
        for item in select_disease_candidates(case, load_disease_catalog(), limit=16)
    ]


def planned_exams(case: dict) -> list[str]:
    disease_catalog = load_disease_catalog()
    examination_catalog = load_examination_catalog()
    knowledge = load_knowledge_registry()
    case_axes = axes(case)
    case["diagnosis_axes"] = case_axes
    candidates = select_disease_candidates(case, disease_catalog, limit=16)
    return select_exam_plan(
        case_state=case,
        disease_candidates=candidates,
        diagnosis_axes=case_axes,
        examination_catalog=examination_catalog,
        item_name_map=build_name_map(flatten_examination_catalog(examination_catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=5,
    )["examinations"]


def issue_codes(result: dict) -> set[str]:
    return {item["code"] for item in result["issues"]}


def test_parenthesized_official_name_resolves_short_name_and_abbreviation() -> None:
    official_map = build_name_map(["室间隔缺损（VSD）"])

    assert match_standard_name("室间隔缺损", official_map) == "室间隔缺损（VSD）"
    assert match_standard_name("VSD", official_map) == "室间隔缺损（VSD）"


def test_parenthesized_alias_keeps_full_abbreviation_and_rejects_fragments() -> None:
    official_map = build_name_map(
        [
            "（1,3）-β-D-葡聚糖检测（G试验）",
            "肺通气/灌注显像（V/Q）",
            "X三体综合征（47,XXX）",
        ]
    )

    assert match_standard_name("G试验", official_map) == "（1,3）-β-D-葡聚糖检测（G试验）"
    assert match_standard_name("V/Q", official_map) == "肺通气/灌注显像（V/Q）"
    assert match_standard_name("47,XXX", official_map) == "X三体综合征（47,XXX）"
    for invalid_fragment in ["1", "3", "1,3", "V", "Q", "47", "XXX"]:
        assert match_standard_name(invalid_fragment, official_map) == ""


def test_verified_clinical_aliases_reach_context_and_axis_candidates() -> None:
    disease_catalog = load_disease_catalog()
    official_diseases = flatten_disease_catalog(disease_catalog)
    official_map = build_name_map(official_diseases)
    alias_rules = load_knowledge_registry()["alias_map"]
    case = state("7岁儿童一年来逐渐出现夜盲，黄昏经常撞物，阳光下畏光。")
    diagnostic_context = {
        "case_features": {
            "demographics": [{"label": "儿童", "evidence": "7岁儿童", "confidence": "high"}],
            "symptom_clusters": [
                {"label": "进行性夜盲", "evidence": "一年来逐渐出现夜盲，黄昏经常撞物", "confidence": "high"}
            ],
        },
        "differential": [{"raw_name": "视网膜色素变性", "rank": 1}],
    }

    assert alias_to_official("视网膜色素变性", alias_rules, official_map) == "遗传性视网膜营养不良"
    context_candidates = normalize_candidates_from_diagnostic_context(
        diagnostic_context,
        literal_candidates=[],
        disease_catalog=disease_catalog,
        official_disease_map=official_map,
        alias_rules=alias_rules,
        limit=8,
        trusted_case_text="7岁儿童一年来逐渐出现夜盲，黄昏经常撞物。",
    )
    assert "遗传性视网膜营养不良" in {item["disease"] for item in context_candidates}

    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "pediatric_night_blindness",
                    "evidence": ["7岁儿童", "黄昏经常撞物"],
                    "candidate_official_names": ["视网膜色素变性"],
                    "exam_intents": ["儿童进行性夜盲视网膜功能评估"],
                }
            ]
        },
        case_state=case,
        official_diseases=official_diseases,
        alias_rules=alias_rules,
    )
    assert "遗传性视网膜营养不良" in consult["diagnosis_axes"][0]["candidate_official_names"]


def test_unsubstantiated_diagnostic_context_differentials_do_not_enter_candidates() -> None:
    disease_catalog = load_disease_catalog()
    official_map = build_name_map(flatten_disease_catalog(disease_catalog))
    alias_rules = load_knowledge_registry()["alias_map"]

    for raw_name in ["骨髓炎", "慢性化脓性中耳炎"]:
        candidates = normalize_candidates_from_diagnostic_context(
            {"case_features": {}, "differential": [{"raw_name": raw_name, "rank": 1}]},
            literal_candidates=[],
            disease_catalog=disease_catalog,
            official_disease_map=official_map,
            alias_rules=alias_rules,
            limit=8,
            trusted_case_text="患者仅有短暂头晕，无骨痛、耳流脓或听力下降。",
        )
        assert candidates == []


def test_diagnostic_context_rejects_two_character_near_neighbor_overlap() -> None:
    disease_catalog = load_disease_catalog()
    official_map = build_name_map(flatten_disease_catalog(disease_catalog))
    alias_rules = load_knowledge_registry()["alias_map"]
    cases = [
        ("骨髓炎", "骨髓穿刺结果正常，患者没有骨痛或局部红肿。"),
        ("慢性化脓性中耳炎", "皮肤伤口有化脓，患者没有耳流脓或听力下降。"),
    ]

    for raw_name, trusted_text in cases:
        candidates = normalize_candidates_from_diagnostic_context(
            {"case_features": {}, "differential": [{"raw_name": raw_name, "rank": 1}]},
            literal_candidates=[],
            disease_catalog=disease_catalog,
            official_disease_map=official_map,
            alias_rules=alias_rules,
            limit=8,
            trusted_case_text=trusted_text,
        )
        assert candidates == []


def test_diagnostic_context_rejects_explicitly_excluded_full_names() -> None:
    disease_catalog = load_disease_catalog()
    official_map = build_name_map(flatten_disease_catalog(disease_catalog))
    alias_rules = load_knowledge_registry()["alias_map"]
    for raw_name, trusted_text in [
        ("室间隔缺损", "已排除室间隔缺损。"),
        ("白内障", "排除白内障。"),
    ]:
        candidates = normalize_candidates_from_diagnostic_context(
            {"case_features": {}, "differential": [{"raw_name": raw_name, "rank": 1}]},
            literal_candidates=[],
            disease_catalog=disease_catalog,
            official_disease_map=official_map,
            alias_rules=alias_rules,
            limit=8,
            trusted_case_text=trusted_text,
        )
        assert candidates == []


def test_verified_cross_case_aliases_map_to_official_catalog_names() -> None:
    official_map = build_name_map(flatten_disease_catalog(load_disease_catalog()))
    alias_rules = load_knowledge_registry()["alias_map"]
    expected = {
        "慢性化脓性中耳炎": "化脓性中耳炎",
        "军团菌肺炎": "军团菌病",
        "踝关节外侧副韧带损伤": "踝关节扭伤",
        "副溶血性弧菌肠炎": "副溶血性弧菌食物中毒",
    }

    for raw_name, official_name in expected.items():
        assert alias_to_official(raw_name, alias_rules, official_map) == official_name
    assert alias_to_official("副溶血性弧菌感染", alias_rules, official_map) == ""


def test_doctor_screening_question_and_patient_denial_do_not_pollute_candidates() -> None:
    case = {
        "chat_history": [
            {
                "from": "doctor",
                "text": "请补充是否有高血压、1型或2型糖尿病，以及目前用药。",
            },
            {
                "from": "patient",
                "text": "没有高血压，也没有糖尿病，目前没有长期用药。",
            },
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
    }

    names = {
        item["disease"]
        for item in select_disease_candidates(case, load_disease_catalog(), limit=32)
    }

    assert not any("糖尿病" in name or "高血压" in name for name in names)


def test_doctor_question_cannot_validate_llm_axis_after_patient_denial() -> None:
    case = {
        "chat_history": [
            {"from": "doctor", "text": "是否反复耳流脓并伴听力下降？"},
            {"from": "patient", "text": "没有耳流脓，听力也没有下降。"},
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
    }
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "suppurative_ear_axis",
                    "evidence": ["耳流脓", "听力下降"],
                    "candidate_official_names": ["慢性化脓性中耳炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        alias_rules=load_knowledge_registry()["alias_map"],
    )

    assert consult["diagnosis_axes"] == []


def test_unrelated_verified_alias_is_not_promotable_from_generic_evidence() -> None:
    case = state("患者发热并咳嗽两天，无耳部症状。")
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "generic_infection_axis",
                    "evidence": ["发热", "咳嗽"],
                    "candidate_official_names": ["慢性化脓性中耳炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        alias_rules=load_knowledge_registry()["alias_map"],
    )

    assert consult["diagnosis_axes"]
    assert consult["diagnosis_axes"][0]["promotable_candidate_official_names"] == []


def test_duplicate_llm_evidence_cannot_satisfy_two_evidence_gate() -> None:
    case = state("患者仅有发热。")
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "duplicated_evidence_axis",
                    "evidence": ["发热", "发热"],
                    "candidate_official_names": ["慢性化脓性中耳炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        alias_rules=load_knowledge_registry()["alias_map"],
    )

    assert consult["diagnosis_axes"] == []


def test_negated_evidence_sentences_cannot_validate_llm_axis() -> None:
    case = state("没有耳流脓。听力没有下降。")
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "negated_suppurative_ear_axis",
                    "evidence": ["没有耳流脓", "听力没有下降"],
                    "candidate_official_names": ["慢性化脓性中耳炎"],
                    "exam_intents": ["化脓性中耳病变直视评估"],
                }
            ]
        },
        case_state=case,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        alias_rules=load_knowledge_registry()["alias_map"],
    )

    assert consult["diagnosis_axes"] == []


def test_evidence_validated_llm_axis_candidate_enters_final_pool() -> None:
    catalog = load_disease_catalog()
    merged = merge_axis_disease_candidates(
        [{"disease": "白内障", "score": 50, "source": "official_catalog", "rank": 2}],
        diagnosis_axes=[
            {
                "axis_id": "pediatric_night_blindness",
                "source": "llm",
                "validated": True,
                "evidence": ["儿童进行性夜盲", "黄昏经常撞物"],
                "candidate_official_names": ["遗传性视网膜营养不良"],
                "llm_candidate_official_names": ["遗传性视网膜营养不良"],
                "promotable_candidate_official_names": ["遗传性视网膜营养不良"],
            }
        ],
        disease_catalog=catalog,
        limit=8,
    )

    assert "遗传性视网膜营养不良" in {item["disease"] for item in merged}


def test_persisted_llm_axis_candidate_stays_below_rule_axis_priority() -> None:
    case = state("儿童视觉问题已记录，需进一步鉴别。")
    case["diagnosis_axes"] = [
        {
            "axis_id": "pediatric_night_blindness",
            "source": "llm",
            "validated": True,
            "evidence": ["儿童", "黄昏撞物"],
            "candidate_official_names": ["遗传性视网膜营养不良"],
            "llm_candidate_official_names": ["遗传性视网膜营养不良"],
            "promotable_candidate_official_names": ["遗传性视网膜营养不良"],
        }
    ]

    candidates = select_disease_candidates(case, load_disease_catalog(), limit=16)
    retinal = next(item for item in candidates if item["disease"] == "遗传性视网膜营养不良")

    assert retinal["source"] == "diagnosis_axis_llm"
    assert retinal["score"] < 90


def test_exact_unrelated_official_name_cannot_use_validated_axis_as_hitchhike() -> None:
    merged = merge_axis_disease_candidates(
        [{"disease": "流行性感冒", "score": 54, "source": "official_catalog"}],
        diagnosis_axes=[
            {
                "axis_id": "generic_fever_axis",
                "source": "llm",
                "validated": True,
                "evidence": ["发热", "咳嗽"],
                "candidate_official_names": ["骨髓炎"],
                "llm_candidate_official_names": ["骨髓炎"],
                "promotable_candidate_official_names": [],
            }
        ],
        disease_catalog=load_disease_catalog(),
        limit=8,
    )

    assert [item["disease"] for item in merged] == ["流行性感冒"]


def test_mixed_rule_and_llm_axis_keeps_candidate_source_priority_separate() -> None:
    merged_axes = merge_diagnosis_axes(
        [
            {
                "axis_id": "shared_axis",
                "source": "llm",
                "validated": True,
                "evidence": ["儿童进行性夜盲", "黄昏撞物"],
                "candidate_official_names": ["化脓性中耳炎"],
                "llm_candidate_official_names": ["化脓性中耳炎"],
                "promotable_candidate_official_names": ["化脓性中耳炎"],
            }
        ],
        [
            {
                "axis_id": "shared_axis",
                "source": "rule",
                "evidence": ["儿童进行性夜盲"],
                "candidate_official_names": ["遗传性视网膜营养不良"],
                "rule_candidate_official_names": ["遗传性视网膜营养不良"],
            }
        ],
    )
    candidates = inject_axis_differentials(
        [],
        case_state={"diagnosis_axes": merged_axes},
        disease_catalog=load_disease_catalog(),
        limit=8,
    )
    by_name = {item["disease"]: item for item in candidates}

    assert by_name["遗传性视网膜营养不良"]["score"] == 90
    assert by_name["化脓性中耳炎"]["score"] < 90


def test_axis_consult_uses_current_validated_axes_instead_of_stale_prior_llm_axes() -> None:
    agent = object.__new__(MyDoctorAgent)
    agent.official_diseases = ["室间隔缺损（VSD）", "遗传性视网膜营养不良"]
    agent.examination_catalog = {}
    agent.knowledge = {"alias_map": load_knowledge_registry()["alias_map"]}
    agent._call_llm = AsyncMock(
        return_value={
            "diagnosis_axes": [
                {
                    "axis_id": "current_retinal_axis",
                    "evidence": ["7岁儿童", "黄昏经常撞物"],
                    "candidate_official_names": ["视网膜色素变性"],
                    "exam_intents": ["儿童进行性夜盲视网膜功能评估"],
                }
            ]
        }
    )
    case = state("7岁儿童黄昏经常撞物，既往会诊还提示室间隔缺损方向。")
    case["diagnosis_axes"] = [
        {
            "axis_id": "prior_cardiac_axis",
            "source": "llm",
            "validated": True,
            "evidence": ["儿童", "室间隔缺损方向"],
            "candidate_official_names": ["室间隔缺损（VSD）"],
            "llm_candidate_official_names": ["室间隔缺损（VSD）"],
            "promotable_candidate_official_names": ["室间隔缺损（VSD）"],
        }
    ]

    result = asyncio.run(
        agent._diagnostic_axis_consult(
            case_state=case,
            disease_candidates=[],
            memory_notes=[],
            patient_id="offline",
            prompt_name="offline_axis_consult",
        )
    )

    axis_ids = {item["axis_id"] for item in result["diagnosis_axes"]}
    assert "current_retinal_axis" in axis_ids
    assert "prior_cardiac_axis" not in axis_ids
    assert axis_ids == {item["axis_id"] for item in case["diagnosis_axes"]}


def test_prior_llm_axis_is_not_persisted_when_current_consult_drops_it() -> None:
    agent = object.__new__(MyDoctorAgent)
    agent.official_diseases = ["化脓性中耳炎"]
    agent.examination_catalog = {}
    agent.knowledge = {"alias_map": load_knowledge_registry()["alias_map"]}
    agent._call_llm = AsyncMock(return_value={"diagnosis_axes": []})
    case = state("患者明确没有耳流脓，听力没有下降。")
    case["diagnosis_axes"] = [
        {
            "axis_id": "stale_suppurative_ear_axis",
            "source": "llm",
            "validated": True,
            "evidence": ["耳流脓", "听力下降"],
            "candidate_official_names": ["化脓性中耳炎"],
            "llm_candidate_official_names": ["化脓性中耳炎"],
            "promotable_candidate_official_names": ["化脓性中耳炎"],
        }
    ]

    result = asyncio.run(
        agent._diagnostic_axis_consult(
            case_state=case,
            disease_candidates=[],
            memory_notes=[],
            patient_id="offline",
            prompt_name="offline_axis_consult",
        )
    )

    assert "stale_suppurative_ear_axis" not in {
        item["axis_id"] for item in result["diagnosis_axes"]
    }


def test_colloquial_neonatal_feeding_stress_requires_echocardiography() -> None:
    case = state(
        "宝宝出生第2天开始吃奶时呼吸快、出汗，后来吃奶容易累，要停下来休息。"
        "最近安静时也喘得厉害，哭或吃奶时嘴巴周围发暗。",
        "宝宝刚出生，没有药物过敏，也没在长期用药，之前体检发现心脏有杂音。",
    )
    case["ordered_examinations"] = ["体格检查", "生命体征", "脉搏血氧饱和度监测（SpO2）"]
    case["examination_results"] = {
        "生命体征": {"status": "abnormal", "result": {"心率": "175次/分"}},
        "脉搏血氧饱和度监测（SpO2）": {"status": "normal", "result": {"SpO2": "98%"}},
    }

    axis_ids = {item["axis_id"] for item in axes(case)}
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "infant_congenital_structural_heart_disease" in axis_ids
    assert "elbow_overuse_enthesopathy" not in axis_ids
    assert "infant_chd_echocardiography" in gap_ids
    assert "超声心动图" in exams
    assert select_next_clinical_action(case)["action"] == "order_examination"


def test_negated_neonatal_cardiorespiratory_features_do_not_open_chd_axis() -> None:
    case = state("新生儿没有呼吸急促，吃奶不困难，也没有出汗或口唇发绀。")

    assert "infant_congenital_structural_heart_disease" not in {
        item["axis_id"] for item in axes(case)
    }
    assert "infant_chd_echocardiography" not in {
        item["gap_id"] for item in open_coverage_gaps(case)
    }


def test_generic_neonatal_chd_pattern_does_not_inject_rare_anatomic_subtype() -> None:
    case = state(
        "新生儿吃奶时呼吸快、出汗、容易累，哭闹时口周发暗，既往听到心脏杂音。"
    )

    names = set(candidate_names(case))
    assert "先天性心脏病" in names
    assert "三房心" not in names


def test_progressive_child_night_blindness_requires_retinal_workup_not_generic_genetics() -> None:
    case = state(
        "我今年7岁，大概一年前开始天黑时看不清，最近黄昏经常撞到东西，"
        "晚上走路要牵着大人的手，阳光太亮也觉得刺眼。"
    )
    axis_ids = {item["axis_id"] for item in axes(case)}
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "pediatric_progressive_night_blindness" in axis_ids
    assert "遗传性视网膜营养不良" in candidate_names(case)
    assert "pediatric_night_blindness_retinal_workup" in gap_ids
    assert {"眼底镜检查", "视网膜电图（ERG）"} <= exams
    assert "视野检查" not in exams
    assert "裂隙灯检查" not in exams
    assert "基因检测" not in exams


def test_night_blindness_first_line_does_not_force_full_four_exam_bundle() -> None:
    case = state("7岁儿童一年来逐渐夜盲，黄昏撞物，暗处行走困难。")
    exams = set(planned_exams(case))

    assert {"眼底镜检查", "视网膜电图（ERG）"} <= exams
    assert "视野检查" not in exams
    assert "裂隙灯检查" not in exams


def test_neonatal_structure_intent_prioritizes_echo_without_ecg_xray_bundle() -> None:
    case = state(
        "新生儿吃奶时呼吸快、出汗、容易累，哭闹时口周发暗，既往听到心脏杂音。"
    )
    exams = set(planned_exams(case))

    assert "超声心动图" in exams
    assert "心电图（ECG）" not in exams
    assert "胸部X线检查（CXR）" not in exams


def test_generic_pneumonia_wording_does_not_trigger_water_aerosol_pathogen_rule() -> None:
    exams = exams_for_intent(
        "明确肺部感染部位及严重程度以区分肺炎与支气管炎",
        load_knowledge_registry()["exam_intent_map"],
    )

    assert "病原体抗原检测" not in exams


def test_water_aerosol_severe_pneumonia_requires_imaging_and_targeted_pathogen_test() -> None:
    case = state(
        "21岁，近期在酒店接触热水浴池和集中空调冷却水。4天前突发寒战高热和肌肉酸痛，"
        "随后干咳转黄痰，气短和胸痛持续加重。"
    )
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "water_aerosol_severe_pneumonia_pathogen" in {item["axis_id"] for item in axes(case)}
    assert "water_aerosol_pneumonia_imaging_pathogen" in gap_ids
    assert {"胸部X线检查（CXR）", "病原体抗原检测"} <= exams
    assert "军团菌病" in candidate_names(case)


def test_raw_seafood_watery_diarrhea_requires_stool_culture() -> None:
    case = state(
        "吃了生蚝和冰虾后16小时突然腹部绞痛、频繁水样腹泻和呕吐，"
        "伴发冷、尿量减少。"
    )
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "seafood_acute_watery_diarrhea_pathogen" in {item["axis_id"] for item in axes(case)}
    assert "seafood_gastroenteritis_stool_pathogen" in gap_ids
    assert "粪便培养" in exams
    assert "副溶血性弧菌食物中毒" in candidate_names(case)


def test_chronic_purulent_otorrhea_requires_otoscopy_and_official_candidate() -> None:
    case = state(
        "左耳流脓和听力下降反复3个月，感冒后加重，分泌物黄绿色带血丝，"
        "伴隐痛、闷胀和耳鸣。"
    )
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "chronic_suppurative_middle_ear" in {item["axis_id"] for item in axes(case)}
    assert "化脓性中耳炎" in candidate_names(case)
    assert "suppurative_middle_ear_otoscopy" in gap_ids
    assert "耳镜检查" in exams


def test_unrelated_exposures_do_not_open_new_high_risk_axes() -> None:
    cases = [
        state("7岁儿童阳光下有些畏光，但暗处看东西正常，也没有撞物。"),
        state("成年人每天正常淋浴，目前没有发热、咳嗽、气短或胸痛。"),
        state("吃了熟虾后没有腹痛、腹泻或呕吐。"),
    ]

    blocked = {
        "pediatric_progressive_night_blindness",
        "water_aerosol_severe_pneumonia_pathogen",
        "seafood_acute_watery_diarrhea_pathogen",
    }
    for case in cases:
        assert blocked.isdisjoint({item["axis_id"] for item in axes(case)})


def test_normal_shower_or_cooked_shellfish_do_not_create_exposure_specific_axes() -> None:
    ordinary_shower = state("在家正常淋浴后出现高热、咳嗽、气短和胸痛。")
    cooked_shellfish = state("吃了烤熟的生蚝后出现腹泻和呕吐。")

    assert "water_aerosol_severe_pneumonia_pathogen" not in {
        item["axis_id"] for item in axes(ordinary_shower)
    }
    assert "seafood_acute_watery_diarrhea_pathogen" not in {
        item["axis_id"] for item in axes(cooked_shellfish)
    }


def test_seafood_exposure_requires_consumption_and_cooked_state_is_not_raw() -> None:
    cases = {
        "吃了熟的生蚝后腹泻呕吐": False,
        "清蒸生蚝后腹泻呕吐": False,
        "烤的生蚝后腹泻呕吐": False,
        "不确定是否吃过生蚝，腹泻呕吐": False,
        "不生吃生蚝，腹泻呕吐": False,
        "避免生吃生蚝，最近腹泻呕吐": False,
        "医生提醒不要吃生蚝，最近腹泻呕吐": False,
        "未曾吃过生蚝，最近腹泻呕吐": False,
        "不曾吃生蚝，最近腹泻呕吐": False,
        "不要吃生蚝，最近腹泻呕吐": False,
        "询问吃生蚝是否安全，最近腹泻呕吐": False,
        "可能吃过生蚝，最近腹泻呕吐": False,
        "好像吃过生蚝，最近腹泻呕吐": False,
        "大概吃过生蚝，最近腹泻呕吐": False,
        "风险提示食用生蚝后可能腹泻": False,
        "医生建议吃生蚝，最近腹泻呕吐": False,
        "禁止吃生蚝，最近腹泻呕吐": False,
        "医生说生食海鲜后会腹泻，本人没吃": False,
        "宣教内容：生食海鲜可致腹泻": False,
        "未煮熟生蚝后来煮熟后吃，腹泻呕吐": False,
        "生吃了蔬菜，吃了熟生蚝后腹泻呕吐": False,
        "吃了煮的生蚝后腹泻呕吐": False,
        "吃了蒸的生蚝后腹泻呕吐": False,
        "吃了蒸好的生蚝后腹泻呕吐": False,
        "吃了熟贝类后腹泻呕吐": False,
        "吃了烤制生蚝后腹泻呕吐": False,
        "吃了罐头生蚝后腹泻呕吐": False,
        "吃了充分烹饪的生蚝后腹泻呕吐": False,
        "吃了海鲜后腹泻呕吐": False,
        "吃了贝类后腹泻呕吐": False,
        "小时候吃过生蚝，最近腹泻呕吐": False,
        "家属吃了生蚝，患者最近腹泻呕吐": False,
        "朋友吃生蚝，患者只是陪同": False,
        "吃了生蚝和冰虾后腹部绞痛、水样腹泻和呕吐": True,
        "生蚝刺身后腹部绞痛、水样腹泻和呕吐": True,
        "未经加热的海鲜后腹部绞痛、水样腹泻和呕吐": True,
        "未充分加热的贝类后腹部绞痛、水样腹泻和呕吐": True,
        "生食过海鲜后腹部绞痛、水样腹泻和呕吐": True,
    }

    for text, expected in cases.items():
        assert has_consumed_high_risk_seafood(text) is expected, text


def test_cooked_shellfish_cannot_resurrect_llm_seafood_axis() -> None:
    case = state("吃了烤熟的生蚝后出现腹泻和呕吐。")
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "seafood_acute_watery_diarrhea_pathogen",
                    "evidence": ["烤熟的生蚝", "腹泻"],
                    "candidate_official_names": ["副溶血性弧菌食物中毒"],
                    "exam_intents": ["生食海鲜相关胃肠病原评估"],
                }
            ]
        },
        case_state=case,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        alias_rules=load_knowledge_registry()["alias_map"],
    )

    assert "seafood_acute_watery_diarrhea_pathogen" not in {
        item["axis_id"] for item in consult["diagnosis_axes"]
    }


def test_negative_targeted_pathogen_result_prunes_legionella_candidate() -> None:
    case = state(
        "21岁，日常淋浴后出现高热、咳嗽、气短和胸痛。"
    )
    case["ordered_examinations"] = ["胸部X线检查（CXR）", "病原体抗原检测"]
    case["examination_results"] = {
        "胸部X线检查（CXR）": {
            "status": "abnormal",
            "result": {"所见": "双肺炎性浸润"},
        },
        "病原体抗原检测": {
            "status": "abnormal",
            "result": {"军团菌尿抗原": "阴性", "甲型流感抗原": "阳性"},
        },
    }

    assert "water_aerosol_pneumonia_imaging_pathogen" not in {
        item["gap_id"] for item in open_coverage_gaps(case)
    }
    assert "军团菌病" not in candidate_names(case)


def test_negative_stool_culture_prunes_vibrio_food_poisoning_candidate() -> None:
    case = state("吃生蚝后突然腹部绞痛、频繁水样腹泻和呕吐。")
    case["ordered_examinations"] = ["粪便培养"]
    case["examination_results"] = {
        "粪便培养": {
            "status": "abnormal",
            "result": {"培养结果": "未检出副溶血性弧菌", "诺如病毒": "阳性"},
        }
    }

    assert "seafood_gastroenteritis_stool_pathogen" not in {
        item["gap_id"] for item in open_coverage_gaps(case)
    }
    assert "副溶血性弧菌食物中毒" not in candidate_names(case)


def test_normal_retinal_function_prunes_inherited_retinal_dystrophy_candidate() -> None:
    case = state("7岁儿童逐渐夜盲、黄昏撞物，检查发现维生素A缺乏。")
    case["ordered_examinations"] = ["眼底镜检查", "视网膜电图（ERG）"]
    case["examination_results"] = {
        "眼底镜检查": {"status": "normal", "result": {"眼底": "正常"}},
        "视网膜电图（ERG）": {"status": "normal", "result": {"ERG": "正常"}},
    }

    assert "pediatric_night_blindness_retinal_workup" not in {
        item["gap_id"] for item in open_coverage_gaps(case)
    }
    assert "遗传性视网膜营养不良" not in candidate_names(case)


def test_positive_target_results_with_negative_reference_ranges_keep_candidates() -> None:
    legionella = state(
        "近期在酒店接触热水浴池，随后高热、咳嗽、气短和胸痛。"
    )
    legionella["ordered_examinations"] = ["胸部X线检查（CXR）", "病原体抗原检测"]
    legionella["examination_results"] = {
        "胸部X线检查（CXR）": {"status": "abnormal", "result": {"所见": "肺炎性浸润"}},
        "病原体抗原检测": {
            "status": "abnormal",
            "result": {"军团菌尿抗原": "阳性（参考：阴性）"},
        },
    }
    vibrio = state("生吃了生蚝和生虾后突然腹部绞痛、水样腹泻和呕吐。")
    vibrio["ordered_examinations"] = ["粪便培养"]
    vibrio["examination_results"] = {
        "粪便培养": {
            "status": "abnormal",
            "result": {"副溶血性弧菌": "检出（参考：未检出）"},
        }
    }
    retinal = state("7岁儿童逐渐夜盲、黄昏撞物、暗处行走困难。")
    retinal["ordered_examinations"] = ["眼底镜检查", "视网膜电图（ERG）"]
    retinal["examination_results"] = {
        "眼底镜检查": {"status": "abnormal", "result": {"眼底": "视网膜色素改变"}},
        "视网膜电图（ERG）": {
            "status": "abnormal",
            "result": {"ERG": "杆体反应幅值降低（参考：正常）"},
        },
    }

    assert "军团菌病" in candidate_names(legionella)
    assert "副溶血性弧菌食物中毒" in candidate_names(vibrio)
    assert "遗传性视网膜营养不良" in candidate_names(retinal)


def test_targeted_result_negative_phrasing_is_not_positive_by_substring() -> None:
    for value in ["杆体反应幅值未降低", "杆体反应幅值不降低", "杆体反应幅值未升高"]:
        assert (
            targeted_result_pair_status(
                "ERG",
                value,
                payload_status="abnormal",
                target_markers=["ERG", "杆体反应"],
            )
            == "negative"
        )


def test_reference_stripping_requires_reference_value_syntax() -> None:
    assert result_value_without_reference("阳性（参考：阴性）") == "阳性"
    assert result_value_without_reference("参考者报告阳性") == "参考者报告阳性"


def test_reference_range_without_colon_cannot_flip_positive_culture() -> None:
    value = "大肠埃希菌生长（参考范围无生长）"
    assert result_value_without_reference(value) == "大肠埃希菌生长"
    assert result_value_without_reference("阳性（参考范围未见）") == "阳性"
    assert result_value_without_reference("检出（参考范围无异常）") == "检出"
    assert result_value_without_reference("异常（参考范围未发现）") == "异常"
    assert (
        urine_culture_evidence_status(
            {
                "case_text": "尿频尿急",
                "examination_results": {
                    "尿培养": {
                        "status": "abnormal",
                        "result": {"培养结果": value},
                    }
                },
            }
        )
        == "positive"
    )


def test_cardiac_irreversible_intervention_requires_positive_anatomic_evidence() -> None:
    case_features = {
        "case_text": "新生儿吃奶时呼吸快、出汗，口周发暗。",
        "patient_text": "新生儿吃奶时呼吸快、出汗，口周发暗。",
        "positive_findings": ["新生儿", "喂养困难", "口周发暗"],
        "candidate_diagnoses": ["动脉导管未闭", "室间隔缺损（VSD）"],
        "diagnosis_axes": [{"axis_id": "infant_congenital_structural_heart_disease"}],
        "examination_results": {
            "生命体征": {"status": "abnormal", "result": {"心率": "175次/分"}},
            "脉搏血氧饱和度监测（SpO2）": {"status": "normal", "result": {"SpO2": "98%"}},
        },
    }
    result = final_verifier(
        diagnosis="动脉导管未闭",
        examinations=["生命体征", "脉搏血氧饱和度监测（SpO2）"],
        treatment_plan="建议尽早行经皮导管封堵术或外科结扎，并使用布洛芬关闭动脉导管。",
        official_diseases=["动脉导管未闭", "室间隔缺损（VSD）"],
        examination_catalog={"基础检查": ["生命体征", "脉搏血氧饱和度监测（SpO2）"]},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)
    assert "封堵" not in result["patched_treatment"]
    assert "结扎" not in result["patched_treatment"]
    assert "关闭动脉导管" not in result["patched_treatment"]


def test_cataract_surgery_requires_objective_lens_opacity() -> None:
    case_features = {
        "case_text": "7岁儿童进行性夜盲，黄昏经常撞物，畏光。",
        "patient_text": "7岁儿童进行性夜盲，黄昏经常撞物，畏光。",
        "positive_findings": ["儿童进行性夜盲"],
        "candidate_diagnoses": ["白内障", "遗传性视网膜营养不良"],
        "diagnosis_axes": [{"axis_id": "pediatric_progressive_night_blindness"}],
        "examination_results": {
            "基因检测": {"status": "normal", "result": {"致病性变异": "未检测到"}},
        },
    }
    result = final_verifier(
        diagnosis="白内障",
        examinations=["基因检测"],
        treatment_plan="尽快行超声乳化吸除术并植入人工晶体，术后开展弱视训练。",
        official_diseases=["白内障", "遗传性视网膜营养不良"],
        examination_catalog={"遗传检查": ["基因检测"]},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)
    assert "超声乳化" not in result["patched_treatment"]
    assert "人工晶体" not in result["patched_treatment"]


def test_positive_confirmatory_evidence_keeps_indicated_procedure_path() -> None:
    heart_features = {
        "case_text": "新生儿喂养困难。",
        "positive_findings": ["新生儿喂养困难"],
        "candidate_diagnoses": ["动脉导管未闭"],
        "examination_results": {
            "超声心动图": {
                "status": "abnormal",
                "result": {"所见": "动脉导管未闭，左向右分流，左心容量负荷增加"},
            }
        },
    }
    heart = final_verifier(
        diagnosis="动脉导管未闭",
        examinations=["超声心动图"],
        treatment_plan="由儿童心脏专科评估经皮导管封堵术。",
        official_diseases=["动脉导管未闭"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features=heart_features,
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(heart)
    assert "封堵" in heart["patched_treatment"]

    eye_features = {
        "case_text": "儿童视力下降。",
        "positive_findings": ["视力下降"],
        "candidate_diagnoses": ["白内障"],
        "examination_results": {
            "裂隙灯检查": {
                "status": "abnormal",
                "result": {"晶状体": "中央致密混浊，遮挡视轴"},
            }
        },
    }
    eye = final_verifier(
        diagnosis="白内障",
        examinations=["裂隙灯检查"],
        treatment_plan="儿童眼科评估超声乳化吸除术联合人工晶体植入。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features=eye_features,
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(eye)
    assert "人工晶体" in eye["patched_treatment"]


def test_cardiac_procedure_requires_evidence_for_same_anatomic_lesion() -> None:
    case_features = {
        "case_text": "新生儿喂养困难。",
        "candidate_diagnoses": ["动脉导管未闭", "室间隔缺损（VSD）"],
        "examination_results": {
            "超声心动图": {
                "status": "abnormal",
                "result": {"所见": "膜周部室间隔缺损，左向右分流"},
            }
        },
    }
    result = final_verifier(
        diagnosis="动脉导管未闭",
        examinations=["超声心动图"],
        treatment_plan="建议经皮导管封堵动脉导管。",
        official_diseases=["动脉导管未闭", "室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features=case_features,
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)
    assert "封堵动脉导管" not in result["patched_treatment"]


def test_vsd_repair_without_echo_is_blocked() -> None:
    result = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="尽早安排室间隔缺损外科修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)
    assert "外科修补" not in result["patched_treatment"]


def test_uncertain_exam_wording_is_not_confirmatory_evidence() -> None:
    heart = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="建议室间隔缺损外科修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": {"提示": "需排除室间隔缺损"}}
            },
        },
        safety_profiles=[],
    )
    eye = final_verifier(
        diagnosis="白内障",
        examinations=["裂隙灯检查"],
        treatment_plan="建议超声乳化并行晶体植入术。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["白内障"],
            "examination_results": {
                "裂隙灯检查": {"status": "abnormal", "result": {"提示": "考虑白内障"}}
            },
        },
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(heart)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(eye)


def test_explicitly_excluded_findings_are_not_confirmatory_evidence() -> None:
    for finding in [
        "排除室间隔缺损",
        "已排除室间隔缺损",
        "未能排除室间隔缺损",
        "室间隔缺损已排除",
        "室间隔缺损未排除",
        "室间隔缺损尚未排除",
    ]:
        result = final_verifier(
            diagnosis="室间隔缺损（VSD）",
            examinations=["超声心动图"],
            treatment_plan="建议室间隔缺损外科修补术。",
            official_diseases=["室间隔缺损（VSD）"],
            examination_catalog={"心脏检查": ["超声心动图"]},
            exam_plan_trace=[],
            case_features={
                "candidate_diagnoses": ["室间隔缺损（VSD）"],
                "examination_results": {
                    "超声心动图": {"status": "abnormal", "result": {"结论": finding}}
                },
            },
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)

    eye = final_verifier(
        diagnosis="白内障",
        examinations=["裂隙灯检查"],
        treatment_plan="建议白内障囊外摘除术。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["白内障"],
            "examination_results": {
                "裂隙灯检查": {"status": "abnormal", "result": {"结论": "排除白内障"}}
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(eye)


def test_remote_condition_wording_does_not_hide_active_procedure_recommendation() -> None:
    result = final_verifier(
        diagnosis="动脉导管未闭",
        examinations=[],
        treatment_plan="对不能通过药物控制者建议经皮导管封堵术。",
        official_diseases=["动脉导管未闭"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["动脉导管未闭"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)
    assert "封堵" not in result["patched_treatment"]


def test_irreversible_gate_preserves_safe_treatment_fragments_in_same_sentence() -> None:
    result = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan=(
            "给予呋塞米并监测心衰，同时建议室间隔缺损外科修补术，"
            "并继续高热量少量多次喂养。"
        ),
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "外科修补" not in result["patched_treatment"]
    assert "呋塞米" in result["patched_treatment"]
    assert "监测心衰" in result["patched_treatment"]
    assert "高热量少量多次喂养" in result["patched_treatment"]


def test_additional_cardiac_and_cataract_procedure_terms_are_blocked() -> None:
    cardiac_plans = ["建议法洛四联症根治术。", "建议外科矫治术。"]
    for plan in cardiac_plans:
        result = final_verifier(
            diagnosis="法洛四联症",
            examinations=[],
            treatment_plan=plan,
            official_diseases=["法洛四联症"],
            examination_catalog={"心脏检查": ["超声心动图"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": ["法洛四联症"], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)

    eye = final_verifier(
        diagnosis="白内障",
        examinations=[],
        treatment_plan="建议白内障囊外摘除术。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["白内障"], "examination_results": {}},
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(eye)


def test_standard_irreversible_procedure_synonyms_are_blocked() -> None:
    cases = [
        ("动脉导管未闭", "建议PDA结扎术。"),
        ("法洛四联症", "建议右心室流出道重建术。"),
        ("白内障", "建议人工晶状体植入术。"),
        ("白内障", "建议白内障摘除术。"),
        ("白内障", "建议白内障吸除术。"),
        ("白内障", "建议晶状体吸除术。"),
        ("白内障", "建议晶状体切除术。"),
    ]
    for diagnosis, treatment_plan in cases:
        result = final_verifier(
            diagnosis=diagnosis,
            examinations=[],
            treatment_plan=treatment_plan,
            official_diseases=[diagnosis],
            examination_catalog={"心脏检查": ["超声心动图"], "眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": [diagnosis], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result), treatment_plan


def test_tof_requires_explicit_or_combined_anatomic_evidence() -> None:
    for finding in ["右室流出道梗阻", "肺动脉狭窄", "主动脉骑跨"]:
        result = final_verifier(
            diagnosis="法洛四联症",
            examinations=["超声心动图"],
            treatment_plan="建议法洛四联症根治术。",
            official_diseases=["法洛四联症"],
            examination_catalog={"心脏检查": ["超声心动图"]},
            exam_plan_trace=[],
            case_features={
                "candidate_diagnoses": ["法洛四联症"],
                "examination_results": {
                    "超声心动图": {"status": "abnormal", "result": {"结论": finding}}
                },
            },
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result), finding

    explicit = final_verifier(
        diagnosis="法洛四联症",
        examinations=["超声心动图"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": {"结论": "明确诊断TOF"}}
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(explicit)


def test_tof_combined_anatomy_must_come_from_one_consistent_exam_result() -> None:
    contradictory = final_verifier(
        diagnosis="法洛四联症",
        examinations=["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "经胸超声心动图（TTE）": {
                    "status": "abnormal",
                    "result": {"结论": "右室流出道梗阻，未见室间隔缺损"},
                },
                "经食管超声心动图（TEE）": {
                    "status": "abnormal",
                    "result": {"结论": "室间隔缺损，未见右室流出道梗阻"},
                },
            },
        },
        safety_profiles=[],
    )
    consistent = final_verifier(
        diagnosis="法洛四联症",
        examinations=["超声心动图"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "右室流出道梗阻并室间隔缺损"},
                }
            },
        },
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(contradictory)
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(consistent)


def test_irreversible_gate_preserves_monitoring_and_rehabilitation_before_procedure() -> None:
    heart = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="加强营养支持并监测心衰后行VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    eye = final_verifier(
        diagnosis="白内障",
        examinations=[],
        treatment_plan="继续弱视训练后评估人工晶体植入。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["白内障"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "营养支持" in heart["patched_treatment"]
    assert "监测心衰" in heart["patched_treatment"]
    assert "VSD修补" not in heart["patched_treatment"]
    assert "弱视训练" in eye["patched_treatment"]
    assert "人工晶体" not in eye["patched_treatment"]


def test_historical_or_postoperative_procedure_mentions_are_not_active() -> None:
    cases = [
        ("法洛四联症", "不建议根治术，继续监测。"),
        ("法洛四联症", "不需要根治术，继续监测。"),
        ("法洛四联症", "不考虑根治术，继续监测。"),
        ("法洛四联症", "既往行法洛四联症根治术，目前随访。"),
        ("法洛四联症", "法洛四联症根治术后定期随访。"),
        ("白内障", "术后监测人工晶体位置并复查眼压。"),
        ("白内障", "人工晶体位置监测并复查眼压。"),
        ("白内障", "已植入人工晶体，继续弱视训练。"),
        ("白内障", "已完成白内障囊外摘除术，继续弱视训练。"),
        ("白内障", "白内障囊外摘除术后继续弱视训练。"),
    ]

    for diagnosis, treatment_plan in cases:
        result = final_verifier(
            diagnosis=diagnosis,
            examinations=[],
            treatment_plan=treatment_plan,
            official_diseases=[diagnosis],
            examination_catalog={"心脏检查": ["超声心动图"], "眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": [diagnosis], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result), treatment_plan


def test_completed_or_declined_procedures_are_not_active_but_new_plans_are() -> None:
    inactive_cases = [
        ("法洛四联症", "法洛四联症根治术已完成，目前随访。"),
        ("法洛四联症", "法洛四联症根治术已顺利完成，目前随访。"),
        ("法洛四联症", "法洛四联症根治术后恢复良好，继续随访。"),
        ("法洛四联症", "目前不再行根治术，继续监测。"),
        ("法洛四联症", "患者拒绝根治术，继续监测。"),
        ("白内障", "人工晶体已植入，继续弱视训练。"),
        ("白内障", "人工晶体植入术后位置稳定，继续弱视训练。"),
        ("白内障", "人工晶体度数计算后复查。"),
    ]
    for diagnosis, treatment_plan in inactive_cases:
        result = final_verifier(
            diagnosis=diagnosis,
            examinations=[],
            treatment_plan=treatment_plan,
            official_diseases=[diagnosis],
            examination_catalog={"心脏检查": ["超声心动图"], "眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": [diagnosis], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result), treatment_plan

    planned = final_verifier(
        diagnosis="法洛四联症",
        examinations=[],
        treatment_plan="既往行姑息术，术后计划行法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["法洛四联症"], "examination_results": {}},
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(planned)


def test_procedure_gate_handles_postoperative_active_and_unrelated_procedures() -> None:
    active_after_history = final_verifier(
        diagnosis="法洛四联症",
        examinations=[],
        treatment_plan="姑息术后行法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["法洛四联症"], "examination_results": {}},
        safety_profiles=[],
    )
    active_after_completed_exam = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="超声检查已完成后行VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    unrelated = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="合并腹股沟疝，建议腹股沟疝修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    not_indicated = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="目前未达到VSD修补术指征，继续随访。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    postoperative_care = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="VSD修补术后感染需给予抗生素。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(active_after_history)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(active_after_completed_exam)
    assert "腹股沟疝修补术" in unrelated["patched_treatment"]
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(not_indicated)
    assert "抗生素" in postoperative_care["patched_treatment"]


def test_confirmatory_result_ignores_contradictory_or_historical_text() -> None:
    contradictory = final_verifier(
        diagnosis="法洛四联症",
        examinations=["超声心动图"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"所见": "右室流出道梗阻并室间隔缺损", "结论": "未见右室流出道梗阻"},
                }
            },
        },
        safety_profiles=[],
    )
    historical = final_verifier(
        diagnosis="法洛四联症",
        examinations=["超声心动图"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": {"结论": "TOF待明确，既往VSD已修补"}}
            },
        },
        safety_profiles=[],
    )
    postoperative = final_verifier(
        diagnosis="法洛四联症",
        examinations=["超声心动图"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": {"结论": "法洛四联症根治术后"}}
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(contradictory)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(historical)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(postoperative)


def test_cataract_subtypes_share_the_irreversible_procedure_gate() -> None:
    for diagnosis in ["先天性白内障", "外伤性白内障", "双眼白内障", "年龄相关性白内障"]:
        result = final_verifier(
            diagnosis=diagnosis,
            examinations=[],
            treatment_plan="建议白内障摘除术并植入人工晶体。",
            official_diseases=[diagnosis],
            examination_catalog={"眼科检查": ["裂隙灯检查"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": [diagnosis], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result), diagnosis


def test_conflicting_reports_and_same_field_conflicts_fail_closed() -> None:
    cross_report = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"],
        treatment_plan="建议VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "经胸超声心动图（TTE）": {"status": "abnormal", "result": {"结论": "室间隔缺损"}},
                "经食管超声心动图（TEE）": {"status": "normal", "result": {"结论": "未见室间隔缺损"}},
            },
        },
        safety_profiles=[],
    )
    same_field = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="建议VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": "室间隔缺损，复核未见室间隔缺损"}
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(cross_report)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(same_field)


def test_conclusion_arbitration_falls_back_only_when_target_is_absent() -> None:
    fallback = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="建议VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"所见": "室间隔缺损，左向右分流", "检查结论": "建议专科随访"},
                }
            },
        },
        safety_profiles=[],
    )
    overridden = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="建议VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"所见": "室间隔缺损", "超声结论": "未见室间隔缺损"},
                }
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(fallback)
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(overridden)


def test_generic_cardiac_procedure_families_are_blocked_without_evidence() -> None:
    cases = [
        ("动脉导管未闭", "建议经导管封堵术。"),
        ("动脉导管未闭", "拟行导管介入封堵术。"),
        ("动脉导管未闭", "建议行结扎术。"),
        ("室间隔缺损（VSD）", "建议行补片修补术。"),
        ("法洛四联症", "建议行完全矫治术。"),
        ("法洛四联症", "建议行根治性修复术。"),
    ]
    for diagnosis, treatment_plan in cases:
        result = final_verifier(
            diagnosis=diagnosis,
            examinations=[],
            treatment_plan=treatment_plan,
            official_diseases=[diagnosis],
            examination_catalog={"心脏检查": ["超声心动图"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": [diagnosis], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result), treatment_plan


def test_same_clause_cardiac_context_does_not_capture_unrelated_surgery() -> None:
    result = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="待心脏功能稳定后行腹股沟疝修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result)
    assert "腹股沟疝修补术" in result["patched_treatment"]


def test_confirmed_coexisting_lesion_procedure_is_not_removed() -> None:
    result = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="由心脏专科评估PDA封堵术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）", "动脉导管未闭"],
            "examination_results": {
                "超声心动图": {"status": "abnormal", "result": {"结论": "室间隔缺损并动脉导管未闭"}}
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result)
    assert "PDA封堵术" in result["patched_treatment"]


def test_explicit_name_scope_rejects_family_and_resolved_history() -> None:
    assert explicit_name_scope("母亲患白内障，患者无白内障", "白内障") == "negative"
    assert explicit_name_scope("既往患白内障，已行手术", "白内障") == "negative"
    assert explicit_name_scope("曾诊断室间隔缺损，现已排除", "室间隔缺损") == "negative"


def test_cancelled_or_discussed_procedures_are_not_active() -> None:
    plans = [
        "不主张行VSD修补术，继续随访。",
        "仅讨论VSD修补术风险，暂不实施。",
        "患者放弃VSD修补术，继续随访。",
        "VSD修补术已取消，继续随访。",
        "尚未决定是否行VSD修补术，继续评估。",
        "原计划VSD修补术现已取消，继续随访。",
    ]
    for treatment_plan in plans:
        result = final_verifier(
            diagnosis="室间隔缺损（VSD）",
            examinations=[],
            treatment_plan=treatment_plan,
            official_diseases=["室间隔缺损（VSD）"],
            examination_catalog={"心脏检查": ["超声心动图"]},
            exam_plan_trace=[],
            case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
            safety_profiles=[],
        )
        assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result), treatment_plan


def test_postoperative_residual_lesion_is_current_confirmatory_evidence() -> None:
    result = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=["超声心动图"],
        treatment_plan="由心脏专科评估VSD修补术。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["室间隔缺损（VSD）"],
            "examination_results": {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "既往VSD修补术后现见室间隔缺损残余分流"},
                }
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" not in issue_codes(result)


def test_tof_conflicting_positive_and_negative_reports_fail_closed() -> None:
    result = final_verifier(
        diagnosis="法洛四联症",
        examinations=["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"],
        treatment_plan="建议法洛四联症根治术。",
        official_diseases=["法洛四联症"],
        examination_catalog={"心脏检查": ["经胸超声心动图（TTE）", "经食管超声心动图（TEE）"]},
        exam_plan_trace=[],
        case_features={
            "candidate_diagnoses": ["法洛四联症"],
            "examination_results": {
                "经胸超声心动图（TTE）": {"status": "abnormal", "result": {"结论": "明确诊断TOF"}},
                "经食管超声心动图（TEE）": {"status": "normal", "result": {"结论": "未见法洛四联症"}},
            },
        },
        safety_profiles=[],
    )
    assert "irreversible_intervention_without_confirmatory_evidence" in issue_codes(result)


def test_irreversible_gate_preserves_safe_postoperative_suffixes() -> None:
    heart = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="建议VSD修补术后继续高热量喂养并监测心衰。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )
    eye = final_verifier(
        diagnosis="白内障",
        examinations=[],
        treatment_plan="建议人工晶体植入术后继续弱视训练并监测眼压。",
        official_diseases=["白内障"],
        examination_catalog={"眼科检查": ["裂隙灯检查"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["白内障"], "examination_results": {}},
        safety_profiles=[],
    )
    strengthened_care = final_verifier(
        diagnosis="室间隔缺损（VSD）",
        examinations=[],
        treatment_plan="建议VSD修补术后加强营养支持并监测心衰。",
        official_diseases=["室间隔缺损（VSD）"],
        examination_catalog={"心脏检查": ["超声心动图"]},
        exam_plan_trace=[],
        case_features={"candidate_diagnoses": ["室间隔缺损（VSD）"], "examination_results": {}},
        safety_profiles=[],
    )

    assert "VSD修补" not in heart["patched_treatment"]
    assert "高热量喂养" in heart["patched_treatment"]
    assert "监测心衰" in heart["patched_treatment"]
    assert "人工晶体植入" not in eye["patched_treatment"]
    assert "弱视训练" in eye["patched_treatment"]
    assert "监测眼压" in eye["patched_treatment"]
    assert "VSD修补" not in strengthened_care["patched_treatment"]
    assert "加强营养支持" in strengthened_care["patched_treatment"]
    assert "监测心衰" in strengthened_care["patched_treatment"]
