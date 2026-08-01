"""Reference-catalog coverage gaps derived from reusable clinical mechanisms."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_evidence_backed_diagnosis_guard,
    build_name_map,
    extract_intake_facts,
    flatten_disease_catalog,
    flatten_examination_catalog,
    has_methemoglobin_risk_pattern,
    has_oxidant_exposure,
    has_pediatric_airway_compression_pattern,
    has_symptomatic_anemia_loss_pattern,
    intake_facts_text,
    load_disease_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    marker_present_not_negated,
    normalize_name,
    open_coverage_gaps,
    required_differential_from_case,
    select_diagnosis_axes,
    select_disease_candidates,
    select_exam_plan,
)


def state(text: str) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": text}],
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
        for item in select_disease_candidates(case, load_disease_catalog(), limit=12)
    ]


def planned_exams(case: dict) -> list[str]:
    disease_catalog = load_disease_catalog()
    examination_catalog = load_examination_catalog()
    knowledge = load_knowledge_registry()
    case_axes = axes(case)
    case["diagnosis_axes"] = case_axes
    candidates = select_disease_candidates(case, disease_catalog, limit=12)
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


def assert_official_outputs(case: dict, disease: str, examinations: set[str]) -> None:
    assert disease in set(flatten_disease_catalog(load_disease_catalog()))
    official_exams = set(flatten_examination_catalog(load_examination_catalog()))
    assert examinations <= official_exams


def test_cyanosis_with_oxidant_opens_dyshemoglobin_gap() -> None:
    case = state(
        "2岁男童从出生起唇甲发青，近三天使用利多卡因喷剂后明显加重，"
        "哭闹时更青，但指夹血氧读数98%。"
    )
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))
    axis_ids = {item["axis_id"] for item in axes(case)}
    names = candidate_names(case)
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert has_methemoglobin_risk_pattern(facts_text)
    assert "cyanosis_oxidant_dyshemoglobinemia" in axis_ids
    assert "先天性高铁血红蛋白血症" in names
    assert "dyshemoglobin_oxygenation" in gap_ids
    assert {"动脉血气（ABG）", "红细胞酶检测"} <= exams
    assert_official_outputs(case, "先天性高铁血红蛋白血症", exams)


def test_cyanosis_without_oxidant_does_not_open_dyshemoglobin_axis() -> None:
    case = state("2岁孩子出生后活动时嘴唇发青，哭闹后更明显，从未接触局部麻醉药或其他氧化剂。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_methemoglobin_risk_pattern(facts_text)
    assert "cyanosis_oxidant_dyshemoglobinemia" not in {
        item["axis_id"] for item in axes(case)
    }
    assert "先天性高铁血红蛋白血症" not in required_differential_from_case(case)


def test_postposed_oxidant_denial_does_not_open_dyshemoglobin_axis() -> None:
    case = state("2岁孩子嘴唇发青，但利多卡因没用过，也没有接触其他氧化剂。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_methemoglobin_risk_pattern(facts_text)
    assert "cyanosis_oxidant_dyshemoglobinemia" not in {
        item["axis_id"] for item in axes(case)
    }


def test_postposed_medication_denial_does_not_negate_unrelated_symptoms() -> None:
    assert marker_present_not_negated("孩子咳嗽没使用过药物", ["咳嗽"])
    assert marker_present_not_negated("孩子发热没用过退烧药", ["发热"])
    assert not has_oxidant_exposure("利多卡因没用过")


def test_toddler_airway_and_swallowing_pressure_requires_chest_imaging() -> None:
    case = state(
        "2岁幼儿三个月来呼吸声越来越重，平躺时明显加重并有喘鸣，"
        "每次进食都会干呕，偶尔嘴唇发暗。"
    )
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))
    axis_ids = {item["axis_id"] for item in axes(case)}
    names = candidate_names(case)
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert has_pediatric_airway_compression_pattern(facts_text)
    assert "pediatric_airway_esophageal_compression" in axis_ids
    assert "先天性纵隔囊肿" in names
    assert "pediatric_mediastinal_structure" in gap_ids
    assert "胸部X线检查（CXR）" in exams
    assert "胸部CT扫描（Chest CT）" not in exams
    assert_official_outputs(case, "先天性纵隔囊肿", exams)


def test_abnormal_cxr_escalates_mediastinal_gap_to_ct() -> None:
    case = state("2岁幼儿三个月来呼吸声越来越重，平卧加重并有喘鸣，进食时反复干呕。")
    case["ordered_examinations"] = ["胸部X线检查（CXR）"]
    case["examination_results"] = {
        "胸部X线检查（CXR）": {
            "status": "abnormal",
            "result": {"所见": "纵隔影增宽，建议进一步胸部CT检查"},
        }
    }

    gaps = open_coverage_gaps(case)
    exams = set(planned_exams(case))

    assert "pediatric_mediastinal_ct" in {item["gap_id"] for item in gaps}
    assert "胸部CT扫描（Chest CT）" in exams
    assert "胸部X线检查（CXR）" not in exams


def test_normal_cxr_closes_mediastinal_structure_gap_without_ct() -> None:
    case = state("2岁幼儿三个月来呼吸声越来越重，平卧加重并有喘鸣，进食时反复干呕。")
    case["ordered_examinations"] = ["胸部X线检查（CXR）"]
    case["examination_results"] = {
        "胸部X线检查（CXR）": {
            "status": "normal",
            "result": {"所见": "纵隔影正常，未提示占位或气道受压"},
        }
    }

    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "pediatric_mediastinal_structure" not in gap_ids
    assert "pediatric_mediastinal_ct" not in gap_ids
    assert "胸部CT扫描（Chest CT）" not in exams


def test_abnormal_cxr_with_negated_mediastinal_findings_does_not_escalate_ct() -> None:
    case = state("2岁幼儿三个月来呼吸声越来越重，平卧加重并有喘鸣，进食时反复干呕。")
    case["ordered_examinations"] = ["胸部X线检查（CXR）"]
    case["examination_results"] = {
        "胸部X线检查（CXR）": {
            "status": "abnormal",
            "result": {
                "肺纹理": "增多",
                "纵隔占位": "未发现",
                "纵隔异常": "未提示",
                "建议": "不建议进一步胸部CT",
            },
        }
    }

    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert "pediatric_mediastinal_ct" not in gap_ids
    assert "胸部CT扫描（Chest CT）" not in exams


def test_short_upper_respiratory_illness_does_not_open_mediastinal_axis() -> None:
    case = state("5岁儿童感冒三天，流鼻涕和偶尔咳嗽，平卧不加重，进食正常，也没有喘鸣。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_pediatric_airway_compression_pattern(facts_text)
    assert "pediatric_airway_esophageal_compression" not in {
        item["axis_id"] for item in axes(case)
    }
    assert "先天性纵隔囊肿" not in required_differential_from_case(case)


def test_acute_child_airway_symptoms_do_not_open_chronic_compression_axis() -> None:
    case = state("3岁幼儿今天突然呼吸声重，平卧时加重，吃东西会干呕。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_pediatric_airway_compression_pattern(facts_text)
    assert "pediatric_airway_esophageal_compression" not in {
        item["axis_id"] for item in axes(case)
    }


def test_same_day_persistent_words_do_not_create_chronic_compression_axis() -> None:
    patient_text = "3岁幼儿今天持续咳嗽，平卧加重，进食时反复干呕。"
    case = state(patient_text)
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_pediatric_airway_compression_pattern(normalize_name(patient_text))
    assert not has_pediatric_airway_compression_pattern(facts_text)
    assert "pediatric_airway_esophageal_compression" not in {
        item["axis_id"] for item in axes(case)
    }
    assert "先天性纵隔囊肿" not in required_differential_from_case(case)


def test_adult_age_does_not_match_child_numeric_age_substrings() -> None:
    case = state("22岁男性呼吸声重三个月，平卧加重，进食时干呕。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_pediatric_airway_compression_pattern(facts_text)
    assert "pediatric_airway_esophageal_compression" not in {
        item["axis_id"] for item in axes(case)
    }


def test_symptomatic_anemia_with_loss_risk_requires_cbc_and_iron_studies() -> None:
    case = state(
        "29岁女性产后逐渐乏力，最近爬楼就气促并反复头晕，"
        "还有痔疮反复便血。"
    )
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))
    axis_ids = {item["axis_id"] for item in axes(case)}
    names = candidate_names(case)
    gap_ids = {item["gap_id"] for item in open_coverage_gaps(case)}
    exams = set(planned_exams(case))

    assert has_symptomatic_anemia_loss_pattern(facts_text)
    assert "symptomatic_anemia_chronic_blood_loss" in axis_ids
    assert "缺铁性贫血" in names
    assert "symptomatic_anemia_cbc_iron" in gap_ids
    assert {"全血细胞计数（CBC）", "铁代谢检查"} <= exams
    assert_official_outputs(case, "缺铁性贫血", exams)


def test_isolated_fatigue_does_not_force_iron_deficiency_axis() -> None:
    case = state("昨晚没睡好，今天有些乏力，没有活动后气促或头晕，也否认任何出血。")
    facts_text = normalize_name(intake_facts_text(extract_intake_facts(case)))

    assert not has_symptomatic_anemia_loss_pattern(facts_text)
    assert "symptomatic_anemia_chronic_blood_loss" not in {
        item["axis_id"] for item in axes(case)
    }
    assert "缺铁性贫血" not in required_differential_from_case(case)


def test_doctor_questions_cannot_self_create_patient_evidence() -> None:
    case = {
        "chat_history": [
            {
                "from": "doctor",
                "text": (
                    "是否使用利多卡因后紫绀加重？是否长期呼吸声重、平卧加重并进食干呕？"
                    "是否产后逐渐乏力、活动后气促、头晕并有痔疮出血？"
                ),
            },
            {"from": "patient", "text": "以上情况都没有，也没有接触这些药物。"},
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "exam_decision_trace": [],
    }
    axis_ids = {item["axis_id"] for item in axes(case)}

    assert "cyanosis_oxidant_dyshemoglobinemia" not in axis_ids
    assert "pediatric_airway_esophageal_compression" not in axis_ids
    assert "symptomatic_anemia_chronic_blood_loss" not in axis_ids


def test_objective_results_reconcile_symptom_mimics_to_supported_etiology() -> None:
    methemoglobin = state("幼儿唇甲发青，使用利多卡因后加重，常规血氧读数正常。")
    methemoglobin["ordered_examinations"] = ["动脉血气（ABG）", "红细胞酶检测"]
    methemoglobin["examination_results"] = {
        "动脉血气（ABG）": {
            "status": "abnormal",
            "result": {"高铁血红蛋白": "12%，明显升高"},
        },
        "红细胞酶检测": {
            "status": "abnormal",
            "result": {"细胞色素b5还原酶": "明显降低"},
        },
    }
    assert apply_evidence_backed_diagnosis_guard(
        "法洛四联症",
        methemoglobin,
        [{"disease": "法洛四联症"}, {"disease": "先天性高铁血红蛋白血症"}],
    ) == "先天性高铁血红蛋白血症"

    mediastinal = state("2岁幼儿呼吸声重三个月，平卧加重，进食干呕。")
    mediastinal["ordered_examinations"] = ["胸部CT扫描（Chest CT）"]
    mediastinal["examination_results"] = {
        "胸部CT扫描（Chest CT）": {
            "status": "abnormal",
            "result": {"所见": "前上纵隔囊性占位，气道受压约40%"},
        }
    }
    assert apply_evidence_backed_diagnosis_guard(
        "腺样体肥大",
        mediastinal,
        [{"disease": "腺样体肥大"}, {"disease": "先天性纵隔囊肿"}],
    ) == "先天性纵隔囊肿"

    anemia = state("产后逐渐乏力、活动后气促和反复头晕，并有痔疮反复便血。")
    anemia["ordered_examinations"] = ["全血细胞计数（CBC）", "铁代谢检查"]
    anemia["examination_results"] = {
        "全血细胞计数（CBC）": {
            "status": "abnormal",
            "result": {"血红蛋白": "明显降低", "形态": "小细胞低色素"},
        },
        "铁代谢检查": {
            "status": "abnormal",
            "result": {"铁蛋白": "明显降低"},
        },
    }
    assert apply_evidence_backed_diagnosis_guard(
        "慢性疲劳综合征",
        anemia,
        [{"disease": "慢性疲劳综合征"}, {"disease": "缺铁性贫血"}],
    ) == "缺铁性贫血"


def test_elevated_methemoglobin_with_normal_enzyme_does_not_force_congenital_diagnosis() -> None:
    case = state("幼儿唇甲发青，使用利多卡因后加重。")
    case["ordered_examinations"] = ["动脉血气（ABG）", "红细胞酶检测"]
    case["examination_results"] = {
        "动脉血气（ABG）": {
            "status": "abnormal",
            "result": {"高铁血红蛋白": "12%，明显升高"},
        },
        "红细胞酶检测": {
            "status": "normal",
            "result": {"细胞色素b5还原酶": "正常"},
        },
    }

    assert apply_evidence_backed_diagnosis_guard(
        "法洛四联症",
        case,
        [{"disease": "法洛四联症"}, {"disease": "先天性高铁血红蛋白血症"}],
    ) == "法洛四联症"


def test_methemoglobin_without_enzyme_evidence_does_not_force_congenital_diagnosis() -> None:
    case = state("幼儿唇甲发青，使用利多卡因后加重。")
    case["ordered_examinations"] = ["动脉血气（ABG）"]
    case["examination_results"] = {
        "动脉血气（ABG）": {
            "status": "abnormal",
            "result": {"高铁血红蛋白": "12%，明显升高"},
        }
    }

    assert apply_evidence_backed_diagnosis_guard(
        "法洛四联症",
        case,
        [{"disease": "法洛四联症"}, {"disease": "先天性高铁血红蛋白血症"}],
    ) == "法洛四联症"


def test_postposed_negative_mediastinal_results_do_not_force_cyst_diagnosis() -> None:
    result_formats = [
        {"所见": "未提示纵隔占位"},
        {"所见": "不支持纵隔占位"},
        {"所见": "未检出纵隔占位"},
        {"纵隔占位": "未发现", "气道受压": "未提示"},
    ]
    for result in result_formats:
        case = state("2岁幼儿呼吸声重三个月，平卧加重，进食干呕。")
        case["ordered_examinations"] = ["胸部CT扫描（Chest CT）"]
        case["examination_results"] = {
            "胸部CT扫描（Chest CT）": {
                "status": "abnormal",
                "result": result,
            }
        }

        assert apply_evidence_backed_diagnosis_guard(
            "腺样体肥大",
            case,
            [{"disease": "腺样体肥大"}, {"disease": "先天性纵隔囊肿"}],
        ) == "腺样体肥大"


def test_normal_imaging_status_cannot_force_mediastinal_cyst() -> None:
    case = state("2岁幼儿呼吸声重三个月，平卧加重，进食干呕。")
    case["ordered_examinations"] = ["胸部CT扫描（Chest CT）"]
    case["examination_results"] = {
        "胸部CT扫描（Chest CT）": {
            "status": "normal",
            "result": {"所见": "前上纵隔囊性占位，气道受压"},
        }
    }

    assert apply_evidence_backed_diagnosis_guard(
        "腺样体肥大",
        case,
        [{"disease": "腺样体肥大"}, {"disease": "先天性纵隔囊肿"}],
    ) == "腺样体肥大"


def test_diagnosis_reconcile_does_not_override_without_objective_support() -> None:
    case = state("2岁幼儿呼吸声重三个月，平卧加重，进食干呕。")

    assert apply_evidence_backed_diagnosis_guard(
        "腺样体肥大",
        case,
        [{"disease": "腺样体肥大"}, {"disease": "先天性纵隔囊肿"}],
    ) == "腺样体肥大"

    methemoglobin = state("幼儿唇甲发青，使用利多卡因后加重。")
    methemoglobin["ordered_examinations"] = ["动脉血气（ABG）", "红细胞酶检测"]
    methemoglobin["examination_results"] = {
        "动脉血气（ABG）": {
            "status": "normal",
            "result": {"高铁血红蛋白": "未升高"},
        },
        "红细胞酶检测": {
            "status": "normal",
            "result": {"细胞色素b5还原酶": "未降低"},
        },
    }
    assert apply_evidence_backed_diagnosis_guard(
        "法洛四联症",
        methemoglobin,
        [{"disease": "法洛四联症"}, {"disease": "先天性高铁血红蛋白血症"}],
    ) == "法洛四联症"

    anemia = state("产后逐渐乏力、活动后气促和反复头晕，并有痔疮反复便血。")
    anemia["ordered_examinations"] = ["全血细胞计数（CBC）", "铁代谢检查"]
    anemia["examination_results"] = {
        "全血细胞计数（CBC）": {
            "status": "normal",
            "result": {"血红蛋白": "未降低"},
        },
        "铁代谢检查": {
            "status": "normal",
            "result": {"铁蛋白": "未降低"},
        },
    }
    assert apply_evidence_backed_diagnosis_guard(
        "慢性疲劳综合征",
        anemia,
        [{"disease": "慢性疲劳综合征"}, {"disease": "缺铁性贫血"}],
    ) == "慢性疲劳综合征"

    case["ordered_examinations"] = ["胸部CT扫描（Chest CT）"]
    case["examination_results"] = {
        "胸部CT扫描（Chest CT）": {
            "status": "normal",
            "result": {"所见": "纵隔未见囊肿或占位，气道无压迫"},
        }
    }
    assert apply_evidence_backed_diagnosis_guard(
        "腺样体肥大",
        case,
        [{"disease": "腺样体肥大"}, {"disease": "先天性纵隔囊肿"}],
    ) == "腺样体肥大"
