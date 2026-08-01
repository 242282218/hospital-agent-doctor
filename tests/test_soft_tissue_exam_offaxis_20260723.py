from __future__ import annotations

from agent.legacy_orchestrator import (
    build_name_map,
    exam_applicable_to_case,
    exams_for_intent,
    flatten_examination_catalog,
    has_acute_lower_extremity_soft_tissue_infection_pattern,
    load_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    select_exam_plan,
    should_suppress_exam,
)


CATALOG = {
    "实验室": [
        "全血细胞计数（CBC）",
        "C反应蛋白（CRP）",
        "抗磷脂抗体（APA）组合检测",
        "抗核抗体（ANA）谱",
    ],
    "影像": [
        "软组织超声",
        "四肢血管超声",
        "前列腺超声",
        "胸部X线检查（CXR）",
    ],
    "体格检查": ["直肠指检（DRE）", "口咽部检查"],
    "病原": ["咽拭子培养", "尿培养"],
}


def _plan(
    patient_text: str,
    *,
    axes: list[dict] | None = None,
    disease_candidates: list[dict] | None = None,
    max_items: int = 6,
    examination_catalog: dict | None = None,
) -> dict:
    catalog = examination_catalog or CATALOG
    knowledge = load_knowledge_registry()
    return select_exam_plan(
        case_state={
            "chat_history": [{"from": "patient", "text": patient_text}],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
        },
        disease_candidates=disease_candidates or [],
        diagnosis_axes=axes or [],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=max_items,
    )


def test_patient_00435_fake_dvt_intent_prefers_severity_not_apa() -> None:
    text = (
        "大概两天前左小腿突然开始不对劲。先是脚踝附近发红发烫，然后慢慢往上肿，摸着很痛。"
        "浑身发冷又发热。我有高血压、膝关节炎和慢性腿肿，还有足癣。对青霉素过敏。"
    )
    assert has_acute_lower_extremity_soft_tissue_infection_pattern(text)
    gap_ids = {
        item["gap_id"]
        for item in open_coverage_gaps(
            {
                "chat_history": [{"from": "patient", "text": text}],
                "ordered_examinations": [],
                "examination_results": {},
            }
        )
    }
    assert "lower_extremity_soft_tissue_infection_severity" in gap_ids

    result = _plan(
        text,
        axes=[
            {
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "status": "suspected",
                "priority": "high",
                "clinical_role": "current_problem",
                "exam_intents": [
                    "下肢软组织感染严重度评估",
                    "鉴别深静脉血栓与软组织感染",
                ],
            }
        ],
    )
    exams = set(result["examinations"])
    assert exams & {"全血细胞计数（CBC）", "C反应蛋白（CRP）", "软组织超声"}
    assert "抗磷脂抗体（APA）组合检测" not in exams
    assert should_suppress_exam("抗磷脂抗体（APA）组合检测", text) is True


def test_soft_tissue_intent_map_excludes_bare_thrombosis_apa() -> None:
    knowledge = load_knowledge_registry()
    rules = knowledge["exam_intent_map"]
    dvt_soft = set(exams_for_intent("鉴别深静脉血栓与软组织感染", rules))
    assert dvt_soft & {"全血细胞计数（CBC）", "C反应蛋白（CRP）", "软组织超声"}
    assert "抗磷脂抗体（APA）组合检测" not in dvt_soft

    apa = set(exams_for_intent("评估血栓和妊娠风险", rules))
    assert "抗磷脂抗体（APA）组合检测" in apa
    assert "抗磷脂抗体（APA）组合检测" in set(exams_for_intent("抗磷脂", rules))


def test_sle_positive_control_still_allows_apa() -> None:
    text = "日晒后脸上红斑皮疹，手腕和手指关节肿痛晨僵，反复口腔溃疡、脱发、低热。"
    case_state = {
        "chat_history": [{"from": "patient", "text": text}],
        "examination_results": {
            "抗核抗体（ANA）谱": {"result": {"ANA": "1:1280 阳性", "抗Sm抗体": "阳性"}},
            "补体成分分析": {"result": {"C3": "降低", "C4": "降低"}},
            "尿液分析（UA）": {"result": {"尿蛋白": "++"}},
        },
        "ordered_examinations": [],
    }
    knowledge = load_knowledge_registry()
    catalog = load_examination_catalog()
    result = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "系统性红斑狼疮", "score": 100, "source": "test"}],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=8,
    )
    assert "抗磷脂抗体（APA）组合检测" in result["examinations"]
    combined = text + " ANA 补体 尿蛋白"
    assert should_suppress_exam("抗磷脂抗体（APA）组合检测", combined) is False


def test_pediatric_airway_abscess_wording_blocks_prostate_ultrasound() -> None:
    text = "2岁幼儿昨晚突然发烧，喉咙剧痛不肯吃东西，还流口水。声音变闷，下巴下面有点肿。"
    assert exam_applicable_to_case("前列腺超声", text) is False
    assert should_suppress_exam("前列腺超声", text) is True

    knowledge = load_knowledge_registry()
    rules = knowledge["exam_intent_map"]
    assert "前列腺超声" not in set(exams_for_intent("有无脓肿形成", rules))
    assert "前列腺超声" not in set(exams_for_intent("评估有无脓肿形成", rules))

    result = _plan(
        text,
        axes=[
            {
                "axis_id": "pediatric_deep_pharyngeal_airway_danger",
                "status": "suspected",
                "priority": "red_flag",
                "exam_intents": ["儿童气道危险评估", "有无脓肿形成"],
            }
        ],
    )
    assert "前列腺超声" not in result["examinations"]


def test_adult_male_prostatitis_allows_prostate_ultrasound() -> None:
    text = "45岁男性尿频尿急尿痛伴会阴痛两天，发热寒战，排尿困难差点尿不出来。"
    assert exam_applicable_to_case("前列腺超声", text) is True
    assert should_suppress_exam("前列腺超声", text) is False

    knowledge = load_knowledge_registry()
    catalog = load_examination_catalog()
    result = select_exam_plan(
        case_state={
            "chat_history": [{"from": "patient", "text": text}],
            "ordered_examinations": [],
            "examination_results": {},
        },
        disease_candidates=[{"disease": "急性细菌性前列腺炎", "score": 100, "source": "test"}],
        diagnosis_axes=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        diagnosis_exam_profiles=knowledge["diagnosis_exam_profiles"],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=8,
    )
    assert "前列腺超声" in result["examinations"]
