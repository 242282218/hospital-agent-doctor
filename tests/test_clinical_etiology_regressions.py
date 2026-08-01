"""Offline regressions for high-risk etiologic diagnosis axes."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    build_name_map,
    extract_intake_facts,
    flatten_examination_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    prune_unsupported_disease_candidates,
    select_diagnosis_axes,
    select_disease_candidates,
    load_disease_catalog,
    select_exam_plan,
    select_required_intake_question,
)


def state(*replies: str, exams: dict | None = None) -> dict:
    history = []
    for index, reply in enumerate(replies):
        history.extend([{"from": "doctor", "text": f"q{index}"}, {"from": "patient", "text": reply}])
    return {
        "chat_history": history,
        "ordered_examinations": list((exams or {}).keys()),
        "invalid_examinations": [],
        "examination_results": exams or {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def axis(state_value: dict, axis_id: str) -> dict:
    return next(
        item for item in select_diagnosis_axes(extract_intake_facts(state_value))
        if item["axis_id"] == axis_id
    )


def exams(state_value: dict) -> list[str]:
    catalog = load_examination_catalog()
    return select_exam_plan(
        case_state=state_value,
        disease_candidates=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
    )["examinations"]


def test_post_spinal_surgery_positional_bilious_vomiting_recalls_sma_syndrome() -> None:
    case = state(
        "脊柱侧弯手术后，餐后半小时腹胀并吐胆汁，躺着更难受，蜷缩或左侧卧好转，大便减少。"
    )

    selected = axis(case, "post_spinal_surgery_positional_duodenal_obstruction")
    planned = exams(case)

    assert "肠系膜上动脉压迫综合征" in selected["candidate_official_names"]
    assert "上消化道造影（UGI）" in planned
    assert "增强腹部CT扫描（Abdominal CECT）" in planned
    assert "血清电解质" in planned


def test_sma_pattern_downgrades_consequence_only_diagnoses() -> None:
    case = state("脊柱侧弯手术后餐后胆汁性呕吐，蜷缩和左侧卧缓解。")
    candidates = [
        {"disease": "低血压", "score": 90, "rank": 1},
        {"disease": "肠系膜上动脉压迫综合征", "score": 40, "rank": 2},
    ]

    pruned = prune_unsupported_disease_candidates(candidates, case)

    assert pruned[0]["disease"] == "肠系膜上动脉压迫综合征"
    assert next(item for item in pruned if item["disease"] == "低血压")["role"] == "consequence"


def test_white_pupil_with_visual_tracking_loss_keeps_retinoblastoma_until_definitive_imaging() -> None:
    case = state(
        "1岁儿童左眼偶尔发白、内斜，追物能力越来越差。",
        exams={
            "红光反射检查": {"status": "normal", "result": {"所见": "本次红反射明亮对称"}},
            "斜视评估": {"status": "normal", "result": {"所见": "本次未见持续偏斜"}},
        },
    )

    selected = axis(case, "pediatric_leukocoria_retinoblastoma_until_excluded")
    planned = exams(case)

    assert "视网膜母细胞瘤" in selected["candidate_official_names"]
    assert "先天性白内障" in selected["candidate_official_names"]
    assert "眼部超声" in planned
    # First-line only until ocular ultrasound is completed; fundoscopy is add-on.
    assert "眼底镜检查" not in planned
    assert "脑和眼眶MRI" not in planned


def test_negative_screening_exam_does_not_promote_strabismus_over_leukocoria_cancer() -> None:
    case = state(
        "1岁儿童反复左眼发白、内斜和追物差。",
        exams={"红光反射检查": {"status": "normal", "result": {"所见": "检查当下红反射正常"}}},
    )
    candidates = [
        {"disease": "斜视", "score": 85, "rank": 1},
        {"disease": "视网膜母细胞瘤", "score": 50, "rank": 2},
    ]

    pruned = prune_unsupported_disease_candidates(candidates, case)

    assert pruned[0]["disease"] == "视网膜母细胞瘤"
    assert next(item for item in pruned if item["disease"] == "斜视")["role"] == "symptom_or_secondary"


def test_severe_headache_calcification_with_intracranial_pressure_clues_prioritizes_neurocysticercosis() -> None:
    case = state(
        "突发剧烈头痛、反复呕吐，弯腰加重，来自卫生条件较差地区并吃过未熟猪肉。",
        exams={
            "头颅平扫CT（Plain Head CT）": {
                "status": "abnormal",
                "result": {"所见": "多发脑实质钙化并梗阻性脑积水"},
            },
            "囊虫抗体检测": {"status": "abnormal", "result": {"所见": "特异性抗体阳性"}},
        },
    )
    candidates = [
        {"disease": "偏头痛", "score": 90, "rank": 1},
        {"disease": "神经囊虫病", "score": 55, "rank": 2},
    ]

    pruned = prune_unsupported_disease_candidates(candidates, case)

    assert pruned[0]["disease"] == "神经囊虫病"
    assert next(item for item in pruned if item["disease"] == "偏头痛")["role"] == "unsafe_symptom_closure"


def test_sma_pattern_handles_separate_scoliosis_and_recent_surgery_statements() -> None:
    case = state(
        "餐后腹胀并吐胆汁，躺着更难受，蜷缩能好点。",
        "以前有脊柱侧弯，刚做完手术。",
    )

    selected = axis(case, "post_spinal_surgery_positional_duodenal_obstruction")

    assert "肠系膜上动脉压迫综合征" in selected["candidate_official_names"]


def test_verified_axis_candidates_enter_official_disease_candidate_pool() -> None:
    cases = [
        (
            state("1岁儿童反复左眼发白、内斜，追物能力越来越差。"),
            "视网膜母细胞瘤",
        ),
        (
            state(
                "餐后腹胀并吐胆汁，躺着更难受，蜷缩能好点。",
                "以前有脊柱侧弯，刚做完手术。",
            ),
            "肠系膜上动脉压迫综合征",
        ),
    ]
    catalog = load_disease_catalog()

    for case, expected in cases:
        case["diagnosis_axes"] = select_diagnosis_axes(extract_intake_facts(case))
        candidates = select_disease_candidates(case, catalog, limit=8)
        names = [item["disease"] for item in candidates]

        assert expected in names
        assert next(item for item in candidates if item["disease"] == expected)["source"] == "diagnosis_axis"


def test_acute_pressure_headache_with_intracranial_calcifications_opens_secondary_cause_axis() -> None:
    case = state(
        "两天前突发剧烈头痛，反复呕吐，弯腰时明显加重，脑子变慢。",
        exams={
            "头颅平扫CT（Plain Head CT）": {
                "status": "abnormal",
                "result": {"颅内钙化": "多发点状钙化", "出血": "未见颅内出血"},
            }
        },
    )

    selected = axis(case, "acute_pressure_headache_intracranial_calcification")
    question = select_required_intake_question(case)
    planned = exams(case)

    assert "神经囊虫病" in selected["candidate_official_names"]
    assert "未熟猪肉" in question or "囊虫" in question
    assert "增强脑部MRI" in planned
    assert "囊虫抗体检测" in planned
    assert "脑脊液（CSF）压力测定" in planned


def test_incidental_calcification_without_pressure_headache_does_not_open_secondary_axis() -> None:
    case = state(
        "体检偶然发现单发颅内钙化，目前没有头痛、呕吐或认知变化。",
        exams={
            "头颅平扫CT（Plain Head CT）": {"status": "abnormal", "result": {"颅内钙化": "单发钙化"}}
        },
    )

    axis_ids = {item["axis_id"] for item in select_diagnosis_axes(extract_intake_facts(case))}

    assert "acute_pressure_headache_intracranial_calcification" not in axis_ids
    assert "囊虫" not in select_required_intake_question(case)
