"""Offline regressions for high-risk etiology recall and bounded clinical control."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    apply_axis_risk_gate,
    build_name_map,
    extract_intake_facts,
    flatten_examination_catalog,
    load_examination_catalog,
    load_knowledge_registry,
    open_coverage_gaps,
    select_diagnosis_axes,
    select_exam_plan,
    select_next_clinical_action,
    select_required_intake_question,
    should_run_prefinal_axis_review,
)


def case_state(*patient_replies: str, examinations: dict | None = None) -> dict:
    history = []
    for index, reply in enumerate(patient_replies, start=1):
        history.extend(
            [
                {"from": "doctor", "text": f"问题{index}"},
                {"from": "patient", "text": reply},
            ]
        )
    return {
        "chat_history": history,
        "ordered_examinations": list((examinations or {}).keys()),
        "invalid_examinations": [],
        "examination_results": examinations or {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def axis_by_id(state: dict, axis_id: str) -> dict:
    axes = select_diagnosis_axes(extract_intake_facts(state))
    return next(axis for axis in axes if axis["axis_id"] == axis_id)


def planned_exams(state: dict) -> list[str]:
    catalog = load_examination_catalog()
    plan = select_exam_plan(
        case_state=state,
        disease_candidates=[],
        examination_catalog=catalog,
        item_name_map=build_name_map(flatten_examination_catalog(catalog)),
        exam_intent_rules=load_knowledge_registry()["exam_intent_map"],
    )
    return plan["examinations"]


def test_immunosuppressed_progressive_dyspnea_opens_lung_axis_and_not_ent_panel() -> None:
    state = case_state(
        "鼻塞流涕、咳嗽，最近气越来越不够用。",
        "长期服用免疫抑制剂和全身激素。",
    )

    axis = axis_by_id(state, "immunosuppressed_progressive_lower_respiratory_infection")
    exams = planned_exams(state)

    assert "呼吸道合胞病毒肺炎" in axis["candidate_official_names"]
    assert "胸部X线检查（CXR）" in exams
    assert "脉搏血氧饱和度监测（SpO2）" in exams
    assert "鼻咽拭子病毒核酸检测" in exams
    assert "前鼻镜检查" not in exams


def test_immunosuppressed_dyspnea_requires_infection_and_aspiration_followup() -> None:
    state = case_state(
        "咳嗽和呼吸短促，气越来越不够用。",
        "长期服用免疫抑制剂和激素。",
    )

    question = select_required_intake_question(state)

    assert "发热" in question
    assert "吞咽" in question or "呛咳" in question


def test_seizure_with_intracranial_calcifications_opens_neurocysticercosis_axis() -> None:
    state = case_state(
        "11岁，突然剧烈头痛、呕吐并抽搐。",
        examinations={
            "头颅平扫CT（Plain Head CT）": {
                "status": "abnormal",
                "result": {"所见": "颅内多发点状钙化，未见占位或急性出血"},
            }
        },
    )

    axis = axis_by_id(state, "seizure_with_intracranial_calcifications")
    question = select_required_intake_question(state)
    exams = planned_exams(state)

    assert "神经囊虫病" in axis["candidate_official_names"]
    assert "未熟猪肉" in question or "猪带绦虫" in question
    assert "囊虫抗体检测" in exams
    assert "脑电图（EEG）" in exams


def test_decompensated_liver_disease_opens_etiology_and_hcc_risk() -> None:
    state = case_state(
        "恶心、食欲差，眼睛黄了，肚子胀得厉害。",
        examinations={
            "腹部超声": {
                "status": "abnormal",
                "result": {"所见": "肝脏结节状、脾大、中量腹水并提示门静脉高压"},
            }
        },
    )

    axis = axis_by_id(state, "decompensated_cirrhosis_etiology_and_hcc_risk")
    question = select_required_intake_question(state)
    exams = planned_exams(state)

    assert "乙型肝炎（HBV感染）" in axis["candidate_official_names"]
    assert "肝细胞癌（肝癌）" in axis["candidate_official_names"]
    assert "乙肝" in question or "肝炎" in question
    assert "乙型肝炎病毒（HBV）检测组合" in exams
    assert "肝功能检查（LFTs）" in exams
    assert "凝血功能全套" in exams


def test_childhood_epilepsy_requires_developmental_and_prior_regimen_history() -> None:
    state = case_state(
        "从小就有癫痫样发作，最近三个月更频繁。",
        "以前吃左乙拉西坦，但最近不规律。",
    )

    question = select_required_intake_question(state)

    assert "发育" in question or "学习" in question
    assert "停药" in question or "效果" in question


def test_developmental_clues_promote_fragile_x_axis_without_using_it_as_case_fact() -> None:
    state = case_state(
        "从小反复癫痫样发作，最近变频繁。",
        "有学习障碍，基因检查提示FMR1异常，既往左乙拉西坦控制不佳。",
    )

    axis = axis_by_id(state, "developmental_genetic_epilepsy")

    assert "脆性X综合征" in axis["candidate_official_names"]
    assert "癫痫" in axis["candidate_official_names"]
    assert len(axis["evidence"]) >= 2


def test_axis_risk_gate_adds_respiratory_and_aspiration_safety_goals() -> None:
    result = apply_axis_risk_gate(
        "继续原免疫抑制剂和激素，居家观察。",
        {
            "positive_findings": ["免疫抑制", "进行性气短", "吞咽困难"],
            "red_flags": ["呼吸困难"],
            "organ_risk": ["误吸风险"],
            "diagnosis_axes": [
                {
                    "axis_id": "immunosuppressed_progressive_lower_respiratory_infection",
                    "source": "rule",
                    "evidence": ["免疫抑制", "进行性气短", "吞咽困难"],
                    "candidate_official_names": ["呼吸道合胞病毒肺炎"],
                    "treatment_risks": ["immunosuppressed_respiratory_infection_unclosed"],
                }
            ],
        },
        diagnosis="肺炎",
    )

    assert any(issue["code"] == "immunosuppressed_respiratory_infection_unclosed" for issue in result["issues"])
    patch_text = " ".join(result["patches"])
    assert "氧合" in patch_text or "血氧" in patch_text
    assert "防误吸" in patch_text


def test_deterministic_action_flow_and_prefinal_gate_do_not_need_llm() -> None:
    initial = case_state()
    after_chief = case_state("持续咳嗽三天。")
    after_core = case_state("持续咳嗽三天。", "无过敏，无基础病，目前未用药。")
    after_exam = case_state(
        "持续咳嗽三天。",
        "无过敏，无基础病，目前未用药。",
        examinations={"胸部X线检查（CXR）": {"status": "normal", "result": {"所见": "未见异常"}}},
    )

    assert select_next_clinical_action(initial)["action"] == "ask_patient"
    assert select_next_clinical_action(after_chief)["action"] == "ask_patient"
    assert select_next_clinical_action(after_core)["action"] == "order_examination"
    assert select_next_clinical_action(after_exam)["action"] == "final_diagnosis"
    assert should_run_prefinal_axis_review(after_exam) is False

    risky = case_state("免疫抑制治疗期间咳嗽并进行性气短。")
    assert open_coverage_gaps(risky)
    assert should_run_prefinal_axis_review(risky) is True

