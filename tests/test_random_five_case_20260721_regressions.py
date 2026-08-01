"""Regressions from the 2026-07-21 random five-case run."""

from __future__ import annotations

import pytest

from agent.legacy_orchestrator import (
    alias_to_official,
    build_name_map,
    feature_evidence_grounded,
    load_knowledge_registry,
    merge_axis_disease_candidates,
    select_disease_candidates,
    validate_axis_consult,
)


def case_state(
    patient_text: str,
    *,
    examination_results: dict | None = None,
    diagnostic_context: dict | None = None,
) -> dict:
    return {
        "chat_history": [{"from": "patient", "text": patient_text}],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": examination_results or {},
        "exam_decision_trace": [],
        "diagnostic_context": diagnostic_context or {},
    }


@pytest.mark.parametrize(
    ("official_name", "patient_text", "examination_results", "diagnostic_context", "axis"),
    [
        (
            "肺癌",
            "患者60岁，10天前开始干咳，最近频繁，偶尔有血丝痰，右侧胸痛并活动后气短。",
            {
                "胸部X线检查（CXR）": {
                    "status": "abnormal",
                    "result": {
                        "右上肺野": "右上叶见毛刺状肿块",
                        "右肺门": "右肺门影增大，提示淋巴结肿大",
                    },
                }
            },
            {
                "case_features": {
                    "symptom_clusters": [
                        {"label": "呼吸道症状伴咯血", "evidence": "干咳伴血丝痰", "confidence": "high"}
                    ],
                    "exam_evidence": [
                        {"label": "肺部占位性病变", "evidence": "右上叶毛刺状肿块", "confidence": "high"}
                    ],
                    "red_flags": [
                        {"label": "肺门淋巴结肿大", "evidence": "右肺门影增大", "confidence": "high"}
                    ],
                },
                "differential": [{"raw_name": "支气管肺癌", "rank": 1}],
                "normalization_suggestions": [
                    {
                        "raw_name": "支气管肺癌",
                        "suggested_official_name": "支气管肺癌",
                        "confidence": "high",
                        "supporting_feature_labels": ["肺部占位性病变", "肺门淋巴结肿大"],
                    }
                ],
            },
            {
                "axis_id": "lung_malignancy_suspected",
                "evidence": ["干咳", "血丝痰", "右上叶见毛刺状肿块"],
                "candidate_official_names": ["肺癌", "原发性支气管肺癌"],
            },
        ),
        (
            "抽动障碍",
            "孩子6岁，频繁眨眼并清嗓子，动作前眼睛和喉咙发痒发紧，紧张时加重，安静时减轻。",
            {
                "神经系统检查": {
                    "status": "normal",
                    "result": {"结论": "神经系统检查未见异常"},
                }
            },
            {
                "case_features": {
                    "symptom_clusters": [
                        {"label": "运动性抽动", "evidence": "频繁眨眼", "confidence": "high"},
                        {"label": "发声性抽动", "evidence": "清嗓子", "confidence": "high"},
                    ]
                },
                "differential": [{"raw_name": "妥瑞氏症", "rank": 1}],
                "normalization_suggestions": [
                    {
                        "raw_name": "妥瑞氏症",
                        "suggested_official_name": "抽动障碍",
                        "confidence": "medium",
                        "supporting_feature_labels": ["运动性抽动", "发声性抽动"],
                    }
                ],
            },
            {
                "axis_id": "tic_disorder_axis",
                "evidence": ["频繁眨眼", "清嗓子", "紧张时加重"],
                "candidate_official_names": ["抽动障碍"],
            },
        ),
    ],
)
def test_evidence_backed_exact_official_axis_enters_candidate_pool(
    official_name: str,
    patient_text: str,
    examination_results: dict,
    diagnostic_context: dict,
    axis: dict,
) -> None:
    case = case_state(
        patient_text,
        examination_results=examination_results,
        diagnostic_context=diagnostic_context,
    )
    knowledge = load_knowledge_registry()
    catalog = {"target": [official_name]}

    consult = validate_axis_consult(
        {"diagnosis_axes": [axis]},
        case_state=case,
        official_diseases=[official_name],
        alias_rules=knowledge["alias_map"],
    )

    assert consult["diagnosis_axes"]
    validated_axis = consult["diagnosis_axes"][0]
    assert validated_axis["promotable_candidate_official_names"] == [official_name]
    candidates = merge_axis_disease_candidates(
        [],
        diagnosis_axes=consult["diagnosis_axes"],
        disease_catalog=catalog,
        limit=8,
    )
    assert [(item["disease"], item["source"]) for item in candidates] == [
        (official_name, "diagnosis_axis_llm")
    ]
    case["diagnosis_axes"] = consult["diagnosis_axes"]
    selected = select_disease_candidates(case, catalog, limit=8)
    selected_candidate = next(item for item in selected if item["disease"] == official_name)
    assert selected_candidate["matched_evidence"] == validated_axis["evidence"]
    assert selected_candidate["evidence_polarity"] == "positive"

    tampered_axis = dict(axis, evidence=["不存在的证据一", "不存在的证据二"])
    rejected = validate_axis_consult(
        {"diagnosis_axes": [tampered_axis]},
        case_state=case,
        official_diseases=[official_name],
        alias_rules=knowledge["alias_map"],
    )
    assert rejected["diagnosis_axes"] == []


def test_exact_unrelated_official_axis_still_cannot_hitchhike() -> None:
    case = case_state("患者发热并咳嗽两天。")
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "generic_fever_axis",
                    "evidence": ["发热", "咳嗽"],
                    "candidate_official_names": ["骨髓炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=["骨髓炎"],
        alias_rules=[],
    )

    assert consult["diagnosis_axes"]
    assert consult["diagnosis_axes"][0]["promotable_candidate_official_names"] == []


def test_surface_form_llm_axis_promotes_to_official_catalog_name() -> None:
    """LLM often emits 支气管肺癌 while catalog official is 肺癌; must still promote."""
    case = case_state(
        "患者60岁，10天前开始干咳，最近频繁，偶尔有血丝痰，右侧胸痛并活动后气短。",
        examination_results={
            "胸部X线检查（CXR）": {
                "status": "abnormal",
                "result": {
                    "右上肺野": "右上叶见毛刺状肿块",
                    "右肺门": "右肺门影增大，提示淋巴结肿大",
                },
            }
        },
    )
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "lung_malignancy_suspected",
                    "evidence": ["干咳", "血丝痰", "右上叶见毛刺状肿块"],
                    "candidate_official_names": ["支气管肺癌"],
                    "priority": "high",
                    "clinical_role": "current_problem",
                    "status": "suspected",
                }
            ]
        },
        case_state=case,
        official_diseases=["肺癌", "肺炎"],
        alias_rules=[],
    )
    assert consult["diagnosis_axes"]
    axis = consult["diagnosis_axes"][0]
    assert "肺癌" in axis["candidate_official_names"]
    assert axis["promotable_candidate_official_names"] == ["肺癌"]
    merged = merge_axis_disease_candidates(
        [],
        diagnosis_axes=consult["diagnosis_axes"],
        disease_catalog={"呼吸科": ["肺癌", "肺炎"]},
        limit=8,
    )
    assert any(item["disease"] == "肺癌" for item in merged)


def test_fabricated_context_feature_labels_cannot_promote_unrelated_axis() -> None:
    case = case_state(
        "患者发热并咳嗽两天。",
        diagnostic_context={
            "case_features": {
                "symptom_clusters": [
                    {"label": "局部骨痛", "evidence": "胫骨局部压痛", "confidence": "high"}
                ],
                "exam_evidence": [
                    {"label": "骨感染影像", "evidence": "MRI提示骨髓炎", "confidence": "high"}
                ],
            },
            "differential": [{"raw_name": "骨髓炎", "rank": 1}],
            "normalization_suggestions": [
                {
                    "raw_name": "骨髓炎",
                    "suggested_official_name": "骨髓炎",
                    "confidence": "high",
                    "supporting_feature_labels": ["局部骨痛", "骨感染影像"],
                }
            ],
        },
    )
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "fabricated_bone_infection_axis",
                    "evidence": ["发热", "咳嗽"],
                    "candidate_official_names": ["骨髓炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=["骨髓炎"],
        alias_rules=[],
    )

    assert consult["diagnosis_axes"]
    assert consult["diagnosis_axes"][0]["promotable_candidate_official_names"] == []


def test_real_four_gram_cannot_ground_fabricated_context_features() -> None:
    case = case_state(
        "患者发热并咳嗽两天。",
        diagnostic_context={
            "case_features": {
                "symptom_clusters": [
                    {
                        "label": "局部骨痛",
                        "evidence": "患者发热后出现胫骨局部压痛",
                        "confidence": "high",
                    }
                ],
                "exam_evidence": [
                    {
                        "label": "骨感染影像",
                        "evidence": "发热并咳后MRI提示骨髓炎",
                        "confidence": "high",
                    }
                ],
            },
            "differential": [{"raw_name": "骨髓炎", "rank": 1}],
            "normalization_suggestions": [
                {
                    "raw_name": "骨髓炎",
                    "suggested_official_name": "骨髓炎",
                    "confidence": "high",
                    "supporting_feature_labels": ["局部骨痛", "骨感染影像"],
                }
            ],
        },
    )
    consult = validate_axis_consult(
        {
            "diagnosis_axes": [
                {
                    "axis_id": "fabricated_bone_infection_axis",
                    "evidence": ["发热", "咳嗽"],
                    "candidate_official_names": ["骨髓炎"],
                }
            ]
        },
        case_state=case,
        official_diseases=["骨髓炎"],
        alias_rules=[],
    )

    assert consult["diagnosis_axes"]
    assert consult["diagnosis_axes"][0]["promotable_candidate_official_names"] == []


def test_feature_evidence_grounding_preserves_supported_insertions_and_conjunctions() -> None:
    assert feature_evidence_grounded(
        "右上叶毛刺状肿块",
        "胸片示右上叶见毛刺状肿块。",
    )
    assert feature_evidence_grounded(
        "干咳伴血丝痰",
        "患者持续干咳，偶有血丝痰。",
    )


def test_negative_exam_label_does_not_create_joint_effusion_candidate() -> None:
    normal_case = case_state(
        "踝后方疼痛。",
        examination_results={
            "肌肉骨骼超声": {
                "status": "normal",
                "result": {"关节积液": "无关节积液"},
            }
        },
    )
    abnormal_case = case_state(
        "踝关节肿痛。",
        examination_results={
            "肌肉骨骼超声": {
                "status": "abnormal",
                "result": {"关节积液": "可见关节积液"},
            }
        },
    )
    catalog = {"骨科": ["关节积液"]}

    assert select_disease_candidates(normal_case, catalog, limit=8) == []
    abnormal = select_disease_candidates(abnormal_case, catalog, limit=8)
    assert [item["disease"] for item in abnormal] == ["关节积液"]
    assert abnormal[0]["matched_evidence"]
    assert abnormal[0]["evidence_polarity"] == "positive"


@pytest.mark.parametrize(
    "result_text",
    [
        "关节积液已消失",
        "既往关节积液已吸收",
        "疑为关节积液",
        "倾向关节积液",
    ],
)
def test_noncurrent_or_uncertain_exam_does_not_create_joint_effusion_candidate(
    result_text: str,
) -> None:
    case = case_state(
        "踝后方不适。",
        examination_results={
            "肌肉骨骼超声": {
                "status": "abnormal",
                "result": {"结论": result_text},
            }
        },
    )

    assert select_disease_candidates(case, {"骨科": ["关节积液"]}, limit=8) == []


@pytest.mark.parametrize(
    "result_text",
    [
        "可见少量关节积液",
        "关节腔内中量积液",
        "较既往关节积液增多",
    ],
)
def test_current_positive_exam_keeps_joint_effusion_candidate(result_text: str) -> None:
    case = case_state(
        "踝关节肿痛。",
        examination_results={
            "肌肉骨骼超声": {
                "status": "abnormal",
                "result": {"结论": result_text},
            }
        },
    )

    candidates = select_disease_candidates(case, {"骨科": ["关节积液"]}, limit=8)
    assert [item["disease"] for item in candidates] == ["关节积液"]


def test_normal_field_in_mixed_abnormal_panel_does_not_create_candidate() -> None:
    case = case_state(
        "例行复查。",
        examination_results={
            "全血细胞计数（CBC）": {
                "status": "abnormal",
                "result": {
                    "白细胞": "5.0×10^9/L",
                    "血小板": "50×10^9/L（降低）",
                },
            }
        },
    )
    catalog = {"血液科": ["白细胞减少症", "血小板减少症"]}

    names = {
        item["disease"] for item in select_disease_candidates(case, catalog, limit=8)
    }
    assert "白细胞减少症" not in names
    assert "血小板减少症" in names


def test_normal_eye_screening_note_does_not_create_eye_disease_candidates() -> None:
    normal_case = case_state(
        "孩子频繁眨眼并清嗓子。",
        examination_results={
            "外眼检查": {
                "result": {
                    "结膜": "结膜清晰，无充血或分泌物",
                    "眼红光反射": "双眼可见，明亮且对称（用于筛查视网膜母细胞瘤/白内障）。",
                    "眼位": "无斜视",
                    "视网膜": "视网膜平伏",
                },
            }
        },
    )
    abnormal_case = case_state(
        "孩子眼红并有分泌物。",
        examination_results={
            "外眼检查": {
                "status": "abnormal",
                "result": {"结膜": "结膜充血伴脓性分泌物"},
            }
        },
    )
    catalog = {"眼科": ["结膜炎", "白内障", "斜视", "视网膜母细胞瘤", "视网膜脱离"]}

    normal_names = {
        item["disease"] for item in select_disease_candidates(normal_case, catalog, limit=8)
    }
    assert normal_names == set()
    abnormal_names = {
        item["disease"] for item in select_disease_candidates(abnormal_case, catalog, limit=8)
    }
    assert "结膜炎" in abnormal_names


@pytest.mark.parametrize(
    ("raw_name", "official_name"),
    [
        ("支气管肺癌", "肺癌"),
        ("原发性支气管肺癌", "肺癌"),
        ("复发性单纯疱疹", "单纯疱疹病毒感染"),
        ("唇疱疹", "单纯疱疹病毒感染"),
        ("herpes labialis", "单纯疱疹病毒感染"),
    ],
)
def test_verified_clinical_aliases_reach_official_catalog(
    raw_name: str,
    official_name: str,
) -> None:
    aliases = load_knowledge_registry()["alias_map"]
    official_map = build_name_map(["肺癌", "单纯疱疹病毒感染"])

    assert alias_to_official(raw_name, aliases, official_map) == official_name
