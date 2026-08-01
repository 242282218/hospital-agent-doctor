from __future__ import annotations

from agent.legacy_orchestrator import (
    build_name_map,
    flatten_examination_catalog,
    load_knowledge_registry,
    select_exam_plan,
)


CATALOG = {
    "影像": ["胸部X线检查（CXR）", "胸部CT扫描（Chest CT）", "肾血管造影", "眼部超声"],
    "病原": ["咽拭子培养", "抗酸杆菌染色（AFB）", "脑脊液（CSF）病原体检测组合"],
    "实验室": [
        "淋巴细胞亚群分析", "HIV病毒载量检测", "C反应蛋白（CRP）",
        "B型利钠肽（BNP）", "N末端B型利钠肽原（NT-proBNP）",
        "抗中性粒细胞胞质抗体（ANCA）谱", "尿液分析（UA）", "肾功能检查（RFTs）",
    ],
    "眼科": ["裂隙灯检查", "眼底镜检查"],
    "体格检查": ["口咽部检查", "皮肤检查", "肛门镜检查"],
}


def plan(case_state: dict, axes: list[dict], *, limit: int = 5) -> dict:
    knowledge = load_knowledge_registry()
    return select_exam_plan(
        case_state=case_state,
        disease_candidates=[],
        diagnosis_axes=axes,
        examination_catalog=CATALOG,
        item_name_map=build_name_map(flatten_examination_catalog(CATALOG)),
        diagnosis_exam_profiles=[],
        exam_intent_rules=knowledge["exam_intent_map"],
        max_items=limit,
    )


def test_one_axis_defaults_to_one_first_line_exam() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "HIV免疫抑制，亚急性偏瘫；MRI示不对称白质病灶。"}]},
        [{
            "axis_id": "immunosuppressed_subacute_multifocal_white_matter_disease",
            "status": "suspected",
            "exam_intents": ["PML病原与免疫状态评估"],
        }],
    )
    assert len(result["examinations"]) == 1
    assert result["examinations"][0] == "脑脊液（CSF）病原体检测组合"


def test_semantically_equivalent_completed_exam_is_not_reordered() -> None:
    result = plan(
        {
            "chat_history": [{"from": "patient", "text": "白瞳且红光反射消失。"}],
            "ordered_examinations": ["眼部超声"],
            "examination_results": {"眼部超声": {"status": "abnormal", "result": {"晶状体": "混浊"}}},
        },
        [{
            "axis_id": "pediatric_leukocoria_retinoblastoma_until_excluded",
            "status": "suspected",
            "exam_intents": ["儿童白瞳病因评估"],
        }],
    )
    assert "眼部超声" not in result["examinations"]


def test_no_supported_exam_returns_empty_plan() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "轻微非特异不适。"}]},
        [],
    )
    assert result["examinations"] == []
    assert result["reason_codes"] == ["no_supported_exam"]




def test_hfref_axis_prefers_one_natriuretic_peptide() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "LVEF 35%，端坐呼吸、双下肢水肿。"}]},
        [{
            "axis_id": "reduced_ejection_fraction_heart_failure",
            "status": "confirmed",
            "priority": "high",
            "clinical_role": "current_problem",
            "exam_intents": ["心衰容量与器官功能评估"],
        }],
    )
    assert set(result["examinations"]) & {"B型利钠肽（BNP）", "N末端B型利钠肽原（NT-proBNP）"}
    assert not ({"B型利钠肽（BNP）", "N末端B型利钠肽原（NT-proBNP）"} <= set(result["examinations"]))


def test_pediatric_airway_axis_keeps_safety_assessment_and_strep_pathogen_in_parallel() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "2岁幼儿高热、流涎、吞咽困难、声音闷。"}]},
        [{
            "axis_id": "pediatric_deep_pharyngeal_airway_danger",
            "status": "suspected",
            "priority": "red_flag",
            "clinical_role": "current_problem",
            "exam_intents": ["儿童气道危险评估", "链球菌病原评估"],
        }],
    )
    assert result["examinations"] == ["口咽部检查", "咽拭子培养"]


def test_cavitary_tuberculosis_route_preserves_afb_before_broad_vasculitis_capacity() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "HIV免疫抑制，慢性咳嗽咯血、盗汗消瘦，胸部CT示右上叶厚壁空洞。"}]},
        [],
    )
    assert "抗酸杆菌染色（AFB）" in result["examinations"]
    assert len(result["examinations"]) <= 5


def test_perioral_axis_uses_skin_exam_without_nasal_screening() -> None:
    result = plan(
        {"chat_history": [{"from": "patient", "text": "长期面部外用激素后口周和鼻翼红斑丘疹灼热，无鼻塞流涕。"}]},
        [{
            "axis_id": "topical_steroid_associated_perioral_dermatitis",
            "status": "suspected",
            "priority": "high",
            "clinical_role": "current_problem",
            "exam_intents": ["局部面部皮损评估"],
        }],
    )
    assert result["examinations"] == ["皮肤检查"]
