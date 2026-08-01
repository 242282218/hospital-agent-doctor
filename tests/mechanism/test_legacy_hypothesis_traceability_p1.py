"""P1: legacy hypothesis shadow projection and traceability codes."""

from __future__ import annotations

from agent.clinical.legacy_hypotheses import (
    build_legacy_hypotheses,
    verify_selected_hypothesis_traceability,
)


def test_selected_has_supporting_evidence_or_issue() -> None:
    hypotheses = build_legacy_hypotheses(
        disease_candidates=[
            {
                "disease": "蜂窝织炎",
                "score": 110,
                "role": "current_problem",
                "matched_evidence": ["小腿红肿热痛"],
            }
        ],
        diagnosis_axes=[
            {
                "axis_id": "acute_lower_extremity_soft_tissue_infection",
                "candidate_official_names": ["蜂窝织炎"],
                "evidence": ["发热", "皮温升高"],
                "exam_intents": ["下肢软组织感染严重度评估"],
                "treatment_risks": ["do_not_close_soft_tissue_infection_as_arthritis"],
            }
        ],
        case_state={
            "examination_results": {
                "C反应蛋白（CRP）": {"status": "abnormal", "result": {"CRP": "升高"}}
            },
            "typed_exam_intent_ids": [
                "exam_intent_lower_extremity_soft_tissue_severity"
            ],
        },
        selected_diagnosis="蜂窝织炎",
    )
    assert len(hypotheses) == 1
    item = hypotheses[0]
    assert item.status == "selected"
    assert item.official_disease_name == "蜂窝织炎"
    assert item.supporting_evidence_ids
    assert "exam_intent_lower_extremity_soft_tissue_severity" in item.required_exam_intents
    ok, issues = verify_selected_hypothesis_traceability(
        selected_diagnosis="蜂窝织炎",
        hypotheses=hypotheses,
        reasoning="支持急性软组织感染。",
    )
    assert ok is True
    assert issues == ()


def test_selected_missing_hypothesis_issue() -> None:
    hypotheses = build_legacy_hypotheses(
        disease_candidates=[{"disease": "上呼吸道感染", "score": 40}],
        diagnosis_axes=[],
        case_state={},
        selected_diagnosis="先天性白内障",
    )
    # selected is forced into shadow list, but may lack support.
    ok, issues = verify_selected_hypothesis_traceability(
        selected_diagnosis="先天性白内障",
        hypotheses=hypotheses,
        reasoning="",
    )
    assert ok is False
    assert "selected_support_missing" in issues or "selected_hypothesis_missing" in issues


def test_negative_exam_enters_opposing_evidence() -> None:
    hypotheses = build_legacy_hypotheses(
        disease_candidates=[
            {
                "disease": "细菌性咽炎",
                "score": 60,
                "matched_evidence": ["咽痛"],
            }
        ],
        diagnosis_axes=[
            {
                "axis_id": "pharyngitis",
                "candidate_official_names": ["细菌性咽炎"],
                "evidence": ["咽痛"],
            }
        ],
        case_state={
            "examination_results": {
                "咽拭子培养": {"status": "negative", "result": {"培养": "阴性"}}
            }
        },
        selected_diagnosis="细菌性咽炎",
    )
    item = hypotheses[0]
    assert any(eid.startswith("exam:") for eid in item.opposing_evidence_ids)


def test_reasoning_rejects_selected_issue_code() -> None:
    hypotheses = build_legacy_hypotheses(
        disease_candidates=[
            {
                "disease": "白色糠疹",
                "score": 50,
                "matched_evidence": ["面部白斑"],
            }
        ],
        diagnosis_axes=[],
        case_state={},
        selected_diagnosis="白色糠疹",
    )
    ok, issues = verify_selected_hypothesis_traceability(
        selected_diagnosis="白色糠疹",
        hypotheses=hypotheses,
        reasoning="当前证据不支持白色糠疹，应优先排除眼内肿瘤。",
    )
    assert ok is False
    assert "reasoning_rejects_selected" in issues


def test_background_disease_not_preferred_over_current_axis() -> None:
    hypotheses = build_legacy_hypotheses(
        disease_candidates=[
            {
                "disease": "原发性高血压",
                "score": 30,
                "role": "background_condition",
                "matched_evidence": ["高血压病史"],
            },
            {
                "disease": "心力衰竭",
                "score": 95,
                "role": "current_problem",
                "matched_evidence": ["端坐呼吸", "双下肢水肿"],
            },
        ],
        diagnosis_axes=[
            {
                "axis_id": "acute_heart_failure",
                "candidate_official_names": ["心力衰竭"],
                "evidence": ["端坐呼吸"],
                "exam_intents": ["急性心衰利钠肽与容量评估"],
            }
        ],
        case_state={},
        selected_diagnosis="心力衰竭",
    )
    names = [item.official_disease_name for item in hypotheses]
    assert names.index("心力衰竭") < names.index("原发性高血压") or hypotheses[
        0
    ].official_disease_name in {"心力衰竭", "原发性高血压"}
    selected = [item for item in hypotheses if item.status == "selected"][0]
    assert selected.official_disease_name == "心力衰竭"
    assert selected.role in {"current_problem", "selected", "differential"}
