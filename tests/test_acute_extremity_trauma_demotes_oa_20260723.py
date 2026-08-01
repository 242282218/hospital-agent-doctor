"""Shared-root: acute extremity trauma demotes chronic OA labels."""

from __future__ import annotations

from agent.legacy_orchestrator import (
    has_acute_extremity_trauma_pattern,
    prune_unsupported_disease_candidates,
)


def test_wrist_fall_is_acute_extremity_trauma() -> None:
    text = "72岁患者跌倒后右腕剧痛肿胀，手撑地，伴拇指侧麻木。"
    assert has_acute_extremity_trauma_pattern(text) is True


def test_chronic_knee_oa_history_alone_is_not_acute_trauma() -> None:
    text = "长期膝关节炎病史，无新发外伤，无红肿剧痛。"
    assert has_acute_extremity_trauma_pattern(text) is False


def test_acute_wrist_trauma_demotes_osteoarthritis_promotes_fracture() -> None:
    text = (
        "72岁跌倒后右腕剧痛肿胀畸形，手撑地，拇指侧麻木疑正中神经受累。"
        "既往骨关节炎。"
    )
    candidates = prune_unsupported_disease_candidates(
        [
            {"disease": "骨关节炎", "score": 90, "source": "catalog_match"},
            {"disease": "桡骨远端骨折", "score": 40, "source": "catalog_match"},
            {"disease": "腕管综合征", "score": 30, "source": "catalog_match"},
        ],
        {
            "chat_history": [{"from": "patient", "text": text}],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {
                "腕部X线检查": {
                    "status": "abnormal",
                    "result": {"描述": "可见退变改变，需排除骨折"},
                }
            },
            "exam_decision_trace": [],
        },
    )
    by_name = {item["disease"]: item for item in candidates}
    assert by_name["骨关节炎"]["role"] == "background_history"
    assert int(by_name["骨关节炎"]["score"] or 0) <= 20
    assert int(by_name["桡骨远端骨折"]["score"] or 0) >= 100
    assert by_name["桡骨远端骨折"]["role"] == "current_problem"
