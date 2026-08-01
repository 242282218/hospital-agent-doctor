from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent.clinical.model import (
    BudgetState,
    ClinicalBlackboard,
    EvidenceItem,
    ExamIntent,
    HypothesisItem,
    InformationGap,
    TreatmentState,
    WorkflowState,
)
from agent.clinical.shadow import ShadowBlackboardProjector


def test_blackboard_model_is_immutable_and_hashable() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-1",
        concept="chief_complaint",
        value="发热三天",
        kind="patient_statement",
        source_ref="message://patient/1",
    )
    board = ClinicalBlackboard(evidence_ledger=(evidence,))

    with pytest.raises(FrozenInstanceError):
        evidence.status = "validated"  # type: ignore[misc]

    assert board.revision == 0
    assert len(board.snapshot_hash()) == 64
    assert board.snapshot_hash() == board.snapshot_hash()


def test_blackboard_contains_all_stage_three_sections() -> None:
    board = ClinicalBlackboard()

    assert board.evidence_ledger == ()
    assert board.hypothesis_set == ()
    assert board.information_gaps == ()
    assert board.examination_state == ()
    assert board.treatment_state == TreatmentState()
    assert board.workflow_state == WorkflowState()
    assert board.budget_state == BudgetState()


def test_shadow_projects_subject_time_polarity_conflict_and_exam_status() -> None:
    trace = {
        "trace_revision": 4,
        "chat_history": [
            {"from": "patient", "text": "父亲有高血压，我本人没有高血压。"},
            {"from": "patient", "text": "目前服用布洛芬，没有药物过敏。"},
            {"from": "patient", "text": "后来又说我确诊过高血压。"},
        ],
        "case_features": {
            "family_history": ["家族高血压"],
            "personal_history": ["无高血压", "高血压"],
            "medications": ["布洛芬"],
            "drug_allergies": ["无药物过敏"],
        },
        "ordered_examinations": ["血常规", "胸部CT"],
        "invalid_examinations": ["胸部CT"],
        "examination_results": {
            "血常规": {"status": "abnormal", "result": {"白细胞": "升高"}},
        },
    }
    before = deepcopy(trace)

    snapshot = ShadowBlackboardProjector().project(trace)

    assert trace == before
    assert snapshot.blackboard.revision == 4
    assert any(item.subject == "family" for item in snapshot.blackboard.evidence_ledger)
    assert any(item.polarity == "negative" for item in snapshot.blackboard.evidence_ledger)
    assert any(item.status == "conflicted" for item in snapshot.blackboard.evidence_ledger)
    statuses = {item.catalog_leaf_name: item.status for item in snapshot.blackboard.examination_state}
    assert statuses == {"血常规": "resulted", "胸部CT": "invalid"}
    assert not any(item.value == "无效检查" for item in snapshot.blackboard.evidence_ledger)


def test_shadow_expresses_hypotheses_gaps_and_final_without_changing_submission() -> None:
    trace = {
        "trace_revision": 8,
        "chat_history": [{"from": "patient", "text": "发热咳嗽三天。"}],
        "disease_candidates": [{"disease": "肺炎", "score": 90}],
        "diagnosis_axes": [
            {
                "axis_id": "respiratory_infection",
                "candidate_official_names": ["肺炎"],
                "exam_intents": ["胸部结构影像"],
                "treatment_risks": [],
            }
        ],
        "coverage_gaps": [
            {
                "gap_id": "chest-imaging",
                "intent": "胸部结构影像",
                "decision_impact": "diagnosis",
                "acquisition_route": "examination",
                "priority": "high",
            }
        ],
        "final_plan": {
            "diagnosis": "肺炎",
            "treatment_plan": "抗感染并监测呼吸。",
            "reasoning": "结合病史和检查。",
        },
        "required_fact_keys": ["patient_statement"],
    }
    final_payload = {
        "diagnosis": ["肺炎"],
        "treatment_plan": "抗感染并监测呼吸。",
        "reasoning": "结合病史和检查。",
    }
    projector = ShadowBlackboardProjector()
    snapshot = projector.project(trace)
    diff = projector.compare(trace, snapshot, final_payload=final_payload)

    assert snapshot.blackboard.hypothesis_set[0].status == "selected"
    assert snapshot.blackboard.information_gaps[0].status == "open"
    assert snapshot.blackboard.treatment_state.draft_text == final_payload["treatment_plan"]
    assert diff.missing_fact_keys == ()
    assert diff.final_field_differences == ()
    assert diff.final_submission_changed is False


def test_historical_shadow_replay_has_zero_final_diff() -> None:
    path = Path("docs/架构迁移基线/阶段3-Shadow历史投影摘要.json")
    assert path.exists()
    import json

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["replayable_runs"] > 0
    assert summary["final_diff_nonzero_runs"] == 0
    assert summary["required_field_expression_rate"] == 1.0


def test_stage_three_report_records_zero_case_budget() -> None:
    path = Path("docs/架构迁移基线/阶段3-Blackboard-Shadow-Mode验证报告.md")
    text = path.read_text(encoding="utf-8")

    assert "病例数：0" in text
    assert "LLM 调用数：0" in text
    assert "evaluation 调用数：0" in text
    assert "Token 近似成本：0" in text
    assert "dict-case-state 生命周期：shadowed，未删除" in text


def test_unused_domain_types_are_exportable() -> None:
    assert ExamIntent(exam_intent_id="x").status == "proposed"
    assert HypothesisItem(hypothesis_id="h", official_disease_name="x").status == "active"
    assert InformationGap(gap_id="g", intent="x").status == "open"
