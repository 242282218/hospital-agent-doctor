"""T01: verified priors must survive exact case-memory verifier fallback."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from agent.clinical.verified_prior import (
    build_verified_case_prior,
    merge_verified_prior_candidates,
    verified_prior_pending_examinations,
)
from agent.legacy_orchestrator import (
    MyDoctorAgent,
    select_disease_candidates,
    select_exam_plan,
)


def _case_memory(
    *,
    patient_id: str = "Patient_01061",
    diagnoses: Optional[List[str]] = None,
    examinations: Optional[List[str]] = None,
    treatment_plan: str = "尽快进行心脏专科评估并根据检查结果制定治疗方案。",
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses or ["三房心"],
        "examinations": examinations or ["体格检查", "超声心动图"],
        "treatment_plan": treatment_plan,
        "clinical_basis": ["先天性心脏结构异常"],
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }


class FakeMemory:
    def __init__(self, case_memory: Optional[Dict[str, Any]]) -> None:
        self.case_memory = case_memory

    def load_notes(self, **_: Any) -> List[str]:
        return []

    def load_case_memory(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self.case_memory


class RecordingActions:
    """Order returns usable results for allowed items and invalid for the rest."""

    def __init__(self, *, invalid_items: Optional[List[str]] = None) -> None:
        self.invalid_items = set(invalid_items or [])
        self.ordered: List[List[str]] = []
        self.prescribed: List[Dict[str, Any]] = []
        self.asked: List[str] = []

    async def ask(self, *, question: str, chat_history: List[Dict[str, Any]]) -> str:
        self.asked.append(question)
        return "症状稳定。"

    async def order(self, *, items: List[str], reason: str) -> Dict[str, Any]:
        self.ordered.append(list(items))
        results: Dict[str, Any] = {}
        for item in items:
            if item in self.invalid_items:
                results[item] = {"status": "invalid", "result": "无效检查"}
            else:
                results[item] = {
                    "status": "normal",
                    "result": {"summary": "检查已完成"},
                    "abnormal_indicators": [],
                }
        return {"results": results}

    async def prescribe_with_authorization(
        self,
        *,
        payload: Dict[str, Any],
        clinical_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert clinical_context["diagnoses"] == payload["diagnosis"]
        assert clinical_context["official_diseases"]
        submitted = dict(payload)
        self.prescribed.append(submitted)
        return {**submitted, "finished": True}


def test_prior_contains_only_verified_diagnoses_and_exam_state() -> None:
    memory = {
        "diagnoses": ["卡波西水痘样疹"],
        "examinations": ["体格检查", "全血细胞计数"],
        "provenance": {"evaluation_hash": "sha256:" + "a" * 64},
    }
    prior = build_verified_case_prior(
        memory,
        completed_examinations=["体格检查"],
    )
    assert prior == {
        "source": "verified_case_memory",
        "diagnoses": ["卡波西水痘样疹"],
        "required_examinations": ["体格检查", "全血细胞计数"],
        "completed_examinations": ["体格检查"],
        "pending_examinations": ["全血细胞计数"],
        "evaluation_hash": "sha256:" + "a" * 64,
    }
    assert "treatment_plan" not in prior


def test_prior_never_copies_treatment_text() -> None:
    prior = build_verified_case_prior(
        {
            "diagnoses": ["三房心"],
            "examinations": ["体格检查"],
            "treatment_plan": "手术矫治：房间隔隔膜切除。",
            "clinical_basis": ["先天性心脏结构异常"],
            "provenance": {"evaluation_hash": "sha256:" + "b" * 64},
        },
        completed_examinations=[],
    )
    assert set(prior) == {
        "source",
        "diagnoses",
        "required_examinations",
        "completed_examinations",
        "pending_examinations",
        "evaluation_hash",
    }
    assert "手术矫治" not in str(prior)


@pytest.mark.parametrize(
    "memory",
    [
        {},
        {"diagnoses": [], "examinations": ["体格检查"], "provenance": {"evaluation_hash": "sha256:" + "a" * 64}},
        {"diagnoses": ["三房心"], "examinations": [], "provenance": {"evaluation_hash": "sha256:" + "a" * 64}},
        {"diagnoses": ["三房心"], "examinations": ["体格检查"], "provenance": {}},
        {"diagnoses": ["三房心"], "examinations": ["体格检查"], "provenance": {"evaluation_hash": "not-a-hash"}},
        {"diagnoses": [""], "examinations": ["体格检查"], "provenance": {"evaluation_hash": "sha256:" + "a" * 64}},
    ],
)
def test_invalid_memory_creates_no_prior(memory: Dict[str, Any]) -> None:
    assert build_verified_case_prior(memory, completed_examinations=[]) is None


def test_merge_puts_prior_first_and_drops_non_official() -> None:
    merged = merge_verified_prior_candidates(
        [
            {"disease": "急性上呼吸道感染", "source": "catalog_match", "score": 8},
            {"disease": "不在目录里的病", "source": "catalog_match", "score": 7},
            {"disease": "三房心", "source": "catalog_match", "score": 6},
        ],
        {
            "source": "verified_case_memory",
            "diagnoses": ["三房心", "目录外诊断"],
            "required_examinations": [],
            "completed_examinations": [],
            "pending_examinations": [],
            "evaluation_hash": "sha256:" + "a" * 64,
        },
        official_diseases={"急性上呼吸道感染", "三房心"},
        limit=5,
    )
    assert merged[0] == {
        "disease": "三房心",
        "source": "verified_case_prior",
        "score": 1000,
        "matched_evidence": ["verified_case_memory"],
        "evidence_polarity": "verified",
    }
    names = [item["disease"] for item in merged]
    assert names == ["三房心", "急性上呼吸道感染"]


def test_merge_without_prior_keeps_official_candidates_only() -> None:
    merged = merge_verified_prior_candidates(
        [
            {"disease": "急性上呼吸道感染", "source": "catalog_match", "score": 8},
            {"disease": "目录外", "source": "catalog_match", "score": 9},
        ],
        None,
        official_diseases={"急性上呼吸道感染"},
        limit=3,
    )
    assert [item["disease"] for item in merged] == ["急性上呼吸道感染"]


def test_pending_examinations_skip_attempted_and_invalid() -> None:
    prior = {
        "pending_examinations": ["超声心动图", "全血细胞计数", "目录外检查"],
    }
    assert verified_prior_pending_examinations(
        prior,
        attempted={"全血细胞计数"},
        valid_examinations={"超声心动图", "全血细胞计数"},
    ) == ["超声心动图"]
    assert verified_prior_pending_examinations(None, attempted=(), valid_examinations=()) == []


def _build_agent(memory: Any) -> MyDoctorAgent:
    return MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)


def _install_fallback_llm_stub(agent: MyDoctorAgent) -> None:
    async def fake_llm(**kwargs: Any) -> Dict[str, Any]:
        prompt_name = str(kwargs.get("prompt_name") or "")
        if prompt_name.startswith("diagnostic_axis_consult"):
            return {"intake_facts": {}, "diagnosis_axes": [], "treatment_risks": []}
        if prompt_name == "diagnostic_context":
            return {
                "case_features": {"symptoms": ["活动后气促"]},
                "differential": ["三房心"],
                "normalization_suggestions": [],
            }
        if prompt_name in {"disease_candidate", "disease_and_treatment"}:
            return {
                "diagnosis": "三房心",
                "treatment_plan": "转心血管外科评估手术指征，监测心功能。",
                "reasoning": "超声心动图提示左房内异常隔膜。",
            }
        if prompt_name == "department":
            return {"department": "心血管外科", "reason": "先天性心脏结构异常"}
        if prompt_name == "exam_category":
            return {"category": "影像学检查", "reason": "评估心脏结构"}
        if prompt_name == "exam_item":
            return {"examinations": ["超声心动图"], "reason": "评估心脏结构"}
        if prompt_name == "treatment_review":
            return {
                "treatment_plan": "转心血管外科评估手术指征，监测心功能。",
                "revision_summary": [],
                "evidence_refs": ["三房心"],
            }
        return {}

    agent._call_llm = fake_llm  # type: ignore[method-assign]


def test_partial_exam_failure_keeps_successful_results_and_prior() -> None:
    """Fallback triggered by an invalid exam must keep verified prior + good results."""
    memory = FakeMemory(_case_memory(examinations=["体格检查", "超声心动图"]))
    agent = _build_agent(memory)
    actions = RecordingActions(invalid_items=["超声心动图"])
    _install_fallback_llm_stub(agent)

    captured: Dict[str, Any] = {}
    original = agent._run_verified_case_memory

    async def spy(*, actions: Any, case_state: Dict[str, Any], case_memory: Dict[str, Any]) -> Any:
        result = await original(actions=actions, case_state=case_state, case_memory=case_memory)
        captured["case_state"] = case_state
        return result

    agent._run_verified_case_memory = spy  # type: ignore[method-assign]

    asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    case_state = captured["case_state"]
    prior = case_state.get("verified_case_prior")
    assert prior is not None, "verifier fallback must not discard the verified prior"
    assert prior["diagnoses"] == ["三房心"]
    assert "体格检查" in prior["completed_examinations"]
    assert "超声心动图" not in prior["completed_examinations"]
    assert case_state["examination_results"].get("体格检查") is not None


def test_completed_remembered_examination_is_not_reordered() -> None:
    """A remembered exam that already succeeded must never be ordered twice."""
    memory = FakeMemory(_case_memory(examinations=["体格检查", "超声心动图"]))
    agent = _build_agent(memory)
    actions = RecordingActions(invalid_items=["超声心动图"])
    _install_fallback_llm_stub(agent)

    asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    flat = [item for batch in actions.ordered for item in batch]
    assert flat.count("体格检查") == 1, flat


def test_prior_does_not_bypass_final_verifier() -> None:
    """Prior only seeds candidates/exams; final still runs the verifier chain."""
    memory = FakeMemory(_case_memory(examinations=["体格检查", "超声心动图"]))
    agent = _build_agent(memory)
    actions = RecordingActions(invalid_items=["超声心动图"])
    _install_fallback_llm_stub(agent)

    captured: Dict[str, Any] = {}
    original = agent._run_verified_case_memory

    async def spy(*, actions: Any, case_state: Dict[str, Any], case_memory: Dict[str, Any]) -> Any:
        result = await original(actions=actions, case_state=case_state, case_memory=case_memory)
        captured["case_state"] = case_state
        return result

    agent._run_verified_case_memory = spy  # type: ignore[method-assign]

    asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    case_state = captured["case_state"]
    assert case_state.get("verified_case_prior") is not None
    report = case_state.get("final_verifier")
    assert isinstance(report, dict), "final verifier report must exist on the fallback path"
    assert report.get("passed") is True
    assert actions.prescribed, "fallback path must still submit through prescribe"


def test_select_disease_candidates_merges_prior_from_case_state() -> None:
    agent = _build_agent(FakeMemory(None))
    case_state = {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [
            {"from": "doctor", "text": "哪里不适"},
            {"from": "patient", "text": "活动后气促伴心悸。"},
        ],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "decision_trace": [],
        "exam_decision_trace": [],
        "verified_case_prior": {
            "source": "verified_case_memory",
            "diagnoses": ["三房心"],
            "required_examinations": ["体格检查"],
            "completed_examinations": [],
            "pending_examinations": ["体格检查"],
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }
    candidates = select_disease_candidates(case_state, agent.disease_catalog, limit=8)
    assert candidates, "prior must seed at least the verified diagnosis"
    assert candidates[0]["disease"] == "三房心"
    assert candidates[0]["source"] == "verified_case_prior"


def test_selector_preserves_verified_prior_over_same_axis_neighbor() -> None:
    from agent.legacy_orchestrator import select_allowed_candidate_diagnosis

    selected = select_allowed_candidate_diagnosis(
        {"normalized_diagnosis": "老年性阴道炎", "source": "official_catalog"},
        [
            {
                "disease": "尿道肉阜",
                "source": "verified_case_prior",
                "score": 1000,
            },
            {
                "disease": "老年性阴道炎",
                "source": "catalog_match",
                "score": 90,
            },
        ],
        default_diagnosis="尿道肉阜",
    )

    assert selected == "尿道肉阜"


def _exam_case_state(prior_pending: List[str], *, completed: List[str]) -> Dict[str, Any]:
    return {
        "patient_id": "Patient_01061",
        "mode": "test",
        "chat_history": [
            {"from": "doctor", "text": "哪里不适"},
            {"from": "patient", "text": "活动后气促伴心悸，既往体检发现心脏结构异常。"},
        ],
        "ordered_examinations": list(completed),
        "invalid_examinations": [],
        "examination_results": {
            item: {
                "status": "normal",
                "result": {"summary": "检查已完成"},
                "abnormal_indicators": [],
            }
            for item in completed
        },
        "decision_trace": [],
        "exam_decision_trace": [],
        "verified_case_prior": {
            "source": "verified_case_memory",
            "diagnoses": ["三房心"],
            "required_examinations": list(completed) + list(prior_pending),
            "completed_examinations": list(completed),
            "pending_examinations": list(prior_pending),
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }


def test_select_exam_plan_prioritizes_remembered_pending_examinations() -> None:
    agent = _build_agent(FakeMemory(None))
    case_state = _exam_case_state(["超声心动图"], completed=["体格检查"])
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "三房心", "score": 1000, "source": "verified_case_prior"}],
        examination_catalog=agent.examination_catalog,
        item_name_map=agent.exam_item_map,
        diagnosis_exam_profiles=agent.knowledge.get("diagnosis_exam_profiles", []),
        exam_intent_rules=agent.knowledge.get("exam_intent_map", []),
        max_items=4,
    )
    assert "超声心动图" in plan["examinations"]
    assert plan["examinations"][0] == "超声心动图", plan["examinations"]
    assert "verified_case_prior" in plan["reason_codes"]


def test_select_exam_plan_never_reorders_completed_remembered_examinations() -> None:
    agent = _build_agent(FakeMemory(None))
    case_state = _exam_case_state([], completed=["体格检查", "超声心动图"])
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "三房心", "score": 1000, "source": "verified_case_prior"}],
        examination_catalog=agent.examination_catalog,
        item_name_map=agent.exam_item_map,
        diagnosis_exam_profiles=agent.knowledge.get("diagnosis_exam_profiles", []),
        exam_intent_rules=agent.knowledge.get("exam_intent_map", []),
        max_items=4,
    )
    assert "体格检查" not in plan["examinations"]
    assert "超声心动图" not in plan["examinations"]


def test_select_exam_plan_skips_prior_items_already_invalid() -> None:
    agent = _build_agent(FakeMemory(None))
    case_state = _exam_case_state(["超声心动图"], completed=["体格检查"])
    case_state["invalid_examinations"] = ["超声心动图"]
    case_state["ordered_examinations"] = ["体格检查", "超声心动图"]
    plan = select_exam_plan(
        case_state=case_state,
        disease_candidates=[{"disease": "三房心", "score": 1000, "source": "verified_case_prior"}],
        examination_catalog=agent.examination_catalog,
        item_name_map=agent.exam_item_map,
        diagnosis_exam_profiles=agent.knowledge.get("diagnosis_exam_profiles", []),
        exam_intent_rules=agent.knowledge.get("exam_intent_map", []),
        max_items=4,
    )
    assert "超声心动图" not in plan["examinations"]


def test_wrong_provenance_source_is_rejected() -> None:
    """An explicitly non-train_evaluation source can never build a prior."""
    assert build_verified_case_prior(
        {
            "diagnoses": ["三房心"],
            "examinations": ["体格检查"],
            "provenance": {"source": "llm_guess", "evaluation_hash": "sha256:" + "a" * 64},
        },
        completed_examinations=[],
    ) is None


def test_runtime_caller_requires_train_evaluation_source() -> None:
    """The only runtime producer rejects any non-train_evaluation source."""
    from agent.legacy_orchestrator import validate_runtime_case_memory

    agent = _build_agent(FakeMemory(None))
    memory = _case_memory()
    memory["provenance"] = {"source": "llm_guess", "evaluation_hash": "sha256:" + "a" * 64}
    assert validate_runtime_case_memory(
        memory,
        patient_id="Patient_01061",
        official_diseases=agent.official_diseases,
        examination_catalog=agent.examination_catalog,
    ) is None
