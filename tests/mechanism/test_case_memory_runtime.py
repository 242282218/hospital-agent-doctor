from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import agent.legacy_orchestrator as legacy_orchestrator
from agent.legacy_orchestrator import MAX_EXAMS_PER_ACTION, MyDoctorAgent
from agent.memory import VerifiedOnlyMemory


class FakeMemory:
    def __init__(self, case_memory: Optional[Dict[str, Any]]) -> None:
        self.case_memory = case_memory
        self.case_lookups: List[str] = []

    def load_notes(self, **_: Any) -> List[str]:
        return []

    def load_case_memory(self, patient_id: str) -> Optional[Dict[str, Any]]:
        self.case_lookups.append(patient_id)
        return self.case_memory


class FakeActions:
    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        *,
        patient_reply: str = "发热、咳嗽，症状稳定。",
    ) -> None:
        self.responses = list(responses or [])
        self.patient_reply = patient_reply
        self.asked: List[str] = []
        self.ordered: List[List[str]] = []
        self.prescribed: List[Dict[str, Any]] = []

    async def ask(self, *, question: str, chat_history: List[Dict[str, Any]]) -> str:
        self.asked.append(question)
        return self.patient_reply

    async def order(self, *, items: List[str], reason: str) -> Dict[str, Any]:
        self.ordered.append(list(items))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return _successful_exam_response(items)

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


def _successful_exam_response(items: List[str]) -> Dict[str, Any]:
    return {
        "results": {
            item: {
                "status": "normal",
                "result": {"summary": "检查已完成"},
                "abnormal_indicators": [],
            }
            for item in items
        }
    }


def _case_memory(
    *,
    patient_id: str = "Patient_01061",
    diagnoses: Optional[List[str]] = None,
    examinations: Optional[List[str]] = None,
    treatment_plan: str = "尽快进行心脏专科评估并根据检查结果制定治疗方案。",
    clinical_basis: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses or ["三房心"],
        "examinations": examinations or ["体格检查", "超声心动图"],
        "treatment_plan": treatment_plan,
        "clinical_basis": clinical_basis or ["先天性心脏结构异常"],
        "provenance": {
            "source": "train_evaluation",
            "evaluation_hash": "sha256:" + "a" * 64,
        },
    }


def _empty_case_state(patient_id: str = "Patient_01061", mode: str = "test") -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "mode": mode,
        "memory_notes": [],
        "chat_history": [],
        "ordered_examinations": [],
        "invalid_examinations": [],
        "examination_results": {},
        "decision_trace": [],
        "exam_decision_trace": [],
    }


def _build_agent(memory: Any) -> MyDoctorAgent:
    return MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)


def _catalog_order(agent: MyDoctorAgent, names: List[str]) -> List[str]:
    requested = set(names)
    return [
        name
        for items in agent.examination_catalog.values()
        for name in items
        if name in requested
    ]


def _forbid_llm(agent: MyDoctorAgent) -> None:
    async def forbidden_llm(**_: Any) -> Dict[str, Any]:
        raise AssertionError("case-memory hit must not call LLM")

    agent._call_llm = forbidden_llm  # type: ignore[method-assign]


def _install_legacy_llm_stub(agent: MyDoctorAgent) -> None:
    async def fake_llm(**kwargs: Any) -> Dict[str, Any]:
        prompt_name = str(kwargs.get("prompt_name") or "")
        if prompt_name.startswith("diagnostic_axis_consult"):
            return {
                "intake_facts": {},
                "diagnosis_axes": [],
                "treatment_risks": [],
            }
        if prompt_name == "diagnostic_context":
            return {
                "case_features": {"symptoms": ["发热", "咳嗽"]},
                "differential": ["急性上呼吸道感染"],
                "normalization_suggestions": [],
            }
        if prompt_name == "disease_candidate":
            return {
                "diagnosis": "急性上呼吸道感染",
                "treatment_plan": "对症退热、补液休息，观察呼吸困难与脱水。",
                "reasoning": "当前症状和检查支持急性呼吸道感染。",
            }
        if prompt_name == "department":
            return {"department": "呼吸内科", "reason": "急性呼吸道症状"}
        if prompt_name == "disease_and_treatment":
            return {
                "diagnosis": "急性上呼吸道感染",
                "treatment_plan": "对症退热、补液休息，观察呼吸困难与脱水。",
                "reasoning": "当前症状和检查支持急性呼吸道感染。",
            }
        if prompt_name == "exam_category":
            return {"category": "实验室检查-血液", "reason": "评估感染"}
        if prompt_name == "exam_item":
            return {
                "examinations": ["全血细胞计数（CBC）"],
                "reason": "评估感染",
            }
        if prompt_name == "treatment_review":
            return {
                "treatment_plan": "对症退热、补液休息，观察呼吸困难与脱水。",
                "revision_summary": [],
                "evidence_refs": ["急性上呼吸道感染"],
            }
        return {}

    agent._call_llm = fake_llm  # type: ignore[method-assign]


def test_test_mode_legacy_hit_keeps_prior_and_falls_back_to_clinical_loop() -> None:
    memory = FakeMemory(_case_memory())
    actions = FakeActions()
    agent = _build_agent(memory)
    _install_legacy_llm_stub(agent)

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    assert memory.case_lookups == ["Patient_01061"]
    assert actions.ordered == [["体格检查", "超声心动图"]]
    assert len(actions.prescribed) == 1
    assert result["finished"] is True


def test_memory_miss_uses_existing_clinical_loop() -> None:
    memory = FakeMemory(None)
    actions = FakeActions()
    agent = _build_agent(memory)
    _install_legacy_llm_stub(agent)

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_99999",
            mode="test",
        )
    )

    assert memory.case_lookups == ["Patient_99999"]
    assert actions.asked
    assert actions.ordered == []
    assert len(actions.prescribed) == 1
    assert result["finished"] is True


@pytest.mark.parametrize(
    ("force_fallback", "expected_prompt"),
    [
        (False, "disease_candidate"),
        (True, "disease_and_treatment"),
    ],
    ids=["disease_candidates", "fallback"],
)
def test_accepted_unsafe_review_only_prescribes_verifier_approved_fallback(
    monkeypatch: pytest.MonkeyPatch,
    force_fallback: bool,
    expected_prompt: str,
) -> None:
    actions = FakeActions(patient_reply="发热、咳嗽，症状稳定。")
    agent = _build_agent(FakeMemory(None))
    _install_legacy_llm_stub(agent)
    prompt_names: List[str] = []
    legacy_llm = agent._call_llm

    async def recording_llm(**kwargs: Any) -> Dict[str, Any]:
        prompt_names.append(str(kwargs.get("prompt_name") or ""))
        return await legacy_llm(**kwargs)

    agent._call_llm = recording_llm  # type: ignore[method-assign]
    if force_fallback:
        monkeypatch.setattr(
            legacy_orchestrator,
            "normalize_candidates_from_diagnostic_context",
            lambda *_args, **_kwargs: [],
        )

    verifier_calls: List[str] = []
    approved_plans: List[str] = []
    captured_state: Dict[str, Any] = {}
    unsafe_review_plan = "未验证的高风险治疗方案。"

    async def accepted_unsafe_review(**kwargs: Any) -> str:
        case_state = kwargs["case_state"]
        original = str(kwargs["treatment_plan"])
        captured_state["value"] = case_state
        case_state["treatment_review_decision"] = (
            legacy_orchestrator.treatment_review_decision(
                status="accepted",
                original=original,
                treatment_plan=unsafe_review_plan,
            )
        )
        return unsafe_review_plan

    agent._review_treatment_plan = accepted_unsafe_review  # type: ignore[method-assign]

    def unresolved_verifier(**kwargs: Any) -> Dict[str, Any]:
        treatment_plan = str(kwargs["treatment_plan"])
        verifier_calls.append(treatment_plan)
        if treatment_plan == legacy_orchestrator.CONSERVATIVE_FALLBACK_TREATMENT:
            approved_plans.append(treatment_plan)
            return {
                "passed": True,
                "issues": [],
                "patched_treatment": treatment_plan,
            }
        return {
            "passed": False,
            "issues": [
                {
                    "severity": "must_fix",
                    "patchable": False,
                    "message": "治疗方案仍有不可自动修复的安全问题。",
                }
            ],
            "patched_treatment": treatment_plan,
        }

    monkeypatch.setattr(legacy_orchestrator, "final_verifier", unresolved_verifier)

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_99999",
            mode="test",
        )
    )

    assert expected_prompt in prompt_names
    assert captured_state["value"]["treatment_review_decision"]["status"] == "accepted"
    assert unsafe_review_plan in verifier_calls
    assert len(actions.prescribed) == 1
    submitted_plan = actions.prescribed[0]["treatment_plan"]
    assert submitted_plan != unsafe_review_plan
    assert submitted_plan in approved_plans
    assert captured_state["value"]["final_verifier"]["passed"] is True
    assert captured_state["value"]["final_verifier"]["patched_treatment"] == submitted_plan
    assert captured_state["value"]["final_plan"]["treatment_plan"] == submitted_plan
    assert result["finished"] is True


def test_unverified_conservative_fallback_is_not_prescribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = FakeActions(patient_reply="发热、咳嗽，症状稳定。")
    agent = _build_agent(FakeMemory(None))
    _install_legacy_llm_stub(agent)

    async def accepted_unsafe_review(**kwargs: Any) -> str:
        return "未验证的高风险治疗方案。"

    agent._review_treatment_plan = accepted_unsafe_review  # type: ignore[method-assign]

    def always_unresolved(**kwargs: Any) -> Dict[str, Any]:
        treatment_plan = str(kwargs["treatment_plan"])
        return {
            "passed": False,
            "issues": [
                {
                    "severity": "must_fix",
                    "patchable": False,
                    "message": "治疗方案仍有不可自动修复的安全问题。",
                }
            ],
            "patched_treatment": treatment_plan,
        }

    monkeypatch.setattr(legacy_orchestrator, "final_verifier", always_unresolved)

    with pytest.raises(legacy_orchestrator.FinalVerificationError):
        asyncio.run(
            agent.run_full_clinical_loop(
                actions=actions,
                patient_id="Patient_99999",
                mode="test",
            )
        )

    assert actions.prescribed == []


def test_memory_miss_prescribes_once_after_patched_treatment_is_reverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = FakeActions()
    agent = _build_agent(FakeMemory(None))
    _install_legacy_llm_stub(agent)
    verifier_calls: List[str] = []
    patched_treatment = "对症退热、补液休息，并监测呼吸困难与脱水警示征象。"

    def converging_verifier(**kwargs: Any) -> Dict[str, Any]:
        treatment_plan = str(kwargs["treatment_plan"])
        verifier_calls.append(treatment_plan)
        if treatment_plan == patched_treatment:
            return {
                "passed": True,
                "issues": [],
                "patched_treatment": patched_treatment,
            }
        return {
            "passed": False,
            "issues": [
                {
                    "severity": "must_fix",
                    "patchable": True,
                    "message": "补充安全监测。",
                }
            ],
            "patched_treatment": (
                treatment_plan if len(verifier_calls) == 1 else patched_treatment
            ),
        }

    monkeypatch.setattr(legacy_orchestrator, "final_verifier", converging_verifier)

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_99999",
            mode="test",
        )
    )

    assert len(verifier_calls) == 3
    assert verifier_calls[-1] == patched_treatment
    assert len(actions.prescribed) == 1
    assert actions.prescribed[0]["treatment_plan"] == patched_treatment
    assert result["finished"] is True


def test_partial_short_path_falls_back_without_reordering_successful_exam() -> None:
    memory = _case_memory(examinations=["体格检查", "超声心动图"])
    response = {
        "results": {
            "体格检查": {
                "status": "normal",
                "result": {"summary": "检查已完成"},
                "abnormal_indicators": [],
            },
            "超声心动图": {
                "status": "invalid",
                "result": {},
                "abnormal_indicators": [],
            },
        }
    }
    actions = FakeActions([response])
    agent = _build_agent(FakeMemory(memory))
    _install_legacy_llm_stub(agent)

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    ordered_items = [item for batch in actions.ordered for item in batch]
    assert ordered_items.count("体格检查") == 1
    assert len(actions.prescribed) == 1
    assert result["finished"] is True


def test_train_mode_does_not_consume_case_memory() -> None:
    memory = FakeMemory(_case_memory())
    actions = FakeActions()
    agent = _build_agent(memory)

    async def stop_before_external_flow(**_: Any) -> Dict[str, Any]:
        raise RuntimeError("legacy train flow reached")

    agent._diagnostic_axis_consult = stop_before_external_flow  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="legacy train flow reached"):
        asyncio.run(
            agent.run_full_clinical_loop(
                actions=actions,
                patient_id="Patient_01061",
                mode="train",
            )
        )

    assert memory.case_lookups == []


@pytest.mark.parametrize(
    "memory",
    [
        _case_memory(patient_id="Patient_99999"),
        _case_memory(diagnoses=[" 三房心 "]),
        _case_memory(diagnoses=["不存在的疾病"]),
        _case_memory(examinations=["超声检查"]),
        _case_memory(examinations=[" 体格检查 "]),
        _case_memory(examinations=["不存在的检查"]),
    ],
)
def test_invalid_case_memory_has_no_side_effects(memory: Dict[str, Any]) -> None:
    actions = FakeActions()
    agent = _build_agent(FakeMemory(memory))

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=_empty_case_state(),
            case_memory=memory,
        )
    )

    assert result is None
    assert actions.asked == []
    assert actions.ordered == []
    assert actions.prescribed == []


def test_remembered_exams_follow_catalog_order_and_batch_limit() -> None:
    requested = [
        "心电图（ECG）",
        "超声心动图",
        "体格检查",
        "血清电解质",
        "生命体征",
        "全血细胞计数（CBC）",
        "胸部X线检查（CXR）",
        "体格检查",
    ]
    memory = _case_memory(examinations=requested)
    actions = FakeActions()
    agent = _build_agent(FakeMemory(memory))
    case_state = _empty_case_state()

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=case_state,
            case_memory=memory,
        )
    )

    catalog_order = _catalog_order(agent, requested)
    assert result is None
    assert case_state["case_memory_fallback_reason"] == "safety_facts_incomplete"
    assert actions.ordered == [
        catalog_order[:MAX_EXAMS_PER_ACTION],
        catalog_order[MAX_EXAMS_PER_ACTION:],
    ]


def test_completed_examination_is_not_ordered_again() -> None:
    memory = _case_memory(examinations=["体格检查", "超声心动图"])
    state = _empty_case_state()
    state["ordered_examinations"] = ["体格检查"]
    state["examination_results"] = _successful_exam_response(["体格检查"])["results"]
    actions = FakeActions()
    agent = _build_agent(FakeMemory(memory))

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=state,
            case_memory=memory,
        )
    )

    assert result is None
    assert state["case_memory_fallback_reason"] == "safety_facts_incomplete"
    assert actions.ordered == [["超声心动图"]]
    assert actions.prescribed == []


def test_partial_response_preserves_success_and_falls_back() -> None:
    memory = _case_memory(examinations=["体格检查", "超声心动图"])
    response = {
        "results": {
            "体格检查": {
                "status": "normal",
                "result": {"summary": "检查已完成"},
                "abnormal_indicators": [],
            },
            "超声心动图": {
                "status": "invalid",
                "result": {},
                "abnormal_indicators": [],
            },
        }
    }
    actions = FakeActions([response])
    state = _empty_case_state()
    agent = _build_agent(FakeMemory(memory))

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=state,
            case_memory=memory,
        )
    )

    assert result is None
    assert "体格检查" in state["examination_results"]
    assert "超声心动图" not in state["examination_results"]
    assert state["invalid_examinations"] == ["超声心动图"]
    assert actions.prescribed == []


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"results": None},
        {"results": []},
        {"results": {}},
        {"results": {"体格检查": {"status": "normal", "result": {}}}},
        {
            "results": {
                "体格检查": {
                    "status": "unavailable",
                    "result": {"summary": "暂不可用"},
                }
            }
        },
    ],
)
def test_malformed_or_unavailable_response_falls_back(response: Dict[str, Any]) -> None:
    memory = _case_memory(examinations=["体格检查"])
    actions = FakeActions([response])
    agent = _build_agent(FakeMemory(memory))

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=_empty_case_state(),
            case_memory=memory,
        )
    )

    assert result is None
    assert actions.prescribed == []


def test_second_batch_failure_preserves_first_batch_without_prescribing() -> None:
    examinations = [
        "体格检查",
        "生命体征",
        "全血细胞计数（CBC）",
        "血清电解质",
        "胸部X线检查（CXR）",
        "超声心动图",
    ]
    memory = _case_memory(examinations=examinations)
    first_batch = examinations[:MAX_EXAMS_PER_ACTION]
    actions = FakeActions(
        [
            _successful_exam_response(first_batch),
            {
                "results": {
                    examinations[-1]: {
                        "status": "invalid",
                        "result": {},
                        "abnormal_indicators": [],
                    }
                }
            },
        ]
    )
    state = _empty_case_state()
    agent = _build_agent(FakeMemory(memory))

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=state,
            case_memory=memory,
        )
    )

    assert result is None
    assert all(item in state["examination_results"] for item in first_batch)
    assert examinations[-1] not in state["examination_results"]
    assert actions.prescribed == []


def test_transport_exception_is_not_retried_or_converted_to_fallback() -> None:
    memory = _case_memory(examinations=["体格检查"])
    actions = FakeActions([TimeoutError("outcome unknown")])
    agent = _build_agent(FakeMemory(memory))

    with pytest.raises(TimeoutError, match="outcome unknown"):
        asyncio.run(
            agent._run_verified_case_memory(
                actions=actions,
                case_state=_empty_case_state(),
                case_memory=memory,
            )
        )

    assert actions.ordered == [["体格检查"]]
    assert actions.prescribed == []


def test_legacy_memory_treatment_is_not_verified_or_prescribed_directly() -> None:
    memory = _case_memory(
        patient_id="Patient_09249",
        diagnoses=["腺病毒性结膜炎"],
        examinations=["裂隙灯检查"],
        treatment_plan="预防性使用局部抗生素滴眼液，每日四次；人工泪液每日六次。",
        clinical_basis=["病毒性结膜炎伴角膜受累"],
    )
    actions = FakeActions()
    agent = _build_agent(FakeMemory(memory))
    state = _empty_case_state("Patient_09249")
    _forbid_llm(agent)

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=state,
            case_memory=memory,
        )
    )

    assert result is None
    assert actions.prescribed == []
    assert state["case_memory_fallback_reason"] == "safety_facts_incomplete"
    assert state["verified_case_prior"]["diagnoses"] == ["腺病毒性结膜炎"]


def test_legacy_memory_treatment_text_never_reaches_direct_payload() -> None:
    memory = _case_memory(
        treatment_plan=(
            "根据 Patient_01061 的 train_evaluation reference，expected 治疗与 ground_truth 一致；"
            "建议尽快进行心脏专科评估。"
        ),
        clinical_basis=[
            "Patient_01061 evaluation expected reference ground truth ground_truth 先天性心脏结构异常"
        ],
    )
    actions = FakeActions()
    agent = _build_agent(FakeMemory(memory))
    state = _empty_case_state()

    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=state,
            case_memory=memory,
        )
    )

    assert result is None
    assert actions.prescribed == []
    assert state["case_memory_fallback_reason"] == "safety_facts_incomplete"
    assert "treatment_plan" not in state["verified_case_prior"]


def test_real_verified_memory_registry_can_drive_exact_hit(tmp_path: Path) -> None:
    memory_content = _case_memory()
    registry = tmp_path / "verified_registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "verified-registry/v1",
                "assets": [
                    {
                        "candidate_id": "case-memory-test",
                        "candidate_type": "case_memory",
                        "content": memory_content,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    memory = VerifiedOnlyMemory(registry)
    actions = FakeActions()
    agent = _build_agent(memory)
    _install_legacy_llm_stub(agent)
    observed_fallbacks: List[tuple[Optional[Dict[str, Any]], Optional[str]]] = []
    original_run_verified_case_memory = agent._run_verified_case_memory

    async def recording_run_verified_case_memory(**kwargs: Any) -> Optional[Dict[str, Any]]:
        result = await original_run_verified_case_memory(**kwargs)
        observed_fallbacks.append(
            (result, kwargs["case_state"].get("case_memory_fallback_reason"))
        )
        return result

    agent._run_verified_case_memory = recording_run_verified_case_memory  # type: ignore[method-assign]

    result = asyncio.run(
        agent.run_full_clinical_loop(
            actions=actions,
            patient_id="Patient_01061",
            mode="test",
        )
    )

    assert result["finished"] is True
    assert observed_fallbacks == [(None, "safety_facts_incomplete")]
    assert len(actions.prescribed) == 1
