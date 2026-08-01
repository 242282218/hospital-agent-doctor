"""Offline regressions for sixteenth-round online-path skeleton failure."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Iterable, List, Optional

import agent.legacy_orchestrator as legacy_orchestrator
from agent.clinical.final_submission import LoadedRuntimeIdentity
from agent.clinical.online_runtime import run_online_clinical_case
from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory
from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack
from tests.typed_rule_test_data import active_diagnosis_priority_pack_payload


TEST_RUNTIME_IDENTITY = LoadedRuntimeIdentity(
    status="strict_verified",
    identity_hash="sixteenth-round-test-release",
)


class FakeDoctorActions:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        self.calls.append(("ask_patient", input_data.get("question")))
        return (
            "前天开始发热，喉咙痛，咳嗽加重，昨晚睡不好。"
            "没有胸痛，没有咯血。"
        )

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        item_list = list(items)
        self.calls.append(("order_examination", item_list, reason))
        return {
            "results": {
                name: {
                    "status": "abnormal",
                    "result": {"flag": "阳性"},
                    "abnormal_indicators": ["flag"],
                }
                for name in item_list
            }
        }

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: Any,
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        self.calls.append(("prescribe_treatment", diagnosis, treatment_plan))
        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "finished": True,
        }

    async def evaluation(
        self,
        patient_id: str,
        final_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(("evaluation", patient_id))
        return {"diagnosisAccuracy": 0.5}


def test_run_agent_still_routes_through_online_runtime() -> None:
    source = inspect.getsource(MyDoctorAgent._run_agent)
    assert "run_online_clinical_case" in source
    assert "case_state" not in source


def test_online_runtime_uses_full_clinical_loop_engine() -> None:
    source = inspect.getsource(run_online_clinical_case)
    assert "run_full_clinical_loop" in source
    assert "clinical_engine" in source or "full_clinical_loop" in source
    # Skeleton-only CBC seed path must not be the sole clinical engine.
    assert "seed-exam-gap" not in source
    assert "basic_labs" not in source




def test_full_clinical_loop_not_undifferentiated_when_llm_returns_axis() -> None:
    async def scenario() -> Dict[str, Any]:
        agent = MyDoctorAgent(
            config={"log_llm_prompts": False},
            memory=VerifiedOnlyMemory(),
            runtime_identity=TEST_RUNTIME_IDENTITY,
        )
        agent.actions = FakeDoctorActions()
        prompt_names: List[str] = []

        async def fake_llm(**kwargs: Any) -> Dict[str, Any]:
            name = str(kwargs.get("prompt_name") or "")
            prompt_names.append(name)
            if name == "next_action":
                asked = sum(1 for item in prompt_names if item == "next_action")
                ordered = any(call[0] == "order_examination" for call in agent.actions.calls)
                if asked == 1:
                    return {
                        "action": "ask_patient",
                        "question": "还有哪些伴随症状？",
                        "reason": "补史",
                    }
                if ordered:
                    return {"action": "final_diagnosis", "question": "", "reason": "可终"}
                return {"action": "order_examination", "question": "", "reason": "需检查"}
            if name.startswith("diagnostic_axis_consult"):
                return {
                    "intake_facts": {},
                    "diagnosis_axes": [
                        {
                            "axis_id": "acute_respiratory_illness",
                            "priority_diseases": ["急性上呼吸道感染"],
                        }
                    ],
                    "treatment_risks": [],
                }
            if name == "diagnostic_context":
                return {
                    "case_features": {"symptoms": ["发热", "咳嗽", "咽痛"]},
                    "differential": ["急性上呼吸道感染", "急性支气管炎"],
                    "normalization_suggestions": [],
                }
            if name == "disease_candidate":
                return {
                    "diagnosis": "急性上呼吸道感染",
                    "treatment_plan": "对症退热、补液休息，观察呼吸困难与脱水。",
                    "reasoning": "急性呼吸道症状为主，优先上感路径。",
                }
            if name in {"exam_category", "exam_item"}:
                return {
                    "category": "实验室检查",
                    "examinations": ["全血细胞计数（CBC）"],
                    "reason": "炎症评估",
                }
            if name == "treatment_review":
                return {
                    "treatment_plan": "对症退热、补液休息，观察呼吸困难与脱水。",
                    "reasoning": "保持安全对症。",
                }
            return {}

        agent._call_llm = fake_llm  # type: ignore[method-assign]
        result = await agent.test(patient_id="offline-path-case")
        result["_prompt_names"] = prompt_names
        result["_action_kinds"] = [call[0] for call in agent.actions.calls]
        return result

    result = asyncio.run(scenario())
    diagnosis = result.get("diagnosis") or []
    if isinstance(diagnosis, list):
        diagnosis_text = " ".join(str(x) for x in diagnosis)
    else:
        diagnosis_text = str(diagnosis)

    assert result.get("authority") == "legacy_full_loop"
    assert result.get("clinical_engine") == "full_clinical_loop"
    assert result.get("finished") is True
    assert "未分化临床综合征" not in diagnosis_text
    # Accept either concrete respiratory leaf; exact leaf may vary by candidate merge order.
    assert ("上呼吸道感染" in diagnosis_text) or ("急性上呼吸道感染" in diagnosis_text) or ("急性支气管炎" in diagnosis_text)
    assert "ask_patient" in result["_action_kinds"]
    assert "order_examination" not in result["_action_kinds"]
    assert "prescribe_treatment" in result["_action_kinds"]
    # A5 rejects an axis without a mapped structured intent; it must not revive
    # the old category/item LLM fallback or catalog-default examination.
    assert "exam_category" not in result["_prompt_names"]
    assert "exam_item" not in result["_prompt_names"]
    # Intake/action progression is deterministic; LLM is reserved for clinical reasoning.
    assert "next_action" not in result["_prompt_names"]
    assert any(name.startswith("diagnostic_axis_consult") for name in result["_prompt_names"])
    assert len(result["_prompt_names"]) <= 8
    assert "disease_candidate" in result["_prompt_names"] or "disease_and_treatment" in result[
        "_prompt_names"
    ]
    assert "hypothesis_builder" not in result["_prompt_names"]


def test_full_clinical_loop_typed_rule_overrides_background_diagnosis(
    monkeypatch: Any,
) -> None:
    decisions = iter(
        [
            {"action": "ask_patient", "question": "请描述听力症状。", "reason": "collect symptoms"},
            {"action": "order_examination", "question": "", "reason": "controlled intent"},
            {"action": "final_diagnosis", "question": "", "reason": "exam complete"},
        ]
    )
    monkeypatch.setattr(
        legacy_orchestrator,
        "select_next_clinical_action",
        lambda _: next(decisions, {"action": "final_diagnosis", "question": "", "reason": "done"}),
    )

    class HearingActions(FakeDoctorActions):
        async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
            self.calls.append(("ask_patient", input_data.get("question")))
            return "持续耳鸣伴双侧高频听力下降；高血压只是既往病史，与耳部症状无关。"

        async def order_examination(
            self,
            patient_id: str,
            items: Iterable[str],
            reason: str = "",
        ) -> Dict[str, Any]:
            item_list = list(items)
            self.calls.append(("order_examination", item_list, reason))
            return {
                "results": {
                    name: {
                        "status": "abnormal",
                        "result": {"高频听阈": "4kHz 75dB，8kHz 85dB"},
                        "abnormal_indicators": ["高频听阈"],
                    }
                    for name in item_list
                }
            }

    async def scenario() -> Dict[str, Any]:
        agent = MyDoctorAgent(
            config={"log_llm_prompts": False},
            memory=VerifiedOnlyMemory(),
            runtime_identity=TEST_RUNTIME_IDENTITY,
            rule_pack=parse_compiled_rule_pack(active_diagnosis_priority_pack_payload()),
        )
        actions = HearingActions()
        agent.actions = actions

        async def controlled_axis_consult(**_: Any) -> Dict[str, Any]:
            return {
                "diagnosis_axes": [
                    {
                        "axis_id": "acute_audio_vestibular_differential",
                        "status": "suspected",
                        "missing_evidence": ["听力损失程度"],
                        "exam_intents": ["exam_intent_quantitative_hearing_assessment"],
                    }
                ],
                "treatment_risks": [],
            }

        agent._diagnostic_axis_consult = controlled_axis_consult  # type: ignore[method-assign]

        async def fake_llm(**kwargs: Any) -> Dict[str, Any]:
            name = str(kwargs.get("prompt_name") or "")
            if name.startswith("diagnostic_axis_consult"):
                return {
                    "intake_facts": {},
                    "diagnosis_axes": [
                        {
                            "axis_id": "acute_audio_vestibular_differential",
                            "status": "suspected",
                            "missing_evidence": ["听力损失程度"],
                            "exam_intents": [
                                "exam_intent_quantitative_hearing_assessment"
                            ],
                        }
                    ],
                    "treatment_risks": [],
                }

            if name == "diagnostic_context":
                return {
                    "case_features": {"symptoms": ["耳鸣", "高频听力下降"]},
                    "differential": ["原发性高血压", "耳鸣"],
                    "normalization_suggestions": [],
                }
            if name == "disease_candidate":
                return {
                    "diagnosis": "原发性高血压",
                    "treatment_plan": "评估听力损失程度，避免噪声暴露并安排耳鼻喉科随访。",
                    "reasoning": "模型误将基础病作为主诊断。",
                }
            if name in {"exam_category", "exam_item"}:
                return {
                    "category": "功能检查",
                    "examinations": ["听力测定"],
                    "reason": "明确高频听力损失",
                }
            if name == "treatment_review":
                return {
                    "treatment_plan": "评估听力损失程度，避免噪声暴露并安排耳鼻喉科随访。",
                    "reasoning": "按耳部当前问题处理。",
                }
            return {}

        agent._call_llm = fake_llm  # type: ignore[method-assign]
        result = await agent.test(patient_id="offline-typed-rule-case")
        result["_calls"] = list(actions.calls)
        return result

    result = asyncio.run(scenario())

    assert result["diagnosis"] == ["耳鸣"]
    prescription_calls = [call for call in result["_calls"] if call[0] == "prescribe_treatment"]
    assert len(prescription_calls) == 1
    assert prescription_calls[0][1] == ["耳鸣"]
