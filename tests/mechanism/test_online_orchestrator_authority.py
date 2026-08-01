from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.clinical.final_submission import LoadedRuntimeIdentity
from agent.clinical.online_runtime import run_online_clinical_case
from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory, build_memory


TEST_RUNTIME_IDENTITY = LoadedRuntimeIdentity(
    status="strict_verified",
    identity_hash="online-authority-test-release",
)


class FakeDoctorActions:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        self.calls.append(("ask_patient", patient_id, input_data.get("question")))
        return "发热咳嗽三天，没有胸痛。"

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        item_list = list(items)
        self.calls.append(("order_examination", patient_id, item_list, reason))
        return {
            "results": {
                name: {"status": "abnormal", "result": {"flag": "阳性"}, "abnormal_indicators": ["flag"]}
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
        self.calls.append(("prescribe_treatment", patient_id, diagnosis, treatment_plan))
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


def test_production_run_agent_source_uses_online_orchestrator_only() -> None:
    source = inspect.getsource(MyDoctorAgent._run_agent)
    assert "run_online_clinical_case" in source
    assert "case_state" not in source
    assert "select_required_intake_question" not in source
    assert "append_case_reflection" not in source


def test_build_memory_is_verified_only_not_markdown() -> None:
    registry_path = Path(
        "releases/release_C_runtime_20260730-458743cac829/verified_registry.json"
    )
    memory = build_memory({"memory": {"verified_registry_path": str(registry_path)}})
    assert isinstance(memory, VerifiedOnlyMemory)
    try:
        memory.append_case_reflection(patient_id="x", evaluation_reflection={})
        assert False, "online write must fail"
    except RuntimeError:
        pass


def test_online_runtime_mutates_only_via_orchestrator_and_gateway() -> None:
    async def scenario() -> Dict[str, Any]:
        agent = MyDoctorAgent(
            config={"log_llm_prompts": False},
            memory=VerifiedOnlyMemory(),
            runtime_identity=TEST_RUNTIME_IDENTITY,
        )
        agent.actions = FakeDoctorActions()
        # Avoid real LLM path.
        async def no_llm(**kwargs):
            return {}

        agent._call_llm = no_llm  # type: ignore[method-assign]
        result = await agent.test(patient_id="transport-only")
        return result

    result = asyncio.run(scenario())
    assert result.get("authority") == "legacy_full_loop"
    assert result.get("finished") is True
    # Action observations advance the metadata revision; it must not remain a
    # static initial snapshot for the whole case.
    assert result.get("blackboard_revision", 0) > 0
    assert result.get("snapshot_hash")
    assert result.get("run_seal", {}).get("trace_hash")
    actions = agent_actions_from_result = None
    # Recover fake from agent if needed
    agent = MyDoctorAgent(
        config={},
        memory=VerifiedOnlyMemory(),
        runtime_identity=TEST_RUNTIME_IDENTITY,
    )
    fake = FakeDoctorActions()
    agent.actions = fake

    async def scenario2() -> FakeDoctorActions:
        async def no_llm(**kwargs):
            return {}

        agent._call_llm = no_llm  # type: ignore[method-assign]
        await run_online_clinical_case(
            agent=agent,
            patient_id="transport-only",
            mode="test",
            valid_examinations=["体格检查", "全血细胞计数（CBC）", "血常规"],
            official_diseases=["肺炎", "上呼吸道感染"],
            exam_intent_map=[],
        )
        return fake

    fake = asyncio.run(scenario2())
    kinds = [call[0] for call in fake.calls]
    assert "ask_patient" in kinds or "order_examination" in kinds
    assert "prescribe_treatment" in kinds
    assert "evaluation" not in kinds  # test mode
