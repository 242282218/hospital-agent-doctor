"""P1: action_sequence distinguishes new same-payload actions from explicit retries."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytest

import agent.clinical.online_runtime as online_runtime
from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalSubmissionCoordinator,
    LoadedRuntimeIdentity,
)
from agent.clinical.online_runtime import OnlineActionBridge
from agent.clinical.orchestrator import ClinicalOrchestrator
from agent.runtime.action_gateway import ActionGateway, build_action_command
from agent.runtime.sdk_adapter import SdkActionAdapter


class CountingActions:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        self.calls.append(("ask", input_data.get("question")))
        return "ok"

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        self.calls.append(("order", list(items), reason))
        return {"results": {name: {"status": "normal", "result": "正常"} for name in items}}

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: Any,
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        self.calls.append(("prescribe", diagnosis, treatment_plan))
        return {
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
        raise AssertionError("no evaluation")


def test_same_payload_different_sequence_is_new_action() -> None:
    async def scenario() -> CountingActions:
        actions = CountingActions()
        gateway = ActionGateway(
            adapter=SdkActionAdapter(actions=actions, patient_id="seq-case"),
            valid_examinations={"血常规"},
        )
        first = build_action_command(
            case_run_id="run-seq",
            blackboard_revision=0,
            action_sequence=1,
            action_type="ask_patient",
            payload={"question": "是否发热？"},
        )
        second = build_action_command(
            case_run_id="run-seq",
            blackboard_revision=0,
            action_sequence=2,
            action_type="ask_patient",
            payload={"question": "是否发热？"},
        )
        assert first.command_id != second.command_id
        await gateway.execute(first, chat_history=[])
        await gateway.execute(second, chat_history=[])
        return actions

    actions = asyncio.run(scenario())
    assert len(actions.calls) == 2


def test_same_command_explicit_retry_is_idempotent() -> None:
    async def scenario() -> CountingActions:
        actions = CountingActions()
        gateway = ActionGateway(
            adapter=SdkActionAdapter(actions=actions, patient_id="seq-case"),
            valid_examinations={"血常规"},
        )
        command = build_action_command(
            case_run_id="run-seq",
            blackboard_revision=0,
            action_sequence=1,
            action_type="ask_patient",
            payload={"question": "是否发热？"},
        )
        first = await gateway.execute(command, chat_history=[])
        second = await gateway.execute(command, chat_history=[])
        assert first is second
        return actions

    actions = asyncio.run(scenario())
    assert len(actions.calls) == 1


def test_online_bridge_sequence_is_monotonic_across_action_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_id = "sequence-test-release"

    def make_test_coordinator(*, registry, runtime_identity, clinical_context):
        _ = clinical_context
        return FinalSubmissionCoordinator(
            registry=registry,
            runtime_identity=runtime_identity,
            apply_safety=lambda payload, context: payload,
            run_legacy_verifier=lambda payload: {"passed": True},
            run_five_dimension_gate=lambda payload: (payload, {"passed": True}),
            converge=lambda payload: payload,
        )

    monkeypatch.setattr(online_runtime, "build_case_coordinator", make_test_coordinator)

    async def scenario() -> List[int]:
        actions = CountingActions()
        gateway = ActionGateway(
            adapter=SdkActionAdapter(actions=actions, patient_id="seq-case"),
            valid_examinations={"血常规"},
        )
        orchestrator = ClinicalOrchestrator(case_run_id="run-seq")
        bridge = OnlineActionBridge(
            gateway=gateway,
            orchestrator=orchestrator,
            final_registry=FinalAuthorizationRegistry(release_identity_hash=release_id),
            runtime_identity=LoadedRuntimeIdentity(
                status="strict_verified",
                identity_hash=release_id,
            ),
        )
        sequences: List[int] = []

        # Capture sequences via gateway command fingerprints by monkeypatching execute.
        original = gateway.execute

        async def tracking_execute(command, *, chat_history=()):
            sequences.append(int(command.action_sequence))
            return await original(command, chat_history=chat_history)

        gateway.execute = tracking_execute  # type: ignore[method-assign]
        await bridge.ask(question="q1", chat_history=[])
        await bridge.order(items=["血常规"], reason="r1")
        await bridge.prescribe_with_authorization(
            payload={
                "diagnosis": ["上呼吸道感染"],
                "treatment_plan": "对症支持。",
                "reasoning": "依据症状。",
            },
            clinical_context={},
        )
        return sequences

    sequences = asyncio.run(scenario())
    assert sequences == [1, 2, 3]


def test_successful_prescribe_is_not_retried_when_trace_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_id = "trace-degraded-release"

    def make_test_coordinator(*, registry, runtime_identity, clinical_context):
        _ = clinical_context
        return FinalSubmissionCoordinator(
            registry=registry,
            runtime_identity=runtime_identity,
            apply_safety=lambda payload, context: payload,
            run_legacy_verifier=lambda payload: {"passed": True},
            run_five_dimension_gate=lambda payload: (payload, {"passed": True}),
            converge=lambda payload: payload,
        )

    monkeypatch.setattr(online_runtime, "build_case_coordinator", make_test_coordinator)

    async def scenario() -> tuple[CountingActions, Dict[str, Any]]:
        actions = CountingActions()
        gateway = ActionGateway(
            adapter=SdkActionAdapter(actions=actions, patient_id="trace-case"),
            valid_examinations=(),
        )
        writes = 0

        def flaky_sink(event: Mapping[str, Any]) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("disk full")

        bridge = OnlineActionBridge(
            gateway=gateway,
            orchestrator=ClinicalOrchestrator(case_run_id="trace-run"),
            event_sink=flaky_sink,
            final_registry=FinalAuthorizationRegistry(release_identity_hash=release_id),
            runtime_identity=LoadedRuntimeIdentity(
                status="strict_verified",
                identity_hash=release_id,
            ),
        )
        result = await bridge.prescribe_with_authorization(
            payload={
                "diagnosis": ["上呼吸道感染"],
                "treatment_plan": "对症支持并监测。",
                "reasoning": "依据症状。",
            },
            clinical_context={},
        )
        return actions, result

    actions, result = asyncio.run(scenario())
    assert [call[0] for call in actions.calls] == ["prescribe"]
    assert result["finished"] is True
    assert result["trace_degraded"] is True
