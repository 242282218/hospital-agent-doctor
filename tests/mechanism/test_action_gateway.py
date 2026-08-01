from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pytest

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    FinalSubmissionCoordinator,
    LoadedRuntimeIdentity,
    build_prescribe_command,
)
from agent.runtime.action_gateway import (
    ActionCommand,
    ActionGateway,
    ObservationEnvelope,
    build_action_command,
)
from agent.runtime.sdk_adapter import SdkActionAdapter


class FakeDoctorActions:
    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.failure: Optional[BaseException] = None

    def _raise_if_needed(self) -> None:
        if self.failure is not None:
            raise self.failure

    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        self.calls.append(("ask_patient", patient_id, input_data))
        self._raise_if_needed()
        return "患者回答"

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        item_list = list(items)
        self.calls.append(("order_examination", patient_id, item_list, reason))
        self._raise_if_needed()
        return {
            "results": {
                name: {
                    "result": "正常" if name != "胸部CT" else "无效检查",
                    "status": "normal" if name != "胸部CT" else "invalid",
                    "abnormal_indicators": [],
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
        self.calls.append(
            ("prescribe_treatment", patient_id, diagnosis, treatment_plan, reasoning)
        )
        self._raise_if_needed()
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
        raise AssertionError("Gateway must not call evaluation")


def make_gateway(actions: FakeDoctorActions) -> ActionGateway:
    adapter = SdkActionAdapter(actions=actions, patient_id="transport-case-id")
    return ActionGateway(adapter=adapter, valid_examinations={"血常规", "胸部CT"})


def make_verified_prescribe_command(*, action_sequence: int):
    release_id = "test-release"
    registry = FinalAuthorizationRegistry(release_identity_hash=release_id)
    coordinator = FinalSubmissionCoordinator(
        registry=registry,
        runtime_identity=LoadedRuntimeIdentity(
            status="strict_verified",
            identity_hash=release_id,
        ),
        apply_safety=lambda payload, context: payload,
        run_legacy_verifier=lambda payload: {"passed": True},
        run_five_dimension_gate=lambda payload: (payload, {"passed": True}),
        converge=lambda payload: payload,
    )
    payload = FinalPayload(
        diagnoses=("肺炎",),
        treatment_plan="抗感染并监测。",
        reasoning="结合病史和检查。",
    )
    verified = coordinator.verify(payload=payload, clinical_context={})
    prepared = coordinator.authorize(
        verified=verified,
        case_run_id="run-1",
        revision=0,
        snapshot_hash="snapshot-0",
    )
    return build_prescribe_command(
        registry=registry,
        ticket=prepared.ticket,
        case_run_id="run-1",
        blackboard_revision=0,
        action_sequence=action_sequence,
        payload={
            "diagnosis": list(verified.payload.diagnoses),
            "treatment_plan": verified.payload.treatment_plan,
            "reasoning": verified.payload.reasoning,
        },
        snapshot_hash="snapshot-0",
        legacy_verifier_hash=verified.legacy_verifier_hash,
        five_dimension_gate_hash=verified.five_dimension_gate_hash,
        issue_codes=verified.issue_codes,
    )


def test_gateway_preserves_sdk_argument_shape_and_success_values() -> None:
    async def scenario() -> tuple[FakeDoctorActions, list[Any]]:
        actions = FakeDoctorActions()
        gateway = make_gateway(actions)
        chat_history = [{"from": "patient", "text": "发热三天"}]

        ask = await gateway.execute(
            build_action_command(
                case_run_id="run-1",
                blackboard_revision=0,
                action_sequence=1,
                action_type="ask_patient",
                payload={"question": "是否咳嗽？"},
            ),
            chat_history=chat_history,
        )
        exam = await gateway.execute(
            build_action_command(
                case_run_id="run-1",
                blackboard_revision=0,
                action_sequence=2,
                action_type="order_examination",
                payload={"items": ["血常规", "胸部CT"], "reason": "鉴别感染"},
            )
        )
        final = await gateway.execute(
            make_verified_prescribe_command(action_sequence=3)
        )
        return actions, [ask.raw_result, exam.raw_result, final.raw_result]

    actions, values = asyncio.run(scenario())

    assert actions.calls == [
        (
            "ask_patient",
            "transport-case-id",
            {"question": "是否咳嗽？", "chat_history": [{"from": "patient", "text": "发热三天"}]},
        ),
        ("order_examination", "transport-case-id", ["血常规", "胸部CT"], "鉴别感染"),
        (
            "prescribe_treatment",
            "transport-case-id",
            ["肺炎"],
            "抗感染并监测。",
            "结合病史和检查。",
        ),
    ]
    assert values[0] == "患者回答"
    assert values[1]["results"]["胸部CT"]["status"] == "invalid"
    assert values[2]["finished"] is True


def test_gateway_rejects_directly_constructed_prescribe_command() -> None:
    async def scenario() -> FakeDoctorActions:
        actions = FakeDoctorActions()
        gateway = make_gateway(actions)
        command = ActionCommand(
            command_id="cmd-forged",
            case_run_id="run-1",
            blackboard_revision=0,
            action_sequence=1,
            action_type="prescribe_treatment",
            payload={
                "diagnosis": ["肺炎"],
                "treatment_plan": "未授权方案",
                "reasoning": "无",
            },
            reason_evidence_ids=(),
            idempotency_key="forged",
        )
        with pytest.raises(ValueError, match="not authorized"):
            await gateway.execute(command)
        return actions

    assert asyncio.run(scenario()).calls == []


def test_verified_prescribe_payload_is_deeply_immutable() -> None:
    command = make_verified_prescribe_command(action_sequence=2)

    with pytest.raises(TypeError):
        command.payload["treatment_plan"] = "授权后篡改"  # type: ignore[index]
    with pytest.raises(TypeError):
        command.payload["diagnosis"][0] = "错误诊断"  # type: ignore[index]


def test_gateway_preserves_optional_structured_findings_in_raw_result() -> None:
    class StructuredFindingActions(FakeDoctorActions):
        async def order_examination(
            self,
            patient_id: str,
            items: Iterable[str],
            reason: str = "",
        ) -> Dict[str, Any]:
            self.calls.append(("order_examination", patient_id, list(items), reason))
            return {
                "results": {
                    "血常规": {
                        "status": "abnormal",
                        "result": {"opaque": True},
                        "structured_findings": [
                            {
                                "schema_version": "exam-axis-evidence-contract/v1",
                                "finding_code": "controlled_respiratory_finding_against_ocular_axis",
                                "polarity": "present",
                                "target_system_id": "respiratory",
                                "source_evidence_id": "sdk:exam:controlled:001",
                            }
                        ],
                    }
                }
            }

    async def scenario() -> ObservationEnvelope:
        actions = StructuredFindingActions()
        gateway = make_gateway(actions)
        return await gateway.execute(
            build_action_command(
                case_run_id="run-structured-findings",
                blackboard_revision=0,
                action_sequence=1,
                action_type="order_examination",
                payload={"items": ["血常规"], "reason": "受控 finding 传输"},
            )
        )

    envelope = asyncio.run(scenario())
    assert envelope.observation.status == "succeeded"
    assert envelope.raw_result["results"]["血常规"]["status"] == "abnormal"
    assert envelope.raw_result["results"]["血常规"]["structured_findings"] == [
        {
            "schema_version": "exam-axis-evidence-contract/v1",
            "finding_code": "controlled_respiratory_finding_against_ocular_axis",
            "polarity": "present",
            "target_system_id": "respiratory",
            "source_evidence_id": "sdk:exam:controlled:001",
        }
    ]


def test_command_payload_rejects_patient_id() -> None:
    with pytest.raises(ValueError, match="patient_id"):
        build_action_command(
            case_run_id="run-1",
            blackboard_revision=0,
                action_sequence=4,
            action_type="ask_patient",
            payload={"patient_id": "forbidden", "question": "主诉？"},
        )


def test_gateway_rejects_non_catalog_exam_without_sdk_call() -> None:
    async def scenario() -> tuple[FakeDoctorActions, Any]:
        actions = FakeDoctorActions()
        gateway = make_gateway(actions)
        envelope = await gateway.execute(
            build_action_command(
                case_run_id="run-1",
                blackboard_revision=0,
                action_sequence=5,
                action_type="order_examination",
                payload={"items": ["不存在检查"], "reason": "验证目录"},
            )
        )
        return actions, envelope

    actions, envelope = asyncio.run(scenario())

    assert actions.calls == []
    assert envelope.observation.status == "invalid"
    assert envelope.raw_result["results"]["不存在检查"]["status"] == "invalid"


def test_gateway_deduplicates_same_command() -> None:
    async def scenario() -> FakeDoctorActions:
        actions = FakeDoctorActions()
        gateway = make_gateway(actions)
        command = build_action_command(
            case_run_id="run-1",
            blackboard_revision=0,
                action_sequence=6,
            action_type="ask_patient",
            payload={"question": "是否咳嗽？"},
        )
        first = await gateway.execute(command, chat_history=[])
        second = await gateway.execute(command, chat_history=[])
        assert first is second
        return actions

    actions = asyncio.run(scenario())
    assert len(actions.calls) == 1


def test_gateway_records_outcome_unknown_and_reraises_same_exception() -> None:
    async def scenario() -> tuple[FakeDoctorActions, ActionGateway, BaseException]:
        actions = FakeDoctorActions()
        failure = TimeoutError("transport timeout")
        actions.failure = failure
        gateway = make_gateway(actions)
        command = build_action_command(
            case_run_id="run-1",
            blackboard_revision=0,
                action_sequence=7,
            action_type="ask_patient",
            payload={"question": "是否咳嗽？"},
        )
        for _ in range(2):
            try:
                await gateway.execute(command, chat_history=[])
            except BaseException as exc:
                assert exc is failure
        return actions, gateway, failure

    actions, gateway, failure = asyncio.run(scenario())

    assert len(actions.calls) == 1
    assert gateway.trace[-1].observation.status == "outcome_unknown"
    assert gateway.trace[-1].error_type == type(failure).__name__




def test_gateway_reconciles_ambiguous_result_without_dispatching_again() -> None:
    class ReconcileAdapter:
        def __init__(self) -> None:
            self.dispatch_calls = 0
            self.reconcile_calls = 0

        async def dispatch(self, command: ActionCommand, *, chat_history: Sequence[Mapping[str, Any]]) -> str:
            self.dispatch_calls += 1
            raise TimeoutError("response lost")

        async def reconcile(self, command: ActionCommand, *, chat_history: Sequence[Mapping[str, Any]]) -> str:
            self.reconcile_calls += 1
            return "recovered answer"

    async def scenario() -> tuple[ReconcileAdapter, ObservationEnvelope]:
        adapter = ReconcileAdapter()
        gateway = ActionGateway(adapter=adapter, valid_examinations=())
        command = build_action_command(
            case_run_id="run-reconcile",
            blackboard_revision=0,
            action_sequence=1,
            action_type="ask_patient",
            payload={"question": "咳嗽？"},
        )
        with pytest.raises(TimeoutError):
            await gateway.execute(command)
        envelope = await gateway.reconcile(command)
        return adapter, envelope

    adapter, envelope = asyncio.run(scenario())
    assert adapter.dispatch_calls == 1
    assert adapter.reconcile_calls == 1
    assert envelope.observation.status == "succeeded"
    assert envelope.raw_result == "recovered answer"


def test_gateway_reconcile_requires_adapter_support() -> None:
    async def scenario() -> None:
        actions = FakeDoctorActions()
        gateway = make_gateway(actions)
        command = build_action_command(
            case_run_id="run-no-reconcile",
            blackboard_revision=0,
            action_sequence=1,
            action_type="ask_patient",
            payload={"question": "咳嗽？"},
        )
        actions.failure = TimeoutError("response lost")
        with pytest.raises(TimeoutError):
            await gateway.execute(command)
        with pytest.raises(RuntimeError, match="reconciliation"):
            await gateway.reconcile(command)

    asyncio.run(scenario())


class BlockingAdapter:
    """Test adapter that blocks so concurrent calls share one future."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def dispatch(
        self,
        command: Any,
        *,
        chat_history: Sequence[Mapping[str, Any]],
    ) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return "patient-answer"


def test_same_command_concurrent_execute_dispatches_once() -> None:
    async def scenario() -> tuple[int, ObservationEnvelope, ObservationEnvelope]:
        adapter = BlockingAdapter()
        gateway = ActionGateway(adapter=adapter, valid_examinations=())
        command = build_action_command(
            case_run_id="run-1",
            blackboard_revision=0,
            action_sequence=8,
            action_type="ask_patient",
            payload={"question": "咳嗽？"},
        )
        first = asyncio.create_task(gateway.execute(command))
        await adapter.started.wait()
        second = asyncio.create_task(gateway.execute(command))
        # ensure the second task has reached the inflight-sharing branch
        await asyncio.sleep(0)
        adapter.release.set()
        left, right = await asyncio.gather(first, second)
        return adapter.calls, left, right

    calls, left, right = asyncio.run(scenario())
    assert calls == 1, "adapter must be called exactly once under concurrency"
    assert left == right, "both callers must observe the same envelope"


def test_concurrent_provider_error_both_waiters_see_same_exception() -> None:
    async def scenario() -> tuple[int, list[Optional[BaseException]]]:
        from agent.runtime.action_gateway import ObservationEnvelope  # noqa: F401
        adapter = BlockingAdapter()
        failure = RuntimeError("provider outage")
        observed: list[Optional[BaseException]] = []

        async def failing_dispatch(
            command: Any,
            *,
            chat_history: Sequence[Mapping[str, Any]],
        ) -> str:
            adapter.calls += 1
            adapter.started.set()
            await adapter.release.wait()
            raise failure

        adapter.dispatch = failing_dispatch  # type: ignore[assignment]
        gateway = ActionGateway(adapter=adapter, valid_examinations=())
        command = build_action_command(
            case_run_id="run-1",
            blackboard_revision=0,
            action_sequence=9,
            action_type="ask_patient",
            payload={"question": "咳嗽？"},
        )
        first = asyncio.create_task(gateway.execute(command))
        await adapter.started.wait()
        second = asyncio.create_task(gateway.execute(command))
        await asyncio.sleep(0)
        adapter.release.set()
        for task in (first, second):
            try:
                await task
                observed.append(None)
            except BaseException as exc:
                observed.append(exc)
        return adapter.calls, observed

    calls, observed = asyncio.run(scenario())
    assert calls == 1, "adapter called exactly once despite dual failure"
    assert observed[0] is observed[1] is not None
    assert observed[0] is observed[1], "both waiters share the same error object"
    assert all(isinstance(o, RuntimeError) and str(o) == "provider outage" for o in observed)


def test_cancelled_producer_cancels_waiter_and_cleans_inflight() -> None:
    async def scenario() -> tuple[ActionGateway, int]:
        adapter = BlockingAdapter()
        gateway = ActionGateway(adapter=adapter, valid_examinations=())
        command = build_action_command(
            case_run_id="run-cancel",
            blackboard_revision=0,
            action_sequence=1,
            action_type="ask_patient",
            payload={"question": "咳嗽？"},
        )
        producer = asyncio.create_task(gateway.execute(command))
        await adapter.started.wait()
        waiter = asyncio.create_task(gateway.execute(command))
        await asyncio.sleep(0)
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=1)
        return gateway, adapter.calls

    gateway, calls = asyncio.run(scenario())
    assert calls == 1
    assert gateway._inflight == {}
    assert gateway._inflight_command == {}
