from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union


ACTION_PAYLOAD_KEYS = {
    "ask_patient": {"question"},
    "order_examination": {"items", "reason"},
    "prescribe_treatment": {"diagnosis", "treatment_plan", "reasoning"},
}
_PRESCRIBE_CAPABILITY = object()


@dataclass(frozen=True)
class ActionCommand:
    command_id: str
    case_run_id: str
    blackboard_revision: int
    action_sequence: int
    action_type: str
    payload: Mapping[str, Any]
    reason_evidence_ids: Tuple[str, ...]
    idempotency_key: str
    _authorization_capability: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class PatientObservation:
    command_id: str
    status: str
    answer_ref: str


@dataclass(frozen=True)
class ExaminationItemObservation:
    requested_name: str
    status: str
    result_ref: str


@dataclass(frozen=True)
class ExaminationObservation:
    command_id: str
    status: str
    items: Tuple[ExaminationItemObservation, ...]


@dataclass(frozen=True)
class FinalReceipt:
    command_id: str
    status: str
    finished: bool
    final_result_ref: str


ActionObservation = Union[PatientObservation, ExaminationObservation, FinalReceipt]


@dataclass(frozen=True)
class ObservationEnvelope:
    command: ActionCommand
    dispatch_status: str
    received_at: str
    observation: ActionObservation
    raw_result: Any
    error_type: str = ""


class ActionTransport(Protocol):
    async def dispatch(
        self,
        command: ActionCommand,
        *,
        chat_history: Sequence[Mapping[str, Any]],
    ) -> Any:
        pass


@dataclass(frozen=True)
class _DispatchRecord:
    envelope: ObservationEnvelope
    error: Optional[Exception]


def _build_validated_command(
    *,
    case_run_id: str,
    blackboard_revision: int,
    action_sequence: int,
    action_type: str,
    payload: Mapping[str, Any],
    reason_evidence_ids: Iterable[str] = (),
    authorization_capability: Any = None,
) -> ActionCommand:
    # Private construction path shared by the public factory and A1's dormant
    # prescribe command factory. The public factory keeps its fail-open shape;
    # A2 will make prescribe fail-closed in build_action_command itself.
    clean_payload = dict(payload)
    if action_type == "prescribe_treatment" and authorization_capability is not _PRESCRIBE_CAPABILITY:
        raise ValueError("prescribe_treatment requires authorization capability")
    if "patient_id" in clean_payload:
        raise ValueError("patient_id is forbidden in ActionCommand payload")
    sequence = int(action_sequence)
    if sequence < 1:
        raise ValueError("action_sequence must be >= 1")
    evidence_ids = tuple(str(item) for item in reason_evidence_ids)
    # Sequence distinguishes a new same-payload action from an explicit retry
    # that reuses the same ActionCommand object/idempotency_key.
    fingerprint = {
        "case_run_id": str(case_run_id),
        "action_sequence": sequence,
        "action_type": str(action_type),
        "payload": clean_payload,
        "reason_evidence_ids": evidence_ids,
    }
    digest = hashlib.sha256(_canonical_json(fingerprint).encode("utf-8")).hexdigest()
    command = ActionCommand(
        command_id="cmd-%s" % digest[:24],
        case_run_id=str(case_run_id),
        blackboard_revision=int(blackboard_revision),
        action_sequence=sequence,
        action_type=str(action_type),
        payload=_freeze_mapping(clean_payload),
        reason_evidence_ids=evidence_ids,
        idempotency_key=digest,
        _authorization_capability=authorization_capability,
    )
    _validate_command(command)
    return command


def build_action_command(
    *,
    case_run_id: str,
    blackboard_revision: int,
    action_sequence: int,
    action_type: str,
    payload: Mapping[str, Any],
    reason_evidence_ids: Iterable[str] = (),
) -> ActionCommand:
    if action_type == "prescribe_treatment":
        raise ValueError(
            "prescribe_treatment must be built through final_submission.build_prescribe_command"
        )
    return _build_validated_command(
        case_run_id=case_run_id,
        blackboard_revision=blackboard_revision,
        action_sequence=action_sequence,
        action_type=action_type,
        payload=payload,
        reason_evidence_ids=reason_evidence_ids,
    )


class ActionGateway:
    def __init__(
        self,
        *,
        adapter: ActionTransport,
        valid_examinations: Iterable[str],
    ) -> None:
        self._adapter = adapter
        self._valid_examinations = {
            str(item).strip() for item in valid_examinations if str(item).strip()
        }
        self._records: Dict[str, _DispatchRecord] = {}
        self._trace: list[ObservationEnvelope] = []
        # In-flight futures keyed by idempotency_key so concurrent dispatches
        # of the SAME command share one adapter call; same key with a different
        # command is rejected before dispatch.
        self._inflight: Dict[str, "asyncio.Future[ObservationEnvelope]"] = {}
        self._inflight_command: Dict[str, ActionCommand] = {}
        self._dispatch_lock = asyncio.Lock()

    @property
    def trace(self) -> Tuple[ObservationEnvelope, ...]:
        return tuple(self._trace)

    async def execute(
        self,
        command: ActionCommand,
        *,
        chat_history: Sequence[Mapping[str, Any]] = (),
    ) -> ObservationEnvelope:
        _validate_command(command)
        key = command.idempotency_key
        # Critical section: decide cached / inflight-match / create-new. We never
        # await inside the lock. If another caller is already dispatching the
        # SAME command we capture its future and await it OUTSIDE the lock so the
        # producer can reacquire the lock to resolve it (no deadlock).
        await_future: Optional["asyncio.Future[ObservationEnvelope]"] = None
        async with self._dispatch_lock:
            cached = self._records.get(key)
            if cached is not None:
                if cached.envelope.command != command:
                    raise ValueError("idempotency_key reused for a different command")
                if cached.error is not None:
                    raise cached.error
                return cached.envelope
            existing = self._inflight.get(key)
            if existing is not None:
                if self._inflight_command[key] != command:
                    raise ValueError("idempotency_key reused for a different command")
                await_future = existing
            else:
                future = asyncio.get_event_loop().create_future()
                self._inflight[key] = future
                self._inflight_command[key] = command
        if await_future is not None:
            # Waiter shares the in-flight future; shield so the producer still
            # resolves even if a waiter is cancelled.
            return await asyncio.shield(await_future)
        # Producer path: dispatch exactly once. Whatever happens, we resolve the
        # future and persist the record so later callers see the cached result.
        envelope: ObservationEnvelope
        error: Optional[Exception]
        try:
            invalid_items = self._invalid_exam_items(command)
            if invalid_items:
                envelope = _local_invalid_envelope(command, invalid_items)
                error = None
            else:
                try:
                    raw_result = await self._adapter.dispatch(
                        command,
                        chat_history=chat_history,
                    )
                except Exception as exc:
                    envelope = _outcome_unknown_envelope(command, exc)
                    error = exc
                else:
                    envelope = _success_envelope(command, raw_result)
                    error = None
        except BaseException as exc:  # cancellation must also resolve shared waiters
            async with self._dispatch_lock:
                self._inflight.pop(key, None)
                self._inflight_command.pop(key, None)
            if not future.done():
                if isinstance(exc, asyncio.CancelledError):
                    future.cancel()
                else:
                    future.set_exception(exc)
            raise
        async with self._dispatch_lock:
            self._records[key] = _DispatchRecord(envelope, error)
            self._trace.append(envelope)
            self._inflight.pop(key, None)
            self._inflight_command.pop(key, None)
            if error is not None and not future.done():
                future.set_exception(error)
            elif not future.done():
                future.set_result(envelope)
        # Leaving the lock: re-raise the captured error so the first caller
        # sees the same exception, just like the cached path does.
        if error is not None:
            future.exception()  # mark producer-set exception as retrieved
            raise error
        return envelope

    def _invalid_exam_items(self, command: ActionCommand) -> Tuple[str, ...]:
        if command.action_type != "order_examination":
            return ()
        return tuple(
            item
            for item in _text_items(command.payload.get("items"))
            if item not in self._valid_examinations
        )

    async def reconcile(
        self,
        command: ActionCommand,
        *,
        chat_history: Sequence[Mapping[str, Any]] = (),
    ) -> ObservationEnvelope:
        """Resolve an ambiguous transport result without blindly re-dispatching.

        Adapters may expose an optional ``reconcile`` method backed by a remote
        receipt/status endpoint. A missing reconciler is explicit: the gateway
        never guesses whether a side effect happened and never retries it here.
        """
        _validate_command(command)
        key = command.idempotency_key
        async with self._dispatch_lock:
            record = self._records.get(key)
            if record is None:
                raise ValueError("cannot reconcile an unknown command")
            if record.envelope.command != command:
                raise ValueError("idempotency_key reused for a different command")
            if record.error is None:
                return record.envelope
        reconciler = getattr(self._adapter, "reconcile", None)
        if not callable(reconciler):
            raise RuntimeError("adapter does not support outcome reconciliation")
        raw_result = await reconciler(command, chat_history=chat_history)
        envelope = _success_envelope(command, raw_result)
        async with self._dispatch_lock:
            self._records[key] = _DispatchRecord(envelope, None)
            self._trace.append(envelope)
        return envelope

    def _save(
        self,
        command: ActionCommand,
        envelope: ObservationEnvelope,
        error: Optional[Exception],
    ) -> None:
        self._records[command.idempotency_key] = _DispatchRecord(envelope, error)
        self._trace.append(envelope)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expected_command_identity(command: ActionCommand) -> Tuple[str, str]:
    fingerprint = {
        "case_run_id": str(command.case_run_id),
        "action_sequence": int(command.action_sequence),
        "action_type": str(command.action_type),
        "payload": dict(command.payload),
        "reason_evidence_ids": tuple(command.reason_evidence_ids),
    }
    digest = hashlib.sha256(_canonical_json(fingerprint).encode("utf-8")).hexdigest()
    return "cmd-%s" % digest[:24], digest


def _validate_command(command: ActionCommand) -> None:
    if not isinstance(command, ActionCommand):
        raise ValueError("command must be an ActionCommand")
    expected_keys = ACTION_PAYLOAD_KEYS.get(command.action_type)
    if expected_keys is None:
        raise ValueError("unsupported action_type: %s" % command.action_type)
    if (
        command.action_type == "prescribe_treatment"
        and command._authorization_capability is not _PRESCRIBE_CAPABILITY
    ):
        raise ValueError("prescribe_treatment command is not authorized")
    expected_command_id, expected_key = _expected_command_identity(command)
    if command.command_id != expected_command_id or command.idempotency_key != expected_key:
        raise ValueError("ActionCommand fingerprint mismatch")
    payload = dict(command.payload)
    if "patient_id" in payload:
        raise ValueError("patient_id is forbidden in ActionCommand payload")
    if set(payload) != expected_keys:
        raise ValueError("invalid payload keys for %s" % command.action_type)
    if command.blackboard_revision < 0:
        raise ValueError("blackboard_revision must be non-negative")
    _validate_payload_values(command.action_type, payload)


def _validate_payload_values(action_type: str, payload: Mapping[str, Any]) -> None:
    if action_type == "ask_patient" and not str(payload["question"]).strip():
        raise ValueError("ask_patient question must not be empty")
    if action_type == "order_examination" and not _text_items(payload["items"]):
        raise ValueError("order_examination items must not be empty")
    if action_type == "prescribe_treatment":
        if not _text_items(payload["diagnosis"]):
            raise ValueError("prescribe_treatment diagnosis must not be empty")
        if not str(payload["treatment_plan"]).strip():
            raise ValueError("prescribe_treatment treatment_plan must not be empty")


def _text_items(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [value]
    return tuple(str(item).strip() for item in values if str(item).strip())


def _received_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_ref(command_id: str, field: str) -> str:
    return "runtime://%s/%s" % (command_id, field)


def _local_invalid_envelope(
    command: ActionCommand,
    invalid_items: Sequence[str],
) -> ObservationEnvelope:
    items = tuple(
        ExaminationItemObservation(name, "invalid", "") for name in invalid_items
    )
    raw_result = {
        "results": {
            name: {
                "result": "无效检查",
                "status": "invalid",
                "abnormal_indicators": [],
            }
            for name in invalid_items
        }
    }
    return ObservationEnvelope(
        command=command,
        dispatch_status="not_sent",
        received_at=_received_at(),
        observation=ExaminationObservation(command.command_id, "invalid", items),
        raw_result=raw_result,
    )


def _outcome_unknown_envelope(
    command: ActionCommand,
    error: Exception,
) -> ObservationEnvelope:
    if command.action_type == "ask_patient":
        observation: ActionObservation = PatientObservation(
            command.command_id, "outcome_unknown", ""
        )
    elif command.action_type == "order_examination":
        observation = ExaminationObservation(command.command_id, "outcome_unknown", ())
    else:
        observation = FinalReceipt(command.command_id, "outcome_unknown", False, "")
    return ObservationEnvelope(
        command=command,
        dispatch_status="unknown",
        received_at=_received_at(),
        observation=observation,
        raw_result=None,
        error_type=type(error).__name__,
    )


def _success_envelope(command: ActionCommand, raw_result: Any) -> ObservationEnvelope:
    if command.action_type == "ask_patient":
        observation: ActionObservation = PatientObservation(
            command.command_id,
            "succeeded",
            _runtime_ref(command.command_id, "answer"),
        )
    elif command.action_type == "order_examination":
        observation = _examination_observation(command, raw_result)
    else:
        finished = bool(raw_result.get("finished")) if isinstance(raw_result, dict) else False
        observation = FinalReceipt(
            command.command_id,
            "succeeded" if finished else "unavailable",
            finished,
            _runtime_ref(command.command_id, "final_result"),
        )
    return ObservationEnvelope(
        command=command,
        dispatch_status="sent",
        received_at=_received_at(),
        observation=observation,
        raw_result=raw_result,
    )


def _examination_observation(
    command: ActionCommand,
    raw_result: Any,
) -> ExaminationObservation:
    results = raw_result.get("results", {}) if isinstance(raw_result, dict) else {}
    items = []
    for name in _text_items(command.payload["items"]):
        row = results.get(name, {}) if isinstance(results, dict) else {}
        status = str(row.get("status") or "unavailable") if isinstance(row, dict) else "unavailable"
        if status not in {"normal", "abnormal", "invalid", "unavailable"}:
            status = "unavailable"
        result_ref = "" if status == "unavailable" else _runtime_ref(command.command_id, name)
        items.append(ExaminationItemObservation(name, status, result_ref))
    statuses = {item.status for item in items}
    if statuses == {"invalid"}:
        overall_status = "invalid"
    elif statuses <= {"normal", "abnormal"}:
        overall_status = "succeeded"
    else:
        overall_status = "partial"
    return ExaminationObservation(command.command_id, overall_status, tuple(items))
