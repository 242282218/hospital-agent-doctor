"""Online clinical path: ActionGateway + full clinical loop.

Honest architecture note (P1-1): the multi-step doctor loop in
`MyDoctorAgent.run_full_clinical_loop` is the real clinical authority.
ClinicalOrchestrator/Blackboard are retained only as run metadata carriers
(revision, snapshot hash, seal). They do not gate diagnosis, exam, or treatment
decisions online.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from uuid import uuid4

from agent.clinical.orchestrator import ClinicalOrchestrator
from agent.clinical.model import RuntimeEvent
from agent.observability.run_trace import RunTraceStore
from agent.observability.runtime_events import (
    SequencedEventSink,
    safe_action_command_event,
    safe_action_observation_event,
    safe_budget_event,
    safe_case_end_event,
    safe_case_error_event,
    safe_final_submission_event,
    safe_llm_attempt_event,
)
from agent.runtime import (
    ActionGateway,
    EvaluationAttemptStore,
    EvaluationCollector,
    SdkActionAdapter,
    build_action_command,
)

from agent.clinical.final_submission import (
    FinalAuthorizationRegistry,
    FinalPayload,
    LoadedRuntimeIdentity,
    build_prescribe_command,
)
from agent.clinical.final_submission_adapters import build_case_coordinator

BASE_DIR = Path(__file__).resolve().parents[2]


class OnlineActionBridge:
    """Gateway wrapper that stamps revision and emits action sequence events."""

    def __init__(
        self,
        *,
        gateway: ActionGateway,
        orchestrator: ClinicalOrchestrator,
        event_sink: Any = None,
        final_registry: Optional[FinalAuthorizationRegistry] = None,
        runtime_identity: Optional[LoadedRuntimeIdentity] = None,
    ) -> None:
        self._gateway = gateway
        self._orchestrator = orchestrator
        self._event_sink = event_sink
        self._trace_degraded = False
        self._action_sequence = 0
        # A2: per-case capability registry + runtime identity bind every
        # prescribe command to a verified authorization ticket. Without a
        # verified identity NO prescribe command can be built here.
        self._final_registry = final_registry
        self._runtime_identity = runtime_identity

    def _next_sequence(self) -> int:
        self._action_sequence += 1
        return self._action_sequence

    def _emit_event(self, event: Mapping[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event)
        except Exception:
            self._trace_degraded = True

    def _emit_command(self, command: Any, action_sequence: int) -> None:
        self._emit_event(
            safe_action_command_event(command, action_sequence=action_sequence)
        )

    def _advance_runtime_revision(self, *, action_type: str, command_id: str) -> None:
        self._orchestrator.apply_runtime_event(
            RuntimeEvent(
                event_id="observation-%s" % command_id,
                event_type="action_observation:%s" % action_type,
                input_revision=self._orchestrator.snapshot.revision,
                payload={"command_id": command_id},
            )
        )

    def _emit_observation(
        self,
        envelope: Any,
        *,
        action_sequence: int,
        command_id: str,
    ) -> None:
        self._emit_event(
            safe_action_observation_event(
                envelope,
                action_sequence=action_sequence,
                command_id=command_id,
            )
        )

    async def ask(self, *, question: str, chat_history: Sequence[Mapping[str, Any]]) -> str:
        action_sequence = self._next_sequence()
        command = build_action_command(
            case_run_id=self._orchestrator.case_run_id,
            blackboard_revision=self._orchestrator.snapshot.revision,
            action_sequence=action_sequence,
            action_type="ask_patient",
            payload={"question": question},
        )
        self._emit_command(command, action_sequence)
        try:
            envelope = await self._gateway.execute(command, chat_history=chat_history)
        except Exception as exc:
            self._emit_observation(
                {
                    "dispatch_status": "error",
                    "observation_status": "error",
                    "raw_result": {"error_type": type(exc).__name__},
                    "command_id": command.command_id,
                },
                action_sequence=action_sequence,
                command_id=command.command_id,
            )
            self._advance_runtime_revision(
                action_type=command.action_type, command_id=command.command_id
            )
            raise
        self._emit_observation(
            envelope, action_sequence=action_sequence, command_id=command.command_id
        )
        self._advance_runtime_revision(
            action_type=command.action_type, command_id=command.command_id
        )
        return str(envelope.raw_result)

    async def order(self, *, items: List[str], reason: str) -> Dict[str, Any]:
        action_sequence = self._next_sequence()
        command = self._orchestrator.build_exam_command(
            items=tuple(items),
            reason=reason,
            action_sequence=action_sequence,
        )
        self._emit_command(command, action_sequence)
        try:
            envelope = await self._gateway.execute(command)
        except Exception as exc:
            self._emit_observation(
                {
                    "dispatch_status": "error",
                    "observation_status": "error",
                    "raw_result": {"error_type": type(exc).__name__},
                    "command_id": command.command_id,
                },
                action_sequence=action_sequence,
                command_id=command.command_id,
            )
            self._advance_runtime_revision(
                action_type=command.action_type, command_id=command.command_id
            )
            raise
        self._emit_observation(
            envelope, action_sequence=action_sequence, command_id=command.command_id
        )
        self._advance_runtime_revision(
            action_type=command.action_type, command_id=command.command_id
        )
        return dict(envelope.raw_result or {})

    @property
    def case_run_id(self) -> str:
        return self._orchestrator.case_run_id

    @property
    def blackboard_revision(self) -> int:
        return self._orchestrator.snapshot.revision

    @property
    def snapshot_hash(self) -> str:
        return self._orchestrator.snapshot.snapshot_hash()

    def _require_final_authorization(self) -> None:
        if self._final_registry is None or self._runtime_identity is None:
            raise RuntimeError(
                "prescribe_treatment requires a verified runtime identity and a per-case "
                "authorization registry; this OnlineActionBridge was constructed without "
                "either and cannot bypass the A2 fail-closed submission gate."
            )
        if self._runtime_identity.status != "strict_verified":
            raise RuntimeError(
                "prescribe_treatment requires a strict_verified runtime identity (%s)"
                % self._runtime_identity.status
            )
        if not self._runtime_identity.identity_hash:
            raise RuntimeError("release identity_hash is empty")

    async def prescribe_with_authorization(
        self,
        *,
        payload: Mapping[str, Any],
        clinical_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """A2 unified submission entry: verify -> authorize -> commit.

        The bridge reads case_run_id / revision / snapshot_hash from the orchestrator; it
        never accepts those bindings from the caller. It also never accepts a caller-made
        verification receipt: only the case-bound coordinator can verify this payload.
        """
        self._require_final_authorization()
        diagnoses = tuple(
            str(item).strip()
            for item in (payload.get("diagnosis") or [])
            if str(item).strip()
        )
        if not diagnoses:
            raise ValueError("prescribe_with_authorization diagnosis must not be empty")
        final_payload = FinalPayload(
            diagnoses=diagnoses,
            treatment_plan=str(payload.get("treatment_plan") or ""),
            reasoning=str(payload.get("reasoning") or ""),
        )
        if not final_payload.treatment_plan.strip():
            raise ValueError("prescribe_with_authorization treatment_plan must not be empty")
        coordinator = build_case_coordinator(
            registry=self._final_registry,
            runtime_identity=self._runtime_identity,
            clinical_context=clinical_context,
        )
        verified = coordinator.verify(
            payload=final_payload,
            clinical_context=clinical_context,
        )
        prepared = coordinator.authorize(
            verified=verified,
            case_run_id=self.case_run_id,
            revision=self.blackboard_revision,
            snapshot_hash=self.snapshot_hash,
        )
        action_sequence = self._next_sequence()
        command = build_prescribe_command(
            registry=self._final_registry,
            ticket=prepared.ticket,
            case_run_id=self.case_run_id,
            blackboard_revision=self.blackboard_revision,
            action_sequence=action_sequence,
            payload={
                "diagnosis": list(verified.payload.diagnoses),
                "treatment_plan": verified.payload.treatment_plan,
                "reasoning": verified.payload.reasoning,
            },
            snapshot_hash=self.snapshot_hash,
            legacy_verifier_hash=verified.legacy_verifier_hash,
            five_dimension_gate_hash=verified.five_dimension_gate_hash,
            issue_codes=verified.issue_codes,
        )
        self._emit_command(command, action_sequence)
        try:
            envelope = await self._gateway.execute(command)
        except Exception as exc:
            self._emit_observation(
                {
                    "dispatch_status": "error",
                    "observation_status": "error",
                    "raw_result": {"error_type": type(exc).__name__},
                    "command_id": command.command_id,
                },
                action_sequence=action_sequence,
                command_id=command.command_id,
            )
            self._advance_runtime_revision(
                action_type=command.action_type, command_id=command.command_id
            )
            raise
        self._emit_observation(
            envelope, action_sequence=action_sequence, command_id=command.command_id
        )
        self._advance_runtime_revision(
            action_type=command.action_type, command_id=command.command_id
        )
        self._emit_event(
            safe_final_submission_event(
                payload_hash=verified.payload_hash,
                legacy_verifier_hash=verified.legacy_verifier_hash,
                five_dimension_gate_hash=verified.five_dimension_gate_hash,
                issue_codes=verified.issue_codes,
                patch_count=verified.patch_count,
                authorization_result="issued",
                command_id=command.command_id,
                action_sequence=action_sequence,
            )
        )
        result = dict(envelope.raw_result or {})
        if self._trace_degraded:
            result["trace_degraded"] = True
        return result

    async def prescribe(
        self,
        *,
        diagnosis: List[str],
        treatment_plan: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        """A2 fail-closed: the legacy direct prescribe shortcut is forbidden.

        All production paths must use prescribe_with_authorization so the verified payload
        is bound to a fresh authorization ticket. This shim stays only as an explicit
        fail-closed rail — it always raises — so any caller accidentally bypassing A2
        surfaces immediately instead of silently writing an unauthorized prescribe command.
        """
        raise RuntimeError(
            "OnlineActionBridge.prescribe no longer bypasses the A2 submit pipeline; "
            "use prescribe_with_authorization(payload, clinical_context) instead."
        )


async def run_online_clinical_case(
    *,
    agent: Any,
    patient_id: str,
    mode: str,
    valid_examinations: Sequence[str],
    official_diseases: Sequence[str],
    exam_intent_map: Optional[Sequence[Mapping[str, Any]]] = None,
    trace_root: Optional[Path] = None,
    evaluation_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one online case through the full clinical loop.

    Blackboard/orchestrator provide case_run_id + revision stamping for the
    ActionGateway. Clinical decisions come only from run_full_clinical_loop.
    This function is the sole RunTrace owner for seal/case_end/case_error.
    """
    if mode not in {"train", "test"}:
        raise ValueError("mode must be train or test")
    _ = official_diseases
    _ = exam_intent_map

    case_run_id = uuid4().hex
    orchestrator = ClinicalOrchestrator(case_run_id=case_run_id)
    sdk_adapter = SdkActionAdapter(actions=agent.actions, patient_id=patient_id)
    gateway = ActionGateway(adapter=sdk_adapter, valid_examinations=valid_examinations)
    evaluation_collector = EvaluationCollector(
        adapter=sdk_adapter,
        mode=mode,
        case_run_id=case_run_id,
        store=EvaluationAttemptStore(
            Path(evaluation_root)
            if evaluation_root is not None
            else BASE_DIR / "outputs" / "evaluation_attempts"
        ),
    )
    trace = RunTraceStore(
        (Path(trace_root) / case_run_id)
        if trace_root is not None
        else BASE_DIR / "outputs" / "run_traces" / case_run_id,
        run_id=case_run_id,
    )
    sink = SequencedEventSink(append=trace.append, case_run_id=case_run_id)
    # A2: per-case capability registry bound to the loaded release identity.
    runtime_identity = getattr(agent, "_final_runtime_identity", None)
    if runtime_identity is None or runtime_identity.status != "strict_verified":
        # Historical releases can still run ask/order for replay, but only a strict
        # runtime identity may mint a prescription authorization capability.
        final_registry = None
    else:
        final_registry = FinalAuthorizationRegistry(
            release_identity_hash=runtime_identity.identity_hash
        )
    actions = OnlineActionBridge(
        gateway=gateway,
        orchestrator=orchestrator,
        event_sink=sink,
        final_registry=final_registry,
        runtime_identity=runtime_identity,
    )
    catalog_leaves: Set[str] = {str(x) for x in valid_examinations}

    release_meta: Dict[str, Any] = {}
    if isinstance(getattr(agent, "config", None), dict):
        release_meta = dict((agent.config or {}).get("release_pack") or {})
    rule_pack = getattr(agent, "rule_pack", None)
    typed_rule_count = int(getattr(rule_pack, "rule_count", 0) or 0)
    sink(
        {
            "type": "case_start",
            "mode": mode,
            "authority": "legacy_full_loop",
            "clinical_engine": "full_clinical_loop",
            "catalog_leaf_count": len(catalog_leaves),
            "release_pack_hash": release_meta.get("pack_hash") or "",
            "typed_rule_count": typed_rule_count
            or int(release_meta.get("typed_rule_count") or 0),
            "runtime_identity_status": getattr(runtime_identity, "status", "legacy_unverified"),
            "runtime_identity_hash": getattr(runtime_identity, "identity_hash", ""),
            "runtime_code_hash": release_meta.get("runtime_code_hash") or "",
            "prompt_pack_hash": release_meta.get("prompt_pack_hash") or "",
            "authority_policy_hash": release_meta.get("authority_policy_hash") or "",
        }
    )

    if not hasattr(agent, "run_full_clinical_loop"):
        raise RuntimeError("agent missing run_full_clinical_loop; cannot run online clinical path")

    seal = None
    emitted_attempt_ids: Set[str] = set()

    def emit_llm_attempts() -> None:
        for row in getattr(agent, "llm_call_audit", []) or []:
            if not isinstance(row, Mapping):
                continue
            call_id = str(row.get("call_id") or "")
            if not call_id or call_id in emitted_attempt_ids:
                continue
            sink(safe_llm_attempt_event(row))
            emitted_attempt_ids.add(call_id)

    def seal_once():
        nonlocal seal
        if seal is None and not trace.is_sealed:
            seal = trace.seal()
        return seal

    try:
        final_result = await agent.run_full_clinical_loop(
            actions=actions,
            patient_id=patient_id,
            mode=mode,
            evaluation_collector=evaluation_collector if mode == "train" else None,
            event_sink=sink,
        )
        emit_llm_attempts()
        sink(
            safe_budget_event(
                getattr(agent, "_case_state_budget", {}) or {},
                getattr(agent, "llm_call_audit", []) or [],
            )
        )
        diagnosis = final_result.get("diagnosis") or []
        if isinstance(diagnosis, str):
            diagnosis_names = [diagnosis] if diagnosis.strip() else []
        else:
            diagnosis_names = [str(x) for x in list(diagnosis) if str(x).strip()]
        sink(
            safe_case_end_event(
                finished=bool(final_result.get("finished", True)),
                diagnosis_names=diagnosis_names,
                action_count=len(gateway.trace),
                llm_attempts=int(
                    (getattr(agent, "_case_state_budget", {}) or {}).get("attempt")
                    or len(getattr(agent, "llm_call_audit", []) or [])
                ),
            )
        )
        seal_once()
        out = dict(final_result)
        out["case_run_id"] = case_run_id
        out["blackboard_revision"] = orchestrator.snapshot.revision
        out["snapshot_hash"] = orchestrator.snapshot.snapshot_hash()
        out["run_seal"] = {
            "trace_hash": seal.trace_hash if seal else "",
            "event_count": seal.event_count if seal else 0,
            "run_id": seal.run_id if seal else case_run_id,
        }
        # Honest authority label: clinical decisions are made by the legacy full loop.
        out["authority"] = "legacy_full_loop"
        out["clinical_engine"] = "full_clinical_loop"
        reasoning = str(final_result.get("reasoning") or "")
        if reasoning and not out.get("reasoning"):
            out["reasoning"] = reasoning
        return out
    except Exception as exc:
        emit_llm_attempts()
        sink(
            safe_budget_event(
                getattr(agent, "_case_state_budget", {}) or {},
                getattr(agent, "llm_call_audit", []) or [],
            )
        )
        sink(
            safe_case_error_event(
                error_type=type(exc).__name__,
                stage="online_clinical_case",
                action_count=len(gateway.trace),
            )
        )
        seal_once()
        raise
    finally:
        if not trace.is_sealed:
            seal_once()
