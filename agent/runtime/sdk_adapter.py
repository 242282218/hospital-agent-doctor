from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence

from .action_gateway import ActionCommand


class DoctorActionsProtocol(Protocol):
    async def ask_patient(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        pass

    async def order_examination(
        self,
        patient_id: str,
        items: Iterable[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        pass

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: Any,
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        pass

    async def evaluation(
        self,
        patient_id: str,
        final_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pass


class SdkActionAdapter:
    def __init__(self, *, actions: DoctorActionsProtocol, patient_id: str) -> None:
        self._actions = actions
        self._patient_id = str(patient_id)

    async def dispatch(
        self,
        command: ActionCommand,
        *,
        chat_history: Sequence[Mapping[str, Any]],
    ) -> Any:
        payload = dict(command.payload)
        if command.action_type == "ask_patient":
            return await self._actions.ask_patient(
                self._patient_id,
                {
                    "question": str(payload["question"]),
                    "chat_history": [dict(item) for item in chat_history],
                },
            )
        if command.action_type == "order_examination":
            return await self._actions.order_examination(
                self._patient_id,
                list(payload["items"]),
                reason=str(payload.get("reason") or ""),
            )
        if command.action_type == "prescribe_treatment":
            return await self._actions.prescribe_treatment(
                patient_id=self._patient_id,
                diagnosis=list(payload["diagnosis"]),
                treatment_plan=str(payload["treatment_plan"]),
                reasoning=str(payload.get("reasoning") or ""),
            )
        raise ValueError("unsupported action_type: %s" % command.action_type)

    async def collect_evaluation(self, final_result: Dict[str, Any]) -> Dict[str, Any]:
        return await self._actions.evaluation(
            patient_id=self._patient_id,
            final_result=final_result,
        )
