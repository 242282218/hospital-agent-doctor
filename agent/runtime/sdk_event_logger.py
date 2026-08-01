"""Privacy boundary for the SDK's file-based event logger.

The SDK logger is a functional output boundary, not an audit trace. Clinical
submission files remain available to the evaluator, while operational events
and evaluation reports are reduced to non-clinical metadata.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from hospital_agent_sdk.event_logger import EventLogger as SdkEventLogger


_SENSITIVE_KEYS = frozenset(
    {
        "input",
        "content",
        "results",
        "result",
        "reason",
        "diagnosis",
        "treatment_plan",
        "reasoning",
        "final_result",
        "report",
        "error",
        "service_base_url",
        "output_log_path",
        "patient_id",
        "target",
    }
)


def _hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:%s" % hashlib.sha256(body.encode("utf-8")).hexdigest()


def _opaque(value: Any) -> str:
    return _hash(str(value))


def _safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {
        "payload_hash": _hash(payload),
        "field_count": len(payload),
    }
    for key in (
        "mode",
        "phase",
        "case_index",
        "finished",
        "message_type",
        "results_count",
    ):
        if key in payload and key not in _SENSITIVE_KEYS:
            value = payload[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
    for key in ("items", "invalid_items"):
        if key in payload:
            value = payload[key]
            if isinstance(value, (list, tuple)):
                safe["%s_count" % key] = len(value)
    return safe


class RedactingEventLogger(SdkEventLogger):
    """Drop clinical payloads before operational SDK events reach disk."""

    def __init__(self, output_dir: Union[str, Path]):
        super().__init__(output_dir)
        self._privacy_lock = threading.RLock()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def write_event(
        self,
        event_type: str,
        agent_id: str,
        patient_id: str,
        status: str = "success",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_payload = dict(payload or {})
        with self._privacy_lock:
            self._event_index += 1
            event = {
                "event_index": self._event_index,
                "timestamp": self._timestamp(),
                "event_type": str(event_type),
                "agent_id_ref": _opaque(agent_id),
                "patient_ref": _opaque(patient_id),
                "status": str(status),
                "payload": _safe_payload(raw_payload),
            }
            self._append_jsonl(self.events_path, event)
            return event

    def write_evaluation_result(self, patient_id: str, report: Dict[str, Any]) -> None:
        with self._privacy_lock:
            record = {
                "timestamp": self._timestamp(),
                "patient_ref": _opaque(patient_id),
                "report_hash": _hash(report),
                "report_field_count": len(report),
            }
            self._append_jsonl(self.evaluation_results_path, record)

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        with self._privacy_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def install_sdk_event_logger() -> None:
    """Install the adapter at the SDK runtime's imported class reference."""
    import hospital_agent_sdk.runtime as sdk_runtime

    current = getattr(sdk_runtime, "EventLogger", None)
    if current is RedactingEventLogger:
        return
    if current is not SdkEventLogger:
        raise RuntimeError("unsupported hospital-agent-sdk EventLogger binding")
    sdk_runtime.EventLogger = RedactingEventLogger


__all__ = ["RedactingEventLogger", "install_sdk_event_logger"]
