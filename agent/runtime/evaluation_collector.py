from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .evaluation_attempt_store import EvaluationAttemptStore


class EvaluationTransport(Protocol):
    async def collect_evaluation(self, final_result: Dict[str, Any]) -> Dict[str, Any]:
        pass


@dataclass(frozen=True)
class EvaluationAttachment:
    clinical_result_hash: str
    report: Dict[str, Any]


class EvaluationCollector:
    def __init__(
        self,
        *,
        adapter: EvaluationTransport,
        mode: str,
        case_run_id: str = "",
        store: Optional[EvaluationAttemptStore] = None,
    ) -> None:
        if mode not in {"train", "test"}:
            raise ValueError("mode must be train or test")
        self._adapter = adapter
        self._mode = mode
        self._case_run_id = str(case_run_id or "")
        self._store = store
        self._attempted = False

    async def collect(self, final_result: Dict[str, Any]) -> EvaluationAttachment:
        if self._mode != "train":
            raise RuntimeError("evaluation is available only in train mode")
        if self._attempted:
            raise RuntimeError("evaluation already attempted for this case")
        if not bool(final_result.get("finished")):
            raise ValueError("evaluation requires finished=true final_result")

        canonical = json.dumps(
            final_result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        clinical_result_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        if self._store is not None:
            if not self._case_run_id:
                raise ValueError("case_run_id is required when attempt store is enabled")
            try:
                self._store.create_once(self._case_run_id, clinical_result_hash)
            except FileExistsError as exc:
                self._attempted = True
                raise RuntimeError("evaluation already attempted for this case") from exc

        self._attempted = True
        report = await self._adapter.collect_evaluation(final_result)
        return EvaluationAttachment(
            clinical_result_hash=clinical_result_hash,
            report=dict(report),
        )
