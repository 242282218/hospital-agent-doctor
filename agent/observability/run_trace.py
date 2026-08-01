from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class SealReceipt:
    trace_hash: str
    event_count: int
    sealed_at: str
    run_id: str
    schema_version: str = "run-trace-seal/v1"


class RunTraceStore:
    """Append-only run.jsonl with external run.seal.json (seal not inside trace)."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "run.jsonl"
        self._seal_path = self._root / "run.seal.json"
        self._sealed = False
        self._count = 0
        if self._path.exists():
            self._count = sum(1 for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip())

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seal_path(self) -> Path:
        return self._seal_path

    @property
    def is_sealed(self) -> bool:
        return bool(self._sealed or self._seal_path.exists())

    @property
    def event_count(self) -> int:
        return int(self._count)

    def append(self, event: Mapping[str, Any]) -> None:
        if self._sealed or self._seal_path.exists():
            raise RuntimeError("run trace already sealed")
        body = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(body)
            handle.write("\n")
        self._count += 1

    def seal(self) -> SealReceipt:
        if self._seal_path.exists():
            raise RuntimeError("run trace already sealed")
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")
        raw = self._path.read_bytes()
        trace_hash = sha256(raw).hexdigest()
        sealed_at = datetime.now(timezone.utc).isoformat()
        receipt = SealReceipt(
            trace_hash=trace_hash,
            event_count=self._count,
            sealed_at=sealed_at,
            run_id=self._run_id,
        )
        payload = {
            "schema_version": receipt.schema_version,
            "run_id": receipt.run_id,
            "trace_hash": receipt.trace_hash,
            "event_count": receipt.event_count,
            "sealed_at": receipt.sealed_at,
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._seal_path.write_text(body, encoding="utf-8")
        self._sealed = True
        return receipt
