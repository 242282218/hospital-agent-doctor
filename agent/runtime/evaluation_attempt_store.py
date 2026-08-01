from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class EvaluationAttemptStore:
    """Atomic per-case evaluation attempt markers (exclusive create)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, case_run_id: str) -> Path:
        safe_id = str(case_run_id).strip()
        if not safe_id or "/" in safe_id or "\\" in safe_id or ".." in safe_id:
            raise ValueError("invalid case_run_id")
        return self._root / ("%s.json" % safe_id)

    def create_once(self, case_run_id: str, final_result_hash: str) -> Path:
        path = self.path_for(case_run_id)
        payload: Dict[str, Any] = {
            "case_run_id": str(case_run_id),
            "final_result_hash": str(final_result_hash),
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # Exclusive create: fails if attempt already recorded (including prior failures).
        with path.open("x", encoding="utf-8") as handle:
            handle.write(body)
            handle.write("\n")
        return path

    def exists(self, case_run_id: str) -> bool:
        return self.path_for(case_run_id).exists()
