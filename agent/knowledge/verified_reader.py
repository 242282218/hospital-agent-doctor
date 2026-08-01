from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class VerifiedKnowledgeReader:
    def __init__(self, root: Path, expected_hashes: Optional[Mapping[str, str]] = None) -> None:
        self._root = Path(root)
        self._expected = dict(expected_hashes or {})

    def read_json(self, relative_path: str) -> Dict[str, Any]:
        path = self._root / relative_path
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        expected = self._expected.get(relative_path.replace("\\", "/"))
        if expected and expected != digest:
            raise ValueError("knowledge hash mismatch for %s" % relative_path)
        return json.loads(raw.decode("utf-8"))

    def read_verified_rules(self, relative_path: str) -> list[dict[str, Any]]:
        payload = self.read_json(relative_path)
        if not isinstance(payload, dict):
            raise ValueError("verified rules payload must be an object")
        if "rules" not in payload:
            raise ValueError("verified rules payload must contain rules")
        rules = payload["rules"]
        if not isinstance(rules, list):
            raise ValueError("verified rules must be a list")
        if any(not isinstance(rule, dict) for rule in rules):
            raise ValueError("verified rules must contain objects")
        return [rule for rule in rules if rule.get("status") == "verified"]
