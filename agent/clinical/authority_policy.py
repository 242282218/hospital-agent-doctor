"""Clinical authority-stage policy shared by legacy and orchestrator runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Tuple


_STAGES = ("intake", "exam", "diagnosis", "treatment_final")
_VALID_PREFIXES = {
    ("legacy", "legacy", "legacy", "legacy"),
    ("orchestrator", "legacy", "legacy", "legacy"),
    ("orchestrator", "orchestrator", "legacy", "legacy"),
    ("orchestrator", "orchestrator", "orchestrator", "legacy"),
    ("orchestrator", "orchestrator", "orchestrator", "orchestrator"),
}


@dataclass(frozen=True)
class ClinicalAuthorityPolicy:
    intake: str
    exam: str
    diagnosis: str
    treatment_final: str

    def values(self) -> Tuple[str, str, str, str]:
        return (self.intake, self.exam, self.diagnosis, self.treatment_final)

    def canonical_payload(self) -> dict:
        return dict(zip(_STAGES, self.values()))

    @property
    def identity_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return sha256(encoded.encode("utf-8")).hexdigest()


def parse_clinical_authority_policy(value: str) -> ClinicalAuthorityPolicy:
    parts = tuple(part.strip() for part in str(value or "").split(","))
    if parts not in _VALID_PREFIXES:
        raise ValueError("invalid clinical authority prefix")
    return ClinicalAuthorityPolicy(*parts)


__all__ = ["ClinicalAuthorityPolicy", "parse_clinical_authority_policy"]
