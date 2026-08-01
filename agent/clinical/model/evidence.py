from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    concept: str
    value: str
    kind: str
    subject: str = "patient"
    temporality: str = "current"
    polarity: str = "unknown"
    source_ref: str = ""
    source_evidence_ids: Tuple[str, ...] = ()
    confidence: str = "medium"
    status: str = "raw"
    created_by: str = ""
