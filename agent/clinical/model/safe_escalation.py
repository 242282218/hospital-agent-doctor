from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SafeEscalationPlan:
    submission_diagnosis: str
    supporting_evidence_ids: Tuple[str, ...]
    unresolved_ids: Tuple[str, ...]
    disposition: str
    reason: str
