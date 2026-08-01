from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class InformationGap:
    gap_id: str
    intent: str
    related_hypothesis_ids: Tuple[str, ...] = ()
    decision_impact: str = "diagnosis"
    acquisition_route: str = "ask"
    priority: str = "normal"
    status: str = "open"
    closure_evidence_ids: Tuple[str, ...] = ()
