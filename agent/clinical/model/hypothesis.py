from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HypothesisItem:
    hypothesis_id: str
    official_disease_name: str
    role: str = "differential"
    confidence: str = "low"
    supporting_evidence_ids: Tuple[str, ...] = ()
    opposing_evidence_ids: Tuple[str, ...] = ()
    open_gap_ids: Tuple[str, ...] = ()
    required_exam_intents: Tuple[str, ...] = ()
    treatment_risk_tags: Tuple[str, ...] = ()
    status: str = "active"
