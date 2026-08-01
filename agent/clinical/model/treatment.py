from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TreatmentItem:
    item_id: str
    category: str
    stage: str
    intent: str
    related_hypothesis_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    constraint_ids: Tuple[str, ...] = ()
    status: str = "proposed"


@dataclass(frozen=True)
class TreatmentState:
    urgency_and_disposition: str = ""
    treatment_items: Tuple[TreatmentItem, ...] = ()
    draft_text: str = ""
    verifier_issues: Tuple[str, ...] = ()
