from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExamIntent:
    exam_intent_id: str
    related_gap_ids: Tuple[str, ...] = ()
    related_hypothesis_ids: Tuple[str, ...] = ()
    requirement: str = "optional"
    catalog_leaf_name: str = ""
    reason: str = ""
    status: str = "proposed"
    result_evidence_ids: Tuple[str, ...] = ()
    waived_by: str = ""
    waiver_reason: str = ""
    waiver_evidence_ids: Tuple[str, ...] = ()
