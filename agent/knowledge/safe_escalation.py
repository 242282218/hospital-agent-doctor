"""Table-driven safe-escalation closure text.

`build_safe_escalation_plan` used to be a 281-line if/elif chain of 46 axis ids
whose branches did nothing but assign a Chinese closure paragraph. That is data,
not logic: every new axis meant editing code, and the chain had already drifted
out of sync with its `validate_safe_escalation_plan` mirror.

The closure paragraphs now live in `safe_escalation_plans.json`. Only the dynamic
default (which formats `closure_requirement` at runtime) stays in code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

_TABLE_PATH = Path(__file__).resolve().parent / "safe_escalation_plans.json"

_CACHE: Optional[Dict[str, str]] = None


class SafeEscalationTableError(ValueError):
    """Raised when the closure table is missing or malformed."""


def _load() -> Dict[str, str]:
    raw = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SafeEscalationTableError("closure table must be an object")
    plans = raw.get("plans")
    if not isinstance(plans, list) or not plans:
        raise SafeEscalationTableError("closure table must contain a non-empty plans list")
    table: Dict[str, str] = {}
    for entry in plans:
        if not isinstance(entry, Mapping):
            raise SafeEscalationTableError("each plan entry must be an object")
        axis_id = str(entry.get("axis_id") or "").strip()
        closure = str(entry.get("closure") or "").strip()
        if not axis_id:
            raise SafeEscalationTableError("plan entry missing axis_id")
        if not closure:
            raise SafeEscalationTableError("plan entry %s missing closure" % axis_id)
        if axis_id in table:
            raise SafeEscalationTableError("duplicate axis_id in closure table: %s" % axis_id)
        table[axis_id] = closure
    return table


def closure_table() -> Dict[str, str]:
    """Parsed axis_id -> closure paragraph table (cached, fail-closed)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return dict(_CACHE)


def closure_for_axis(axis_id: str) -> str:
    """Closure paragraph for `axis_id`, or "" when the axis has no table entry.

    An empty result means the caller must fall back to the dynamic default; it is
    never a silent pass.
    """
    return closure_table().get(str(axis_id or "").strip(), "")


def known_axis_ids() -> Tuple[str, ...]:
    return tuple(sorted(closure_table()))


def table_path() -> Path:
    return _TABLE_PATH


def _reset_cache_for_tests() -> None:
    global _CACHE
    _CACHE = None


__all__ = [
    "SafeEscalationTableError",
    "closure_for_axis",
    "closure_table",
    "known_axis_ids",
    "table_path",
]
