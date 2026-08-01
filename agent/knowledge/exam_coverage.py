"""Data-driven exam coverage predicates.

The `exams_cover_*` family in `legacy_orchestrator` was 27 hand-written functions
over the same shape: substring-match a marker list against the completed
examination names. Marker lists are clinical knowledge, not control flow, so they
live in `exam_coverage_rules.json` and the predicates are built from that table.

Adding coverage for a new concept is a JSON edit, which is what makes reflection
output (Reflexion `next_action_rule`) mechanically applicable instead of requiring
a new function per lesson.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

RULES_PATH = Path(__file__).with_name("exam_coverage_rules.json")

_VALID_MODES = frozenset({"any", "any_excluding", "all_groups", "any_group_pair"})


class ExamCoverageRuleError(ValueError):
    """Raised when the coverage rule table is malformed."""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        value = value.values()
    out: List[str] = []
    for item in value:
        text = _clean(item)
        if text:
            out.append(text)
    return out


@lru_cache(maxsize=1)
def load_coverage_rules(path: str = "") -> Dict[str, Dict[str, Any]]:
    """Load and validate the coverage rule table, keyed by rule id."""
    source = Path(path) if path else RULES_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ExamCoverageRuleError("coverage rules payload must be an object")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ExamCoverageRuleError("coverage rules must be a non-empty list")

    table: Dict[str, Dict[str, Any]] = {}
    for raw in rules:
        if not isinstance(raw, Mapping):
            raise ExamCoverageRuleError("each coverage rule must be an object")
        rule_id = _clean(raw.get("id"))
        if not rule_id:
            raise ExamCoverageRuleError("coverage rule missing id")
        if rule_id in table:
            raise ExamCoverageRuleError("duplicate coverage rule id: %s" % rule_id)
        mode = _clean(raw.get("mode")) or "any"
        if mode not in _VALID_MODES:
            raise ExamCoverageRuleError("unknown coverage mode %r for %s" % (mode, rule_id))

        markers = _text_list(raw.get("markers"))
        groups = [_text_list(group) for group in (raw.get("groups") or [])]
        groups = [group for group in groups if group]
        if mode in {"all_groups", "any_group_pair"}:
            if len(groups) < 2:
                raise ExamCoverageRuleError("%s mode needs >=2 groups: %s" % (mode, rule_id))
        elif not markers:
            raise ExamCoverageRuleError("coverage rule %s needs markers" % rule_id)

        table[rule_id] = {
            "id": rule_id,
            "mode": mode,
            "markers": tuple(markers),
            "groups": tuple(tuple(group) for group in groups),
            "excluding": tuple(_text_list(raw.get("excluding"))),
            "exact_names": tuple(_text_list(raw.get("exact_names"))),
        }
    return table


def _matches(name: str, markers: Sequence[str]) -> bool:
    return any(marker in name for marker in markers)


def exams_match_rule(examinations: Iterable[str], rule: Mapping[str, Any]) -> bool:
    """Evaluate one coverage rule against completed examination names."""
    names = [_clean(exam) for exam in _text_list(examinations)]
    if not names:
        return False
    exact = rule.get("exact_names") or ()
    if exact and any(name in exact for name in names):
        return True

    mode = rule.get("mode") or "any"
    if mode == "any":
        return any(_matches(name, rule["markers"]) for name in names)
    if mode == "any_excluding":
        excluding = rule.get("excluding") or ()
        return any(
            _matches(name, rule["markers"]) and not _matches(name, excluding)
            for name in names
        )
    if mode == "all_groups":
        # Every group must be satisfied by the SAME examination name.
        return any(
            all(_matches(name, group) for group in rule["groups"]) for name in names
        )
    if mode == "any_group_pair":
        return any(
            any(_matches(name, group) for name in names) for group in rule["groups"]
        )
    raise ExamCoverageRuleError("unhandled coverage mode: %s" % mode)


def covers(rule_id: str, examinations: Iterable[str]) -> bool:
    """Public entry: does the completed exam set cover `rule_id`?"""
    table = load_coverage_rules()
    rule = table.get(rule_id)
    if rule is None:
        raise ExamCoverageRuleError("unknown coverage rule id: %s" % rule_id)
    return exams_match_rule(examinations, rule)


def build_predicates() -> Dict[str, Callable[[Iterable[str]], bool]]:
    """Build one `exams_cover_<id>`-style predicate per rule."""
    table = load_coverage_rules()

    def _make(rule: Mapping[str, Any]) -> Callable[[Iterable[str]], bool]:
        def predicate(examinations: Iterable[str]) -> bool:
            return exams_match_rule(examinations, rule)

        predicate.__name__ = "exams_cover_%s" % rule["id"]
        predicate.__doc__ = "Coverage predicate generated from exam_coverage_rules.json."
        return predicate

    return {rule_id: _make(rule) for rule_id, rule in table.items()}


def rule_ids() -> Tuple[str, ...]:
    return tuple(sorted(load_coverage_rules()))
