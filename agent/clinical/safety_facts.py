"""Typed, provenance-bound current safety facts for case-memory fast paths."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


SAFETY_FACT_KINDS = frozenset(
    {
        "allergy",
        "current_medication",
        "contraindication",
        "confirmed_resistance",
        "pregnancy_status",
        "renal_function",
        "hepatic_function",
        "comorbidity",
    }
)
SAFETY_FACT_POLARITIES = frozenset({"present", "absent", "unknown"})
SAFETY_FACT_TEMPORALITIES = frozenset({"current"})
SAFETY_FACT_FIELDS = frozenset(
    {
        "fact_id",
        "kind",
        "value",
        "polarity",
        "source_ref",
        "source_evidence_ids",
        "temporality",
    }
)


@dataclass(frozen=True)
class SafetyFact:
    fact_id: str
    kind: str
    value: str
    polarity: str
    source_ref: str
    source_evidence_ids: Tuple[str, ...]
    temporality: str = "current"


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def _parse_fact(raw: Any) -> Optional[SafetyFact]:
    if not isinstance(raw, Mapping) or set(raw) != SAFETY_FACT_FIELDS:
        return None
    fact_id = _clean_text(raw.get("fact_id"))
    kind = _clean_text(raw.get("kind"))
    value = _clean_text(raw.get("value"))
    polarity = _clean_text(raw.get("polarity"))
    source_ref = _clean_text(raw.get("source_ref"))
    temporality = _clean_text(raw.get("temporality"))
    evidence = raw.get("source_evidence_ids")
    if (
        not fact_id
        or kind not in SAFETY_FACT_KINDS
        or not value
        or polarity not in SAFETY_FACT_POLARITIES
        or not source_ref
        or temporality not in SAFETY_FACT_TEMPORALITIES
        or not isinstance(evidence, list)
        or not evidence
    ):
        return None
    evidence_ids = tuple(_clean_text(item) for item in evidence)
    if any(item is None for item in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
        return None
    return SafetyFact(
        fact_id=fact_id,
        kind=kind,
        value=value,
        polarity=polarity,
        source_ref=source_ref,
        source_evidence_ids=evidence_ids,
        temporality=temporality,
    )


def canonical_safety_facts_hash(facts: Sequence[SafetyFact]) -> str:
    canonical = [asdict(fact) for fact in sorted(facts, key=lambda item: item.fact_id)]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_case_memory_safety_facts(
    raw_facts: Any,
    raw_hash: Any,
) -> Optional[Tuple[SafetyFact, ...]]:
    """Accept only a complete, conflict-free, hash-bound fact set."""
    if not isinstance(raw_facts, list) or not raw_facts:
        return None
    facts = tuple(_parse_fact(item) for item in raw_facts)
    if any(fact is None for fact in facts):
        return None
    parsed = tuple(fact for fact in facts if fact is not None)
    fact_ids = [fact.fact_id for fact in parsed]
    if len(set(fact_ids)) != len(fact_ids):
        return None
    semantic_polarities = {}
    for fact in parsed:
        key = (fact.kind, fact.value, fact.temporality)
        existing = semantic_polarities.setdefault(key, fact.polarity)
        if existing != fact.polarity:
            return None
    expected_hash = canonical_safety_facts_hash(parsed)
    if raw_hash != expected_hash:
        return None
    return tuple(sorted(parsed, key=lambda item: item.fact_id))


def safety_facts_to_case_features(facts: Sequence[SafetyFact]) -> dict[str, Any]:
    """Project explicit current positive facts onto legacy safety inputs."""
    current = [fact for fact in facts if fact.temporality == "current"]
    features: dict[str, Any] = {
        "safety_facts": [asdict(fact) for fact in current],
    }
    for kind, feature_key in (
        ("allergy", "drug_allergies"),
        ("contraindication", "contraindicated_drugs"),
    ):
        values = [fact.value for fact in current if fact.kind == kind and fact.polarity == "present"]
        if values:
            features[feature_key] = values
    resistance = [
        fact.value
        for fact in current
        if fact.kind == "confirmed_resistance" and fact.polarity == "present"
    ]
    if resistance:
        features["anti_infective_provenance"] = {
            "ast": [],
            "cultures": [],
            "confirmed_resistance": resistance,
            "empiric": None,
        }
    return features
