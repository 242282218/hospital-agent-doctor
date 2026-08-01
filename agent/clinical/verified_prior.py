"""Verified case-memory priors that survive exact-memory fallback.

The exact case-memory fast path may abort late (verifier, five-dimension gate,
partial examination failure). Diagnoses and examinations from a validated
train-evaluation asset stay trustworthy even then, so they are kept as a prior
for the full clinical loop. Treatment text is never carried over: it must be
re-derived through the normal safety chain.
"""
from __future__ import annotations

import re
from typing import Any, Collection, Mapping, Sequence

PRIOR_SOURCE = "verified_case_memory"
PRIOR_CANDIDATE_SOURCE = "verified_case_prior"
PRIOR_CANDIDATE_SCORE = 1000

_EVALUATION_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Treatment-bearing keys must never be copied into a prior.
_FORBIDDEN_PRIOR_KEYS = frozenset({"treatment_plan", "clinical_basis"})


def _clean_items(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return []
        text = item.strip()
        if not text:
            return []
        items.append(text)
    return items


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def build_verified_case_prior(
    validated_memory: Mapping[str, Any],
    *,
    completed_examinations: Sequence[str],
) -> dict[str, Any] | None:
    """Build a verified prior, or None when the memory is not trustworthy."""
    if not isinstance(validated_memory, Mapping):
        return None

    provenance = validated_memory.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    # Callers pass memory already validated by validate_runtime_case_memory(), so
    # source is optional here but must never contradict the verified origin.
    # An explicitly wrong source is rejected. An absent source is tolerated
    # because the only runtime caller is validate_runtime_case_memory(), which
    # already hard-requires source == "train_evaluation"; the plan spec builds
    # priors from that post-validation shape.
    source = str(provenance.get("source") or "")
    if source and source != "train_evaluation":
        return None
    evaluation_hash = str(provenance.get("evaluation_hash") or "")
    if not _EVALUATION_HASH_RE.match(evaluation_hash):
        return None

    diagnoses = _unique(_clean_items(validated_memory.get("diagnoses")))
    required = _unique(_clean_items(validated_memory.get("examinations")))
    if not diagnoses or not required:
        return None

    completed_set = {
        item.strip()
        for item in completed_examinations or ()
        if isinstance(item, str) and item.strip()
    }
    prior = {
        "source": PRIOR_SOURCE,
        "diagnoses": diagnoses,
        "required_examinations": required,
        "completed_examinations": [item for item in required if item in completed_set],
        "pending_examinations": [item for item in required if item not in completed_set],
        "evaluation_hash": evaluation_hash,
    }
    assert not _FORBIDDEN_PRIOR_KEYS.intersection(prior)
    return prior


def refresh_verified_case_prior(
    prior: Mapping[str, Any] | None,
    *,
    completed_examinations: Sequence[str],
) -> dict[str, Any] | None:
    """Recompute completed/pending after each examination response merge."""
    if not isinstance(prior, Mapping):
        return None
    required = _unique(_clean_items(prior.get("required_examinations")))
    if not required:
        return None
    completed_set = {
        item.strip()
        for item in completed_examinations or ()
        if isinstance(item, str) and item.strip()
    }
    updated = dict(prior)
    updated["completed_examinations"] = [item for item in required if item in completed_set]
    updated["pending_examinations"] = [
        item for item in required if item not in completed_set
    ]
    return updated


def verified_prior_diagnoses(
    prior: Mapping[str, Any] | None,
    *,
    official_diseases: Collection[str],
) -> list[str]:
    if not isinstance(prior, Mapping):
        return []
    official = set(official_diseases)
    return [
        name
        for name in _unique(_clean_items(prior.get("diagnoses")))
        if name in official
    ]


def merge_verified_prior_candidates(
    candidates: Sequence[Mapping[str, Any]],
    prior: Mapping[str, Any] | None,
    *,
    official_diseases: Collection[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Put verified prior diagnoses first, keeping catalog membership mandatory."""
    official = set(official_diseases)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in verified_prior_diagnoses(prior, official_diseases=official):
        if name in seen:
            continue
        merged.append(
            {
                "disease": name,
                "source": PRIOR_CANDIDATE_SOURCE,
                "score": PRIOR_CANDIDATE_SCORE,
                "matched_evidence": [PRIOR_SOURCE],
                "evidence_polarity": "verified",
            }
        )
        seen.add(name)
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        name = str(candidate.get("disease") or "").strip()
        if name and name in official and name not in seen:
            merged.append(dict(candidate))
            seen.add(name)
    return merged[: max(0, int(limit))]


def verified_prior_pending_examinations(
    prior: Mapping[str, Any] | None,
    *,
    attempted: Collection[str],
    valid_examinations: Collection[str],
) -> list[str]:
    """Remembered examinations that are still worth ordering exactly once."""
    if not isinstance(prior, Mapping):
        return []
    attempted_set = {
        item.strip() for item in attempted or () if isinstance(item, str) and item.strip()
    }
    valid_set = {
        item.strip()
        for item in valid_examinations or ()
        if isinstance(item, str) and item.strip()
    }
    return [
        item
        for item in _unique(_clean_items(prior.get("pending_examinations")))
        if item in valid_set and item not in attempted_set
    ]
