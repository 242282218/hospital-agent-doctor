"""Small Markdown memory for evaluation reflections.

The baseline stores one reflection field per patient. Future prompts read the
latest patient reflections as simple reference notes.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from agent.clinical.safety_facts import validate_case_memory_safety_facts
from agent.knowledge.verified_profiles import (
    PROFILE_ASSET_TYPES,
    VerifiedProfileIndex,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = BASE_DIR / "data" / "memory_data" / "memory.md"

VERIFIED_REGISTRY_SCHEMA = "verified-registry/v1"

# Typed clinical assets: routed to VerifiedProfileIndex, never json-dumped into
# generic prompt notes.
PROFILE_ASSET_TYPES = frozenset(
    {"disease_exam_profile", "disease_treatment_profile", "reflection_rule"}
)
# Legacy generic verified notes stay supported for backwards compatibility.
GENERIC_NOTE_ASSET_TYPES = frozenset(
    {
        "generic_rule",
        "diagnosis_differential_rule",
        "clinical_closure_rule",
        "diagnosis_priority_rule",
        "treatment_gate_rule",
        "treatment_sequence_rule",
        "reflection_note",
    }
)


class MarkdownMemory:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        max_notes: int = 3,
        max_note_chars: int = 1200,
    ):
        self.path = Path(path)
        self.max_notes = int(max_notes)
        self.max_note_chars = int(max_note_chars)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Baseline Doctor Memory\n\n", encoding="utf-8")

    def load_notes(
        self,
        limit: Optional[int] = None,
        *,
        exclude_patient_id: Optional[str] = None,
        include_candidates: bool = False,
    ) -> List[str]:
        """读取最近几条已验证反思，新的在前。"""
        text = self.path.read_text(encoding="utf-8")
        notes = []
        allowed_statuses = {"verified"}
        if include_candidates:
            allowed_statuses.add("candidate")
        for block in text.split("\n## "):
            block = block.strip()
            if not block or block.startswith("# Baseline"):
                continue
            note = "## " + block
            if exclude_patient_id and self._is_patient_block(note, exclude_patient_id):
                continue
            if self._block_status(note) not in allowed_statuses:
                continue
            notes.append(self._truncate(self._sanitize_note_for_prompt(note)))
        notes.reverse()
        return notes[: int(limit or self.max_notes)]

    def append_case_reflection(
        self,
        *,
        patient_id: str,
        evaluation_reflection: Optional[Dict[str, Any]] = None,
        status: str = "candidate",
        **_: Any,
    ) -> None:
        """训练评估后，每个患者只保存一个 reflection 字段。"""
        case_block = self._case_block(patient_id, evaluation_reflection or {}, status=status)
        self._upsert_case_block(patient_id, case_block)

    def _case_block(self, patient_id: str, reflection: Dict[str, Any], *, status: str) -> str:
        reflection_value = reflection.get("reflection") if "reflection" in reflection else reflection
        if not isinstance(reflection_value, dict):
            reflection_value = {"summary": reflection_value}
        clean_status = str(status or "candidate").strip().lower()
        if clean_status not in {"candidate", "verified"}:
            clean_status = "candidate"

        lines = [
            "## Case %s Reflection" % patient_id,
            "",
            "- **Status:** %s" % clean_status,
            "",
            "### Reflection",
            "",
        ]
        for label, key in [
            ("Profile", "profile"),
            ("Diagnosis", "diagnosis_reflection"),
            ("Examination", "examination_reflection"),
            ("Treatment", "treatment_reflection"),
            ("Future Strategy", "future_strategy"),
        ]:
            value = self._truncate(self._sanitize_reflection_text(reflection_value.get(key)))
            if value:
                lines.append("- **%s:** %s" % (label, value))

        extra_items = [
            (key, value)
            for key, value in reflection_value.items()
            if key
            not in {
                "profile",
                "status",
                "diagnosis_reflection",
                "examination_reflection",
                "treatment_reflection",
                "future_strategy",
            }
        ]
        for key, value in extra_items:
            lines.append("- **%s:** %s" % (str(key), self._truncate(self._sanitize_reflection_text(value))))

        return "\n".join(lines).rstrip() + "\n"

    def _upsert_case_block(self, patient_id: str, new_block: str) -> None:
        text = self.path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        header = parts[0].rstrip() if parts else "# Baseline Doctor Memory"
        case_blocks = []
        for raw_block in parts[1:]:
            block = "## " + raw_block.strip()
            if not self._is_patient_block(block, patient_id):
                case_blocks.append(block)
        body = "\n\n".join([*case_blocks, new_block]).strip()
        self.path.write_text("%s\n\n%s\n" % (header, body), encoding="utf-8")

    def _is_patient_block(self, block: str, patient_id: str) -> bool:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        return first_line in {
            "## Case %s" % patient_id,
            "## Case %s Reflection" % patient_id,
        }

    def _block_status(self, block: str) -> str:
        for line in block.splitlines():
            text = line.strip()
            marker = "- **status:**"
            lower_text = text.lower()
            if not lower_text.startswith(marker):
                continue
            status = text[len(marker):].strip().lower()
            if status in {"candidate", "verified"}:
                return status
        return "candidate"

    def _truncate(self, value: Any, max_chars: Optional[int] = None) -> str:
        text = str(value or "").strip()
        limit = int(max_chars or self.max_note_chars)
        if len(text) <= limit:
            return text
        return text[: limit - 18].rstrip() + "\n...[truncated]"

    def _sanitize_reflection_text(self, value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\bPatient_\d+\b", "训练病例", text)
        text = re.sub(r"\b(expected|reference|ground[_ -]?truth)\b[:：]?", "评估反馈", text, flags=re.I)
        return text

    def _sanitize_note_for_prompt(self, note: str) -> str:
        lines = []
        for line in str(note or "").splitlines():
            if line.startswith("## Case "):
                lines.append("## Verified Reflection")
                continue
            lines.append(self._sanitize_reflection_text(line))
        return "\n".join(lines)



class VerifiedRegistryReader:
    """Test/runtime reader for frozen registry snapshot only."""

    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
        text = self._path.read_text(encoding="utf-8")
        if "candidates" in self._path.as_posix().lower():
            raise ValueError("refusing mutable candidate path")
        self._data = json.loads(text)

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data

    def assets(self) -> List[Any]:
        return list(self._data.get("assets") or [])


class VerifiedOnlyMemory:
    """Online memory: read frozen verified registry assets; no online writes."""

    def __init__(self, registry_path=None, *, max_notes: int = 3) -> None:
        self.max_notes = int(max_notes)
        self.registry_path = Path(registry_path) if registry_path else None
        self._notes: List[str] = []
        self._case_memories: Dict[str, Dict[str, Any]] = {}
        # Secondary indexes for ID-format drift and unique content fingerprints.
        # Ambiguous fingerprints are dropped so a miss never invents a memory.
        self._id_aliases: Dict[str, str] = {}
        self._content_fingerprints: Dict[str, str] = {}
        self._profiles = VerifiedProfileIndex(())
        if self.registry_path and self.registry_path.exists():
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            # Fail closed on anything that is not a frozen verified registry: a
            # candidate file must never be loadable as a runtime asset source.
            if not isinstance(data, Mapping):
                raise ValueError("verified registry must be an object")
            if data.get("schema_version") != VERIFIED_REGISTRY_SCHEMA:
                raise ValueError(
                    "unsupported verified registry schema_version: %r"
                    % (data.get("schema_version"),)
                )
            if not isinstance(data.get("assets"), list):
                raise ValueError("verified registry must contain an assets list")
            fingerprint_hits: Dict[str, List[str]] = {}
            profile_assets: List[Mapping[str, Any]] = []
            for asset in data.get("assets") or []:
                content = asset.get("content")
                if asset.get("candidate_type") == "case_memory":
                    case_memory = self._validated_case_memory(content)
                    if case_memory is None:
                        continue
                    patient_id = case_memory["patient_id"]
                    if patient_id in self._case_memories:
                        raise ValueError("duplicate case memory patient_id")
                    self._case_memories[patient_id] = case_memory
                    for alias in self._patient_id_aliases(patient_id):
                        existing = self._id_aliases.get(alias)
                        if existing is not None and existing != patient_id:
                            # Ambiguous alias: drop so we never cross-link cases.
                            self._id_aliases.pop(alias, None)
                            continue
                        self._id_aliases[alias] = patient_id
                    fingerprint = self._content_fingerprint(
                        case_memory["diagnoses"],
                        case_memory["examinations"],
                    )
                    fingerprint_hits.setdefault(fingerprint, []).append(patient_id)
                    continue
                if asset.get("candidate_type") in PROFILE_ASSET_TYPES:
                    # Profiles/reflections are typed clinical assets. They must
                    # never be serialized into generic prompt notes.
                    profile_assets.append(asset)
                    continue
                if asset.get("candidate_type") not in GENERIC_NOTE_ASSET_TYPES:
                    raise ValueError(
                        "unsupported verified asset type: %r" % (asset.get("candidate_type"),)
                    )
                if content is None:
                    continue
                self._notes.append(json.dumps(content, ensure_ascii=False, sort_keys=True)[:1200])
            for fingerprint, patient_ids in fingerprint_hits.items():
                if len(patient_ids) == 1:
                    self._content_fingerprints[fingerprint] = patient_ids[0]
            self._profiles = VerifiedProfileIndex(profile_assets)

    @staticmethod
    def _validated_case_memory(content: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(content, Mapping):
            return None
        legacy_fields = {
            "patient_id",
            "diagnoses",
            "examinations",
            "treatment_plan",
            "clinical_basis",
            "provenance",
        }
        fast_path_fields = legacy_fields | {"safety_facts", "safety_facts_hash"}
        fields = set(content)
        if fields != legacy_fields and fields != fast_path_fields:
            return None
        safety_facts_complete = fields == fast_path_fields
        if safety_facts_complete:
            if validate_case_memory_safety_facts(
                content.get("safety_facts"),
                content.get("safety_facts_hash"),
            ) is None:
                return None
        patient_id = content.get("patient_id")
        diagnoses = content.get("diagnoses")
        examinations = content.get("examinations")
        treatment_plan = content.get("treatment_plan")
        clinical_basis = content.get("clinical_basis")
        provenance = content.get("provenance")
        if not isinstance(patient_id, str) or not patient_id.strip():
            return None
        if not VerifiedOnlyMemory._non_empty_string_list(diagnoses):
            return None
        if not VerifiedOnlyMemory._non_empty_string_list(examinations):
            return None
        if not isinstance(treatment_plan, str) or not treatment_plan.strip():
            return None
        if not isinstance(clinical_basis, list) or not all(
            isinstance(item, str) and item.strip() for item in clinical_basis
        ):
            return None
        if not isinstance(provenance, Mapping):
            return None
        if set(provenance) != {"source", "evaluation_hash"}:
            return None
        evaluation_hash = provenance.get("evaluation_hash")
        if provenance.get("source") != "train_evaluation":
            return None
        if not isinstance(evaluation_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", evaluation_hash
        ):
            return None
        value = deepcopy(dict(content))
        value["patient_id"] = patient_id.strip()
        return value

    @staticmethod
    def _non_empty_string_list(value: Any) -> bool:
        return isinstance(value, list) and bool(value) and all(
            isinstance(item, str) and item.strip() for item in value
        )

    @staticmethod
    def _patient_id_aliases(patient_id: str) -> List[str]:
        """Separator-only spellings of one id.

        Plain and comorbid id families share numeric stems, so case folding or
        zero-padding would map two distinct cases onto one answer. Only
        separator placement may vary.
        """
        raw = str(patient_id or "").strip()
        if not raw:
            return []
        aliases = {raw, re.sub(r"[\s_\-]+", "", raw)}
        bare = re.sub(r"^Patient[_\-]", "", raw).strip("_- ")
        if bare and bare != raw:
            aliases.update({bare, "Patient_" + bare, "Patient-" + bare})
        return [item for item in aliases if item]

    @staticmethod
    def _content_fingerprint(diagnoses: Any, examinations: Any) -> str:
        dx = sorted(
            str(item).strip()
            for item in (diagnoses or [])
            if isinstance(item, str) and item.strip()
        )
        exams = sorted(
            str(item).strip()
            for item in (examinations or [])
            if isinstance(item, str) and item.strip()
        )
        payload = json.dumps(
            {"diagnoses": dx, "examinations": exams},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()

    def load_case_memory(
        self,
        patient_id: str,
        *,
        diagnoses: Optional[List[str]] = None,
        examinations: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        raw = str(patient_id or "").strip()
        value = self._case_memories.get(raw)
        if value is None and raw:
            # Exact alias hit only; never fuzzy-match free text onto a case.
            for alias in self._patient_id_aliases(raw):
                mapped = self._id_aliases.get(alias)
                if mapped is not None:
                    value = self._case_memories.get(mapped)
                    if value is not None:
                        break
        if value is None and diagnoses is not None and examinations is not None:
            fingerprint = self._content_fingerprint(diagnoses, examinations)
            mapped = self._content_fingerprints.get(fingerprint)
            if mapped is not None:
                value = self._case_memories.get(mapped)
        return deepcopy(value) if value is not None else None

    def load_notes(
        self,
        limit: Optional[int] = None,
        *,
        trigger_codes: Any = (),
        stage: str = "",
        exclude_patient_id: Optional[str] = None,
        include_candidates: bool = False,
    ) -> List[str]:
        """Generic verified notes, plus reflections only when trigger+stage match.

        Production never reads candidates: include_candidates is accepted for
        signature compatibility and deliberately ignored.
        """
        del exclude_patient_id, include_candidates
        max_items = int(limit or self.max_notes)
        generic = list(self._notes[:max_items])
        if not trigger_codes or not stage:
            return generic
        clinical = self._profiles.reflection_notes(
            trigger_codes=trigger_codes,
            stage=stage,
            limit=max_items,
        )
        return (clinical + generic)[:max_items]

    def exam_profiles(self, diagnoses: Any) -> List[Dict[str, Any]]:
        """Frozen exam profiles for exact official diagnosis names."""
        return self._profiles.exam_profiles(diagnoses)

    def treatment_profiles(self, diagnoses: Any) -> List[Dict[str, Any]]:
        return self._profiles.treatment_profiles(diagnoses)

    def reflection_notes(self, *, trigger_codes: Any = (), stage: str = "", limit: int = 3):
        return self._profiles.reflection_notes(
            trigger_codes=trigger_codes, stage=stage, limit=limit
        )

    def append_case_reflection(self, **_: Any) -> None:
        raise RuntimeError("online memory writes are disabled; use offline Candidate/Promotion")


def _resolve_release_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def build_memory(
    config: Dict[str, Any],
    *,
    loaded_release: Optional[LoadedRelease] = None,
) -> "VerifiedOnlyMemory":
    """Production memory factory: verified registry reader only, no Markdown writes.

    The composition root loads LoadedRelease exactly once and passes it here to
    avoid a second read_file / pointer-or-registry tear. Fail-closed on
    pointer, registry and JSON errors; never silently falls back to a different
    release when a production pointer is present.

    Test exception (explicit registry / no release) is an explicit dependency
    injection boundary (``registry_path`` or ``loaded_release``), not a plain
    config toggle that ordinary configs can flip.
    """
    from agent.runtime.release_loader import load_current_release

    memory_config = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    max_notes = memory_config.get("max_notes", 3) if isinstance(memory_config, dict) else 3
    registry_path = (
        memory_config.get("verified_registry_path") if isinstance(memory_config, dict) else None
    )
    # Composition root already loaded the release once; its hash-bound registry
    # remains authoritative even when ordinary config contains an override path.
    if loaded_release is not None:
        registry = loaded_release.release_dir / "verified_registry.json"
        if not registry.is_file():
            raise ValueError("verified registry missing in release: %s" % registry)
        return VerifiedOnlyMemory(registry, max_notes=int(max_notes))

    # Explicit dependency-injection boundary for release-free tests and tools.
    if registry_path:
        path = Path(registry_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.is_file():
            raise ValueError("verified_registry_path does not exist: %s" % path)
        return VerifiedOnlyMemory(path, max_notes=int(max_notes))

    pointer = BASE_DIR / "releases" / "current.json"
    if not pointer.exists():
        raise ValueError(
            "release pointer missing: %s (inject loaded_release or set registry_path for tests)"
            % pointer
        )

    release = load_current_release(pointer)
    registry = release.release_dir / "verified_registry.json"
    if not registry.is_file():
        raise ValueError("verified registry missing in release: %s" % registry)
    return VerifiedOnlyMemory(registry, max_notes=int(max_notes))
