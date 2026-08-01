"""Aggregate disease profiles from build-partition ground truth only.

Profiles are the sanctioned generalization path: aggregate statistics over
official catalog names and closed codes. Per-case answers, patient ids, grader
reasoning and free-text prescriptions must never reach a profile effect, so the
aggregation deliberately drops anything it cannot map to a catalog leaf or a
registered code.
"""
from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from agent.knowledge.profile_schema import (
    valid_exam_profile_effect,
    valid_treatment_profile_effect,
)
from offline.artifacts import write_immutable_json
from offline.ground_truth_profiles import GroundTruthRecord

EXAM_PROFILE_SCHEMA = "disease-exam-profile/v1"

EXAM_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "diagnosis_name",
        "exam_items",
        "support_case_count",
        "source_receipt_hash",
    }
)
EXAM_ITEM_FIELDS = frozenset({"name", "support_count", "support_ratio", "rank"})


def aggregate_exam_profiles(
    records: Sequence[GroundTruthRecord],
    *,
    partition: Literal["build"],
    exam_catalog_order: Sequence[str],
    source_receipt_hash: str,
    min_support_count: int = 3,
    min_support_ratio: float = 0.35,
    max_examinations: int = 12,
) -> list[dict[str, Any]]:
    """Group build records by diagnosis and keep well-supported catalog leaves."""
    if partition != "build":
        raise ValueError("profiles may only be built from the build partition")
    order_index = {name: index for index, name in enumerate(exam_catalog_order)}

    grouped: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    for record in records:
        if record.partition != "build":
            raise ValueError("held-out record passed to profile builder")
        for diagnosis in set(record.diagnosis_items):
            grouped[diagnosis].append(record)

    profiles: list[dict[str, Any]] = []
    for diagnosis in sorted(grouped):
        disease_records = grouped[diagnosis]
        counts = Counter(
            examination
            for record in disease_records
            # set() so one case never counts an examination twice.
            for examination in set(record.exam_items)
            if examination in order_index
        )
        rows: list[tuple[str, int, float]] = []
        for name, count in counts.items():
            ratio = count / len(disease_records)
            if count >= int(min_support_count) and ratio >= float(min_support_ratio):
                rows.append((name, count, ratio))
        rows.sort(key=lambda row: (-row[2], -row[1], order_index[row[0]], row[0]))
        exam_items = [
            {
                "name": name,
                "support_count": count,
                "support_ratio": round(ratio, 6),
                "rank": index + 1,
            }
            for index, (name, count, ratio) in enumerate(rows[: max(0, int(max_examinations))])
        ]
        if exam_items:
            profiles.append(
                {
                    "schema_version": EXAM_PROFILE_SCHEMA,
                    "diagnosis_name": diagnosis,
                    "exam_items": exam_items,
                    "support_case_count": len(disease_records),
                    "source_receipt_hash": str(source_receipt_hash),
                }
            )
    return profiles


TREATMENT_PROFILE_SCHEMA = "disease-treatment-profile/v1"

TREATMENT_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "diagnosis_name",
        "goal_codes",
        "risk_codes",
        "contraindication_codes",
        "support_stats",
        "source_receipt_hash",
    }
)
SUPPORT_STATS_FIELDS = frozenset(
    {
        "support_case_count",
        "goal_support_counts",
        "risk_support_counts",
        "contraindication_support_counts",
    }
)

# Closed codebooks. Free-text treatment sentences never become codes, so an
# unmappable phrase is counted as quarantine instead of widening the vocabulary.
GOAL_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "antiviral_therapy": ("阿昔洛韦", "伐昔洛韦", "抗病毒"),
        "antibacterial_therapy": ("头孢", "青霉素", "抗菌", "抗生素", "万古霉素"),
        "analgesia_antipyresis": ("对乙酰氨基酚", "退热", "镇痛", "布洛芬"),
        "fluid_and_nutrition": ("补液", "水分", "营养", "饮食"),
        "inpatient_monitoring": ("住院", "监测", "观察", "监护"),
        "surgical_evaluation": ("手术", "外科", "矫治", "切除"),
        "specialist_referral": ("专科", "会诊", "转诊"),
        "immunosuppressant_adjustment": ("免疫抑制", "泼尼松"),
        "wound_and_skin_care": ("皮肤清洁", "创面", "换药"),
        "anticoagulation": ("抗凝", "华法林", "肝素"),
    }
)

RISK_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sepsis_risk": ("脓毒", "败血", "菌血"),
        "neurologic_involvement": ("神经", "中枢"),
        "secondary_bacterial_infection": ("继发感染", "继发细菌", "细菌播散"),
        "immunosuppression": ("免疫抑制", "免疫功能低下"),
        "bleeding_risk": ("出血", "凝血"),
        "airway_compromise": ("呼吸衰竭", "气道", "窒息"),
    }
)

CONTRAINDICATION_CODEBOOK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "systemic_corticosteroid": ("糖皮质激素", "泼尼松", "地塞米松"),
        "live_attenuated_vaccine": ("减毒活疫苗", "活疫苗", "mmr", "bcg"),
        "aspirin_in_children": ("阿司匹林",),
        "tetracycline_class": ("四环素",),
        "codeine_or_tramadol": ("可待因", "曲马多"),
        "nsaid": ("非甾体", "布洛芬", "双氯芬酸"),
        "methotrexate": ("甲氨蝶呤",),
        "nephrotoxic_drug": ("氨基糖苷", "肾毒"),
    }
)

GOAL_MIN_SUPPORT_COUNT = 3
GOAL_MIN_SUPPORT_RATIO = 0.50
CONTRAINDICATION_MIN_SUPPORT_COUNT = 2


def _codes_for_text(
    text: Any,
    codebook: Mapping[str, tuple[str, ...]],
) -> tuple[set[str], bool]:
    """Map free text onto closed codes; report whether it stayed unmapped."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if not normalized.strip():
        return set(), False
    matched = {
        code
        for code, markers in codebook.items()
        if any(
            unicodedata.normalize("NFKC", marker).casefold() in normalized
            for marker in markers
        )
    }
    return matched, not matched


def aggregate_treatment_profiles(
    records: Sequence[GroundTruthRecord],
    *,
    partition: Literal["build"],
    source_receipt_hash: str,
    min_goal_support_count: int = GOAL_MIN_SUPPORT_COUNT,
    min_goal_support_ratio: float = GOAL_MIN_SUPPORT_RATIO,
    min_contraindication_support_count: int = CONTRAINDICATION_MIN_SUPPORT_COUNT,
) -> dict[str, Any]:
    """Aggregate closed goal/risk/contraindication codes from build records only."""
    if partition != "build":
        raise ValueError("profiles may only be built from the build partition")

    grouped: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    for record in records:
        if record.partition != "build":
            raise ValueError("held-out record passed to profile builder")
        for diagnosis in set(record.diagnosis_items):
            grouped[diagnosis].append(record)

    profiles: list[dict[str, Any]] = []
    unmapped_treatment_cases = 0
    unmapped_contraindication_items = 0

    for diagnosis in sorted(grouped):
        disease_records = grouped[diagnosis]
        total = len(disease_records)
        goal_counts: Counter[str] = Counter()
        risk_counts: Counter[str] = Counter()
        contraindication_counts: Counter[str] = Counter()
        for record in disease_records:
            goals, goal_unmapped = _codes_for_text(record.treatment_text, GOAL_CODEBOOK)
            risks, _ = _codes_for_text(record.treatment_text, RISK_CODEBOOK)
            if goal_unmapped:
                unmapped_treatment_cases += 1
            goal_counts.update(goals)
            risk_counts.update(risks)
            case_contraindications: set[str] = set()
            for item in record.contraindication_items:
                codes, unmapped = _codes_for_text(item, CONTRAINDICATION_CODEBOOK)
                if unmapped:
                    unmapped_contraindication_items += 1
                case_contraindications.update(codes)
            contraindication_counts.update(case_contraindications)

        goal_codes = sorted(
            code
            for code, count in goal_counts.items()
            if count >= int(min_goal_support_count)
            and (count / total) >= float(min_goal_support_ratio)
        )
        risk_codes = sorted(
            code
            for code, count in risk_counts.items()
            if count >= int(min_goal_support_count)
            and (count / total) >= float(min_goal_support_ratio)
        )
        contraindication_codes = sorted(
            code
            for code, count in contraindication_counts.items()
            if count >= int(min_contraindication_support_count)
        )
        if not goal_codes and not contraindication_codes:
            continue
        profiles.append(
            {
                "schema_version": TREATMENT_PROFILE_SCHEMA,
                "diagnosis_name": diagnosis,
                "goal_codes": goal_codes,
                "risk_codes": risk_codes,
                "contraindication_codes": contraindication_codes,
                "support_stats": {
                    "support_case_count": total,
                    "goal_support_counts": {code: goal_counts[code] for code in goal_codes},
                    "risk_support_counts": {code: risk_counts[code] for code in risk_codes},
                    "contraindication_support_counts": {
                        code: contraindication_counts[code]
                        for code in contraindication_codes
                    },
                },
                "source_receipt_hash": str(source_receipt_hash),
            }
        )
    return {
        "profiles": profiles,
        "quarantine": {
            "unmapped_treatment_cases": unmapped_treatment_cases,
            "unmapped_contraindication_items": unmapped_contraindication_items,
        },
    }


def _valid_code_list(value: Any, codebook: Mapping[str, tuple[str, ...]]) -> bool:
    return bool(
        isinstance(value, list)
        and all(isinstance(item, str) and item in codebook for item in value)
        and list(value) == sorted(value)
        and len(set(value)) == len(value)
    )


def _valid_support_counts(value: Any, codes: Sequence[str]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == set(codes)
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in value.values()
        )
    )


def _candidate_id(prefix: str, diagnosis: str) -> str:
    """Stable, id-free candidate name derived from the diagnosis only."""
    digest = sha256(diagnosis.encode("utf-8")).hexdigest()[:12]
    return "%s__%s" % (prefix, digest)


def build_profile_candidate_pack(
    loaded: Any,
    *,
    source_receipt: Mapping[str, Any],
    output_root: Path,
    exam_catalog_order: Sequence[str],
) -> list[str]:
    """Write exam/treatment profile candidates for the build partition only.

    Held-out records are never aggregated here: they exist so T07 controls can
    measure the profiles against data that never influenced them.
    """
    from offline.candidates import create_candidate, write_candidate

    if not source_receipt.get("reconciled"):
        raise ValueError("source receipt is not reconciled; refusing to build candidates")
    receipt_hash = str(source_receipt["receipt_hash"])

    build_records = [record for record in loaded.records if record.partition == "build"]
    exam_profiles = aggregate_exam_profiles(
        build_records,
        partition="build",
        exam_catalog_order=exam_catalog_order,
        source_receipt_hash=receipt_hash,
    )
    treatment_result = aggregate_treatment_profiles(
        build_records,
        partition="build",
        source_receipt_hash=receipt_hash,
    )

    root = Path(output_root)
    if root.exists():
        raise FileExistsError("refusing to overwrite candidate pack: %s" % root)
    root.mkdir(parents=True)

    written: list[str] = []
    for profile in exam_profiles:
        candidate = create_candidate(
            candidate_id=_candidate_id("disease_exam_profile", profile["diagnosis_name"]),
            candidate_type="disease_exam_profile",
            proposed_effect=profile,
            evidence={
                "source_receipt_hash": receipt_hash,
                "partition": "build",
                "support_case_count": profile["support_case_count"],
            },
        )
        path = root / ("%s.json" % candidate["candidate_id"])
        write_candidate(path, candidate)
        written.append(path.as_posix())

    for profile in treatment_result["profiles"]:
        candidate = create_candidate(
            candidate_id=_candidate_id("disease_treatment_profile", profile["diagnosis_name"]),
            candidate_type="disease_treatment_profile",
            proposed_effect=profile,
            evidence={
                "source_receipt_hash": receipt_hash,
                "partition": "build",
                "support_case_count": profile["support_stats"]["support_case_count"],
            },
        )
        path = root / ("%s.json" % candidate["candidate_id"])
        write_candidate(path, candidate)
        written.append(path.as_posix())

    write_immutable_json(
        root / "pack_summary.json",
        {
            "schema_version": "profile-candidate-pack/v1",
            "source_receipt_hash": receipt_hash,
            "build_partition_hash": str(source_receipt["build_partition_hash"]),
            "held_out_partition_hash": str(source_receipt["held_out_partition_hash"]),
            "build_record_count": len(build_records),
            "exam_profile_count": len(exam_profiles),
            "treatment_profile_count": len(treatment_result["profiles"]),
            "quarantine": treatment_result["quarantine"],
            "candidate_files": sorted(Path(item).name for item in written),
            "online_actions": [],
        },
    )
    return sorted(written)


def codes_for_text(text: str, codebook):
    """Public alias: map free text onto closed codes."""
    return _codes_for_text(text, codebook)
