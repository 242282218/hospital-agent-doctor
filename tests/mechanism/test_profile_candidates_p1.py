"""T05/T06: disease profiles must aggregate build-partition statistics only.

The profiles are the only generalization path from harvested ground truth to
runtime. They may carry catalog leaf names, aggregate counts and closed codes,
never patient ids, per-case references or free-text prescriptions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from offline.artifacts import canonical_json, content_hash
from offline.candidates import (
    PROFILE_CANDIDATE_TYPES,
    _quarantine_reason,
    create_candidate,
    load_candidate,
    write_candidate,
)
from offline.ground_truth_profiles import GroundTruthRecord
from offline.profile_candidates import (
    build_profile_candidate_pack,
    EXAM_PROFILE_SCHEMA,
    GOAL_CODEBOOK,
    TREATMENT_PROFILE_SCHEMA,
    aggregate_exam_profiles,
    aggregate_treatment_profiles,
)

_RECEIPT_HASH = "sha256:" + "c" * 64

_CATALOG_ORDER = [
    "体格检查",
    "全血细胞计数（CBC）",
    "病毒核酸检测（Viral NAT）",
    "细胞学检查",
    "细菌培养及鉴定",
    "超声心动图",
]


def _record(
    patient_id: str,
    *,
    diagnoses: List[str] | None = None,
    exams: List[str] | None = None,
    partition: str = "build",
    treatment: str = "静脉阿昔洛韦抗病毒治疗；监测继发细菌感染。",
    contraindications: List[str] | None = None,
) -> GroundTruthRecord:
    return GroundTruthRecord(
        patient_id=patient_id,
        diagnosis_items=tuple(diagnoses or ["卡波西水痘样疹"]),
        exam_items=tuple(exams if exams is not None else ["体格检查", "全血细胞计数（CBC）"]),
        treatment_text=treatment,
        contraindication_items=tuple(
            ["糖皮质激素"] if contraindications is None else contraindications
        ),
        source_run="train_harvest_1",
        evaluation_hash="sha256:" + "a" * 64,
        partition=partition,  # type: ignore[arg-type]
    )


def _build_records(count: int, **kwargs: Any) -> List[GroundTruthRecord]:
    return [_record("Patient_%05d" % index, **kwargs) for index in range(1, count + 1)]


def _aggregate(records: List[GroundTruthRecord], **kwargs: Any) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "partition": "build",
        "exam_catalog_order": _CATALOG_ORDER,
        "source_receipt_hash": _RECEIPT_HASH,
    }
    params.update(kwargs)
    return aggregate_exam_profiles(records, **params)


def test_exam_profile_contains_only_catalog_leaves_and_counts() -> None:
    profiles = _aggregate(_build_records(4))
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["schema_version"] == EXAM_PROFILE_SCHEMA
    assert profile["diagnosis_name"] == "卡波西水痘样疹"
    assert profile["support_case_count"] == 4
    assert profile["source_receipt_hash"] == _RECEIPT_HASH
    assert set(profile) == {
        "schema_version",
        "diagnosis_name",
        "exam_items",
        "support_case_count",
        "source_receipt_hash",
    }
    for item in profile["exam_items"]:
        assert set(item) == {"name", "support_count", "support_ratio", "rank"}
        assert item["name"] in _CATALOG_ORDER
        assert isinstance(item["support_count"], int)
        assert 0.0 < float(item["support_ratio"]) <= 1.0
    assert [item["rank"] for item in profile["exam_items"]] == [1, 2]


def test_exam_profile_never_contains_patient_ids_or_references() -> None:
    profiles = _aggregate(_build_records(5))
    blob = canonical_json(profiles)
    for marker in ("Patient_", "patient_id", "ground_truth", "reference", "expected"):
        assert marker not in blob
    # Treatment text must never reach an exam profile.
    assert "阿昔洛韦" not in blob


def test_exam_profile_excludes_low_support_examinations() -> None:
    records = _build_records(4)
    # One-off examination appears in a single case only.
    records.append(
        _record(
            "Patient_00099",
            exams=["体格检查", "全血细胞计数（CBC）", "超声心动图"],
        )
    )
    profiles = _aggregate(records)
    names = [item["name"] for item in profiles[0]["exam_items"]]
    assert "超声心动图" not in names
    assert names == ["体格检查", "全血细胞计数（CBC）"]


def test_exam_profile_is_byte_stable_under_input_reordering() -> None:
    records = _build_records(6)
    forward = _aggregate(records)
    backward = _aggregate(list(reversed(records)))
    assert canonical_json(forward) == canonical_json(backward)
    assert content_hash(forward) == content_hash(backward)


def test_exam_profile_counts_each_case_once_per_examination() -> None:
    records = _build_records(3, exams=["体格检查", "体格检查", "全血细胞计数（CBC）"])
    profiles = _aggregate(records)
    counts = {item["name"]: item["support_count"] for item in profiles[0]["exam_items"]}
    assert counts["体格检查"] == 3


def test_exam_profile_rejects_held_out_records() -> None:
    records = _build_records(3) + [_record("Patient_00100", partition="held_out")]
    with pytest.raises(ValueError, match="held-out"):
        _aggregate(records)


def test_exam_profile_rejects_non_build_partition_argument() -> None:
    with pytest.raises(ValueError, match="build partition"):
        _aggregate(_build_records(3), partition="held_out")


def test_exam_profile_caps_examinations_per_disease() -> None:
    many = ["体格检查", "全血细胞计数（CBC）", "病毒核酸检测（Viral NAT）", "细胞学检查"]
    profiles = _aggregate(_build_records(5, exams=many), max_examinations=2)
    assert len(profiles[0]["exam_items"]) == 2


def test_exam_profile_orders_by_ratio_then_count_then_catalog() -> None:
    records = _build_records(4, exams=["体格检查", "全血细胞计数（CBC）"])
    records.extend(
        _record("Patient_001%02d" % index, exams=["体格检查"]) for index in range(1, 5)
    )
    profiles = _aggregate(records)
    names = [item["name"] for item in profiles[0]["exam_items"]]
    assert names[0] == "体格检查"


def test_exam_profile_groups_multi_diagnosis_records() -> None:
    records = _build_records(3, diagnoses=["卡波西水痘样疹", "三房心"])
    profiles = _aggregate(records)
    assert [profile["diagnosis_name"] for profile in profiles] == sorted(
        ["三房心", "卡波西水痘样疹"]
    )


def test_exam_profile_drops_names_outside_catalog_order() -> None:
    records = _build_records(4, exams=["体格检查", "不在目录的检查"])
    profiles = _aggregate(records)
    names = [item["name"] for item in profiles[0]["exam_items"]]
    assert names == ["体格检查"]


def test_profile_candidate_types_are_registered() -> None:
    assert PROFILE_CANDIDATE_TYPES == frozenset(
        {"disease_exam_profile", "disease_treatment_profile", "reflection_rule"}
    )


def test_valid_exam_profile_effect_is_not_quarantined() -> None:
    profile = _aggregate(_build_records(4))[0]
    evidence = {
        "source_receipt_hash": _RECEIPT_HASH,
        "partition": "build",
        "support_case_count": profile["support_case_count"],
    }
    assert _quarantine_reason("disease_exam_profile", profile, evidence) == ""


def test_exam_profile_with_unknown_field_is_quarantined() -> None:
    profile = _aggregate(_build_records(4))[0]
    profile["unexpected"] = "x"
    evidence = {
        "source_receipt_hash": _RECEIPT_HASH,
        "partition": "build",
        "support_case_count": 4,
    }
    assert _quarantine_reason("disease_exam_profile", profile, evidence) != ""


@pytest.mark.parametrize(
    "leaked",
    [
        {"diagnosis_name": "Patient_01061"},
        {"diagnosis_name": "参考答案：诊断：X 检查：Y 治疗：Z"},
    ],
)
def test_exam_profile_with_leaked_value_is_quarantined(leaked: Dict[str, Any]) -> None:
    profile = _aggregate(_build_records(4))[0]
    profile.update(leaked)
    evidence = {
        "source_receipt_hash": _RECEIPT_HASH,
        "partition": "build",
        "support_case_count": 4,
    }
    assert _quarantine_reason("disease_exam_profile", profile, evidence) != ""


def test_exam_profile_candidate_carries_hashes(tmp_path: Path) -> None:
    profile = _aggregate(_build_records(4))[0]
    candidate = create_candidate(
        candidate_id="disease_exam_profile__kaposi",
        candidate_type="disease_exam_profile",
        proposed_effect=profile,
        evidence={
            "source_receipt_hash": _RECEIPT_HASH,
            "partition": "build",
            "support_case_count": profile["support_case_count"],
        },
    )
    assert candidate["status"] == "candidate"
    assert candidate["candidate_hash"].startswith("sha256:") is False  # raw hex
    assert len(candidate["candidate_hash"]) == 64
    assert len(candidate["effect_hash"]) == 64
    assert candidate["proposed_effect"]["diagnosis_name"] == "卡波西水痘样疹"


def test_exam_profile_min_support_ratio_is_enforced() -> None:
    records = _build_records(10, exams=["体格检查"])
    records.extend(
        _record("Patient_002%02d" % index, exams=["体格检查", "细胞学检查"])
        for index in range(1, 4)
    )
    profiles = _aggregate(records)
    names = [item["name"] for item in profiles[0]["exam_items"]]
    # 3/13 support ratio is below the 0.35 floor even though count reaches 3.
    assert "细胞学检查" not in names


# --- T06: treatment goal / risk / contraindication profiles -------------------


def _treatment_records(
    count: int,
    *,
    treatment: str,
    contraindications: List[str] | None = None,
    diagnoses: List[str] | None = None,
    partition: str = "build",
    start: int = 1,
) -> List[GroundTruthRecord]:
    return [
        _record(
            "Patient_%05d" % index,
            treatment=treatment,
            contraindications=contraindications,
            diagnoses=diagnoses,
            partition=partition,
        )
        for index in range(start, start + count)
    ]


def _aggregate_treatment(records: List[GroundTruthRecord], **kwargs: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "partition": "build",
        "source_receipt_hash": _RECEIPT_HASH,
    }
    params.update(kwargs)
    return aggregate_treatment_profiles(records, **params)


_ACYCLOVIR_PLAN = "静脉注射阿昔洛韦抗病毒；住院监测；补液支持。"


def test_treatment_profile_uses_closed_codes_not_reference_sentences() -> None:
    result = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))
    profile = result["profiles"][0]
    assert profile["schema_version"] == TREATMENT_PROFILE_SCHEMA
    assert set(profile) == {
        "schema_version",
        "diagnosis_name",
        "goal_codes",
        "risk_codes",
        "contraindication_codes",
        "support_stats",
        "source_receipt_hash",
    }
    assert "antiviral_therapy" in profile["goal_codes"]
    assert all(code in GOAL_CODEBOOK for code in profile["goal_codes"])
    blob = canonical_json(profile)
    # The reference sentence itself must never survive into the effect.
    assert "阿昔洛韦" not in blob
    assert "静脉注射" not in blob
    assert "Patient_" not in blob


def test_treatment_profile_effect_has_no_free_text_or_dose() -> None:
    dosed = "阿昔洛韦 10mg/kg 静脉每8小时一次，共14天；住院监测。"
    result = _aggregate_treatment(_treatment_records(4, treatment=dosed))
    blob = canonical_json(result["profiles"])
    for marker in ("mg", "kg", "每8小时", "14天"):
        assert marker not in blob


def test_treatment_profile_quarantines_unmappable_text() -> None:
    result = _aggregate_treatment(
        _treatment_records(
            4,
            treatment="按当地某种未登记的特殊路径处置。",
            contraindications=[],
        )
    )
    assert result["quarantine"]["unmapped_treatment_cases"] == 4
    # Nothing mappable at all means no profile is emitted.
    assert result["profiles"] == []


def test_treatment_profile_keeps_safety_only_profile() -> None:
    """Unmappable goals must not discard aggregated contraindications."""
    result = _aggregate_treatment(
        _treatment_records(
            4,
            treatment="按当地某种未登记的特殊路径处置。",
            contraindications=["糖皮质激素"],
        )
    )
    assert result["quarantine"]["unmapped_treatment_cases"] == 4
    profile = result["profiles"][0]
    assert profile["goal_codes"] == []
    assert profile["contraindication_codes"] == ["systemic_corticosteroid"]


def test_treatment_profile_quarantines_unknown_contraindication() -> None:
    result = _aggregate_treatment(
        _treatment_records(
            4,
            treatment=_ACYCLOVIR_PLAN,
            contraindications=["某种未登记的禁忌物"],
        )
    )
    assert result["quarantine"]["unmapped_contraindication_items"] == 4
    assert result["profiles"][0]["contraindication_codes"] == []


def test_treatment_profile_contraindication_requires_aggregate_support() -> None:
    records = _treatment_records(
        4, treatment=_ACYCLOVIR_PLAN, contraindications=["无法映射项"]
    )
    # A single case mentioning a real contraindication stays below the floor of 2.
    records.append(
        _record(
            "Patient_00099",
            treatment=_ACYCLOVIR_PLAN,
            contraindications=["阿司匹林"],
        )
    )
    single = _aggregate_treatment(records)
    assert single["profiles"][0]["contraindication_codes"] == []

    records.append(
        _record(
            "Patient_00098",
            treatment=_ACYCLOVIR_PLAN,
            contraindications=["阿司匹林"],
        )
    )
    paired = _aggregate_treatment(records)
    assert paired["profiles"][0]["contraindication_codes"] == ["aspirin_in_children"]


def test_treatment_profile_goal_requires_majority_support() -> None:
    records = _treatment_records(6, treatment="住院监测并补液。")
    records.extend(
        _treatment_records(
            2,
            treatment="手术矫治评估；住院监测。",
            start=90,
        )
    )
    profile = _aggregate_treatment(records)["profiles"][0]
    # 2/8 = 0.25 is below the 0.50 goal ratio floor.
    assert "surgical_evaluation" not in profile["goal_codes"]
    assert "inpatient_monitoring" in profile["goal_codes"]


def test_treatment_profile_rejects_held_out_records() -> None:
    records = _treatment_records(3, treatment=_ACYCLOVIR_PLAN)
    records.append(
        _record("Patient_00100", treatment=_ACYCLOVIR_PLAN, partition="held_out")
    )
    with pytest.raises(ValueError, match="held-out"):
        _aggregate_treatment(records)


def test_treatment_profile_rejects_non_build_partition_argument() -> None:
    with pytest.raises(ValueError, match="build partition"):
        _aggregate_treatment(
            _treatment_records(3, treatment=_ACYCLOVIR_PLAN), partition="held_out"
        )


def test_treatment_profile_is_byte_stable_under_reordering() -> None:
    records = _treatment_records(5, treatment=_ACYCLOVIR_PLAN)
    forward = _aggregate_treatment(records)["profiles"]
    backward = _aggregate_treatment(list(reversed(records)))["profiles"]
    assert canonical_json(forward) == canonical_json(backward)


def test_treatment_profile_effect_avoids_global_leakage_keys() -> None:
    profile = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))[
        "profiles"
    ][0]
    for forbidden in ("diagnoses", "treatment_plan", "examinations", "clinical_basis"):
        assert forbidden not in profile


def test_valid_treatment_profile_effect_is_not_quarantined() -> None:
    profile = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))[
        "profiles"
    ][0]
    evidence = {
        "source_receipt_hash": _RECEIPT_HASH,
        "partition": "build",
        "support_case_count": profile["support_stats"]["support_case_count"],
    }
    assert _quarantine_reason("disease_treatment_profile", profile, evidence) == ""


def test_treatment_profile_with_unknown_code_is_quarantined() -> None:
    profile = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))[
        "profiles"
    ][0]
    profile["goal_codes"] = ["not_a_registered_goal"]
    evidence = {"source_receipt_hash": _RECEIPT_HASH, "partition": "build"}
    assert _quarantine_reason("disease_treatment_profile", profile, evidence) != ""


def test_treatment_profile_with_free_text_goal_is_quarantined() -> None:
    profile = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))[
        "profiles"
    ][0]
    profile["goal_codes"] = ["静脉注射阿昔洛韦 10mg/kg 每8小时一次"]
    evidence = {"source_receipt_hash": _RECEIPT_HASH, "partition": "build"}
    assert _quarantine_reason("disease_treatment_profile", profile, evidence) != ""


def test_treatment_profile_candidate_round_trips(tmp_path: Path) -> None:
    profile = _aggregate_treatment(_treatment_records(4, treatment=_ACYCLOVIR_PLAN))[
        "profiles"
    ][0]
    candidate = create_candidate(
        candidate_id="disease_treatment_profile__kaposi",
        candidate_type="disease_treatment_profile",
        proposed_effect=profile,
        evidence={
            "source_receipt_hash": _RECEIPT_HASH,
            "partition": "build",
            "support_case_count": profile["support_stats"]["support_case_count"],
        },
    )
    assert candidate["status"] == "candidate"
    path = tmp_path / "candidate.json"
    write_candidate(path, candidate)
    assert load_candidate(path)["effect_hash"] == candidate["effect_hash"]


# --- pack builder (T05 step 4 / T06 wiring) ---


def _loaded_stub(records):
    from offline.ground_truth_profiles import LoadedGroundTruth

    return LoadedGroundTruth(
        records=tuple(records),
        source_files=("train_harvest_1/evaluation_results.jsonl",),
        source_file_hashes={"train_harvest_1/evaluation_results.jsonl": "sha256:" + "d" * 64},
        raw_row_count=len(records),
        identical_duplicate_count=0,
        rejected_count=0,
        rejected_reasons={},
        conflicting_patient_ids=(),
    )


def _receipt_stub():
    return {
        "schema_version": "ground-truth-source-receipt/v1",
        "receipt_hash": _RECEIPT_HASH,
        "build_partition_hash": "sha256:" + "e" * 64,
        "held_out_partition_hash": "sha256:" + "f" * 64,
        "reconciled": True,
    }


def test_pack_builder_writes_only_build_partition_candidates(tmp_path: Path) -> None:
    records = _build_records(4) + [_record("Patient_00500", partition="held_out")]
    written = build_profile_candidate_pack(
        _loaded_stub(records),
        source_receipt=_receipt_stub(),
        output_root=tmp_path / "pack",
        exam_catalog_order=_CATALOG_ORDER,
    )
    assert written
    blob = "".join(Path(path).read_text(encoding="utf-8") for path in written)
    assert "Patient_" not in blob
    assert "held_out" not in blob
    for path in written:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["status"] == "candidate"
        assert payload["candidate_type"] in {
            "disease_exam_profile",
            "disease_treatment_profile",
        }
        assert payload["evidence"]["partition"] == "build"
        assert payload["evidence"]["source_receipt_hash"] == _RECEIPT_HASH


def test_pack_builder_refuses_to_overwrite(tmp_path: Path) -> None:
    records = _build_records(4)
    build_profile_candidate_pack(
        _loaded_stub(records),
        source_receipt=_receipt_stub(),
        output_root=tmp_path / "pack",
        exam_catalog_order=_CATALOG_ORDER,
    )
    with pytest.raises(FileExistsError):
        build_profile_candidate_pack(
            _loaded_stub(records),
            source_receipt=_receipt_stub(),
            output_root=tmp_path / "pack",
            exam_catalog_order=_CATALOG_ORDER,
        )


def test_pack_builder_rejects_unreconciled_receipt(tmp_path: Path) -> None:
    receipt = _receipt_stub()
    receipt["reconciled"] = False
    with pytest.raises(ValueError, match="reconcil"):
        build_profile_candidate_pack(
            _loaded_stub(_build_records(4)),
            source_receipt=receipt,
            output_root=tmp_path / "pack",
            exam_catalog_order=_CATALOG_ORDER,
        )
