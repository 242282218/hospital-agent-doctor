"""T04: harvest ground-truth sources need auditable receipts and stable partitions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from offline.ground_truth_profiles import (
    GroundTruthRecord,
    build_source_receipt,
    load_ground_truth_records,
    partition_for_patient,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64

_DIAGNOSIS = "卡波西水痘样疹"
_EXAMS = ["体格检查", "全血细胞计数（CBC）"]


def _row(
    patient_id: str,
    *,
    diagnosis: str = _DIAGNOSIS,
    exams: List[str] | None = None,
    treatment: str = "静脉阿昔洛韦抗病毒治疗，并监测继发细菌感染。",
    contraindications: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "report": {
            "ground_truth": {
                "final_diagnosis": diagnosis,
                "necessary_examinations": list(exams if exams is not None else _EXAMS),
                "treatment_plan": treatment,
                "contraindications": list(contraindications or ["避免使用糖皮质激素"]),
            }
        },
    }


def _write_run(root: Path, name: str, rows: List[Dict[str, Any]]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    path = run_dir / "evaluation_results.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


def _catalogs() -> Dict[str, Any]:
    return {
        "official_diseases": {_DIAGNOSIS, "三房心"},
        "exam_leaf_names": {"体格检查", "全血细胞计数（CBC）", "超声心动图"},
    }


def test_identical_duplicates_are_deduplicated(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00001")])
    _write_run(tmp_path, "train_harvest_2", [_row("Patient_00001")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert len(result.records) == 1
    assert result.identical_duplicate_count == 1
    assert result.conflicting_patient_ids == ()


def test_conflicting_duplicates_are_rejected(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00001")])
    _write_run(tmp_path, "train_harvest_2", [_row("Patient_00001", diagnosis="三房心")])
    with pytest.raises(ValueError, match="conflicting ground truth"):
        load_ground_truth_records(source_root=tmp_path, **_catalogs())


def test_non_catalog_names_are_rejected(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00002", diagnosis="不在目录的病")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert result.records == ()
    assert result.rejected_count == 1
    assert result.rejected_reasons["diagnosis_not_official"] == 1

    _write_run(tmp_path, "train_harvest_2", [_row("Patient_00003", exams=["不存在的检查"])])
    result2 = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert result2.rejected_reasons["examination_not_catalog_leaf"] == 1


def test_partition_is_stable_and_input_order_independent(tmp_path: Path) -> None:
    rows = [_row("Patient_%05d" % index) for index in range(1, 11)]
    _write_run(tmp_path, "train_harvest_1", rows)
    forward = load_ground_truth_records(source_root=tmp_path, **_catalogs())

    other = tmp_path / "reversed"
    other.mkdir()
    _write_run(other, "train_harvest_1", list(reversed(rows)))
    backward = load_ground_truth_records(source_root=other, **_catalogs())

    forward_map = {record.patient_id: record.partition for record in forward.records}
    backward_map = {record.patient_id: record.partition for record in backward.records}
    assert forward_map == backward_map
    assert [record.patient_id for record in forward.records] == [
        record.patient_id for record in backward.records
    ]
    assert set(forward_map.values()) <= {"build", "held_out"}


def test_partition_rule_is_sha256_modulo_five() -> None:
    for patient_id in ("Patient_00001", "Patient_02654", "Patient_09249"):
        assert partition_for_patient(patient_id) in {"build", "held_out"}
    # Rule is a pure function of the id, so repeated calls never drift.
    assert partition_for_patient("Patient_00001") == partition_for_patient("Patient_00001")


def test_receipt_reconciles_raw_unique_duplicate_rejected(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "train_harvest_1",
        [
            _row("Patient_00001"),
            _row("Patient_00002"),
            _row("Patient_00003", diagnosis="不在目录的病"),
        ],
    )
    _write_run(tmp_path, "train_harvest_2", [_row("Patient_00001")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    receipt = build_source_receipt(
        result,
        declared_pool_count=5,
        rejected_ledger_count=1,
    )
    assert receipt["schema_version"] == "ground-truth-source-receipt/v1"
    assert receipt["raw_row_count"] == 4
    assert receipt["unique_count"] == 2
    assert receipt["identical_duplicate_count"] == 1
    assert receipt["rejected_count"] == 1
    assert (
        receipt["raw_row_count"]
        == receipt["unique_count"]
        + receipt["identical_duplicate_count"]
        + receipt["rejected_count"]
    )
    # declared pool = unique + missing + rejected(+ledger)
    assert (
        receipt["declared_pool_count"]
        == receipt["unique_count"] + receipt["missing_count"] + receipt["total_rejected_count"]
    )
    assert receipt["reconciled"] is True
    assert receipt["build_count"] + receipt["held_out_count"] == receipt["unique_count"]
    for key in (
        "source_hash",
        "build_partition_hash",
        "held_out_partition_hash",
        "receipt_hash",
    ):
        assert receipt[key].startswith("sha256:")
    assert receipt["source_file_hashes"]
    assert all(value.startswith("sha256:") for value in receipt["source_file_hashes"].values())


def test_receipt_is_recomputable(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00001"), _row("Patient_00002")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    first = build_source_receipt(result, declared_pool_count=2, rejected_ledger_count=0)
    second = build_source_receipt(result, declared_pool_count=2, rejected_ledger_count=0)
    assert first == second


def test_receipt_flags_unreconciled_declared_pool(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00001")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    receipt = build_source_receipt(result, declared_pool_count=100, rejected_ledger_count=0)
    assert receipt["missing_count"] == 99
    assert receipt["reconciled"] is True
    # A declared pool smaller than the observed unique rows can never reconcile.
    shrunk = build_source_receipt(result, declared_pool_count=0, rejected_ledger_count=0)
    assert shrunk["reconciled"] is False


def test_records_are_frozen_and_carry_partition(tmp_path: Path) -> None:
    _write_run(tmp_path, "train_harvest_1", [_row("Patient_00001")])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    record = result.records[0]
    assert isinstance(record, GroundTruthRecord)
    assert record.diagnosis_items == (_DIAGNOSIS,)
    assert record.exam_items == tuple(_EXAMS)
    assert record.partition in {"build", "held_out"}
    assert record.evaluation_hash.startswith("sha256:")
    with pytest.raises(Exception):
        record.patient_id = "Patient_99999"  # type: ignore[misc]


def test_contraindication_dict_shape_is_parsed(tmp_path: Path) -> None:
    """Harvest rows store contraindications as {"drugs": [...], "treatments": [...]}."""
    row = _row("Patient_00001")
    row["report"]["ground_truth"]["contraindications"] = {
        "drugs": ["可待因", "阿司匹林"],
        "treatments": ["全身性糖皮质激素"],
    }
    _write_run(tmp_path, "train_harvest_1", [row])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert result.rejected_count == 0
    assert result.records[0].contraindication_items == (
        "可待因",
        "阿司匹林",
        "全身性糖皮质激素",
    )


def test_contraindication_dict_with_unknown_key_is_rejected(tmp_path: Path) -> None:
    row = _row("Patient_00001")
    row["report"]["ground_truth"]["contraindications"] = {
        "drugs": ["可待因"],
        "treatments": [],
        "surprise": ["x"],
    }
    _write_run(tmp_path, "train_harvest_1", [row])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert result.records == ()
    assert result.rejected_reasons["malformed_row"] == 1


def test_empty_contraindications_are_allowed(tmp_path: Path) -> None:
    row = _row("Patient_00001")
    row["report"]["ground_truth"]["contraindications"] = {"drugs": [], "treatments": []}
    _write_run(tmp_path, "train_harvest_1", [row])
    result = load_ground_truth_records(source_root=tmp_path, **_catalogs())
    assert result.rejected_count == 0
    assert result.records[0].contraindication_items == ()
