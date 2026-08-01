"""T07: profiles may only be promoted through immutable held-out control reports.

The controls are the only thing standing between an aggregate statistic and the
online path, so they must be computed on records that never participated in the
build aggregation, must be recomputable from the stored report, and must fail
closed on any tampering.
"""
from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from offline.artifacts import content_hash, read_json
from offline.candidates import create_candidate, write_candidate
from offline.ground_truth_profiles import GroundTruthRecord
from offline.profile_candidates import (
    aggregate_exam_profiles,
    aggregate_treatment_profiles,
)
from offline.profile_controls import (
    PROFILE_CONTROL_REPORT_SCHEMA,
    ProfileControlReport,
    build_exam_profile_control_report,
    build_treatment_profile_control_report,
    validate_profile_control_report,
    write_control_report,
)
from offline.promotion import approve_candidate, build_registry_snapshot
from offline.release import build_candidate_pack

_RECEIPT_HASH = "sha256:" + "c" * 64
_HELD_OUT_HASH = "sha256:" + "d" * 64

_CATALOG_ORDER = [
    "体格检查",
    "全血细胞计数（CBC）",
    "病毒核酸检测（Viral NAT）",
    "细胞学检查",
    "超声心动图",
]

_DIAGNOSIS = "卡波西水痘样疹"
_PLAN = "静脉注射阿昔洛韦抗病毒；住院监测；补液支持。"


def _record(
    patient_id: str,
    *,
    partition: str,
    diagnoses: List[str] | None = None,
    exams: List[str] | None = None,
    treatment: str = _PLAN,
    contraindications: List[str] | None = None,
) -> GroundTruthRecord:
    return GroundTruthRecord(
        patient_id=patient_id,
        diagnosis_items=tuple(diagnoses or [_DIAGNOSIS]),
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
    return [
        _record("Patient_1%04d" % index, partition="build", **kwargs)
        for index in range(1, count + 1)
    ]


def _held_out_records(count: int, **kwargs: Any) -> List[GroundTruthRecord]:
    return [
        _record("Patient_2%04d" % index, partition="held_out", **kwargs)
        for index in range(1, count + 1)
    ]


def _exam_profiles(records: List[GroundTruthRecord]) -> List[Dict[str, Any]]:
    return aggregate_exam_profiles(
        records,
        partition="build",
        exam_catalog_order=_CATALOG_ORDER,
        source_receipt_hash=_RECEIPT_HASH,
    )


def _treatment_profiles(records: List[GroundTruthRecord]) -> List[Dict[str, Any]]:
    return aggregate_treatment_profiles(
        records,
        partition="build",
        source_receipt_hash=_RECEIPT_HASH,
    )["profiles"]


def _exam_report(
    *,
    build_count: int = 5,
    held_out_count: int = 5,
    candidate_hash: str = "abc",
    **kwargs: Any,
) -> ProfileControlReport:
    profiles = _exam_profiles(_build_records(build_count))
    return build_exam_profile_control_report(
        profiles=profiles,
        held_out_records=_held_out_records(held_out_count, **kwargs),
        candidate_hash=candidate_hash,
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
        exam_catalog_order=_CATALOG_ORDER,
    )


def test_exam_control_report_is_computed_on_held_out_only() -> None:
    report = _exam_report()
    assert isinstance(report, ProfileControlReport)
    assert report.candidate_type == "disease_exam_profile"
    assert report.held_out_partition_hash == _HELD_OUT_HASH
    assert report.leakage_count == 0
    assert report.exam_macro_recall_at_12 == pytest.approx(1.0)
    assert report.exam_macro_precision_at_12 == pytest.approx(1.0)
    assert report.passed is True
    assert report.report_hash.startswith("sha256:")


def test_exam_control_report_rejects_build_partition_records() -> None:
    profiles = _exam_profiles(_build_records(4))
    with pytest.raises(ValueError, match="held-out"):
        build_exam_profile_control_report(
            profiles=profiles,
            held_out_records=_build_records(4),
            candidate_hash="abc",
            source_receipt_hash=_RECEIPT_HASH,
            held_out_partition_hash=_HELD_OUT_HASH,
            exam_catalog_order=_CATALOG_ORDER,
        )


def test_exam_control_report_fails_on_precision_drop() -> None:
    # Profile recommends two examinations; held-out cases only ever need one,
    # so precision collapses relative to the no-profile baseline.
    profiles = _exam_profiles(_build_records(5))
    report = build_exam_profile_control_report(
        profiles=profiles,
        held_out_records=_held_out_records(5, exams=["体格检查"]),
        candidate_hash="abc",
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
        exam_catalog_order=_CATALOG_ORDER,
    )
    assert report.exam_macro_precision_at_12 < 1.0
    assert report.passed is False


def test_treatment_control_report_checks_goal_recall_and_false_positives() -> None:
    profiles = _treatment_profiles(_build_records(5))
    report = build_treatment_profile_control_report(
        profiles=profiles,
        held_out_records=_held_out_records(5),
        candidate_hash="abc",
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
    )
    assert report.candidate_type == "disease_treatment_profile"
    assert report.treatment_goal_macro_recall == pytest.approx(1.0)
    assert report.contraindication_false_positive_count == 0
    assert report.passed is True


def test_treatment_control_report_fails_on_contraindication_false_positive() -> None:
    profiles = _treatment_profiles(_build_records(5))
    # Held-out cases never list a corticosteroid contraindication, so the
    # aggregated contraindication code is a false positive there.
    report = build_treatment_profile_control_report(
        profiles=profiles,
        held_out_records=_held_out_records(5, contraindications=[]),
        candidate_hash="abc",
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
    )
    assert report.contraindication_false_positive_count > 0
    assert report.passed is False


def test_control_report_hash_is_recomputable_and_tamper_evident(tmp_path: Path) -> None:
    report = _exam_report()
    path = write_control_report(tmp_path / "exam.json", report)
    stored = read_json(path)
    assert stored["schema_version"] == PROFILE_CONTROL_REPORT_SCHEMA
    assert validate_profile_control_report(
        stored,
        candidate_type="disease_exam_profile",
        candidate_hash="abc",
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
    )

    tampered = dict(stored)
    # Any metric edit must invalidate the stored hash.
    tampered["exam_macro_precision_at_12"] = 0.1
    with pytest.raises(ValueError, match="report_hash"):
        validate_profile_control_report(
            tampered,
            candidate_type="disease_exam_profile",
            candidate_hash="abc",
            source_receipt_hash=_RECEIPT_HASH,
            held_out_partition_hash=_HELD_OUT_HASH,
        )


def test_control_report_validation_binds_candidate_and_source() -> None:
    report = _exam_report()
    stored = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    with pytest.raises(ValueError, match="candidate_hash"):
        validate_profile_control_report(
            stored,
            candidate_type="disease_exam_profile",
            candidate_hash="different",
            source_receipt_hash=_RECEIPT_HASH,
            held_out_partition_hash=_HELD_OUT_HASH,
        )
    with pytest.raises(ValueError, match="source_receipt_hash"):
        validate_profile_control_report(
            stored,
            candidate_type="disease_exam_profile",
            candidate_hash="abc",
            source_receipt_hash="sha256:" + "9" * 64,
            held_out_partition_hash=_HELD_OUT_HASH,
        )
    with pytest.raises(ValueError, match="held_out_partition_hash"):
        validate_profile_control_report(
            stored,
            candidate_type="disease_exam_profile",
            candidate_hash="abc",
            source_receipt_hash=_RECEIPT_HASH,
            held_out_partition_hash="sha256:" + "8" * 64,
        )


def test_control_report_type_must_match_candidate_type() -> None:
    report = _exam_report()
    with pytest.raises(ValueError, match="candidate_type"):
        validate_profile_control_report(
            report.to_dict(),
            candidate_type="disease_treatment_profile",
            candidate_hash="abc",
            source_receipt_hash=_RECEIPT_HASH,
            held_out_partition_hash=_HELD_OUT_HASH,
        )


def _write_profile_candidate(store: Path, candidate_id: str) -> Dict[str, Any]:
    profile = _exam_profiles(_build_records(5))[0]
    candidate = create_candidate(
        candidate_id=candidate_id,
        candidate_type="disease_exam_profile",
        proposed_effect=profile,
        evidence={
            "source_receipt_hash": _RECEIPT_HASH,
            "partition": "build",
            "support_case_count": profile["support_case_count"],
        },
    )
    store.mkdir(parents=True, exist_ok=True)
    write_candidate(store / ("%s.json" % candidate_id), candidate)
    return candidate


def _report_for_candidate(candidate: Dict[str, Any], **kwargs: Any) -> ProfileControlReport:
    return _exam_report(candidate_hash=candidate["candidate_hash"], **kwargs)


def test_failed_report_cannot_be_approved(tmp_path: Path) -> None:
    store = tmp_path / "candidates"
    candidate = _write_profile_candidate(store, "profile_a")
    # A precision-collapsing held-out set produces passed=False.
    failing = build_exam_profile_control_report(
        profiles=_exam_profiles(_build_records(5)),
        held_out_records=_held_out_records(5, exams=["体格检查"]),
        candidate_hash=candidate["candidate_hash"],
        source_receipt_hash=_RECEIPT_HASH,
        held_out_partition_hash=_HELD_OUT_HASH,
        exam_catalog_order=_CATALOG_ORDER,
    )
    assert failing.passed is False
    control_store = tmp_path / "controls"
    control_store.mkdir()
    write_control_report(control_store / "profile_a.json", failing)

    with pytest.raises(ValueError, match="did not pass"):
        approve_candidate(
            candidate_path=store / "profile_a.json",
            decision_path=tmp_path / "decision.json",
            reviewer="人工复核者",
            rationale="尝试批准失败报告",
            control_store=control_store,
            control_report_ref="profile_a.json",
        )
    assert not (tmp_path / "decision.json").exists()


def test_profile_approval_requires_a_control_report(tmp_path: Path) -> None:
    store = tmp_path / "candidates"
    _write_profile_candidate(store, "profile_b")
    with pytest.raises(ValueError, match="control report"):
        approve_candidate(
            candidate_path=store / "profile_b.json",
            decision_path=tmp_path / "decision_b.json",
            reviewer="人工复核者",
            rationale="缺少 control report",
        )


@pytest.mark.parametrize("bad_ref", ["../escape.json", "/abs/report.json"])
def test_control_report_ref_must_be_relative(tmp_path: Path, bad_ref: str) -> None:
    store = tmp_path / "candidates"
    candidate = _write_profile_candidate(store, "profile_c")
    control_store = tmp_path / "controls"
    control_store.mkdir()
    write_control_report(
        control_store / "profile_c.json", _report_for_candidate(candidate)
    )
    with pytest.raises(ValueError, match="control_report_ref"):
        approve_candidate(
            candidate_path=store / "profile_c.json",
            decision_path=tmp_path / "decision_c.json",
            reviewer="人工复核者",
            rationale="路径逃逸",
            control_store=control_store,
            control_report_ref=bad_ref,
        )


def _approved_profile(tmp_path: Path, candidate_id: str = "profile_d") -> Dict[str, Any]:
    store = tmp_path / "candidates"
    candidate = _write_profile_candidate(store, candidate_id)
    control_store = tmp_path / "controls"
    control_store.mkdir(exist_ok=True)
    write_control_report(
        control_store / ("%s.json" % candidate_id), _report_for_candidate(candidate)
    )
    decision = approve_candidate(
        candidate_path=store / ("%s.json" % candidate_id),
        decision_path=tmp_path / ("decision_%s.json" % candidate_id),
        reviewer="人工复核者",
        rationale="held-out controls 通过",
        control_store=control_store,
        control_report_ref="%s.json" % candidate_id,
    )
    return {
        "candidate": candidate,
        "decision": decision,
        "store": store,
        "control_store": control_store,
        "decision_path": tmp_path / ("decision_%s.json" % candidate_id),
    }


def test_passing_report_enables_approval_with_control_evidence(tmp_path: Path) -> None:
    bundle = _approved_profile(tmp_path)
    decision = bundle["decision"]
    assert decision["decision"] == "approved"
    assert decision["control_report_ref"] == "profile_d.json"
    assert decision["control_report_hash"].startswith("sha256:")
    # Decision hash must still recompute.
    body = {key: value for key, value in decision.items() if key != "decision_hash"}
    assert content_hash(body) == decision["decision_hash"]


def test_registry_snapshot_rejects_tampered_control_report(tmp_path: Path) -> None:
    bundle = _approved_profile(tmp_path)
    report_path = bundle["control_store"] / "profile_d.json"
    tampered = read_json(report_path)
    tampered["exam_macro_precision_at_12"] = 0.1
    report_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        build_registry_snapshot(
            decision_paths=[bundle["decision_path"]],
            candidate_store=bundle["store"],
            output_path=tmp_path / "registry.json",
            control_store=bundle["control_store"],
        )


def test_registry_snapshot_accepts_intact_chain(tmp_path: Path) -> None:
    bundle = _approved_profile(tmp_path)
    registry = build_registry_snapshot(
        decision_paths=[bundle["decision_path"]],
        candidate_store=bundle["store"],
        output_path=tmp_path / "registry.json",
        control_store=bundle["control_store"],
    )
    assert len(registry["assets"]) == 1
    asset = registry["assets"][0]
    assert asset["candidate_type"] == "disease_exam_profile"
    assert asset["content"]["diagnosis_name"] == _DIAGNOSIS
    assert registry["registry_hash"]


def test_registry_snapshot_requires_control_store_for_profiles(tmp_path: Path) -> None:
    bundle = _approved_profile(tmp_path)
    with pytest.raises(ValueError, match="control_store"):
        build_registry_snapshot(
            decision_paths=[bundle["decision_path"]],
            candidate_store=bundle["store"],
            output_path=tmp_path / "registry2.json",
        )


def test_release_pack_records_control_hashes_and_keeps_pointer(tmp_path: Path) -> None:
    bundle = _approved_profile(tmp_path)
    registry = build_registry_snapshot(
        decision_paths=[bundle["decision_path"]],
        candidate_store=bundle["store"],
        output_path=tmp_path / "registry.json",
        control_store=bundle["control_store"],
    )
    pointer = tmp_path / "current.json"
    pointer.write_text('{"release_dir":"releases/old"}', encoding="utf-8")
    pointer_before = pointer.read_bytes()

    manifest = build_candidate_pack(
        release_dir=tmp_path / "release_new",
        code_commit="deadbeef",
        prompt_pack={},
        policy_pack={},
        registry=registry,
        knowledge_hashes={},
        catalog_hashes={},
        control_report_hashes={
            bundle["decision"]["candidate_id"]: bundle["decision"]["control_report_hash"]
        },
    )
    assert manifest["control_report_hashes"] == {
        "profile_d": bundle["decision"]["control_report_hash"]
    }
    assert "gate_report_hash" not in manifest
    # Building a candidate release must never touch the production pointer.
    assert pointer.read_bytes() == pointer_before


def test_release_pack_refuses_to_overwrite_existing_release(tmp_path: Path) -> None:
    args: Dict[str, Any] = {
        "code_commit": "deadbeef",
        "prompt_pack": {},
        "policy_pack": {},
        "registry": {"schema_version": "verified-registry/v1", "assets": []},
        "knowledge_hashes": {},
        "catalog_hashes": {},
        "control_report_hashes": {},
    }
    build_candidate_pack(release_dir=tmp_path / "rel", **args)
    with pytest.raises(FileExistsError):
        build_candidate_pack(release_dir=tmp_path / "rel", **args)


def test_candidate_assets_are_invisible_to_runtime_reader(tmp_path: Path) -> None:
    """A candidate on disk must never be loadable as a verified runtime asset."""
    from agent.memory import VerifiedOnlyMemory

    store = tmp_path / "candidates"
    _write_profile_candidate(store, "profile_e")
    # The runtime reader only accepts a frozen registry, never a candidate file.
    with pytest.raises(Exception):
        VerifiedOnlyMemory(store / "profile_e.json")


def test_controls_script_cannot_self_approve() -> None:
    """The controls script may only emit reports; approval stays human (G6)."""
    import scripts.knowledge.run_profile_controls as runner

    source = pathlib.Path(runner.__file__).read_text(encoding="utf-8")
    # No promotion/approval/pointer API may be reachable from the controls script.
    for forbidden in (
        "approve_candidate",
        "write_promotion_record",
        "switch_release_pointer",
        "build_registry_snapshot",
    ):
        assert forbidden not in source, forbidden

    # And the emitted report carries no reviewer/decision fields at all.
    report = _exam_report()
    stored = report.to_dict()
    assert set(stored) == {
        "schema_version",
        "candidate_type",
        "candidate_hash",
        "source_receipt_hash",
        "held_out_partition_hash",
        "exam_macro_recall_at_12",
        "exam_macro_precision_at_12",
        "baseline_exam_macro_recall_at_12",
        "baseline_exam_macro_precision_at_12",
        "treatment_goal_macro_recall",
        "baseline_treatment_goal_macro_recall",
        "contraindication_false_positive_count",
        "held_out_case_count",
        "evaluated_diagnosis_count",
        "leakage_count",
        "passed",
        "report_hash",
    }
    for forbidden_field in ("reviewer", "rationale", "decision", "approved_by", "approved_at"):
        assert forbidden_field not in stored
