"""T09: reflections come only from a curated offline source and retrieve by trigger/stage.

The reflection path is the narrowest knowledge channel in the system: a short
note, keyed by closed trigger codes and a stage. It must never be synthesized
from harvest data, must fail closed on any path trickery, and must stay
invisible to the runtime until it has passed its own held-out controls and a
human decision.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

import pytest

from agent.memory import VerifiedOnlyMemory
from offline.artifacts import canonical_json, content_hash, read_json
from offline.candidates import create_candidate, write_candidate
from offline.promotion import approve_candidate, build_registry_snapshot
from offline.reflection_sources import (
    CURATED_SOURCE_REF,
    REFLECTION_MIN_SUPPORT_COUNT,
    REFLECTION_RULE_SCHEMA,
    REFLECTION_SOURCE_RECEIPT_SCHEMA,
    ReflectionControlReport,
    ReflectionSourcePathError,
    aggregate_reflection_rules,
    build_reflection_control_report,
    build_reflection_source_receipt,
    load_reflection_sources,
    partition_for_source_id,
    resolve_curated_source_path,
    validate_reflection_control_report,
    write_reflection_control_report,
)
from offline.release import build_candidate_pack

_NOTE = "免疫抑制伴疱疹样皮损时，优先闭合病毒病原和继发细菌感染风险。"


def _source_row(
    source_id: str,
    *,
    note: str = _NOTE,
    trigger_codes: List[str] | None = None,
    stages: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "reflection-source/v1",
        "source_id": source_id,
        "trigger_codes": trigger_codes or ["immunosuppressed_infection", "vesicular_rash"],
        "stages": stages or ["diagnosis", "examination", "treatment"],
        "note": note,
        "positive_controls": [
            {
                "control_id": "pos_1",
                "fact_codes": ["immunosuppressed_infection", "vesicular_rash"],
            },
            {"control_id": "pos_2", "fact_codes": ["immunosuppressed_infection", "fever"]},
        ],
        "near_neighbor_controls": [
            {"control_id": "neg_1", "fact_codes": ["noninfectious_eczema"]},
            {
                "control_id": "neg_2",
                "fact_codes": ["isolated_vesicle_without_systemic_risk"],
            },
        ],
        "provenance": {
            "source_type": "curated_offline",
            "source_ref": "docs/论文精读/Reflexion论文精读.md",
            "reviewer": "人工复核者",
        },
    }


def _write_source(root: Path, rows: List[Dict[str, Any]]) -> Path:
    path = root / "data" / "knowledge_sources" / "reflection_sources.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


# --- Step 2: the only allowed data source, with hard path safety ---------------


def test_curated_source_ref_is_fixed() -> None:
    assert CURATED_SOURCE_REF == PurePosixPath("data/knowledge_sources/reflection_sources.jsonl")


@pytest.mark.parametrize(
    "alias",
    [
        "data\\knowledge_sources\\reflection_sources.jsonl",
        "./data/knowledge_sources/reflection_sources.jsonl",
        "data/../data/knowledge_sources/reflection_sources.jsonl",
        "/data/knowledge_sources/reflection_sources.jsonl",
        "other/reflection_sources.jsonl",
    ],
)
def test_alias_source_refs_are_rejected(tmp_path: Path, alias: str) -> None:
    _write_source(tmp_path, [_source_row("reflection_a_001")])
    with pytest.raises(ReflectionSourcePathError):
        resolve_curated_source_path(project_root=tmp_path, source_ref=alias)


def test_symlinked_source_file_outside_project_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    real = outside / "evil.jsonl"
    real.write_text("{}\n", encoding="utf-8")
    target = project / "data" / "knowledge_sources"
    target.mkdir(parents=True)
    link = target / "reflection_sources.jsonl"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    with pytest.raises(ReflectionSourcePathError):
        resolve_curated_source_path(project_root=project)


def test_junctioned_parent_directory_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "reflection_sources.jsonl").write_text("{}\n", encoding="utf-8")
    (project / "data").mkdir(parents=True)
    junction = project / "data" / "knowledge_sources"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
    )
    if completed.returncode != 0 or not junction.exists():
        pytest.skip("junction creation not permitted in this environment")
    with pytest.raises(ReflectionSourcePathError):
        resolve_curated_source_path(project_root=project)


def test_missing_source_returns_no_curated_source(tmp_path: Path) -> None:
    loaded = load_reflection_sources(project_root=tmp_path)
    assert loaded.status == "no_curated_source"
    assert loaded.records == ()
    assert loaded.raw_count == 0
    assert loaded.rejected_count == 0


# --- Step 3: recomputable receipt ---------------------------------------------


def test_receipt_reconciles_and_is_recomputable(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        [_source_row("reflection_a_%03d" % index) for index in range(1, 6)],
    )
    loaded = load_reflection_sources(project_root=tmp_path)
    assert loaded.status == "ready"
    receipt = build_reflection_source_receipt(loaded)
    assert receipt["schema_version"] == REFLECTION_SOURCE_RECEIPT_SCHEMA
    assert receipt["raw_count"] == receipt["unique_count"] + receipt["rejected_count"]
    assert receipt["unique_count"] == receipt["build_count"] + receipt["held_out_count"]
    body = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    assert "sha256:" + content_hash(body) == receipt["receipt_hash"]
    assert build_reflection_source_receipt(loaded) == receipt


def test_no_source_receipt_is_still_recomputable(tmp_path: Path) -> None:
    loaded = load_reflection_sources(project_root=tmp_path)
    receipt = build_reflection_source_receipt(loaded)
    assert receipt["status"] == "no_curated_source"
    assert receipt["source_file_hash"] is None
    assert receipt["raw_count"] == 0
    assert receipt["unique_count"] == 0


def test_duplicate_source_id_is_rejected_and_blocks_candidates(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        [_source_row("reflection_dup_001"), _source_row("reflection_dup_001")],
    )
    loaded = load_reflection_sources(project_root=tmp_path)
    assert loaded.rejected_count == 1
    rules = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)
    assert rules == []


def test_patient_id_in_source_id_is_rejected(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row("reflection_Patient_01061")])
    loaded = load_reflection_sources(project_root=tmp_path)
    assert loaded.records == ()
    assert loaded.rejected_count == 1


def test_partition_rule_is_stable(tmp_path: Path) -> None:
    for source_id in ("reflection_a_001", "reflection_a_002", "reflection_b_010"):
        assert partition_for_source_id(source_id) in {"build", "held_out"}
        assert partition_for_source_id(source_id) == partition_for_source_id(source_id)


# --- Step 4: candidate aggregation thresholds ---------------------------------


def _build_ids(count: int) -> List[str]:
    """Source ids that all land in the build partition."""
    found: List[str] = []
    index = 0
    while len(found) < count:
        index += 1
        source_id = "reflection_build_%04d" % index
        if partition_for_source_id(source_id) == "build":
            found.append(source_id)
    return found


def _held_out_ids(count: int) -> List[str]:
    found: List[str] = []
    index = 0
    while len(found) < count:
        index += 1
        source_id = "reflection_hold_%04d" % index
        if partition_for_source_id(source_id) == "held_out":
            found.append(source_id)
    return found


def test_no_source_produces_no_candidate(tmp_path: Path) -> None:
    loaded = load_reflection_sources(project_root=tmp_path)
    assert aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64) == []


def test_two_build_records_produce_no_candidate(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row(item) for item in _build_ids(2)])
    loaded = load_reflection_sources(project_root=tmp_path)
    assert aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64) == []


def test_three_build_records_produce_one_candidate(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row(item) for item in _build_ids(3)])
    loaded = load_reflection_sources(project_root=tmp_path)
    rules = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)
    assert len(rules) == 1
    rule = rules[0]
    assert rule["schema_version"] == REFLECTION_RULE_SCHEMA
    assert rule["support_count"] == REFLECTION_MIN_SUPPORT_COUNT
    assert rule["note"] == _NOTE
    assert set(rule) == {
        "schema_version",
        "trigger_codes",
        "stages",
        "note",
        "source_refs",
        "support_count",
        "source_receipt_hash",
    }


def test_different_notes_are_not_merged(tmp_path: Path) -> None:
    ids = _build_ids(6)
    rows = [_source_row(item) for item in ids[:3]]
    rows.extend(_source_row(item, note="另一条完全不同的复盘要点。") for item in ids[3:])
    _write_source(tmp_path, rows)
    loaded = load_reflection_sources(project_root=tmp_path)
    rules = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)
    assert len(rules) == 2
    assert len({rule["note"] for rule in rules}) == 2


def test_candidate_never_contains_patient_or_answer_text(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row(item) for item in _build_ids(3)])
    loaded = load_reflection_sources(project_root=tmp_path)
    rules = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)
    blob = canonical_json(rules)
    for marker in ("Patient_", "ground_truth", "reference", "expected"):
        assert marker not in blob


# --- Step 5: independent reflection controls ---------------------------------


def _ready_sources(tmp_path: Path) -> Any:
    rows = [_source_row(item) for item in _build_ids(3)]
    rows.extend(_source_row(item) for item in _held_out_ids(1))
    _write_source(tmp_path, rows)
    return load_reflection_sources(project_root=tmp_path)


def test_reflection_control_report_uses_its_own_metric_fields(tmp_path: Path) -> None:
    loaded = _ready_sources(tmp_path)
    rule = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)[0]
    report = build_reflection_control_report(
        rule=rule,
        loaded=loaded,
        candidate_hash="abc",
        source_receipt_hash="sha256:" + "e" * 64,
        held_out_partition_hash="sha256:" + "f" * 64,
    )
    assert isinstance(report, ReflectionControlReport)
    assert report.candidate_type == "reflection_rule"
    assert report.positive_pass_count >= 2
    assert report.near_neighbor_pass_count >= 2
    assert report.false_positive_count == 0
    assert report.leakage_count == 0
    assert report.passed is True
    # Exam/treatment metric fields must not leak into a reflection report.
    stored = report.to_dict()
    for forbidden in (
        "exam_macro_recall_at_12",
        "exam_macro_precision_at_12",
        "treatment_goal_macro_recall",
        "contraindication_false_positive_count",
    ):
        assert forbidden not in stored


def test_reflection_control_report_fails_without_held_out_support(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row(item) for item in _build_ids(3)])
    loaded = load_reflection_sources(project_root=tmp_path)
    rule = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)[0]
    report = build_reflection_control_report(
        rule=rule,
        loaded=loaded,
        candidate_hash="abc",
        source_receipt_hash="sha256:" + "e" * 64,
        held_out_partition_hash="sha256:" + "f" * 64,
    )
    assert report.passed is False


def test_reflection_report_is_tamper_evident(tmp_path: Path) -> None:
    loaded = _ready_sources(tmp_path)
    rule = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)[0]
    report = build_reflection_control_report(
        rule=rule,
        loaded=loaded,
        candidate_hash="abc",
        source_receipt_hash="sha256:" + "e" * 64,
        held_out_partition_hash="sha256:" + "f" * 64,
    )
    path = write_reflection_control_report(tmp_path / "r.json", report)
    stored = read_json(path)
    assert validate_reflection_control_report(
        stored,
        candidate_type="reflection_rule",
        candidate_hash="abc",
        source_receipt_hash="sha256:" + "e" * 64,
        held_out_partition_hash="sha256:" + "f" * 64,
    )
    tampered = dict(stored)
    tampered["false_positive_count"] = 99
    with pytest.raises(ValueError, match="report_hash"):
        validate_reflection_control_report(
            tampered,
            candidate_type="reflection_rule",
            candidate_hash="abc",
            source_receipt_hash="sha256:" + "e" * 64,
            held_out_partition_hash="sha256:" + "f" * 64,
        )


def test_failed_reflection_report_cannot_be_approved(tmp_path: Path) -> None:
    _write_source(tmp_path, [_source_row(item) for item in _build_ids(3)])
    loaded = load_reflection_sources(project_root=tmp_path)
    rule = aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)[0]
    candidate = create_candidate(
        candidate_id="reflection_x",
        candidate_type="reflection_rule",
        proposed_effect=rule,
        evidence={
            "source_receipt_hash": "sha256:" + "e" * 64,
            "partition": "build",
            "support_count": rule["support_count"],
        },
    )
    store = tmp_path / "candidates"
    store.mkdir()
    write_candidate(store / "reflection_x.json", candidate)
    failing = build_reflection_control_report(
        rule=rule,
        loaded=loaded,
        candidate_hash=candidate["candidate_hash"],
        source_receipt_hash="sha256:" + "e" * 64,
        held_out_partition_hash="sha256:" + "f" * 64,
    )
    assert failing.passed is False
    control_store = tmp_path / "controls"
    control_store.mkdir()
    write_reflection_control_report(control_store / "reflection_x.json", failing)
    with pytest.raises(ValueError, match="did not pass"):
        approve_candidate(
            candidate_path=store / "reflection_x.json",
            decision_path=tmp_path / "decision.json",
            reviewer="人工复核者",
            rationale="不应通过",
            control_store=control_store,
            control_report_ref="reflection_x.json",
        )
    assert not (tmp_path / "decision.json").exists()


# --- Step 8: full source-to-release round trip -------------------------------


def test_source_to_release_round_trip(tmp_path: Path) -> None:
    loaded = _ready_sources(tmp_path)
    receipt = build_reflection_source_receipt(loaded)
    receipt_hash = receipt["receipt_hash"]

    rules = aggregate_reflection_rules(loaded, source_receipt_hash=receipt_hash)
    assert len(rules) == 1
    rule = rules[0]

    candidate = create_candidate(
        candidate_id="reflection_rt",
        candidate_type="reflection_rule",
        proposed_effect=rule,
        evidence={
            "source_receipt_hash": receipt_hash,
            "partition": "build",
            "support_count": rule["support_count"],
        },
    )
    assert candidate["status"] == "candidate"
    store = tmp_path / "candidates"
    store.mkdir()
    write_candidate(store / "reflection_rt.json", candidate)

    report = build_reflection_control_report(
        rule=rule,
        loaded=loaded,
        candidate_hash=candidate["candidate_hash"],
        source_receipt_hash=receipt_hash,
        held_out_partition_hash=receipt["held_out_partition_hash"],
    )
    assert report.passed is True
    control_store = tmp_path / "controls"
    control_store.mkdir()
    write_reflection_control_report(control_store / "reflection_rt.json", report)

    decision = approve_candidate(
        candidate_path=store / "reflection_rt.json",
        decision_path=tmp_path / "decision_rt.json",
        reviewer="人工复核者",
        rationale="held-out controls 通过",
        control_store=control_store,
        control_report_ref="reflection_rt.json",
    )
    assert decision["control_report_hash"] == report.report_hash

    registry = build_registry_snapshot(
        decision_paths=[tmp_path / "decision_rt.json"],
        candidate_store=store,
        output_path=tmp_path / "registry.json",
        control_store=control_store,
    )
    assert registry["assets"][0]["candidate_type"] == "reflection_rule"

    manifest = build_candidate_pack(
        release_dir=tmp_path / "release",
        code_commit="deadbeef",
        prompt_pack={},
        policy_pack={},
        registry=registry,
        knowledge_hashes={},
        catalog_hashes={},
        control_report_hashes={"reflection_rt": report.report_hash},
    )
    assert manifest["control_report_hashes"]["reflection_rt"] == report.report_hash

    # Tampering at the control layer must break the chain.
    tampered = read_json(control_store / "reflection_rt.json")
    tampered["positive_pass_count"] = 99
    (control_store / "reflection_rt.json").write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        build_registry_snapshot(
            decision_paths=[tmp_path / "decision_rt.json"],
            candidate_store=store,
            output_path=tmp_path / "registry2.json",
            control_store=control_store,
        )


# --- Step 6/7: runtime retrieval by trigger and stage ------------------------


def _reflection_registry(tmp_path: Path, rule: Dict[str, Any]) -> Path:
    path = tmp_path / "verified_registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "verified-registry/v1",
                "assets": [
                    {
                        "candidate_id": "reflection_rt",
                        "candidate_type": "reflection_rule",
                        "content": rule,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _one_rule(tmp_path: Path) -> Dict[str, Any]:
    loaded = _ready_sources(tmp_path / "src")
    return aggregate_reflection_rules(loaded, source_receipt_hash="sha256:" + "e" * 64)[0]


def test_runtime_requires_both_trigger_and_stage(tmp_path: Path) -> None:
    rule = _one_rule(tmp_path)
    memory = VerifiedOnlyMemory(_reflection_registry(tmp_path, rule))
    assert memory.load_notes() == []
    assert memory.load_notes(trigger_codes={"vesicular_rash"}) == []
    assert memory.load_notes(stage="diagnosis") == []
    notes = memory.load_notes(trigger_codes={"vesicular_rash"}, stage="diagnosis")
    assert notes == [rule["note"]]


def test_runtime_stage_must_match(tmp_path: Path) -> None:
    rule = _one_rule(tmp_path)
    rule = dict(rule)
    rule["stages"] = ["treatment"]
    memory = VerifiedOnlyMemory(_reflection_registry(tmp_path, rule))
    assert memory.load_notes(trigger_codes={"vesicular_rash"}, stage="diagnosis") == []
    assert memory.load_notes(trigger_codes={"vesicular_rash"}, stage="treatment") == [
        rule["note"]
    ]


def test_runtime_trigger_must_intersect(tmp_path: Path) -> None:
    rule = _one_rule(tmp_path)
    memory = VerifiedOnlyMemory(_reflection_registry(tmp_path, rule))
    assert memory.load_notes(trigger_codes={"bleeding_tendency"}, stage="diagnosis") == []


def test_runtime_never_reads_candidates(tmp_path: Path) -> None:
    rule = _one_rule(tmp_path)
    candidate = create_candidate(
        candidate_id="reflection_c",
        candidate_type="reflection_rule",
        proposed_effect=rule,
        evidence={
            "source_receipt_hash": "sha256:" + "e" * 64,
            "partition": "build",
            "support_count": rule["support_count"],
        },
    )
    store = tmp_path / "candidates"
    store.mkdir()
    write_candidate(store / "reflection_c.json", candidate)
    with pytest.raises(ValueError):
        VerifiedOnlyMemory(store / "reflection_c.json")


def test_runtime_injects_a_stage_note_only_once(tmp_path: Path) -> None:
    rule = _one_rule(tmp_path)
    memory = VerifiedOnlyMemory(_reflection_registry(tmp_path, rule))
    notes = memory.load_notes(
        trigger_codes={"vesicular_rash", "immunosuppressed_infection"},
        stage="diagnosis",
    )
    assert notes == [rule["note"]]
    assert len(notes) == len(set(notes))


def test_symlink_guard_fires_without_needing_symlink_privileges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the symlink branch even where symlink creation is not permitted.

    The junction test proves the reparse-point branch on Windows; this forces the
    is_symlink() branch directly so the guard is never left unverified.
    """
    _write_source(tmp_path, [_source_row("reflection_a_001")])
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(self: Path) -> bool:
        if self.name == "knowledge_sources":
            return True
        return real_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(ReflectionSourcePathError, match="reparse point"):
        resolve_curated_source_path(project_root=tmp_path)
