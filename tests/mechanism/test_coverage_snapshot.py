from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pytest

from offline.artifacts import canonical_json, content_hash, file_hash, read_json
from offline.coverage_pollution import validate_pollution_receipt
from offline.coverage_snapshot import CoverageInputs, build_coverage_snapshot
from scripts.coverage.build_coverage_snapshot import write_coverage_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_RELEASE = PROJECT_ROOT / "releases" / "release_C_case_memory_20260719_v4"
REAL_OFFLINE_ROOTS = (
    PROJECT_ROOT / "docs" / "离线测试题目",
    PROJECT_ROOT.parent / "docs" / "离线测试题目",
)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _attach_manifest_hash(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "manifest_hash"}
    return {**body, "manifest_hash": content_hash(body)}


def _refresh_trust_manifest(inputs: CoverageInputs) -> None:
    manifest = read_json(inputs.trust_manifest)
    for run in manifest["historical_runs"]:
        run_dir = (
            inputs.train_outputs if run["mode"] == "train" else inputs.test_outputs
        ) / run["run_id"]
        for name in tuple(run["artifact_hashes"]):
            path = run_dir / name
            if path.is_file():
                run["artifact_hashes"][name] = file_hash(path)
    _write_json(inputs.trust_manifest, _attach_manifest_hash(manifest))


def _batch_result(report: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    return {
        "event_type": "BATCH_EVALUATION_RESULT",
        "patient_id": "SYSTEM",
        "status": "success",
        "payload": {
            "output_log_path": "outputs/test/%s" % run_id,
            "report": report,
            "patient_id": "SYSTEM",
        },
    }


def _make_fixture_tree(root: Path) -> CoverageInputs:
    train_outputs = root / "outputs" / "train"
    test_outputs = root / "outputs" / "test"

    anchored_run = "test_anchored"
    anchored_dir = test_outputs / anchored_run
    anchored_report = {
        "counts": {"final_results": 2, "evaluated_patients": 1},
        "treatment_details": [{"patient_id": "Patient_00002", "overall_score": 0.5}],
    }
    anchored_events = [
        {"event_type": "CASE_START", "patient_id": "Patient_00002", "status": "success"},
        {"event_type": "CASE_END", "patient_id": "Patient_00002", "status": "success", "payload": {"finished": True}},
        {"event_type": "CASE_START", "patient_id": "Patient_00007", "status": "success"},
        {"event_type": "AGENT_ERROR", "patient_id": "Patient_00007", "status": "error"},
        _batch_result(anchored_report, anchored_run),
    ]
    anchored_finals = [
        {"patient_id": "Patient_00002", "finished": True, "diagnosis": ["d"]},
        {"patient_id": "Patient_00007", "finished": False, "diagnosis": []},
    ]
    _write_jsonl(anchored_dir / "events.jsonl", anchored_events)
    _write_jsonl(anchored_dir / "final_results.jsonl", anchored_finals)
    _write_json(anchored_dir / "final_results_eval_report.json", anchored_report)

    unanchored_run = "test_unanchored"
    unanchored_dir = test_outputs / unanchored_run
    unanchored_report = {
        "counts": {"final_results": 1, "evaluated_patients": 1},
        "treatment_details": [{"patient_id": "Patient_00003", "overall_score": 0.6}],
    }
    _write_jsonl(
        unanchored_dir / "events.jsonl",
        [
            {"event_type": "CASE_START", "patient_id": "Patient_00003", "status": "success"},
            _batch_result(unanchored_report, unanchored_run),
        ],
    )
    _write_jsonl(
        unanchored_dir / "final_results.jsonl",
        [{"patient_id": "Patient_00003", "finished": True, "diagnosis": ["d"]}],
    )
    _write_json(unanchored_dir / "final_results_eval_report.json", unanchored_report)

    _write_jsonl(
        test_outputs / "test_final_only" / "events.jsonl",
        [{"event_type": "CASE_START", "patient_id": "Patient_00004", "status": "success"}],
    )
    _write_jsonl(
        test_outputs / "test_final_only" / "final_results.jsonl",
        [{"patient_id": "Patient_00004", "finished": True, "diagnosis": ["d"]}],
    )
    _write_jsonl(
        test_outputs / "test_attempt_only" / "events.jsonl",
        [{"event_type": "AGENT_ERROR", "patient_id": "Patient_00005", "status": "error"}],
    )

    train_run = train_outputs / "train_case_memory"
    train_evaluation = {
        "timestamp": "2026-07-19T00:00:00+08:00",
        "patient_id": "Patient_00001",
        "report": {"status": "evaluated", "patientId": "Patient_00001"},
    }
    train_final = {"patient_id": "Patient_00001", "finished": True, "diagnosis": ["d"]}
    _write_jsonl(train_run / "evaluation_results.jsonl", [train_evaluation])
    _write_jsonl(
        train_run / "events.jsonl",
        [
            {
                "event_type": "EVALUATION_REQUEST",
                "patient_id": "Patient_00001",
                "status": "success",
                "payload": {
                    "patient_id": "Patient_00001",
                    "final_result": train_final,
                },
            },
            {
                "event_type": "EVALUATION_RESULT",
                "patient_id": "Patient_00001",
                "status": "success",
                "payload": {
                    "patient_id": "Patient_00001",
                    "report": train_evaluation["report"],
                },
            },
            {
                "event_type": "CASE_END",
                "patient_id": "Patient_00001",
                "status": "success",
                "payload": {
                    "patient_id": "Patient_00001",
                    "finished": True,
                },
            },
        ],
    )
    _write_jsonl(train_run / "final_results.jsonl", [train_final])

    trust_manifest = {
        "schema_version": "architecture-baseline/v1",
        "historical_runs": [
            {
                "run_id": anchored_run,
                "mode": "test",
                "has_evaluation": False,
                "artifact_hashes": {
                    "events.jsonl": file_hash(anchored_dir / "events.jsonl"),
                    "final_results.jsonl": file_hash(anchored_dir / "final_results.jsonl"),
                },
            },
            {
                "run_id": "train_case_memory",
                "mode": "train",
                "has_evaluation": True,
                "artifact_hashes": {
                    name: file_hash(train_run / name)
                    for name in ("evaluation_results.jsonl", "events.jsonl", "final_results.jsonl")
                },
            },
        ],
    }
    trust_manifest_path = _write_json(
        root / "docs" / "manifest.json", _attach_manifest_hash(trust_manifest)
    )

    registry = {
        "schema_version": "verified-registry/v1",
        "assets": [
            {
                "candidate_id": "case-memory-Patient_00001",
                "candidate_type": "case_memory",
                "content": {"patient_id": "Patient_00001"},
            }
        ],
    }
    registry["registry_hash"] = content_hash(registry)
    registry_path = _write_json(root / "releases" / "explicit_v4" / "verified_registry.json", registry)
    release_manifest = {
        "schema_version": "clinical-runtime/v1",
        "registry_hash": file_hash(registry_path),
    }
    release_manifest_path = _write_json(
        root / "releases" / "explicit_v4" / "release_manifest.json", release_manifest
    )

    project_offline = root / "docs" / "offline_questions"
    parent_offline = root.parent / (root.name + "_parent_docs") / "offline_questions"
    project_offline.mkdir(parents=True)
    parent_offline.mkdir(parents=True, exist_ok=True)
    (project_offline / "project.md").write_text(
        "Patient_00001 Patient_00006", encoding="utf-8"
    )
    (parent_offline / "parent.md").write_text(
        "Patient_00003 Patient_00006", encoding="utf-8"
    )

    evidence_path = test_outputs / "test_attempt_only" / "events.jsonl"
    evidence_excerpt = {
        "line": 1,
        "patient_id": "Patient_00005",
        "event_type": "AGENT_ERROR",
        "status": "error",
    }
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00005",
        "run_id": "test_attempt_only",
        "evidence_path": _relative(evidence_path, root),
        "evidence_file_sha256": file_hash(evidence_path),
        "evidence_excerpt": evidence_excerpt,
        "evidence_excerpt_hash": content_hash(evidence_excerpt),
        "pollution_kind": "cross_case_patch",
        "reviewer": "github:24228",
    }
    receipt_path = _write_json(root / "receipts" / "pollution.json", receipt)

    return CoverageInputs(
        project_root=root,
        train_outputs=train_outputs,
        test_outputs=test_outputs,
        trust_manifest=trust_manifest_path,
        registry_path=registry_path,
        release_manifest_path=release_manifest_path,
        offline_question_roots=(project_offline, parent_offline),
        pollution_receipts=(receipt_path,),
    )


def test_pollution_receipt_validates_exact_bytes_excerpt_and_review_fields(tmp_path: Path) -> None:
    evidence = _write_jsonl(
        tmp_path / "outputs" / "test" / "test_run" / "events.jsonl",
        [{"patient_id": "Patient_00010", "event_type": "AGENT_ERROR", "status": "error"}],
    )
    excerpt = {
        "line": 1,
        "patient_id": "Patient_00010",
        "event_type": "AGENT_ERROR",
        "status": "error",
    }
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00010",
        "run_id": "test_run",
        "evidence_path": _relative(evidence, tmp_path),
        "evidence_file_sha256": file_hash(evidence),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "cross_case_patch",
        "reviewer": "github:24228",
    }

    assert validate_pollution_receipt(receipt, project_root=tmp_path) == receipt

    forged_excerpt = {**excerpt, "status": "success"}
    with pytest.raises(ValueError, match="evidence excerpt does not match evidence line"):
        validate_pollution_receipt(
            {
                **receipt,
                "evidence_excerpt": forged_excerpt,
                "evidence_excerpt_hash": content_hash(forged_excerpt),
            },
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="evidence excerpt line"):
        validate_pollution_receipt(
            {
                **receipt,
                "evidence_excerpt": {**excerpt, "line": 2},
                "evidence_excerpt_hash": content_hash({**excerpt, "line": 2}),
            },
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="evidence run_id mismatch"):
        validate_pollution_receipt(
            {**receipt, "run_id": "other_run"},
            project_root=tmp_path,
        )

    json_evidence = _write_json(
        tmp_path / "outputs" / "test" / "json_run" / "evidence.json",
        {"patient_id": "Patient_00010", "status": "error"},
    )
    json_excerpt = {"patient_id": "Patient_00010", "status": "error", "line": 1}
    json_receipt = {
        **receipt,
        "run_id": "json_run",
        "evidence_path": _relative(json_evidence, tmp_path),
        "evidence_file_sha256": file_hash(json_evidence),
        "evidence_excerpt": json_excerpt,
        "evidence_excerpt_hash": content_hash(json_excerpt),
    }
    assert validate_pollution_receipt(json_receipt, project_root=tmp_path) == json_receipt

    for field, value, message in (
        ("evidence_file_sha256", "a" * 64, "evidence file hash mismatch"),
        ("evidence_excerpt_hash", "b" * 64, "evidence excerpt hash mismatch"),
        ("reviewer", "", "reviewer"),
        ("pollution_kind", "keyword_match", "pollution kind"),
    ):
        invalid = {**receipt, field: value}
        with pytest.raises(ValueError, match=message):
            validate_pollution_receipt(invalid, project_root=tmp_path)


def test_pollution_receipt_hashes_and_parses_one_read_bytes_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _write_jsonl(
        tmp_path / "outputs" / "test" / "test_run" / "events.jsonl",
        [{"patient_id": "Patient_00010", "event_type": "AGENT_ERROR", "status": "error"}],
    )
    excerpt = {
        "line": 1,
        "patient_id": "Patient_00010",
        "event_type": "AGENT_ERROR",
        "status": "error",
    }
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00010",
        "run_id": "test_run",
        "evidence_path": _relative(evidence, tmp_path),
        "evidence_file_sha256": file_hash(evidence),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "cross_case_patch",
        "reviewer": "github:24228",
    }
    evidence_path = evidence.resolve()
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    evidence_reads = 0

    def tracked_read_bytes(path: Path) -> bytes:
        nonlocal evidence_reads
        if Path(path).resolve() == evidence_path:
            evidence_reads += 1
            if evidence_reads > 1:
                raise AssertionError("pollution evidence bytes read more than once")
        return original_read_bytes(path)

    def reject_evidence_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if Path(path).resolve() == evidence_path:
            raise AssertionError("pollution evidence must be parsed from the hashed bytes")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(Path, "read_text", reject_evidence_read_text)

    assert validate_pollution_receipt(receipt, project_root=tmp_path) == receipt
    assert evidence_reads == 1


def test_coverage_and_pollution_sources_hash_and_parse_one_bytes_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_fixture_tree(tmp_path / "project")
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    source_reads: dict[Path, int] = {}

    def tracked_read_bytes(path: Path) -> bytes:
        resolved = Path(path).resolve()
        if resolved.is_relative_to(tmp_path.resolve()):
            source_reads[resolved] = source_reads.get(resolved, 0) + 1
        return original_read_bytes(path)

    def reject_source_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if Path(path).resolve().is_relative_to(tmp_path.resolve()):
            raise AssertionError("coverage sources must be parsed from their hashed bytes")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(Path, "read_text", reject_source_read_text)

    snapshot = build_coverage_snapshot(inputs)
    expected_sources = {
        (inputs.project_root / source["path"]).resolve() for source in snapshot["sources"]
    }

    assert set(source_reads) == expected_sources
    assert all(source_reads[path] == 1 for path in expected_sources)


def test_pollution_receipt_layers_excerpt_metadata_from_source_row_content(tmp_path: Path) -> None:
    source_row = {
        "patient_id": "Patient_00010",
        "line": "source-row-line",
        "status": "error",
    }
    evidence = _write_jsonl(
        tmp_path / "outputs" / "test" / "test_run" / "events.jsonl",
        [source_row],
    )
    excerpt = {"line": 1, "row": source_row}
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00010",
        "run_id": "test_run",
        "evidence_path": _relative(evidence, tmp_path),
        "evidence_file_sha256": file_hash(evidence),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "unsupported_clinical_fact",
        "reviewer": "github:24228",
    }

    assert validate_pollution_receipt(receipt, project_root=tmp_path) == receipt
    assert receipt["evidence_excerpt"]["row"]["line"] == "source-row-line"


def test_pollution_receipt_rejects_unknown_fields_absolute_paths_and_parent_escape(
    tmp_path: Path,
) -> None:
    evidence = _write_jsonl(
        tmp_path / "outputs" / "test" / "test_run" / "events.jsonl",
        [{"patient_id": "Patient_00010"}],
    )
    excerpt = {"line": 1, "patient_id": "Patient_00010"}
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00010",
        "run_id": "test_run",
        "evidence_path": _relative(evidence, tmp_path),
        "evidence_file_sha256": file_hash(evidence),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "unsupported_clinical_fact",
        "reviewer": "github:24228",
    }

    with pytest.raises(ValueError, match="unknown fields"):
        validate_pollution_receipt({**receipt, "automatic_keyword": "bad"}, project_root=tmp_path)
    with pytest.raises(ValueError, match="project-relative"):
        validate_pollution_receipt({**receipt, "evidence_path": str(evidence.resolve())}, project_root=tmp_path)
    with pytest.raises(ValueError, match="project-relative"):
        validate_pollution_receipt({**receipt, "evidence_path": "../events.jsonl"}, project_root=tmp_path)
    with pytest.raises(ValueError, match="project-relative"):
        validate_pollution_receipt(
            {**receipt, "evidence_path": "C:/outside/events.jsonl"},
            project_root=tmp_path,
        )


def test_pollution_receipt_rejects_symlink_escape(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / (tmp_path.name + "-outside")
    outside = _write_jsonl(outside_dir / "events.jsonl", [{"patient_id": "Patient_00010"}])
    link = tmp_path / "outputs" / "test" / "test_run"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        if os.name != "nt":
            pytest.skip("symlinks unavailable")
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            pytest.skip("directory junctions unavailable")
    excerpt = {"line": 1, "patient_id": "Patient_00010"}
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_00010",
        "run_id": "test_run",
        "evidence_path": "outputs/test/test_run/events.jsonl",
        "evidence_file_sha256": file_hash(outside),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "unsupported_clinical_fact",
        "reviewer": "github:24228",
    }

    with pytest.raises(ValueError, match="project-relative"):
        validate_pollution_receipt(receipt, project_root=tmp_path)


def test_classification_priority_flags_and_patient_exact_batch_binding(tmp_path: Path) -> None:
    snapshot = build_coverage_snapshot(_make_fixture_tree(tmp_path))
    patients = {item["patient_id"]: item for item in snapshot["patients"]}
    classes = {patient_id: item["primary_class"] for patient_id, item in patients.items()}

    assert classes == {
        "Patient_00001": "case-memory-covered",
        "Patient_00002": "manifest-anchored-batch-evaluated-provenance-only",
        "Patient_00003": "unanchored-evaluated",
        "Patient_00004": "final-only",
        "Patient_00005": "attempt-only",
        "Patient_00006": "offline-test-only",
        "Patient_00007": "attempt-only",
    }
    assert patients["Patient_00001"]["flags"]["offline_test_covered"] is True
    assert patients["Patient_00005"]["flags"]["polluted"] is True
    assert patients["Patient_00007"]["flags"]["batch_evaluated"] is False
    assert snapshot["counts"]["primary"] == {
        "case-memory-covered": 1,
        "manifest-anchored-batch-evaluated-provenance-only": 1,
        "unanchored-evaluated": 1,
        "final-only": 1,
        "attempt-only": 2,
        "offline-test-only": 1,
    }


def test_pollution_receipt_cannot_create_patient_without_any_coverage_occurrence(
    tmp_path: Path,
) -> None:
    inputs = _make_fixture_tree(tmp_path)
    evidence_row = {
        "event_type": "AGENT_ERROR",
        "status": "error",
        "payload": {"patient_id": "Patient_99999"},
    }
    evidence = _write_jsonl(
        inputs.test_outputs / "receipt_only" / "events.jsonl",
        [evidence_row],
    )
    excerpt = {"line": 1, "row": evidence_row}
    receipt = {
        "schema_version": "coverage-pollution-receipt/v1",
        "patient_id": "Patient_99999",
        "run_id": "receipt_only",
        "evidence_path": _relative(evidence, tmp_path),
        "evidence_file_sha256": file_hash(evidence),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_hash": content_hash(excerpt),
        "pollution_kind": "cross_case_patch",
        "reviewer": "github:24228",
    }
    receipt_path = _write_json(tmp_path / "receipts" / "receipt-only.json", receipt)
    shutil.rmtree(inputs.test_outputs / "receipt_only")
    _write_jsonl(evidence, [receipt["evidence_excerpt"]["row"]])
    inputs = CoverageInputs(
        project_root=inputs.project_root,
        train_outputs=inputs.train_outputs,
        test_outputs=inputs.test_outputs,
        trust_manifest=inputs.trust_manifest,
        registry_path=inputs.registry_path,
        release_manifest_path=inputs.release_manifest_path,
        offline_question_roots=inputs.offline_question_roots,
        pollution_receipts=inputs.pollution_receipts + (receipt_path,),
    )

    with pytest.raises(ValueError, match="pollution receipt patient has no coverage occurrence"):
        build_coverage_snapshot(inputs)


def test_jsonl_refs_have_original_file_hash_one_based_line_and_canonical_row_hash(
    tmp_path: Path,
) -> None:
    inputs = _make_fixture_tree(tmp_path)
    snapshot = build_coverage_snapshot(inputs)
    patient = next(item for item in snapshot["patients"] if item["patient_id"] == "Patient_00002")
    ref = next(item for item in patient["refs"] if item["kind"] == "test-final")
    path = tmp_path / ref["path"]
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[ref["line"] - 1])

    assert ref["file_sha256"] == file_hash(path)
    assert ref["line"] == 1
    assert ref["row_hash"] == content_hash(row)
    assert ref["path"] == "outputs/test/test_anchored/final_results.jsonl"
    assert "\\" not in ref["path"]


def test_anchored_run_fails_closed_when_manifest_hash_does_not_match(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path)
    events = inputs.test_outputs / "test_anchored" / "events.jsonl"
    events.write_text(events.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="anchored artifact hash mismatch"):
        build_coverage_snapshot(inputs)


def test_trust_manifest_requires_schema_and_valid_manifest_hash(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path)
    manifest = read_json(inputs.trust_manifest)
    manifest["schema_version"] = "wrong/v1"
    _write_json(inputs.trust_manifest, _attach_manifest_hash(manifest))
    with pytest.raises(ValueError, match="trust manifest schema_version"):
        build_coverage_snapshot(inputs)

    inputs = _make_fixture_tree(tmp_path / "bad_hash")
    manifest = read_json(inputs.trust_manifest)
    manifest["manifest_hash"] = "0" * 64
    _write_json(inputs.trust_manifest, manifest)
    with pytest.raises(ValueError, match="trust manifest hash mismatch"):
        build_coverage_snapshot(inputs)


def test_missing_train_or_test_outputs_fail_closed(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path)
    shutil.rmtree(inputs.train_outputs)
    with pytest.raises(FileNotFoundError, match="train outputs missing"):
        build_coverage_snapshot(inputs)

    inputs = _make_fixture_tree(tmp_path / "missing_test")
    shutil.rmtree(inputs.test_outputs)
    with pytest.raises(FileNotFoundError, match="test outputs missing"):
        build_coverage_snapshot(inputs)


def test_train_evaluation_requires_success_and_every_layer_patient_binding(tmp_path: Path) -> None:
    for mutation, message in (
        ("status", "train evaluation status"),
        ("report_patient", "train evaluation patient binding"),
        ("event_payload", "train evaluation patient binding"),
        ("event_report", "train evaluation report mismatch"),
        ("request_patient", "train evaluation request binding"),
        ("case_end", "train case end binding"),
    ):
        inputs = _make_fixture_tree(tmp_path / mutation)
        run_dir = inputs.train_outputs / "train_case_memory"
        evaluation = json.loads((run_dir / "evaluation_results.jsonl").read_text(encoding="utf-8"))
        events = [json.loads(raw) for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        result_event = next(event for event in events if event["event_type"] == "EVALUATION_RESULT")
        if mutation == "status":
            evaluation["report"]["status"] = "failed"
        elif mutation == "report_patient":
            evaluation["report"]["patientId"] = "Patient_99999"
        elif mutation == "event_payload":
            result_event["payload"]["patient_id"] = "Patient_99999"
        elif mutation == "event_report":
            result_event["payload"]["report"] = {**result_event["payload"]["report"], "score": 0.1}
        elif mutation == "request_patient":
            request = next(event for event in events if event["event_type"] == "EVALUATION_REQUEST")
            request["payload"]["patient_id"] = "Patient_99999"
        else:
            case_end = next(event for event in events if event["event_type"] == "CASE_END")
            case_end["payload"]["finished"] = False
        _write_jsonl(run_dir / "evaluation_results.jsonl", [evaluation])
        _write_jsonl(run_dir / "events.jsonl", events)
        _refresh_trust_manifest(inputs)

        with pytest.raises(ValueError, match=message):
            build_coverage_snapshot(inputs)


def test_manifest_anchored_train_evaluation_is_anchored_evaluated_outside_case_memory(
    tmp_path: Path,
) -> None:
    inputs = _make_fixture_tree(tmp_path)
    registry = read_json(inputs.registry_path)
    registry["assets"] = []
    registry["registry_hash"] = content_hash(
        {key: value for key, value in registry.items() if key != "registry_hash"}
    )
    _write_json(inputs.registry_path, registry)
    release_manifest = read_json(inputs.release_manifest_path)
    release_manifest["registry_hash"] = file_hash(inputs.registry_path)
    _write_json(inputs.release_manifest_path, release_manifest)

    snapshot = build_coverage_snapshot(inputs)
    patient = next(item for item in snapshot["patients"] if item["patient_id"] == "Patient_00001")

    assert patient["primary_class"] == "manifest-anchored-batch-evaluated-provenance-only"
    assert patient["flags"]["batch_evaluated"] is True
    assert patient["flags"]["manifest_anchored"] is True


def test_train_evaluation_requires_unique_finished_final_and_anchored_hash(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path)
    run_dir = inputs.train_outputs / "train_case_memory"
    final = json.loads((run_dir / "final_results.jsonl").read_text(encoding="utf-8"))
    _write_jsonl(run_dir / "final_results.jsonl", [final, final])
    _refresh_trust_manifest(inputs)
    with pytest.raises(ValueError, match="unique finished final"):
        build_coverage_snapshot(inputs)

    inputs = _make_fixture_tree(tmp_path / "duplicate_evaluation")
    run_dir = inputs.train_outputs / "train_case_memory"
    evaluation = json.loads((run_dir / "evaluation_results.jsonl").read_text(encoding="utf-8"))
    _write_jsonl(run_dir / "evaluation_results.jsonl", [evaluation, evaluation])
    _refresh_trust_manifest(inputs)
    with pytest.raises(ValueError, match="unique evaluation row"):
        build_coverage_snapshot(inputs)

    inputs = _make_fixture_tree(tmp_path / "unanchored_evaluation")
    manifest = read_json(inputs.trust_manifest)
    train_run = next(run for run in manifest["historical_runs"] if run["mode"] == "train")
    train_run["artifact_hashes"].pop("evaluation_results.jsonl")
    _write_json(inputs.trust_manifest, _attach_manifest_hash(manifest))
    with pytest.raises(ValueError, match="anchored evaluation hash required"):
        build_coverage_snapshot(inputs)

    inputs = _make_fixture_tree(tmp_path / "has_evaluation_false")
    manifest = read_json(inputs.trust_manifest)
    train_run = next(run for run in manifest["historical_runs"] if run["mode"] == "train")
    train_run["has_evaluation"] = False
    _write_json(inputs.trust_manifest, _attach_manifest_hash(manifest))
    with pytest.raises(ValueError, match="has_evaluation mismatch"):
        build_coverage_snapshot(inputs)


def test_offline_patient_identifier_requires_token_boundaries(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path)
    offline_file = inputs.offline_question_roots[0] / "project.md"
    offline_file.write_text(
        "Patient_00001 Patient_00010X XPatient_00011 Patient_00012-extra Patient_00009",
        encoding="utf-8",
    )
    inputs.offline_question_roots[1].joinpath("parent.md").write_text("", encoding="utf-8")

    snapshot = build_coverage_snapshot(inputs)
    patient_ids = {item["patient_id"] for item in snapshot["patients"]}

    assert "Patient_00010" not in patient_ids
    assert "Patient_00011" not in patient_ids
    assert "Patient_00012" not in patient_ids
    assert "Patient_00009" in patient_ids


def _copy_fixture_tree(source: Path, destination: Path) -> CoverageInputs:
    shutil.copytree(source, destination)
    source_parent_offline = source.parent / (source.name + "_parent_docs")
    destination_parent_offline = destination.parent / (source.name + "_parent_docs")
    shutil.copytree(source_parent_offline, destination_parent_offline)
    os.utime(destination / "outputs" / "test" / "test_anchored" / "events.jsonl", (1_000_000, 1_000_000))
    return CoverageInputs(
        project_root=destination,
        train_outputs=destination / "outputs" / "train",
        test_outputs=destination / "outputs" / "test",
        trust_manifest=destination / "docs" / "manifest.json",
        registry_path=destination / "releases" / "explicit_v4" / "verified_registry.json",
        release_manifest_path=destination / "releases" / "explicit_v4" / "release_manifest.json",
        offline_question_roots=(
            destination / "docs" / "offline_questions",
            destination.parent / (source.name + "_parent_docs") / "offline_questions",
        ),
        pollution_receipts=(destination / "receipts" / "pollution.json",),
    )


def test_snapshot_id_is_deterministic_across_absolute_roots_and_mtimes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    first_inputs = _make_fixture_tree(source_root)
    second_inputs = _copy_fixture_tree(source_root, tmp_path / "other" / "deep" / "copy")

    first = build_coverage_snapshot(first_inputs)
    second = build_coverage_snapshot(second_inputs)

    assert first == second
    assert first["snapshot_id"] == content_hash(
        {key: value for key, value in first.items() if key != "snapshot_id"}
    )
    assert canonical_json(first).find(str(source_root)) == -1


def test_snapshot_id_changes_when_a_source_byte_changes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    inputs = _make_fixture_tree(source_root)
    before = build_coverage_snapshot(inputs)["snapshot_id"]
    offline_file = source_root / "docs" / "offline_questions" / "project.md"
    offline_file.write_text(offline_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    after = build_coverage_snapshot(inputs)["snapshot_id"]

    assert after != before


def test_immutable_writer_reuses_identical_content_and_rejects_different_content(
    tmp_path: Path,
) -> None:
    snapshot = build_coverage_snapshot(_make_fixture_tree(tmp_path / "project"))
    artifact_root = tmp_path / "artifacts"

    first = write_coverage_snapshot(snapshot, artifact_root=artifact_root)
    second = write_coverage_snapshot(snapshot, artifact_root=artifact_root)
    assert first["reused"] is False
    assert second["reused"] is True

    path = Path(first["path"])
    path.write_text(canonical_json({**snapshot, "schema_version": "different/v1"}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs"):
        write_coverage_snapshot(snapshot, artifact_root=artifact_root)


def test_snapshot_writer_publishes_exclusively_without_overwriting_a_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = build_coverage_snapshot(_make_fixture_tree(tmp_path / "project"))
    artifact_root = tmp_path / "artifacts"
    target = artifact_root / snapshot["snapshot_id"] / "coverage_snapshot.json"
    competitor = canonical_json({**snapshot, "schema_version": "competitor/v1"}).encode("utf-8")
    original_link = os.link
    published = False

    def publish_competitor_then_link(source: Path, destination: Path) -> None:
        nonlocal published
        if Path(destination) == target and not published:
            published = True
            target.write_bytes(competitor)
        original_link(source, destination)

    monkeypatch.setattr(os, "link", publish_competitor_then_link)

    with pytest.raises(FileExistsError, match="differs"):
        write_coverage_snapshot(snapshot, artifact_root=artifact_root)

    assert published is True
    assert target.read_bytes() == competitor
    assert not list(target.parent.glob(".coverage_snapshot.json.*.tmp"))


def test_snapshot_writer_flushes_and_fsyncs_temp_before_exclusive_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = build_coverage_snapshot(_make_fixture_tree(tmp_path / "project"))
    artifact_root = tmp_path / "artifacts"
    events: list[tuple[str, Any]] = []
    original_fsync = os.fsync
    original_link = os.link

    def tracked_fsync(fd: int) -> None:
        events.append(("fsync", fd))
        original_fsync(fd)

    def tracked_link(source: Path, destination: Path) -> None:
        events.append(("link", Path(source)))
        assert events and events[-2][0] == "fsync"
        assert Path(source).parent == Path(destination).parent
        original_link(source, destination)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    monkeypatch.setattr(os, "link", tracked_link)

    result = write_coverage_snapshot(snapshot, artifact_root=artifact_root)

    assert result["reused"] is False
    assert [name for name, _ in events] == ["fsync", "link"]


def _real_inputs() -> CoverageInputs:
    return CoverageInputs(
        project_root=PROJECT_ROOT,
        train_outputs=PROJECT_ROOT / "outputs" / "train",
        test_outputs=PROJECT_ROOT / "outputs" / "test",
        trust_manifest=PROJECT_ROOT / "docs" / "架构迁移基线" / "manifest.json",
        registry_path=REAL_RELEASE / "verified_registry.json",
        release_manifest_path=REAL_RELEASE / "release_manifest.json",
        offline_question_roots=REAL_OFFLINE_ROOTS,
    )


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_COVERAGE_TREE") != "1",
    reason="set RUN_REAL_COVERAGE_TREE=1 to verify workspace-local historical artifacts",
)
def test_real_tree_counts_and_explicit_v4_release() -> None:
    snapshot = build_coverage_snapshot(_real_inputs())

    assert snapshot["counts"]["unique_patients"] == 125
    assert snapshot["counts"]["primary"] == {
        "case-memory-covered": 6,
        "manifest-anchored-batch-evaluated-provenance-only": 65,
        "unanchored-evaluated": 46,
        "final-only": 4,
        "attempt-only": 4,
        "offline-test-only": 0,
    }
    assert snapshot["counts"]["sources"] == {
        "test": 122,
        "train": 6,
        "offline": 52,
        "test_offline_intersection": 49,
    }
    release_sources = {item["path"] for item in snapshot["sources"] if item["kind"].startswith("release-")}
    assert release_sources == {
        "releases/release_C_case_memory_20260719_v4/release_manifest.json",
        "releases/release_C_case_memory_20260719_v4/verified_registry.json",
    }
    assert all("releases/current.json" not in item["path"] for item in snapshot["sources"])
    assert len(snapshot["patient_content_ids"]) == 125
    assert snapshot["patient_content_ids"] == {
        item["patient_id"]: content_hash(item) for item in snapshot["patients"]
    }


def test_cli_writes_content_addressed_snapshot_without_online_actions(tmp_path: Path) -> None:
    inputs = _make_fixture_tree(tmp_path / "project")
    artifact_root = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-m",
        "scripts.coverage.build_coverage_snapshot",
        "--project-root",
        str(inputs.project_root),
        "--train-outputs",
        str(inputs.train_outputs),
        "--test-outputs",
        str(inputs.test_outputs),
        "--trust-manifest",
        str(inputs.trust_manifest),
        "--registry",
        str(inputs.registry_path),
        "--release-manifest",
        str(inputs.release_manifest_path),
        "--offline-question-root",
        str(inputs.offline_question_roots[0]),
        "--offline-question-root",
        str(inputs.offline_question_roots[1]),
        "--pollution-receipt",
        str(inputs.pollution_receipts[0]),
        "--artifact-root",
        str(artifact_root),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    snapshot_path = Path(output["path"])
    snapshot = read_json(snapshot_path)

    assert snapshot_path == artifact_root / snapshot["snapshot_id"] / "coverage_snapshot.json"
    assert snapshot_path.exists()
    assert hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest() == file_hash(snapshot_path)
