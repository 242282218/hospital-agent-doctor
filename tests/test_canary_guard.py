from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.test import canary_guard
from scripts.test.canary_guard import (
    build_parser,
    combine_batch_results,
    collect_run_metrics,
    create_batch_evaluation_marker,
    create_run_attempt,
    evaluate_canary_gate,
    freeze_selection,
    historical_patient_snapshot,
    require_test_run_dir,
    selected_batch,
    validate_evaluation_binding,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_minimal_llm_logs(run_dir: Path, patient_ids: list[str]) -> None:
    blocks = []
    for index, patient_id in enumerate(patient_ids):
        blocks.append(
            f"timestamp: 2026-07-16T00:00:{index:02d}+08:00\n"
            "prompt_name: test_prompt\n"
            f"patient_id: {patient_id}\n\n"
            "system_prompt:\n系统\n\nuser_prompt:\n问题\n\nresponse:\n{\"ok\": true}\n"
        )
    prompt_dir = run_dir / "llm_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "calls.txt").write_text(("\n" + "=" * 80 + "\n").join(blocks), encoding="utf-8")


def passing_report(patient_ids: list[str], *, diagnosis_accuracy: float = 1.0) -> dict:
    return {
        "diagnosis_accuracy": diagnosis_accuracy,
        "examination_precision": 1.0,
        "treatment_overall_score": 1.0,
        "treatment_safety": 1.0,
        "treatment_effectiveness_alignment": 1.0,
        "treatment_personalization": 1.0,
        "counts": {"final_results": len(patient_ids), "evaluated_patients": len(patient_ids)},
        "treatment_details": [{"patient_id": patient_id, "safety": 1.0} for patient_id in patient_ids],
    }


def write_evaluated_artifacts(
    project_root: Path,
    selection_path: Path,
    batch: str,
    run_name: str,
    report: dict | None = None,
    gate_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    patient_ids = selected_batch(selection, batch)
    attempt_path = create_run_attempt(
        selection_path,
        batch,
        project_root / "outputs" / "run_attempts",
        gate_path=gate_path,
    )
    run_dir = project_root / "outputs" / "test" / run_name
    final_rows = [{"patient_id": patient_id, "finished": True} for patient_id in patient_ids]
    write_jsonl(run_dir / "final_results.jsonl", final_rows)
    case_events = [
        event
        for patient_id in patient_ids
        for event in [
            {"event_type": "PRESCRIBE_TREATMENT", "patient_id": patient_id},
            {"event_type": "CASE_END", "patient_id": patient_id},
        ]
    ]
    write_jsonl(run_dir / "events.jsonl", case_events)
    write_minimal_llm_logs(run_dir, patient_ids)
    run_receipt_path = canary_guard.create_run_receipt(run_dir, attempt_path)
    metrics_path = run_dir / f"{batch}_metrics.json"
    metrics_path.write_text(
        json.dumps(collect_run_metrics(run_dir.resolve(), expected_ids=patient_ids)),
        encoding="utf-8",
    )
    marker_path = create_batch_evaluation_marker(
        run_dir,
        project_root / "outputs" / "batch_evaluation_attempts",
        run_receipt_path=run_receipt_path,
        metrics_path=metrics_path,
    )
    report = report or passing_report(patient_ids)
    report_path = run_dir / "final_results_eval_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    write_jsonl(
        run_dir / "events.jsonl",
        [
            *case_events,
            {
                "event_type": "BATCH_EVALUATION_REQUEST",
                "status": "success",
                "payload": {"output_log_path": str(run_dir)},
            },
            {
                "event_type": "BATCH_EVALUATION_RESULT",
                "status": "success",
                "payload": {"output_log_path": str(run_dir), "report": report},
            },
        ],
    )
    canary_guard.create_evaluation_receipt(run_dir, marker_path, report_path, report)
    return run_dir, metrics_path, report_path


def test_paired_rerun_manifest_is_baseline_bound_and_single_attempt(tmp_path: Path) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003", "Patient_00004"]
    baseline_receipts = {}
    for patient_id in patient_ids:
        receipt_path = tmp_path / f"{patient_id}.json"
        receipt_path.write_text(json.dumps({"patient_ids": [patient_id]}), encoding="utf-8")
        baseline_receipts[patient_id] = receipt_path
    manifest_path = tmp_path / "paired_manifest.json"

    manifest = canary_guard.create_paired_rerun_manifest(
        pair_id="diagnosis_chain_20260722",
        patient_ids=patient_ids,
        baseline_run_receipts=baseline_receipts,
        candidate_source_sha256="a" * 64,
        output_path=manifest_path,
    )
    assert manifest["run_policy"]["allow_retry"] is False
    assert manifest["run_policy"]["evaluation_count"] == 1
    attempt = canary_guard.create_paired_rerun_attempt(manifest_path, tmp_path / "attempts")
    assert attempt.parent.name == "paired"
    with pytest.raises(FileExistsError):
        canary_guard.create_paired_rerun_attempt(manifest_path, tmp_path / "attempts")

    baseline_receipts[patient_ids[0]].write_text(
        json.dumps({"patient_ids": ["Patient_99999"]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="baseline receipt changed"):
        canary_guard.validate_paired_rerun_manifest(manifest_path)


def test_history_snapshot_and_selection_are_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    write_jsonl(
        output_root / "test" / "old_test" / "events.jsonl",
        [{"patient_id": "Patient_00002"}, {"payload": {"patient_id": "Patient_00001"}}],
    )
    write_jsonl(
        output_root / "train" / "old_train" / "final_results.jsonl",
        [{"patient_id": "Patient_00003"}, {"patient_id": "Patient_Comorbid-00003"}],
    )
    (output_root / "run_attempts").mkdir(parents=True)
    (output_root / "run_attempts" / "attempt.json").write_text(
        json.dumps({"patient_ids": ["Patient_00004"]}),
        encoding="utf-8",
    )
    expected_ids = [
        "Patient_00001",
        "Patient_00002",
        "Patient_00003",
        "Patient_00004",
        "Patient_Comorbid-00003",
    ]
    expected_digest = hashlib.sha256(("\n".join(expected_ids) + "\n").encode()).hexdigest()

    history_ids, history_digest = historical_patient_snapshot(output_root)
    selection_path = tmp_path / "selection.json"
    manifest = freeze_selection(
        candidate_ids=["Patient_00002"] + [f"Patient_{index:05d}" for index in range(4, 15)],
        history_ids=history_ids,
        history_sha256=history_digest,
        output_path=selection_path,
        seed=2026071601,
        expected_history_count=5,
        expected_history_sha256=expected_digest,
    )

    assert manifest["canary_ids"] == ["Patient_00005", "Patient_00006", "Patient_00007"]
    assert manifest["confirmation_ids"] == [f"Patient_{index:05d}" for index in range(8, 15)]
    with pytest.raises(FileExistsError):
        freeze_selection(
            candidate_ids=[],
            history_ids=history_ids,
            history_sha256=history_digest,
            output_path=selection_path,
            seed=2026071601,
            expected_history_count=5,
            expected_history_sha256=expected_digest,
        )


def test_history_snapshot_fails_closed_on_corrupt_attempt(tmp_path: Path) -> None:
    attempt_root = tmp_path / "outputs" / "run_attempts"
    attempt_root.mkdir(parents=True)
    (attempt_root / "corrupt.json").write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid run attempt artifact"):
        historical_patient_snapshot(tmp_path / "outputs")


def test_collect_metrics_counts_actions_repairs_and_approximate_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    write_jsonl(
        run_dir / "final_results.jsonl",
        [
            {
                "patient_id": patient_id,
                "conversation_rounds": 1,
                "ordered_examinations": ["检查A"],
                "finished": True,
            }
            for patient_id in patient_ids
        ],
    )
    write_jsonl(
        run_dir / "events.jsonl",
        [
            {"event_type": "SEND_MESSAGE", "patient_id": "Patient_00001", "payload": {}},
            {
                "event_type": "SCHEDULE_EXAMINATION",
                "patient_id": "Patient_00001",
                "payload": {"items": ["检查A", "检查B"]},
            },
            {
                "event_type": "DO_EXAMINATION",
                "patient_id": "Patient_00001",
                "payload": {"invalid_items": ["错误检查"]},
            },
        ],
    )
    prompt_dir = run_dir / "llm_prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "calls.txt").write_text(
        "timestamp: 2026-07-16T00:00:00+08:00\n"
        "prompt_name: test_prompt\n"
        "patient_id: Patient_00001\n\n"
        "system_prompt:\n系统\n\nuser_prompt:\n问题A\n\nresponse:\n{\"ok\": true}\n\n"
        + "=" * 80
        + "\n"
        "timestamp: 2026-07-16T00:00:01+08:00\n"
        "prompt_name: test_prompt\n"
        "patient_id: Patient_00001\n\n"
        "system_prompt:\n系统\n\nuser_prompt:\n问题B\n\nresponse:\nnot-json\n\n"
        + "=" * 80
        + "\n",
        encoding="utf-8",
    )
    (prompt_dir / "unbound.txt").write_text(
        "timestamp: 2026-07-16T00:00:02+08:00\n"
        "prompt_name: test_prompt\n"
        "patient_id: Patient_99999\n\n"
        "system_prompt:\n系统\n\nuser_prompt:\n问题C\n\nresponse:\n{\"ok\": true}\n",
        encoding="utf-8",
    )

    metrics = collect_run_metrics(run_dir, expected_ids=patient_ids)

    assert metrics["finished_count"] == 3
    assert metrics["conversation_rounds"] == 3
    assert metrics["send_message_count"] == 1
    assert metrics["scheduled_exam_count"] == 2
    assert metrics["invalid_exam_count"] == 1
    assert metrics["llm_logged_calls"] == 3
    assert metrics["repair_calls_inferred"] == 1
    assert metrics["llm_calls_min"] == 4
    assert metrics["unbound_llm_calls"] == 1
    assert metrics["malformed_llm_blocks"] == 0
    assert metrics["approximate_tokens"] > 0
    assert metrics["prompt_log_bytes"] > 0


def test_batch_marker_is_exclusive_and_rejects_previous_evaluation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    marker_root = tmp_path / "markers"
    write_jsonl(run_dir / "final_results.jsonl", [{"patient_id": "Patient_00001", "finished": True}])
    write_jsonl(run_dir / "events.jsonl", [{"event_type": "CASE_END"}])

    marker_path = create_batch_evaluation_marker(run_dir, marker_root)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["sha256"] == marker_path.stem
    with pytest.raises(FileExistsError):
        create_batch_evaluation_marker(run_dir, marker_root)

    marker_path.unlink()
    write_jsonl(run_dir / "events.jsonl", [{"event_type": "BATCH_EVALUATION_REQUEST"}])
    with pytest.raises(RuntimeError, match="already requested"):
        create_batch_evaluation_marker(run_dir, marker_root)

    unfinished_dir = tmp_path / "unfinished"
    write_jsonl(unfinished_dir / "final_results.jsonl", [{"patient_id": "Patient_00002", "finished": "false"}])
    with pytest.raises(RuntimeError, match="unfinished"):
        create_batch_evaluation_marker(unfinished_dir, tmp_path / "unfinished_markers")

    corrupt_root = tmp_path / "corrupt_markers"
    corrupt_root.mkdir()
    (corrupt_root / "corrupt.json").write_text("{", encoding="utf-8")
    clean_run = tmp_path / "clean_run"
    write_jsonl(clean_run / "final_results.jsonl", [{"patient_id": "Patient_00003", "finished": True}])
    with pytest.raises(RuntimeError, match="invalid evaluation marker"):
        create_batch_evaluation_marker(clean_run, corrupt_root)


def test_truncated_llm_block_is_marked_malformed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    patient_ids = ["Patient_00001"]
    write_jsonl(run_dir / "final_results.jsonl", [{"patient_id": patient_ids[0], "finished": True}])
    prompt_dir = run_dir / "llm_prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "truncated.txt").write_text(
        "timestamp: 2026-07-16T00:00:00+08:00\nprompt_name: broken\npatient_id: Patient_00001\n",
        encoding="utf-8",
    )

    metrics = collect_run_metrics(run_dir, expected_ids=patient_ids)

    assert metrics["malformed_llm_blocks"] == 1


def test_canary_gate_enforces_quality_safety_and_llm_limits() -> None:
    metrics = {
        "patient_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
        "expected_count": 3,
        "finished_count": 3,
        "invalid_exam_count": 0,
        "unbound_llm_calls": 0,
        "malformed_llm_blocks": 0,
        "llm_logged_calls": 21,
        "repair_calls_inferred": 0,
        "llm_calls_min": 21,
        "prompt_log_bytes": 1,
        "case_end_count": 3,
        "treatment_event_count": 3,
        "per_patient": {
            "Patient_00001": {"llm_calls_min": 7, "case_end_events": 1, "treatment_events": 1},
            "Patient_00002": {"llm_calls_min": 8, "case_end_events": 1, "treatment_events": 1},
            "Patient_00003": {"llm_calls_min": 6, "case_end_events": 1, "treatment_events": 1},
        },
    }
    report = {
        "diagnosis_accuracy": 0.34,
        "examination_precision": 0.25,
        "treatment_overall_score": 0.50,
        "treatment_safety": 0.90,
        "counts": {"final_results": 3, "evaluated_patients": 3},
        "treatment_details": [
            {"patient_id": patient_id, "safety": 0.8}
            for patient_id in metrics["per_patient"]
        ],
    }

    passed = evaluate_canary_gate(metrics, report, p0_count=0)
    failed = evaluate_canary_gate(metrics, {**report, "treatment_safety": 0.89}, p0_count=0)
    unbound = evaluate_canary_gate({**metrics, "unbound_llm_calls": 1}, report, p0_count=0)
    malformed = evaluate_canary_gate({**metrics, "malformed_llm_blocks": 1}, report, p0_count=0)
    invalid_score = evaluate_canary_gate({**metrics}, {**report, "diagnosis_accuracy": 999}, p0_count=0)
    missing_logs = evaluate_canary_gate(
        {
            **metrics,
            "llm_logged_calls": 0,
            "llm_calls_min": 0,
            "prompt_log_bytes": 0,
            "per_patient": {patient_id: {"llm_calls_min": 0} for patient_id in metrics["patient_ids"]},
        },
        report,
        p0_count=0,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert "treatment_safety" in failed["failed_checks"]
    assert unbound["passed"] is False
    assert "unbound_llm_calls" in unbound["failed_checks"]
    assert malformed["passed"] is False
    assert "malformed_llm_blocks" in malformed["failed_checks"]
    assert invalid_score["passed"] is False
    assert "score_ranges" in invalid_score["failed_checks"]
    assert missing_logs["passed"] is False
    assert "llm_logs_present" in missing_logs["failed_checks"]


def test_gate_rejects_incomplete_report_details() -> None:
    metrics = {
        "patient_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
        "expected_count": 3,
        "finished_count": 3,
        "invalid_exam_count": 0,
        "unbound_llm_calls": 0,
        "malformed_llm_blocks": 0,
        "llm_logged_calls": 3,
        "repair_calls_inferred": 0,
        "llm_calls_min": 3,
        "prompt_log_bytes": 1,
        "per_patient": {
            "Patient_00001": {"llm_calls_min": 1},
            "Patient_00002": {"llm_calls_min": 1},
            "Patient_00003": {"llm_calls_min": 1},
        },
    }
    report = {
        "diagnosis_accuracy": 1,
        "examination_precision": 1,
        "treatment_overall_score": 1,
        "treatment_safety": 1,
        "counts": {"final_results": 3, "evaluated_patients": 3},
        "treatment_details": [{"patient_id": "Patient_00001", "safety": 1}],
    }

    result = evaluate_canary_gate(metrics, report, p0_count=0)

    assert result["passed"] is False
    assert "treatment_details" in result["failed_checks"]


def test_run_attempt_is_exclusive_and_confirmation_requires_verified_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "canary_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    attempts = tmp_path / "attempts"
    canary_attempt = create_run_attempt(selection, "canary", attempts)
    assert canary_attempt.exists()
    with pytest.raises(FileExistsError):
        create_run_attempt(selection, "canary", attempts)

    copied_selection = tmp_path / "selection_copy.json"
    copied_selection.write_text(
        json.dumps({**json.loads(selection.read_text(encoding="utf-8")), "note": "same patients"}, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        create_run_attempt(copied_selection, "canary", attempts)

    failed_gate = tmp_path / "failed_gate.json"
    failed_gate.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="verified Canary gate"):
        create_run_attempt(selection, "confirmation", attempts, gate_path=failed_gate)

    passed_gate = tmp_path / "passed_gate.json"
    passed_gate.write_text(
        json.dumps(
            {
                "passed": True,
                "batch": "canary",
                "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="verified Canary gate"):
        create_run_attempt(selection, "confirmation", attempts, gate_path=passed_gate)

    canary_ids = json.loads(selection.read_text(encoding="utf-8"))["canary_ids"]
    _run_dir, metrics_path, report_path = write_evaluated_artifacts(
        tmp_path,
        selection,
        "canary",
        "canary",
    )
    verified_gate = tmp_path / "verified_gate.json"
    gate = canary_guard.command_gate(
        SimpleNamespace(
            selection=str(selection),
            metrics=str(metrics_path),
            report=str(report_path),
            p0_count=0,
            output=str(verified_gate),
        )
    )
    assert gate["passed"] is True
    assert create_run_attempt(selection, "confirmation", attempts, gate_path=verified_gate).exists()

    rebound_selection = tmp_path / "selection_rebound.json"
    rebound_selection.write_text(
        json.dumps(
            {
                "canary_ids": canary_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(11, 18)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="verified Canary gate"):
        create_run_attempt(
            rebound_selection,
            "confirmation",
            tmp_path / "rebound_attempts",
            gate_path=verified_gate,
        )


def test_selection_requires_exact_disjoint_three_plus_seven() -> None:
    valid = {
        "canary_ids": ["Patient_1", "Patient_2", "Patient_3"],
        "confirmation_ids": [f"Patient_{index}" for index in range(4, 11)],
    }

    assert selected_batch(valid, "canary") == valid["canary_ids"]
    with pytest.raises(RuntimeError, match="exactly 3 Canary"):
        selected_batch({**valid, "canary_ids": valid["canary_ids"] + ["Patient_11"]}, "canary")
    with pytest.raises(RuntimeError, match="must be unique"):
        selected_batch(
            {**valid, "confirmation_ids": ["Patient_3"] + valid["confirmation_ids"][1:]},
            "confirmation",
        )
    with pytest.raises(RuntimeError, match="invalid patient ID"):
        selected_batch({**valid, "canary_ids": ["invalid"] + valid["canary_ids"][1:]}, "canary")


def test_evaluation_binding_requires_selection_ids_and_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    selection = {
        "canary_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
        "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
    }
    write_jsonl(
        run_dir / "final_results.jsonl",
        [{"patient_id": patient_id, "finished": True} for patient_id in selection["canary_ids"]],
    )
    metrics = collect_run_metrics(run_dir.resolve(), expected_ids=selection["canary_ids"])
    validate_evaluation_binding(run_dir, selection, "canary", metrics)
    with pytest.raises(RuntimeError, match="metrics patient IDs"):
        validate_evaluation_binding(run_dir, selection, "canary", {**metrics, "patient_ids": ["Patient_99999"]})


def test_combine_weights_three_plus_seven_and_rejects_duplicate_ids() -> None:
    canary_metrics = {
        "patient_ids": ["Patient_1", "Patient_2", "Patient_3"],
        "invalid_exam_count": 0,
        "llm_calls_min": 17,
        "unbound_llm_calls": 0,
        "malformed_llm_blocks": 0,
        "llm_logged_calls": 17,
        "repair_calls_inferred": 0,
        "prompt_log_bytes": 1,
        "per_patient": {
            "Patient_1": {"llm_calls_min": 6},
            "Patient_2": {"llm_calls_min": 6},
            "Patient_3": {"llm_calls_min": 5},
        },
    }
    confirmation_metrics = {
        "patient_ids": [f"Patient_{index}" for index in range(4, 11)],
        "invalid_exam_count": 0,
        "llm_calls_min": 41,
        "unbound_llm_calls": 0,
        "malformed_llm_blocks": 0,
        "llm_logged_calls": 41,
        "repair_calls_inferred": 0,
        "prompt_log_bytes": 1,
        "per_patient": {
            **{f"Patient_{index}": {"llm_calls_min": 6} for index in range(4, 10)},
            "Patient_10": {"llm_calls_min": 5},
        },
    }
    canary_report = {
        "diagnosis_accuracy": 0.3333,
        "examination_precision": 0.5556,
        "treatment_overall_score": 0.6267,
        "treatment_safety": 1.0,
        "treatment_effectiveness_alignment": 0.6,
        "treatment_personalization": 0.6667,
        "counts": {"final_results": 3, "evaluated_patients": 3},
        "treatment_details": [{"patient_id": patient_id, "safety": 1.0} for patient_id in canary_metrics["patient_ids"]],
    }
    confirmation_report = {
        "diagnosis_accuracy": 0.1429,
        "examination_precision": 0.2857,
        "treatment_overall_score": 0.4343,
        "treatment_safety": 0.7143,
        "treatment_effectiveness_alignment": 0.4571,
        "treatment_personalization": 0.5429,
        "counts": {"final_results": 7, "evaluated_patients": 7},
        "treatment_details": [
            {"patient_id": patient_id, "safety": 0.2 if patient_id in {"Patient_8", "Patient_9"} else 1.0}
            for patient_id in confirmation_metrics["patient_ids"]
        ],
    }

    combined = combine_batch_results(
        [(canary_metrics, canary_report), (confirmation_metrics, confirmation_report)],
        p0_count=0,
    )

    assert combined["diagnosis_accuracy"] == pytest.approx(0.2, abs=1e-4)
    assert combined["examination_precision"] == pytest.approx(0.3667, abs=1e-4)
    assert combined["treatment_overall_score"] == pytest.approx(0.492, abs=1e-4)
    assert combined["treatment_safety"] == pytest.approx(0.8, abs=1e-4)
    assert combined["minimum_patient_safety"] == 0.2
    assert combined["llm_calls_min"] == 58
    assert combined["unbound_llm_calls"] == 0

    missing_logs = {
        **canary_metrics,
        "llm_logged_calls": 0,
        "llm_calls_min": 0,
        "prompt_log_bytes": 0,
        "per_patient": {patient_id: {"llm_calls_min": 0} for patient_id in canary_metrics["patient_ids"]},
    }
    missing_logs_result = combine_batch_results(
        [(missing_logs, canary_report), (confirmation_metrics, confirmation_report)],
        p0_count=0,
    )
    assert missing_logs_result["passed"] is False
    assert "llm_logs_present" in missing_logs_result["failed_checks"]

    duplicate = {**confirmation_metrics, "patient_ids": ["Patient_1"] + confirmation_metrics["patient_ids"][1:]}
    with pytest.raises(RuntimeError, match="duplicate patient IDs"):
        combine_batch_results([(canary_metrics, canary_report), (duplicate, confirmation_report)], p0_count=0)

    five_a = {**canary_metrics, "patient_ids": [f"Patient_A{index}" for index in range(5)]}
    five_b = {**confirmation_metrics, "patient_ids": [f"Patient_B{index}" for index in range(5)]}
    report_a = {**canary_report, "counts": {"evaluated_patients": 5}}
    report_b = {**confirmation_report, "counts": {"evaluated_patients": 5}}
    with pytest.raises(RuntimeError, match="3-case Canary and 7-case confirmation"):
        combine_batch_results([(five_a, report_a), (five_b, report_b)], p0_count=0)

    rounding_report_a = {**canary_report, "diagnosis_accuracy": 0.49996}
    rounding_report_b = {**confirmation_report, "diagnosis_accuracy": 0.49996}
    rounding_result = combine_batch_results(
        [(canary_metrics, rounding_report_a), (confirmation_metrics, rounding_report_b)],
        p0_count=0,
    )
    assert rounding_result["diagnosis_accuracy"] == 0.5
    assert "diagnosis_accuracy" in rounding_result["failed_checks"]

    over_budget = {
        **confirmation_metrics,
        "llm_logged_calls": 48,
        "llm_calls_min": 48,
        "per_patient": {
            **confirmation_metrics["per_patient"],
            "Patient_4": {"llm_calls_min": 13},
        },
    }
    over_budget_result = combine_batch_results(
        [(canary_metrics, canary_report), (over_budget, confirmation_report)],
        p0_count=0,
    )
    assert "llm_calls_per_patient_max" in over_budget_result["failed_checks"]


def test_report_scores_must_be_finite_and_within_unit_interval() -> None:
    patient_ids = ["Patient_1", "Patient_2", "Patient_3"]
    report = passing_report(patient_ids)

    canary_guard.validate_report_binding(report, patient_ids)
    with pytest.raises(RuntimeError, match="invalid aggregate score"):
        canary_guard.validate_report_binding({**report, "diagnosis_accuracy": 999}, patient_ids)
    with pytest.raises(RuntimeError, match="invalid aggregate score"):
        canary_guard.validate_report_binding({**report, "diagnosis_accuracy": float("nan")}, patient_ids)


def test_cli_exposes_read_only_history_and_does_not_allow_custom_run_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_jsonl(
        tmp_path / "outputs" / "test" / "old" / "final_results.jsonl",
        [{"patient_id": "Patient_00001", "finished": True}],
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)

    result = canary_guard.command_history(SimpleNamespace())

    assert result["history_count"] == 1
    assert len(result["history_sha256"]) == 64
    assert build_parser().parse_args(["history"]).command == "history"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run",
                "--selection",
                "selection.json",
                "--batch",
                "canary",
                "--port",
                "9999",
            ]
        )


def test_run_creates_attempt_after_readiness_before_post_and_binds_confirmation_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = tmp_path / "selection.json"
    canary_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    confirmation_ids = [f"Patient_{index:05d}" for index in range(4, 11)]
    selection_path.write_text(
        json.dumps({"canary_ids": canary_ids, "confirmation_ids": confirmation_ids}),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "test" / "run"
    class FakeServer:
        @staticmethod
        def poll():
            return None

    class FakeRunner:
        def start_agent_server(self):
            assert len(list((tmp_path / "outputs" / "run_attempts").glob("*.json"))) == 0
            return FakeServer()

        @staticmethod
        def wait_until_ready() -> None:
            return None

        @staticmethod
        def stop_agent_server(_server: object) -> None:
            return None

    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(canary_guard, "port_is_open", lambda port: False)
    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    monkeypatch.setattr(canary_guard, "runtime_clients", lambda: object())
    def fake_post_test(patient_ids: list[str], port: int) -> dict:
        assert len(list((tmp_path / "outputs" / "run_attempts").glob("*.json"))) == 1
        write_jsonl(
            run_dir / "final_results.jsonl",
            [{"patient_id": patient_id, "finished": True} for patient_id in patient_ids],
        )
        return {
            "result": {"output_dir": str(run_dir.relative_to(tmp_path))},
            "patient_ids": patient_ids,
            "port": port,
        }

    monkeypatch.setattr(canary_guard, "post_test", fake_post_test)

    result = canary_guard.command_run(
        SimpleNamespace(selection=str(selection_path), batch="canary", gate=None)
    )

    assert result["patient_ids"] == canary_ids
    assert Path(result["attempt_path"]).exists()
    assert Path(result["run_receipt_path"]).exists()
    assert len(list((tmp_path / "outputs" / "run_attempts").glob("*.json"))) == 1
    assert not (tmp_path / "outputs" / "canary_run.lock").exists()
    with pytest.raises(RuntimeError, match="verified Canary gate"):
        canary_guard.command_run(
            SimpleNamespace(selection=str(selection_path), batch="confirmation", gate=None)
        )


def test_run_local_preflight_failure_does_not_consume_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(canary_guard, "port_is_open", lambda port: False)
    monkeypatch.setattr(
        canary_guard,
        "load_local_test_runner",
        lambda: (_ for _ in ()).throw(ImportError("broken local runner")),
    )

    with pytest.raises(ImportError, match="broken local runner"):
        canary_guard.command_run(SimpleNamespace(selection=str(selection_path), batch="canary", gate=None))

    assert list((tmp_path / "outputs" / "run_attempts").glob("*.json")) == []


def test_readiness_failure_does_not_claim_patients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )

    class FakeServer:
        @staticmethod
        def poll():
            return None

    class FakeRunner:
        @staticmethod
        def start_agent_server() -> FakeServer:
            return FakeServer()

        @staticmethod
        def wait_until_ready() -> None:
            raise TimeoutError("not ready")

        @staticmethod
        def stop_agent_server(_server: object) -> None:
            return None

    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(canary_guard, "port_is_open", lambda port: False)
    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    monkeypatch.setattr(canary_guard, "runtime_clients", lambda: object())

    with pytest.raises(TimeoutError, match="not ready"):
        canary_guard.command_run(SimpleNamespace(selection=str(selection_path), batch="canary", gate=None))

    assert list((tmp_path / "outputs" / "run_attempts").glob("*.json")) == []
    assert list((tmp_path / "outputs" / "patient_attempts").glob("*.claim")) == []
    assert not (tmp_path / "outputs" / "canary_run.lock").exists()


def test_run_rejects_patient_that_entered_history_before_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "outputs" / "test" / "old" / "final_results.jsonl",
        [{"patient_id": patient_ids[0], "finished": True}],
    )

    class FakeRunner:
        @staticmethod
        def start_agent_server() -> object:
            raise AssertionError("service must not start for a historical patient")

    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(canary_guard, "port_is_open", lambda port: False)
    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    monkeypatch.setattr(canary_guard, "runtime_clients", lambda: object())

    with pytest.raises(RuntimeError, match="already exist in history"):
        canary_guard.command_run(SimpleNamespace(selection=str(selection_path), batch="canary", gate=None))
    assert list((tmp_path / "outputs" / "run_attempts").glob("*.json")) == []


def test_evaluate_validates_selection_and_metrics_before_creating_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "test" / "run"
    write_jsonl(
        run_dir / "final_results.jsonl",
        [{"patient_id": patient_id, "finished": True} for patient_id in patient_ids],
    )
    attempt_path = create_run_attempt(selection_path, "canary", tmp_path / "outputs" / "run_attempts")
    canary_guard.create_run_receipt(run_dir, attempt_path)
    case_events = [
        event
        for patient_id in patient_ids
        for event in [
            {"event_type": "PRESCRIBE_TREATMENT", "patient_id": patient_id},
            {"event_type": "CASE_END", "patient_id": patient_id},
        ]
    ]
    write_jsonl(run_dir / "events.jsonl", case_events)
    write_minimal_llm_logs(run_dir, patient_ids)
    metrics_path = run_dir / "canary_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "patient_ids": ["Patient_99999"],
                "finished_count": 3,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class FakeRunner:
        @staticmethod
        def evaluate_test_output(_run_dir: Path) -> dict:
            assert len(list((tmp_path / "outputs" / "batch_evaluation_attempts").glob("*.json"))) == 1
            report = passing_report(patient_ids)
            write_jsonl(
                run_dir / "events.jsonl",
                [
                    *case_events,
                    {
                        "event_type": "BATCH_EVALUATION_REQUEST",
                        "status": "success",
                        "payload": {"output_log_path": str(run_dir)},
                    },
                    {
                        "event_type": "BATCH_EVALUATION_RESULT",
                        "status": "success",
                        "payload": {"output_log_path": str(run_dir), "report": report},
                    },
                ],
            )
            calls.append(_run_dir)
            return report

    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    args = SimpleNamespace(
        run_dir=str(run_dir),
        selection=str(selection_path),
        batch="canary",
        metrics=str(metrics_path),
        confirm_once=True,
    )

    with pytest.raises(RuntimeError, match="metrics patient IDs"):
        canary_guard.command_evaluate(args)

    assert calls == []
    assert list((tmp_path / "outputs" / "batch_evaluation_attempts").glob("*.json")) == []

    metrics_path.write_text(
        json.dumps(collect_run_metrics(run_dir.resolve(), expected_ids=patient_ids)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        canary_guard,
        "load_local_test_runner",
        lambda: (_ for _ in ()).throw(ImportError("broken evaluator")),
    )
    with pytest.raises(ImportError, match="broken evaluator"):
        canary_guard.command_evaluate(args)
    assert list((tmp_path / "outputs" / "batch_evaluation_attempts").glob("*.json")) == []

    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    monkeypatch.setattr(canary_guard, "runtime_clients", lambda: object())
    result = canary_guard.command_evaluate(args)

    assert calls == [run_dir.resolve()]
    assert Path(result["marker_path"]).exists()
    assert Path(result["evaluation_receipt_path"]).exists()


def test_failed_gate_is_written_once_and_main_returns_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate.json"
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    _run_dir, metrics_path, report_path = write_evaluated_artifacts(
        tmp_path,
        selection_path,
        "canary",
        "run",
        passing_report(patient_ids, diagnosis_accuracy=0),
    )
    argv = [
        "gate",
        "--selection",
        str(selection_path),
        "--metrics",
        str(metrics_path),
        "--report",
        str(report_path),
        "--p0-count",
        "0",
        "--output",
        str(gate_path),
    ]

    assert canary_guard.main(argv) == 1
    artifact = json.loads(gate_path.read_text(encoding="utf-8"))
    assert artifact["passed"] is False
    assert artifact["batch"] == "canary"
    assert artifact["p0_count"] == 0
    assert artifact["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="verified Canary gate"):
        create_run_attempt(
            selection_path,
            "confirmation",
            tmp_path / "failed_confirmation_attempts",
            gate_path=gate_path,
        )
    with pytest.raises(FileExistsError):
        canary_guard.command_gate(build_parser().parse_args(argv))


def test_gate_rejects_fabricated_report_without_evaluation_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "outputs" / "test" / "run"
    write_jsonl(
        run_dir / "final_results.jsonl",
        [{"patient_id": patient_id, "finished": True} for patient_id in patient_ids],
    )
    write_minimal_llm_logs(run_dir, patient_ids)
    metrics_path = run_dir / "canary_metrics.json"
    metrics_path.write_text(
        json.dumps(collect_run_metrics(run_dir.resolve(), expected_ids=patient_ids)),
        encoding="utf-8",
    )
    fabricated_report = tmp_path / "fabricated_report.json"
    fabricated_report.write_text(json.dumps(passing_report(patient_ids)), encoding="utf-8")
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    attempt_path = create_run_attempt(selection_path, "canary", tmp_path / "outputs" / "run_attempts")
    canary_guard.create_run_receipt(run_dir, attempt_path)

    with pytest.raises(RuntimeError, match="evaluation completion receipt"):
        canary_guard.command_gate(
            SimpleNamespace(
                selection=str(selection_path),
                metrics=str(metrics_path),
                report=str(fabricated_report),
                p0_count=0,
                output=str(tmp_path / "gate.json"),
            )
        )


def test_gate_rejects_llm_log_changes_after_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    run_dir, metrics_path, report_path = write_evaluated_artifacts(
        tmp_path,
        selection_path,
        "canary",
        "canary",
    )
    prompt_path = run_dir / "llm_prompts" / "calls.txt"
    prompt_path.write_text(prompt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metrics|receipt|evidence"):
        canary_guard.command_gate(
            SimpleNamespace(
                selection=str(selection_path),
                metrics=str(metrics_path),
                report=str(report_path),
                p0_count=0,
                output=str(tmp_path / "gate.json"),
            )
        )


def test_combine_cli_binds_three_plus_seven_to_selection_and_writes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary_ids = ["Patient_1", "Patient_2", "Patient_3"]
    confirmation_ids = [f"Patient_{index}" for index in range(4, 11)]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"canary_ids": canary_ids, "confirmation_ids": confirmation_ids}),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    _canary_run, canary_metrics, canary_report = write_evaluated_artifacts(
        tmp_path,
        selection_path,
        "canary",
        "canary",
        passing_report(canary_ids),
    )
    gate_path = tmp_path / "passed_gate.json"
    gate = canary_guard.command_gate(
        SimpleNamespace(
            selection=str(selection_path),
            metrics=str(canary_metrics),
            report=str(canary_report),
            p0_count=0,
            output=str(gate_path),
        )
    )
    assert gate["passed"] is True
    _confirmation_run, confirmation_metrics, confirmation_report = write_evaluated_artifacts(
        tmp_path,
        selection_path,
        "confirmation",
        "confirmation",
        passing_report(confirmation_ids),
        gate_path=gate_path,
    )
    output_path = tmp_path / "combined.json"
    args = SimpleNamespace(
        selection=str(selection_path),
        canary_metrics=str(canary_metrics),
        canary_report=str(canary_report),
        confirmation_metrics=str(confirmation_metrics),
        confirmation_report=str(confirmation_report),
        p0_count=0,
        output=str(output_path),
    )

    result = canary_guard.command_combine(args)

    assert result["passed"] is True
    assert result["evaluated_patients"] == 10
    assert result["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert output_path.exists()
    with pytest.raises(FileExistsError):
        canary_guard.command_combine(args)

    mismatched_report = json.loads(confirmation_report.read_text(encoding="utf-8"))
    mismatched_report["treatment_details"][0]["patient_id"] = "Patient_999"
    confirmation_report.write_text(json.dumps(mismatched_report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="report treatment patient IDs"):
        canary_guard.command_combine(
            SimpleNamespace(
                **{
                    **vars(args),
                    "output": str(tmp_path / "mismatched_combined.json"),
                }
            )
        )


def test_test_run_directory_must_stay_under_outputs_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    inside = tmp_path / "outputs" / "test" / "run"

    assert require_test_run_dir(inside) == inside.resolve()
    with pytest.raises(RuntimeError, match="outputs/test"):
        require_test_run_dir(tmp_path / "outside")


def test_global_run_lock_blocks_concurrent_7860_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": ["Patient_00001", "Patient_00002", "Patient_00003"],
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "outputs" / "canary_run.lock").mkdir(parents=True)
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        canary_guard,
        "load_local_test_runner",
        lambda: (_ for _ in ()).throw(AssertionError("loader must not run while lock is held")),
    )

    with pytest.raises(RuntimeError, match="another Canary run"):
        canary_guard.command_run(SimpleNamespace(selection=str(selection_path), batch="canary", gate=None))


def test_evaluate_rejects_run_without_run_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "outputs" / "test" / "run_without_receipt"
    write_jsonl(
        run_dir / "final_results.jsonl",
        [{"patient_id": patient_id, "finished": True} for patient_id in patient_ids],
    )
    write_minimal_llm_logs(run_dir, patient_ids)
    metrics_path = run_dir / "canary_metrics.json"
    metrics_path.write_text(
        json.dumps(collect_run_metrics(run_dir.resolve(), expected_ids=patient_ids)),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        canary_guard,
        "load_local_test_runner",
        lambda: (_ for _ in ()).throw(AssertionError("evaluator must not load without run receipt")),
    )

    with pytest.raises(RuntimeError, match="run receipt"):
        canary_guard.command_evaluate(
            SimpleNamespace(
                run_dir=str(run_dir),
                selection=str(selection_path),
                batch="canary",
                metrics=str(metrics_path),
                confirm_once=True,
            )
        )


def test_wait_for_port_release_tolerates_short_shutdown_race(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter([True, True, False])
    monkeypatch.setattr(canary_guard, "port_is_open", lambda _port: next(states))
    monkeypatch.setattr(canary_guard.time, "sleep", lambda _seconds: None)

    assert canary_guard.wait_for_port_release(canary_guard.AGENT_PORT, timeout_seconds=1) is True


@pytest.mark.parametrize(
    ("failure_kind", "expected_error_type", "run_dir_present"),
    [
        ("http_500", "HTTPError", False),
        ("service_exception", "RuntimeError", True),
        ("incomplete_final", "RuntimeError", True),
    ],
)
def test_online_run_failure_writes_non_overwritable_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_error_type: str,
    run_dir_present: bool,
) -> None:
    patient_ids = ["Patient_00001", "Patient_00002", "Patient_00003"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "canary_ids": patient_ids,
                "confirmation_ids": [f"Patient_{index:05d}" for index in range(4, 11)],
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "outputs" / "test" / f"failed_{failure_kind}"

    class FakeServer:
        @staticmethod
        def poll() -> None:
            return None

    class FakeRunner:
        @staticmethod
        def start_agent_server() -> FakeServer:
            return FakeServer()

        @staticmethod
        def wait_until_ready() -> None:
            return None

        @staticmethod
        def stop_agent_server(_server: object) -> None:
            return None

    def fake_post(_patient_ids: list[str], port: int) -> dict:
        assert port == canary_guard.AGENT_PORT
        if failure_kind == "http_500":
            raise urllib.error.HTTPError(
                url=f"http://127.0.0.1:{port}/test",
                code=500,
                msg="server failure",
                hdrs=None,
                fp=None,
            )
        if failure_kind == "service_exception":
            run_dir.mkdir(parents=True)
            raise RuntimeError("service unavailable")
        write_jsonl(
            run_dir / "final_results.jsonl",
            [{"patient_id": patient_ids[0], "finished": True}],
        )
        return {"result": {"output_dir": str(run_dir.relative_to(tmp_path))}}

    monkeypatch.setattr(canary_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(canary_guard, "port_is_open", lambda _port: False)
    monkeypatch.setattr(canary_guard, "load_local_test_runner", lambda: FakeRunner())
    monkeypatch.setattr(canary_guard, "runtime_clients", lambda: object())
    monkeypatch.setattr(canary_guard, "post_test", fake_post)

    with pytest.raises((urllib.error.HTTPError, RuntimeError)) as raised:
        canary_guard.command_run(SimpleNamespace(selection=str(selection_path), batch="canary", gate=None))

    attempts = list((tmp_path / "outputs" / "run_attempts").glob("*.json"))
    attempt_paths = [path for path in attempts if not path.name.endswith(".failure.json")]
    failure_paths = list((tmp_path / "outputs" / "run_attempts").glob("*.failure.json"))
    assert len(attempt_paths) == 1
    assert len(failure_paths) == 1
    attempt_path = attempt_paths[0]
    failure_path = failure_paths[0]
    _attempt, attempt_hash = canary_guard.read_json_snapshot(attempt_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert type(raised.value).__name__ == expected_error_type
    assert failure["selection"] == str(selection_path.resolve())
    assert failure["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert failure["attempt"] == str(attempt_path.resolve())
    assert failure["attempt_sha256"] == attempt_hash
    assert failure["batch"] == "canary"
    assert failure["patient_ids"] == patient_ids
    assert failure["error_type"] == expected_error_type
    assert failure["error"]
    assert failure["run_dir"] == (str(run_dir.resolve()) if run_dir_present else "")
    assert failure["port_released"] is True
    assert failure["evaluation_performed"] is False
    assert (tmp_path / "outputs" / "patient_attempts" / "Patient_00001.claim").exists()
    assert not (run_dir / "run_receipt.json").exists()

    with pytest.raises(FileExistsError):
        canary_guard.write_json_exclusive(failure_path, failure)
