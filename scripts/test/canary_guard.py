from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import re
import socket
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.legacy_orchestrator import parse_json_object
from hospital_agent_sdk import build_service_clients, load_config, runtime_config_from_project_config


PATIENT_ID_PATTERN = re.compile(r"Patient_[A-Za-z0-9_-]+")
LLM_BLOCK_PATTERN = re.compile(r"(?m)^timestamp:\s*")
AGENT_PORT = 7860
EVALUATION_SCORE_KEYS = [
    "diagnosis_accuracy",
    "examination_precision",
    "treatment_overall_score",
    "treatment_safety",
    "treatment_effectiveness_alignment",
    "treatment_personalization",
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return parse_jsonl_bytes(path.read_bytes())


def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def historical_patient_snapshot(output_root: Path) -> tuple[list[str], str]:
    patient_ids: set[str] = set()
    for mode in ["test", "train"]:
        mode_root = output_root / mode
        if not mode_root.exists():
            continue
        for name in ["events.jsonl", "final_results.jsonl"]:
            for path in mode_root.rglob(name):
                patient_ids.update(PATIENT_ID_PATTERN.findall(path.read_text(encoding="utf-8", errors="ignore")))
    for path in (output_root / "run_attempts").glob("*.json"):
        try:
            attempt_ids = read_json(path).get("patient_ids")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid run attempt artifact: {path}") from exc
        if not isinstance(attempt_ids, list) or not attempt_ids:
            raise RuntimeError(f"invalid run attempt artifact: {path}")
        patient_ids.update(str(item) for item in attempt_ids)
    for path in (output_root / "patient_attempts").glob("*.claim"):
        patient_id = path.stem
        if PATIENT_ID_PATTERN.fullmatch(patient_id) is None:
            raise RuntimeError(f"invalid patient attempt claim: {path}")
        patient_ids.add(patient_id)
    ordered = sorted(patient_ids)
    serialized = ("\n".join(ordered) + ("\n" if ordered else "")).encode("utf-8")
    return ordered, hashlib.sha256(serialized).hexdigest()


def freeze_selection(
    *,
    candidate_ids: list[str],
    history_ids: list[str],
    history_sha256: str,
    output_path: Path,
    seed: int,
    expected_history_count: int,
    expected_history_sha256: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    if len(history_ids) != expected_history_count or history_sha256 != expected_history_sha256:
        raise RuntimeError("historical patient snapshot changed; inspect before selecting new cases")
    history = set(history_ids)
    unseen = []
    for patient_id in candidate_ids:
        if patient_id not in history and patient_id not in unseen:
            unseen.append(patient_id)
    if len(unseen) < 10:
        raise RuntimeError(f"only {len(unseen)} unseen patients available; need 10")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "history_count": len(history_ids),
        "history_sha256": history_sha256,
        "patient_ids": unseen[:10],
        "canary_ids": unseen[:3],
        "confirmation_ids": unseen[3:10],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def split_llm_blocks(text: str) -> list[str]:
    starts = [match.start() for match in LLM_BLOCK_PATTERN.finditer(text)]
    return [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]


def llm_block_fields(block: str) -> tuple[str, str, str, bool]:
    prompt_name_match = re.search(r"(?m)^prompt_name:\s*(\S+)", block)
    patient_match = re.search(r"(?m)^patient_id:\s*(\S+)", block)
    response_match = re.search(r"(?s)\nresponse:\n(.*)$", block)
    prompt_match = re.search(r"(?s)\nsystem_prompt:\n(.*?)\n\nuser_prompt:\n(.*?)\n\nresponse:\n", block)
    patient_id = patient_match.group(1) if patient_match else ""
    response = response_match.group(1).strip() if response_match else ""
    response = re.sub(r"\n={20,}\s*$", "", response).strip()
    prompt = "\n".join(prompt_match.groups()) if prompt_match else ""
    complete = bool(prompt_name_match and patient_id and prompt.strip() and response_match and response)
    return patient_id, prompt, response, complete


def approximate_tokens(text: str) -> float:
    non_ascii = sum(ord(char) > 127 for char in text)
    ascii_count = len(text) - non_ascii
    return non_ascii + ascii_count / 4


def collect_run_metrics(run_dir: Path, *, expected_ids: list[str]) -> dict[str, Any]:
    finals = read_jsonl(run_dir / "final_results.jsonl")
    final_ids = [str(item.get("patient_id") or "") for item in finals]
    if final_ids != expected_ids:
        raise RuntimeError(f"final patient IDs differ from selection: {final_ids!r}")
    per_patient = {
        patient_id: {
            "send_messages": 0,
            "conversation_rounds": 0,
            "scheduled_exams": 0,
            "invalid_exams": 0,
            "case_end_events": 0,
            "treatment_events": 0,
            "llm_calls_min": 0,
        }
        for patient_id in expected_ids
    }
    event_totals = count_event_metrics(run_dir / "events.jsonl", per_patient)
    for item in finals:
        patient_id = str(item.get("patient_id") or "")
        if patient_id in per_patient:
            per_patient[patient_id]["conversation_rounds"] = int(item.get("conversation_rounds") or 0)
    llm_totals = count_llm_metrics(run_dir / "llm_prompts", per_patient)
    return {
        "run_dir": str(run_dir),
        "patient_ids": final_ids,
        "expected_count": len(expected_ids),
        "finished_count": sum(item.get("finished") is True for item in finals),
        "conversation_rounds": sum(int(item.get("conversation_rounds") or 0) for item in finals),
        "ordered_exam_count": sum(len(item.get("ordered_examinations") or []) for item in finals),
        **event_totals,
        **llm_totals,
        "per_patient": per_patient,
    }


def count_event_metrics(events_path: Path, per_patient: dict[str, dict[str, int]]) -> dict[str, int]:
    totals = {
        "send_message_count": 0,
        "scheduled_exam_count": 0,
        "invalid_exam_count": 0,
        "exam_action_count": 0,
        "case_end_count": 0,
        "treatment_event_count": 0,
    }
    for event in read_jsonl(events_path):
        event_type = str(event.get("event_type") or "")
        patient_id = str(event.get("patient_id") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "SEND_MESSAGE":
            totals["send_message_count"] += 1
            increment_patient(per_patient, patient_id, "send_messages", 1)
        elif event_type == "SCHEDULE_EXAMINATION":
            count = len(payload.get("items") or [])
            totals["scheduled_exam_count"] += count
            increment_patient(per_patient, patient_id, "scheduled_exams", count)
        elif event_type == "DO_EXAMINATION":
            totals["exam_action_count"] += len(payload.get("items") or [])
            count = len(payload.get("invalid_items") or [])
            totals["invalid_exam_count"] += count
            increment_patient(per_patient, patient_id, "invalid_exams", count)
        elif event_type == "CASE_END":
            totals["case_end_count"] += 1
            increment_patient(per_patient, patient_id, "case_end_events", 1)
        elif event_type == "PRESCRIBE_TREATMENT":
            totals["treatment_event_count"] += 1
            increment_patient(per_patient, patient_id, "treatment_events", 1)
    return totals


def count_llm_metrics(prompt_dir: Path, per_patient: dict[str, dict[str, int]]) -> dict[str, Any]:
    logged = repairs = unbound = malformed = 0
    token_estimate = 0.0
    paths = sorted(prompt_dir.glob("*.txt")) if prompt_dir.exists() else []
    for path in paths:
        for block in split_llm_blocks(path.read_text(encoding="utf-8", errors="replace")):
            patient_id, prompt, response, complete = llm_block_fields(block)
            logged += 1
            malformed += int(not complete)
            repair = int(parse_json_object(response) is None)
            repairs += repair
            token_estimate += approximate_tokens(prompt + response)
            call_count = 1 + repair
            increment_patient(per_patient, patient_id, "llm_calls_min", call_count)
            if patient_id not in per_patient:
                unbound += call_count
    return {
        "llm_logged_calls": logged,
        "repair_calls_inferred": repairs,
        "llm_calls_min": logged + repairs,
        "unbound_llm_calls": unbound,
        "malformed_llm_blocks": malformed,
        "approximate_tokens": round(token_estimate),
        "prompt_log_bytes": sum(path.stat().st_size for path in paths),
    }


def increment_patient(
    per_patient: dict[str, dict[str, int]],
    patient_id: str,
    field: str,
    amount: int,
) -> None:
    if patient_id in per_patient:
        per_patient[patient_id][field] += amount


def create_batch_evaluation_marker(
    run_dir: Path,
    marker_root: Path,
    *,
    run_receipt_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Path:
    finals_path = run_dir / "final_results.jsonl"
    finals_raw = finals_path.read_bytes()
    finals = parse_jsonl_bytes(finals_raw)
    if not finals or not all(item.get("finished") is True for item in finals):
        raise RuntimeError("run is unfinished; evaluation is forbidden")
    if (run_dir / "final_results_eval_report.json").exists():
        raise RuntimeError("evaluation report already exists")
    if any(
        item.get("event_type") in {"BATCH_EVALUATION_REQUEST", "BATCH_EVALUATION_RESULT"}
        for item in read_jsonl(run_dir / "events.jsonl")
    ):
        raise RuntimeError("batch evaluation was already requested")
    digest = hashlib.sha256(finals_raw).hexdigest()
    receipt_payload: dict[str, Any] = {}
    marker_id = digest
    if run_receipt_path is not None:
        receipt, receipt_hash = read_json_snapshot(run_receipt_path)
        marker_id = receipt_hash
        receipt_payload = {
            "run_receipt": str(run_receipt_path.resolve()),
            "run_receipt_sha256": receipt_hash,
            "selection_sha256": receipt.get("selection_sha256"),
            "batch": receipt.get("batch"),
            "patient_ids": receipt.get("patient_ids"),
        }
        if metrics_path is None:
            raise RuntimeError("metrics artifact is required for a bound evaluation attempt")
        _metrics, metrics_hash = read_json_snapshot(metrics_path)
        receipt_payload.update(
            {
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": metrics_hash,
                "llm_prompts_sha256": directory_sha256(run_dir / "llm_prompts"),
            }
        )
    events_raw = (run_dir / "events.jsonl").read_bytes() if (run_dir / "events.jsonl").exists() else b""
    if events_raw and not events_raw.endswith(b"\n"):
        raise RuntimeError("events log must end with a newline before evaluation")
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_path = marker_root / f"{marker_id}.json"
    if marker_path.exists():
        raise FileExistsError(marker_path)
    reject_other_marker_for_run(marker_root, run_dir)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run_dir),
        "sha256": digest,
        "events_pre_size": len(events_raw),
        "events_pre_sha256": hashlib.sha256(events_raw).hexdigest(),
        **receipt_payload,
    }
    with marker_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    return marker_path


def reject_other_marker_for_run(marker_root: Path, run_dir: Path) -> None:
    normalized_run = str(run_dir.resolve())
    for path in marker_root.glob("*.json"):
        try:
            marked_run = str(Path(str(read_json(path).get("run") or "")).resolve())
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid evaluation marker artifact: {path}") from exc
        if marked_run == normalized_run:
            raise RuntimeError(f"run already has an evaluation marker: {path}")


def selection_sha256(selection_path: Path) -> str:
    return hashlib.sha256(selection_path.read_bytes()).hexdigest()


def create_paired_rerun_manifest(
    *,
    pair_id: str,
    patient_ids: list[str],
    baseline_run_receipts: dict[str, Path],
    candidate_source_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if not pair_id or len(patient_ids) != 4 or len(set(patient_ids)) != len(patient_ids):
        raise RuntimeError("paired rerun requires four unique patient IDs")
    if any(PATIENT_ID_PATTERN.fullmatch(patient_id) is None for patient_id in patient_ids):
        raise RuntimeError("paired rerun manifest contains an invalid patient ID")
    baseline_runs = []
    for patient_id in patient_ids:
        receipt_path = baseline_run_receipts.get(patient_id)
        if receipt_path is None:
            raise RuntimeError("paired rerun is missing baseline evidence for a patient")
        receipt, receipt_hash = read_json_snapshot(receipt_path)
        if patient_id not in receipt.get("patient_ids", []):
            raise RuntimeError("paired rerun baseline receipt does not contain its patient")
        baseline_runs.append(
            {
                "patient_id": patient_id,
                "run_receipt": str(receipt_path.resolve()),
                "run_receipt_sha256": receipt_hash,
                "selection": receipt.get("selection"),
                "selection_sha256": receipt.get("selection_sha256"),
            }
        )
    payload = {
        "schema_version": 1,
        "pair_id": pair_id,
        "patient_ids": patient_ids,
        "baseline_runs": baseline_runs,
        "candidate_source_sha256": candidate_source_sha256,
        "run_policy": {"max_attempts": 1, "evaluation_count": 1, "allow_retry": False},
    }
    write_json_exclusive(output_path, payload)
    return payload


def validate_paired_rerun_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest, manifest_hash = read_json_snapshot(path)
    patient_ids = manifest.get("patient_ids")
    baseline_runs = manifest.get("baseline_runs")
    policy = manifest.get("run_policy") if isinstance(manifest.get("run_policy"), dict) else {}
    if (
        not isinstance(patient_ids, list)
        or len(patient_ids) != 4
        or len(set(patient_ids)) != 4
        or any(PATIENT_ID_PATTERN.fullmatch(str(patient_id)) is None for patient_id in patient_ids)
        or not isinstance(baseline_runs, list)
        or [item.get("patient_id") for item in baseline_runs if isinstance(item, dict)] != patient_ids
        or policy != {"max_attempts": 1, "evaluation_count": 1, "allow_retry": False}
    ):
        raise RuntimeError("paired rerun manifest is invalid")
    for item in baseline_runs:
        receipt_path = Path(str(item.get("run_receipt") or "")).resolve()
        receipt, receipt_hash = read_json_snapshot(receipt_path)
        if receipt_hash != item.get("run_receipt_sha256") or item.get("patient_id") not in receipt.get("patient_ids", []):
            raise RuntimeError("paired rerun baseline receipt changed")
    return manifest, manifest_hash


def create_paired_rerun_attempt(manifest_path: Path, attempt_root: Path) -> Path:
    manifest, manifest_hash = validate_paired_rerun_manifest(manifest_path)
    key = hashlib.sha256(
        f"paired:{manifest['pair_id']}:{manifest_hash}:{','.join(manifest['patient_ids'])}".encode()
    ).hexdigest()
    path = attempt_root / "paired" / f"{key}.json"
    payload = {
        "schema_version": 1,
        "kind": "paired_rerun",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "pair_id": manifest["pair_id"],
        "patient_ids": manifest["patient_ids"],
        "candidate_source_sha256": manifest["candidate_source_sha256"],
    }
    write_json_exclusive(path, payload)
    return path


def create_run_attempt(
    selection_path: Path,
    batch: str,
    attempt_root: Path,
    *,
    gate_path: Path | None = None,
) -> Path:
    manifest, selection_hash = read_json_snapshot(selection_path)
    return create_run_attempt_from_snapshot(
        selection_path,
        manifest,
        selection_hash,
        batch,
        attempt_root,
        gate_path=gate_path,
    )


def create_run_attempt_from_snapshot(
    selection_path: Path,
    manifest: dict[str, Any],
    selection_hash: str,
    batch: str,
    attempt_root: Path,
    *,
    gate_path: Path | None = None,
) -> Path:
    patient_ids = selected_batch(manifest, batch)
    gate_payload: dict[str, Any] = {}
    if batch == "confirmation":
        if gate_path is None or not gate_path.exists():
            raise RuntimeError("confirmation requires a verified Canary gate")
        gate_hash = validate_passed_canary_gate(
            gate_path,
            selection_path,
            manifest,
            selection_hash,
        )
        gate_payload = {"gate": str(gate_path.resolve()), "gate_sha256": gate_hash}
    key = hashlib.sha256(f"{batch}:{','.join(sorted(patient_ids))}".encode()).hexdigest()
    attempt_root.mkdir(parents=True, exist_ok=True)
    path = attempt_root / f"{key}.json"
    if path.exists():
        raise FileExistsError(path)
    claim_patient_ids(patient_ids, attempt_root.parent / "patient_attempts")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path.resolve()),
        "selection_sha256": selection_hash,
        "batch": batch,
        "patient_ids": patient_ids,
        **gate_payload,
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def claim_patient_ids(patient_ids: list[str], claim_root: Path) -> None:
    claim_root.mkdir(parents=True, exist_ok=True)
    for patient_id in patient_ids:
        path = claim_root / f"{patient_id}.claim"
        try:
            path.open("x").close()
        except FileExistsError as exc:
            raise RuntimeError(f"patient ID already has a run attempt claim: {patient_id}") from exc


def validate_evaluation_binding(
    run_dir: Path,
    selection: dict[str, Any],
    batch: str,
    metrics: dict[str, Any],
) -> None:
    expected_ids = selected_batch(selection, batch)
    final_ids = [str(item.get("patient_id") or "") for item in read_jsonl(run_dir / "final_results.jsonl")]
    if final_ids != expected_ids:
        raise RuntimeError("final patient IDs differ from selection")
    if metrics.get("patient_ids") != expected_ids:
        raise RuntimeError("metrics patient IDs differ from selection")
    if Path(str(metrics.get("run_dir") or "")).resolve() != run_dir.resolve():
        raise RuntimeError("metrics run directory differs from evaluation run")
    if int(metrics.get("finished_count") or 0) != len(expected_ids):
        raise RuntimeError("metrics show an unfinished run")
    recomputed = collect_run_metrics(run_dir.resolve(), expected_ids=expected_ids)
    if metrics != recomputed:
        raise RuntimeError("metrics artifact differs from run logs")


def combine_batch_results(
    batches: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    p0_count: int,
) -> dict[str, Any]:
    batch_sizes = [len(metrics.get("patient_ids", [])) for metrics, _ in batches]
    if batch_sizes != [3, 7]:
        raise RuntimeError("combine requires a 3-case Canary and 7-case confirmation")
    patient_ids = [patient_id for metrics, _ in batches for patient_id in metrics.get("patient_ids", [])]
    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError("duplicate patient IDs across batches")
    for metrics, report in batches:
        validate_report_binding(report, [str(item) for item in metrics["patient_ids"]])
    total = sum(int(report.get("counts", {}).get("evaluated_patients") or 0) for _, report in batches)
    if total != len(patient_ids):
        raise RuntimeError("evaluation counts differ from batch patient IDs")
    raw_scores = {
        key: sum(
            float(report[key]) * int(report["counts"]["evaluated_patients"])
            for _, report in batches
        )
        / total
        for key in EVALUATION_SCORE_KEYS
    }
    result = {key: round(value, 4) for key, value in raw_scores.items()}
    details = [item for _, report in batches for item in report.get("treatment_details", [])]
    result.update(
        {
            "patient_ids": patient_ids,
            "evaluated_patients": total,
            "minimum_patient_safety": min((float(item.get("safety") or 0) for item in details), default=0.0),
            "invalid_exam_count": sum(int(metrics.get("invalid_exam_count") or 0) for metrics, _ in batches),
            "llm_calls_min": sum(int(metrics.get("llm_calls_min") or 0) for metrics, _ in batches),
            "unbound_llm_calls": sum(int(metrics.get("unbound_llm_calls") or 0) for metrics, _ in batches),
            "malformed_llm_blocks": sum(int(metrics.get("malformed_llm_blocks") or 0) for metrics, _ in batches),
            "p0_count": p0_count,
        }
    )
    checks = {
        "patient_count": total == 10,
        "diagnosis_accuracy": raw_scores["diagnosis_accuracy"] >= 0.50,
        "examination_precision": raw_scores["examination_precision"] >= 0.50,
        "treatment_overall_score": raw_scores["treatment_overall_score"] >= 0.55,
        "treatment_safety": raw_scores["treatment_safety"] >= 0.90,
        "minimum_patient_safety": result["minimum_patient_safety"] >= 0.80,
        "invalid_exam_count": result["invalid_exam_count"] == 0,
        "p0_count": p0_count == 0,
        "llm_calls_min": result["llm_calls_min"] <= 80,
        "unbound_llm_calls": result["unbound_llm_calls"] == 0,
        "malformed_llm_blocks": result["malformed_llm_blocks"] == 0,
    }
    observation_checks = [{**llm_observation_checks(metrics), **run_observation_checks(metrics)} for metrics, _ in batches]
    for name in observation_checks[0]:
        checks[name] = all(item[name] for item in observation_checks)
    result["passed"] = all(checks.values())
    result["failed_checks"] = [name for name, passed in checks.items() if not passed]
    return result


def validate_report_binding(report: dict[str, Any], expected_ids: list[str]) -> None:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    if int(counts.get("final_results") or 0) != len(expected_ids):
        raise RuntimeError("report final result count differs from selection")
    if int(counts.get("evaluated_patients") or 0) != len(expected_ids):
        raise RuntimeError("report evaluated patient count differs from selection")
    details = report.get("treatment_details") if isinstance(report.get("treatment_details"), list) else []
    detail_ids = [str(item.get("patient_id") or "") for item in details if isinstance(item, dict)]
    if len(detail_ids) != len(set(detail_ids)) or sorted(detail_ids) != sorted(expected_ids):
        raise RuntimeError("report treatment patient IDs differ from selection")
    if any(not unit_interval_score(report.get(key)) for key in EVALUATION_SCORE_KEYS):
        raise RuntimeError("evaluation report contains an invalid aggregate score")
    if any(not unit_interval_score(item.get("safety")) for item in details if isinstance(item, dict)):
        raise RuntimeError("evaluation report contains an invalid patient safety score")


def unit_interval_score(value: Any) -> bool:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and 0.0 <= score <= 1.0


def llm_observation_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    per_patient = metrics.get("per_patient") if isinstance(metrics.get("per_patient"), dict) else {}
    patient_calls = [int(item.get("llm_calls_min") or 0) for item in per_patient.values() if isinstance(item, dict)]
    total = int(metrics.get("llm_calls_min") or 0)
    logged = int(metrics.get("llm_logged_calls") or 0)
    repairs = int(metrics.get("repair_calls_inferred") or 0)
    return {
        "llm_logs_present": logged > 0 and int(metrics.get("prompt_log_bytes") or 0) > 0,
        "llm_calls_per_patient_min": bool(patient_calls) and all(count >= 1 for count in patient_calls),
        # Accuracy-first full loop (axis consult + treatment review) often needs >8.
        "llm_calls_per_patient_max": bool(patient_calls) and all(count <= 12 for count in patient_calls),
        "llm_call_accounting": sum(patient_calls) == total == logged + repairs,
        "malformed_llm_blocks": int(metrics.get("malformed_llm_blocks") or 0) == 0,
    }


def run_observation_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    patient_ids = metrics.get("patient_ids") if isinstance(metrics.get("patient_ids"), list) else []
    per_patient = metrics.get("per_patient") if isinstance(metrics.get("per_patient"), dict) else {}
    return {
        "case_end_events": int(metrics.get("case_end_count") or 0) == len(patient_ids)
        and all(int(per_patient.get(patient_id, {}).get("case_end_events") or 0) == 1 for patient_id in patient_ids),
        "treatment_events": int(metrics.get("treatment_event_count") or 0) == len(patient_ids)
        and all(int(per_patient.get(patient_id, {}).get("treatment_events") or 0) == 1 for patient_id in patient_ids),
        "conversation_events": all(
            int(per_patient.get(patient_id, {}).get("send_messages") or 0)
            == int(per_patient.get(patient_id, {}).get("conversation_rounds") or 0)
            for patient_id in patient_ids
        ),
        "examination_events": int(metrics.get("scheduled_exam_count") or 0)
        == int(metrics.get("exam_action_count") or 0)
        == int(metrics.get("ordered_exam_count") or 0),
    }


def require_observable_llm_metrics(metrics: dict[str, Any]) -> None:
    checks = {**llm_observation_checks(metrics), **run_observation_checks(metrics)}
    if (
        not all(checks.values())
        or int(metrics.get("unbound_llm_calls") or 0) != 0
    ):
        raise RuntimeError("LLM observation is incomplete, unbound, malformed, or over budget")


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()) if path.exists() else []:
        digest.update(str(file_path.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_evaluation_event_append(run_dir: Path, marker: dict[str, Any], report: dict[str, Any]) -> str:
    events_raw = (run_dir / "events.jsonl").read_bytes()
    pre_size = int(marker.get("events_pre_size") or 0)
    if pre_size > len(events_raw) or hashlib.sha256(events_raw[:pre_size]).hexdigest() != marker.get("events_pre_sha256"):
        raise RuntimeError("evaluation events no longer preserve the pre-evaluation prefix")
    appended = parse_jsonl_bytes(events_raw[pre_size:])
    if len(appended) != 2:
        raise RuntimeError("evaluation must append exactly one request and one result event")
    request, result = appended
    request_payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
    result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if (
        request.get("event_type") != "BATCH_EVALUATION_REQUEST"
        or request.get("status") != "success"
        or result.get("event_type") != "BATCH_EVALUATION_RESULT"
        or result.get("status") != "success"
        or Path(str(request_payload.get("output_log_path") or "")).resolve() != run_dir
        or Path(str(result_payload.get("output_log_path") or "")).resolve() != run_dir
        or result_payload.get("report") != report
    ):
        raise RuntimeError("evaluation request/result events do not match this run and report")
    return hashlib.sha256(events_raw).hexdigest()


def create_evaluation_receipt(
    run_dir: Path,
    marker_path: Path,
    report_path: Path,
    returned_report: dict[str, Any],
) -> Path:
    run_dir = require_test_run_dir(run_dir)
    marker, marker_hash = read_json_snapshot(marker_path)
    canonical_report_path = (run_dir / "final_results_eval_report.json").resolve()
    if report_path.resolve() != canonical_report_path:
        raise RuntimeError("report must be the canonical evaluation report for this run")
    report, report_hash = read_json_snapshot(canonical_report_path)
    expected_ids = [str(item) for item in marker.get("patient_ids", [])]
    validate_report_binding(report, expected_ids)
    if returned_report != report:
        raise RuntimeError("returned evaluation report differs from canonical report")
    run_receipt_path = Path(str(marker.get("run_receipt") or "")).resolve()
    _run_receipt, run_receipt_hash = read_json_snapshot(run_receipt_path)
    metrics_path = Path(str(marker.get("metrics") or "")).resolve()
    _metrics, metrics_hash = read_json_snapshot(metrics_path)
    if (
        Path(str(marker.get("run") or "")).resolve() != run_dir
        or marker.get("sha256") != hashlib.sha256((run_dir / "final_results.jsonl").read_bytes()).hexdigest()
        or marker.get("run_receipt_sha256") != run_receipt_hash
        or marker.get("metrics_sha256") != metrics_hash
        or marker.get("llm_prompts_sha256") != directory_sha256(run_dir / "llm_prompts")
    ):
        raise RuntimeError("evaluation attempt marker no longer matches run evidence")
    events_post_hash = validate_evaluation_event_append(run_dir, marker, report)
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_receipt": str(run_receipt_path),
        "run_receipt_sha256": run_receipt_hash,
        "evaluation_attempt": str(marker_path.resolve()),
        "evaluation_attempt_sha256": marker_hash,
        "metrics": str(metrics_path),
        "metrics_sha256": metrics_hash,
        "report": str(canonical_report_path),
        "report_sha256": report_hash,
        "returned_report_sha256": canonical_json_sha256(returned_report),
        "events_post_sha256": events_post_hash,
        "llm_prompts_sha256": marker.get("llm_prompts_sha256"),
        "selection_sha256": marker.get("selection_sha256"),
        "batch": marker.get("batch"),
        "patient_ids": expected_ids,
    }
    path = run_dir / "evaluation_receipt.json"
    write_json_exclusive(path, payload)
    return path


def validate_evaluation_receipt(
    run_dir: Path,
    selection_path: Path,
    selection: dict[str, Any],
    selection_hash: str,
    batch: str,
    metrics_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], str, str]:
    run_dir = require_test_run_dir(run_dir)
    _run_receipt, run_receipt_hash = validate_run_receipt(run_dir, selection_path, selection, selection_hash, batch)
    receipt_path = run_dir / "evaluation_receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("evaluation completion receipt is missing")
    receipt, receipt_hash = read_json_snapshot(receipt_path)
    metrics, metrics_hash = read_json_snapshot(metrics_path)
    validate_evaluation_binding(run_dir, selection, batch, metrics)
    require_observable_llm_metrics(metrics)
    marker_path = Path(str(receipt.get("evaluation_attempt") or "")).resolve()
    marker, marker_hash = read_json_snapshot(marker_path)
    canonical_report_path = (run_dir / "final_results_eval_report.json").resolve()
    if report_path.resolve() != canonical_report_path:
        raise RuntimeError("report must be the canonical evaluation report for this run")
    report, report_hash = read_json_snapshot(canonical_report_path)
    validate_report_binding(report, selected_batch(selection, batch))
    events_post_hash = validate_evaluation_event_append(run_dir, marker, report)
    if (
        receipt.get("run_dir") != str(run_dir)
        or receipt.get("run_receipt_sha256") != run_receipt_hash
        or receipt.get("evaluation_attempt_sha256") != marker_hash
        or receipt.get("metrics") != str(metrics_path.resolve())
        or receipt.get("metrics_sha256") != metrics_hash
        or receipt.get("report") != str(canonical_report_path)
        or receipt.get("report_sha256") != report_hash
        or receipt.get("returned_report_sha256") != canonical_json_sha256(report)
        or receipt.get("events_post_sha256") != events_post_hash
        or receipt.get("llm_prompts_sha256") != directory_sha256(run_dir / "llm_prompts")
        or receipt.get("selection_sha256") != selection_hash
        or receipt.get("batch") != batch
        or receipt.get("patient_ids") != selected_batch(selection, batch)
        or Path(str(marker.get("run") or "")).resolve() != run_dir
        or marker.get("sha256") != hashlib.sha256((run_dir / "final_results.jsonl").read_bytes()).hexdigest()
        or marker.get("run_receipt_sha256") != run_receipt_hash
        or marker.get("selection_sha256") != selection_hash
        or marker.get("batch") != batch
        or marker.get("patient_ids") != selected_batch(selection, batch)
        or marker.get("metrics") != str(metrics_path.resolve())
        or marker.get("metrics_sha256") != metrics_hash
        or marker.get("llm_prompts_sha256") != directory_sha256(run_dir / "llm_prompts")
    ):
        raise RuntimeError("evaluation completion receipt no longer matches run evidence")
    return report, report_hash, receipt_hash


def evaluate_canary_gate(metrics: dict[str, Any], report: dict[str, Any], *, p0_count: int) -> dict[str, Any]:
    details = report.get("treatment_details") if isinstance(report.get("treatment_details"), list) else []
    patient_ids = metrics.get("patient_ids") if isinstance(metrics.get("patient_ids"), list) else []
    detail_ids = [str(item.get("patient_id") or "") for item in details]
    per_patient = metrics.get("per_patient") if isinstance(metrics.get("per_patient"), dict) else {}
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    min_safety = min((float(item.get("safety") or 0) for item in details), default=0.0)
    max_llm = max(
        (int(item.get("llm_calls_min") or 0) for item in per_patient.values()),
        default=0,
    )
    checks = {
        "finished": metrics.get("expected_count") == 3 and metrics.get("finished_count") == 3 and len(patient_ids) == 3,
        "per_patient": sorted(per_patient) == sorted(patient_ids),
        "evaluation_counts": counts.get("final_results") == 3 and counts.get("evaluated_patients") == 3,
        "treatment_details": len(detail_ids) == 3 and len(set(detail_ids)) == 3 and sorted(detail_ids) == sorted(patient_ids),
        "invalid_exam_count": int(metrics.get("invalid_exam_count") or 0) == 0,
        "p0_count": p0_count == 0,
        "diagnosis_accuracy": float(report.get("diagnosis_accuracy") or 0) >= 0.33,
        "examination_precision": float(report.get("examination_precision") or 0) >= 0.25,
        "treatment_overall_score": float(report.get("treatment_overall_score") or 0) >= 0.50,
        "treatment_safety": float(report.get("treatment_safety") or 0) >= 0.90,
        "score_ranges": all(
            unit_interval_score(report.get(key))
            for key in ["diagnosis_accuracy", "examination_precision", "treatment_overall_score", "treatment_safety"]
        )
        and all(unit_interval_score(item.get("safety")) for item in details if isinstance(item, dict)),
        "minimum_patient_safety": min_safety >= 0.80,
        "llm_calls_per_patient": max_llm <= 12,
        "unbound_llm_calls": int(metrics.get("unbound_llm_calls") or 0) == 0,
        "malformed_llm_blocks": int(metrics.get("malformed_llm_blocks") or 0) == 0,
        **llm_observation_checks(metrics),
        **run_observation_checks(metrics),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def validate_passed_canary_gate(
    gate_path: Path,
    selection_path: Path,
    selection: dict[str, Any],
    selection_hash: str,
) -> str:
    try:
        gate, gate_hash = read_json_snapshot(gate_path)
        if (
            gate.get("passed") is not True
            or gate.get("batch") != "canary"
            or gate.get("selection_sha256") != selection_hash
            or Path(str(gate.get("selection") or "")).resolve() != selection_path.resolve()
            or gate.get("p0_count") != 0
        ):
            raise RuntimeError("gate metadata mismatch")
        metrics_path = Path(str(gate.get("metrics") or "")).resolve()
        report_path = Path(str(gate.get("report") or "")).resolve()
        metrics, metrics_hash = read_json_snapshot(metrics_path)
        if gate.get("metrics_sha256") != metrics_hash:
            raise RuntimeError("gate metrics hash mismatch")
        run_dir = require_test_run_dir(Path(str(metrics.get("run_dir") or "")))
        report, report_hash, evaluation_receipt_hash = validate_evaluation_receipt(
            run_dir,
            selection_path,
            selection,
            selection_hash,
            "canary",
            metrics_path,
            report_path,
        )
        if gate.get("report_sha256") != report_hash:
            raise RuntimeError("gate report hash mismatch")
        if gate.get("evaluation_receipt_sha256") != evaluation_receipt_hash:
            raise RuntimeError("gate evaluation receipt hash mismatch")
        recomputed = evaluate_canary_gate(metrics, report, p0_count=0)
        if not recomputed["passed"] or gate.get("checks") != recomputed["checks"] or gate.get("failed_checks") != []:
            raise RuntimeError("gate result no longer passes")
        return gate_hash
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("confirmation requires a verified Canary gate bound to this selection") from exc


def load_local_test_runner() -> Any:
    spec = importlib.util.spec_from_file_location("local_test_runner", PROJECT_ROOT / "test.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load local test runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_clients() -> Any:
    config = load_config(PROJECT_ROOT / "config.yaml")
    runtime = runtime_config_from_project_config(config, {"local_test": True})
    return build_service_clients(
        base_url=runtime.service_base_url,
        token=runtime.service_token,
        model_api_key=runtime.model_api_key,
        team_id=runtime.team_id,
        mode="test",
    )


def post_test(patient_ids: list[str], port: int) -> dict[str, Any]:
    body = json.dumps({"patient_ids": patient_ids}).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/test",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("/test returned non-object JSON")
    return payload


def port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port_release(port: int, *, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while port_is_open(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def require_test_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    test_root = (PROJECT_ROOT / "outputs" / "test").resolve()
    if resolved == test_root or not resolved.is_relative_to(test_root):
        raise RuntimeError(f"run directory must be under outputs/test: {resolved}")
    return resolved


@contextmanager
def exclusive_run_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"another Canary run owns the local execution lock: {lock_path}") from exc
    try:
        yield
    finally:
        lock_path.rmdir()


def create_run_receipt(run_dir: Path, attempt_path: Path) -> Path:
    run_dir = require_test_run_dir(run_dir)
    attempt, attempt_hash = read_json_snapshot(attempt_path)
    patient_ids = [str(item) for item in attempt.get("patient_ids", [])]
    finals_path = run_dir / "final_results.jsonl"
    finals = read_jsonl(finals_path)
    final_ids = [str(item.get("patient_id") or "") for item in finals]
    if final_ids != patient_ids or not finals or not all(item.get("finished") is True for item in finals):
        raise RuntimeError("run cannot be bound because final results do not match its attempt")
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "attempt": str(attempt_path.resolve()),
        "attempt_sha256": attempt_hash,
        "selection": attempt.get("selection"),
        "selection_sha256": attempt.get("selection_sha256"),
        "batch": attempt.get("batch"),
        "patient_ids": patient_ids,
        "final_results_sha256": hashlib.sha256(finals_path.read_bytes()).hexdigest(),
    }
    path = run_dir / "run_receipt.json"
    write_json_exclusive(path, payload)
    return path


def create_run_failure_receipt(
    attempt_path: Path,
    error: BaseException,
    *,
    run_dir: Path | None,
    port_released: bool,
) -> Path:
    attempt, attempt_hash = read_json_snapshot(attempt_path)
    payload = {
        "schema_version": 1,
        "kind": "canary_run_failure",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": attempt.get("selection"),
        "selection_sha256": attempt.get("selection_sha256"),
        "attempt": str(attempt_path.resolve()),
        "attempt_sha256": attempt_hash,
        "batch": attempt.get("batch"),
        "patient_ids": attempt.get("patient_ids"),
        "error_type": type(error).__name__,
        "error": str(error),
        "run_dir": str(run_dir.resolve()) if run_dir is not None else "",
        "port_released": port_released,
        "evaluation_performed": False,
    }
    path = attempt_path.with_suffix(".failure.json")
    write_json_exclusive(path, payload)
    return path


def validate_run_receipt(
    run_dir: Path,
    selection_path: Path,
    selection: dict[str, Any],
    selection_hash: str,
    batch: str,
) -> tuple[dict[str, Any], str]:
    run_dir = require_test_run_dir(run_dir)
    receipt_path = run_dir / "run_receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("run receipt is missing")
    receipt, receipt_hash = read_json_snapshot(receipt_path)
    expected_ids = selected_batch(selection, batch)
    finals_path = run_dir / "final_results.jsonl"
    finals = read_jsonl(finals_path)
    final_ids = [str(item.get("patient_id") or "") for item in finals]
    if (
        receipt.get("run_dir") != str(run_dir)
        or receipt.get("selection") != str(selection_path.resolve())
        or receipt.get("selection_sha256") != selection_hash
        or receipt.get("batch") != batch
        or receipt.get("patient_ids") != expected_ids
        or final_ids != expected_ids
        or not all(item.get("finished") is True for item in finals)
        or receipt.get("final_results_sha256") != hashlib.sha256(finals_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("run receipt does not match selection or final results")
    attempt_path = Path(str(receipt.get("attempt") or "")).resolve()
    attempt, attempt_hash = read_json_snapshot(attempt_path)
    if (
        receipt.get("attempt_sha256") != attempt_hash
        or attempt.get("selection") != str(selection_path.resolve())
        or attempt.get("selection_sha256") != selection_hash
        or attempt.get("batch") != batch
        or attempt.get("patient_ids") != expected_ids
    ):
        raise RuntimeError("run receipt does not match its run attempt")
    if batch == "confirmation":
        gate_path = Path(str(attempt.get("gate") or "")).resolve()
        if attempt.get("gate_sha256") != selection_sha256(gate_path):
            raise RuntimeError("confirmation run attempt gate hash changed")
        validate_passed_canary_gate(gate_path, selection_path, selection, selection_hash)
    return receipt, receipt_hash


def write_json_consistent(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def command_history(_args: argparse.Namespace) -> dict[str, Any]:
    patient_ids, digest = historical_patient_snapshot(PROJECT_ROOT / "outputs")
    return {"history_count": len(patient_ids), "history_sha256": digest}


def command_select(args: argparse.Namespace) -> dict[str, Any]:
    history_ids, history_sha = historical_patient_snapshot(PROJECT_ROOT / "outputs")
    if len(history_ids) != args.expected_history_count or history_sha != args.expected_history_sha256:
        raise RuntimeError("historical patient snapshot changed; inspect before selecting new cases")
    candidates = asyncio.run(
        runtime_clients().dataset_client.list_patient_ids(
            patient_count=args.candidate_count,
            random_seed=args.seed,
            selection="random",
        )
    )
    history_ids, history_sha = historical_patient_snapshot(PROJECT_ROOT / "outputs")
    return freeze_selection(
        candidate_ids=candidates,
        history_ids=history_ids,
        history_sha256=history_sha,
        output_path=Path(args.output),
        seed=args.seed,
        expected_history_count=args.expected_history_count,
        expected_history_sha256=args.expected_history_sha256,
    )


def selected_batch(manifest: dict[str, Any], batch: str) -> list[str]:
    if batch not in {"canary", "confirmation"}:
        raise RuntimeError(f"unknown batch: {batch}")
    canary_ids = manifest.get("canary_ids")
    confirmation_ids = manifest.get("confirmation_ids")
    if not isinstance(canary_ids, list) or len(canary_ids) != 3:
        raise RuntimeError("selection manifest must contain exactly 3 Canary patient IDs")
    if not isinstance(confirmation_ids, list) or len(confirmation_ids) != 7:
        raise RuntimeError("selection manifest must contain exactly 7 confirmation patient IDs")
    all_ids = [str(item) for item in canary_ids + confirmation_ids]
    if len(set(all_ids)) != 10:
        raise RuntimeError("selection patient IDs must be unique across 3+7 batches")
    if any(PATIENT_ID_PATTERN.fullmatch(patient_id) is None for patient_id in all_ids):
        raise RuntimeError("selection manifest contains an invalid patient ID")
    return all_ids[:3] if batch == "canary" else all_ids[3:]


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    with exclusive_run_lock(PROJECT_ROOT / "outputs" / "canary_run.lock"):
        return command_run_locked(args)


def command_run_locked(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    if port_is_open(AGENT_PORT):
        raise RuntimeError(f"port {AGENT_PORT} is already in use")
    runner = load_local_test_runner()
    runtime_clients()
    manifest, selection_hash = read_json_snapshot(selection_path)
    patient_ids = selected_batch(manifest, args.batch)
    history_ids, _ = historical_patient_snapshot(PROJECT_ROOT / "outputs")
    if set(patient_ids) & set(history_ids):
        raise RuntimeError("selected patient IDs already exist in history")
    gate_path = Path(args.gate).resolve() if args.gate else None
    if args.batch == "confirmation":
        if gate_path is None:
            raise RuntimeError("confirmation requires a verified Canary gate")
        validate_passed_canary_gate(gate_path, selection_path, manifest, selection_hash)
    server = runner.start_agent_server()
    attempt: Path | None = None
    run_dir: Path | None = None
    failure: Exception | None = None
    test_root = PROJECT_ROOT / "outputs" / "test"
    known_run_dirs = {path.resolve() for path in test_root.iterdir() if path.is_dir()} if test_root.exists() else set()
    try:
        runner.wait_until_ready()
        if server.poll() is not None:
            raise RuntimeError("the started Agent process exited before owning port 7860")
        attempt = create_run_attempt_from_snapshot(
            selection_path,
            manifest,
            selection_hash,
            args.batch,
            PROJECT_ROOT / "outputs" / "run_attempts",
            gate_path=gate_path,
        )
        response = post_test(patient_ids, AGENT_PORT)
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        run_dir = require_test_run_dir(PROJECT_ROOT / str(result.get("output_dir") or ""))
        collect_run_metrics(run_dir, expected_ids=patient_ids)
    except Exception as exc:
        failure = exc
    finally:
        try:
            runner.stop_agent_server(server)
        except Exception as exc:
            if failure is None:
                failure = exc
    port_released = wait_for_port_release(AGENT_PORT)
    if failure is not None and run_dir is None and test_root.exists():
        new_run_dirs = [path.resolve() for path in test_root.iterdir() if path.is_dir() and path.resolve() not in known_run_dirs]
        if len(new_run_dirs) == 1:
            run_dir = new_run_dirs[0]
    if failure is None and not port_released:
        failure = RuntimeError(f"port {AGENT_PORT} remains in use after stopping the service")
    if failure is not None:
        if attempt is not None:
            create_run_failure_receipt(
                attempt,
                failure,
                run_dir=run_dir,
                port_released=port_released,
            )
        raise failure
    if attempt is None or run_dir is None:
        raise RuntimeError("run completed without auditable evidence")
    receipt = create_run_receipt(run_dir, attempt)
    return {
        "run_dir": str(run_dir),
        "patient_ids": patient_ids,
        "attempt_path": str(attempt),
        "run_receipt_path": str(receipt),
    }


def command_metrics(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    manifest, selection_hash = read_json_snapshot(selection_path)
    patient_ids = selected_batch(manifest, args.batch)
    run_dir = require_test_run_dir(Path(args.run_dir))
    validate_run_receipt(run_dir, selection_path, manifest, selection_hash, args.batch)
    metrics = collect_run_metrics(run_dir, expected_ids=patient_ids)
    path = run_dir / f"{args.batch}_metrics.json"
    write_json_consistent(path, metrics)
    return {"metrics_path": str(path), "metrics": metrics}


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_once:
        raise RuntimeError("--confirm-once is required")
    run_dir = require_test_run_dir(Path(args.run_dir))
    selection_path = Path(args.selection).resolve()
    selection, selection_hash = read_json_snapshot(selection_path)
    run_receipt_path = run_dir / "run_receipt.json"
    validate_run_receipt(run_dir, selection_path, selection, selection_hash, args.batch)
    metrics_path = Path(args.metrics).resolve()
    metrics = read_json(metrics_path)
    validate_evaluation_binding(run_dir, selection, args.batch, metrics)
    require_observable_llm_metrics(metrics)
    runner = load_local_test_runner()
    runtime_clients()
    marker = create_batch_evaluation_marker(
        run_dir,
        PROJECT_ROOT / "outputs" / "batch_evaluation_attempts",
        run_receipt_path=run_receipt_path,
        metrics_path=metrics_path,
    )
    report = runner.evaluate_test_output(run_dir)
    report_path = run_dir / "final_results_eval_report.json"
    if not report_path.exists():
        write_json_consistent(report_path, report)
    receipt = create_evaluation_receipt(run_dir, marker, report_path, report)
    return {
        "marker_path": str(marker),
        "report_path": str(report_path),
        "evaluation_receipt_path": str(receipt),
    }


def command_gate(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    selection, selection_hash = read_json_snapshot(selection_path)
    metrics_path = Path(args.metrics).resolve()
    report_path = Path(args.report).resolve()
    metrics, metrics_hash = read_json_snapshot(metrics_path)
    run_dir = require_test_run_dir(Path(str(metrics.get("run_dir") or "")))
    validate_run_receipt(run_dir, selection_path, selection, selection_hash, "canary")
    validate_evaluation_binding(run_dir, selection, "canary", metrics)
    report, report_hash, evaluation_receipt_hash = validate_evaluation_receipt(
        run_dir,
        selection_path,
        selection,
        selection_hash,
        "canary",
        metrics_path,
        report_path,
    )
    result = evaluate_canary_gate(metrics, report, p0_count=args.p0_count)
    result.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "batch": "canary",
            "selection": str(selection_path),
            "selection_sha256": selection_hash,
            "metrics": str(metrics_path),
            "metrics_sha256": metrics_hash,
            "report": str(report_path),
            "report_sha256": report_hash,
            "evaluation_receipt_sha256": evaluation_receipt_hash,
            "p0_count": args.p0_count,
        }
    )
    write_json_exclusive(Path(args.output), result)
    return result


def command_combine(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    selection, selection_hash = read_json_snapshot(selection_path)
    canary_metrics_path = Path(args.canary_metrics).resolve()
    canary_report_path = Path(args.canary_report).resolve()
    confirmation_metrics_path = Path(args.confirmation_metrics).resolve()
    confirmation_report_path = Path(args.confirmation_report).resolve()
    canary_metrics, canary_metrics_hash = read_json_snapshot(canary_metrics_path)
    confirmation_metrics, confirmation_metrics_hash = read_json_snapshot(confirmation_metrics_path)
    canary_run_dir = require_test_run_dir(Path(str(canary_metrics.get("run_dir") or "")))
    confirmation_run_dir = require_test_run_dir(Path(str(confirmation_metrics.get("run_dir") or "")))
    validate_run_receipt(canary_run_dir, selection_path, selection, selection_hash, "canary")
    validate_run_receipt(confirmation_run_dir, selection_path, selection, selection_hash, "confirmation")
    validate_evaluation_binding(
        canary_run_dir,
        selection,
        "canary",
        canary_metrics,
    )
    validate_evaluation_binding(
        confirmation_run_dir,
        selection,
        "confirmation",
        confirmation_metrics,
    )
    canary_report, canary_report_hash, canary_evaluation_receipt_hash = validate_evaluation_receipt(
        canary_run_dir,
        selection_path,
        selection,
        selection_hash,
        "canary",
        canary_metrics_path,
        canary_report_path,
    )
    confirmation_report, confirmation_report_hash, confirmation_evaluation_receipt_hash = validate_evaluation_receipt(
        confirmation_run_dir,
        selection_path,
        selection,
        selection_hash,
        "confirmation",
        confirmation_metrics_path,
        confirmation_report_path,
    )
    result = combine_batch_results(
        [
            (canary_metrics, canary_report),
            (confirmation_metrics, confirmation_report),
        ],
        p0_count=args.p0_count,
    )
    result.update(
        {
            "selection_sha256": selection_hash,
            "canary_metrics_sha256": canary_metrics_hash,
            "canary_report_sha256": canary_report_hash,
            "canary_evaluation_receipt_sha256": canary_evaluation_receipt_hash,
            "confirmation_metrics_sha256": confirmation_metrics_hash,
            "confirmation_report_sha256": confirmation_report_hash,
            "confirmation_evaluation_receipt_sha256": confirmation_evaluation_receipt_hash,
        }
    )
    write_json_exclusive(Path(args.output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard explicit 3-case Canary execution and one-shot evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    history = subparsers.add_parser("history")
    history.set_defaults(handler=command_history)
    select = subparsers.add_parser("select")
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--candidate-count", type=int, default=100)
    select.add_argument("--expected-history-count", type=int, required=True)
    select.add_argument("--expected-history-sha256", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=command_select)
    run = subparsers.add_parser("run")
    run.add_argument("--selection", required=True)
    run.add_argument("--batch", choices=["canary", "confirmation"], required=True)
    run.add_argument("--gate")
    run.set_defaults(handler=command_run)
    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--selection", required=True)
    metrics.add_argument("--batch", choices=["canary", "confirmation"], required=True)
    metrics.add_argument("--run-dir", required=True)
    metrics.set_defaults(handler=command_metrics)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--selection", required=True)
    evaluate.add_argument("--batch", choices=["canary", "confirmation"], required=True)
    evaluate.add_argument("--metrics", required=True)
    evaluate.add_argument("--confirm-once", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--selection", required=True)
    gate.add_argument("--metrics", required=True)
    gate.add_argument("--report", required=True)
    gate.add_argument("--p0-count", type=int, required=True)
    gate.add_argument("--output", required=True)
    gate.set_defaults(handler=command_gate)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--selection", required=True)
    combine.add_argument("--canary-metrics", required=True)
    combine.add_argument("--canary-report", required=True)
    combine.add_argument("--confirmation-metrics", required=True)
    combine.add_argument("--confirmation-report", required=True)
    combine.add_argument("--p0-count", type=int, required=True)
    combine.add_argument("--output", required=True)
    combine.set_defaults(handler=command_combine)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False))
    if args.command in {"gate", "combine"} and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
