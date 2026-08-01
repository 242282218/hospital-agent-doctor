from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, Tuple

from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json
from offline.case_memory import case_memory_candidate, extract_case_memory
from offline.candidates import load_candidate, write_candidate


_SOURCE_FILES = ("evaluation_results.jsonl", "events.jsonl", "final_results.jsonl")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSONL at %s:%d" % (path, line_number)) from exc
        if not isinstance(value, dict):
            raise ValueError("JSONL row must be an object at %s:%d" % (path, line_number))
        value = dict(value)
        value["_source_line"] = line_number
        rows.append(value)
    return rows


def _without_source_line(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_source_line"}


def _aware_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("evaluation timestamp required")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid evaluation timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("evaluation timestamp must include timezone")
    return timestamp


def _unique_event(events: Iterable[Mapping[str, Any]], event_type: str, patient_id: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in events
        if item.get("event_type") == event_type and item.get("patient_id") == patient_id
    ]
    if len(matches) != 1:
        raise ValueError("expected one %s for %s" % (event_type, patient_id))
    event = matches[0]
    if event.get("status") != "success":
        raise ValueError("event not successful: %s" % event_type)
    payload = event.get("payload")
    if not isinstance(payload, Mapping) or payload.get("patient_id") != patient_id:
        raise ValueError("event patient_id mismatch: %s" % event_type)
    return event


def _ground_truth_key(effect: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        tuple(effect.get("diagnoses") or []),
        tuple(effect.get("examinations") or []),
        effect.get("treatment_plan"),
    )


def _trusted_runs(trust_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    runs = trust_manifest.get("historical_runs")
    if not isinstance(runs, list):
        raise ValueError("trust manifest historical_runs required")
    return [
        item
        for item in runs
        if isinstance(item, Mapping)
        and item.get("mode") == "train"
        and item.get("has_evaluation") is True
    ]


def scan_trusted_train_evaluations(
    *,
    train_outputs: Path,
    trust_manifest_path: Path,
    official_diseases: Collection[str],
    valid_examinations: Collection[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    train_outputs = Path(train_outputs)
    trust_manifest_path = Path(trust_manifest_path)
    trust_manifest = read_json(trust_manifest_path)
    if not isinstance(trust_manifest, Mapping):
        raise ValueError("invalid trust manifest")

    records: List[Dict[str, Any]] = []
    trusted_run_receipts = []
    for run in _trusted_runs(trust_manifest):
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise ValueError("trusted run_id required")
        run_dir = train_outputs / run_id
        artifact_hashes = run.get("artifact_hashes")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError("trusted artifact hashes required")
        actual_hashes = {}
        for name in _SOURCE_FILES:
            path = run_dir / name
            if not path.exists():
                raise FileNotFoundError("trusted source artifact missing: %s" % path)
            actual_hash = file_hash(path)
            actual_hashes[name] = actual_hash
            if actual_hash != artifact_hashes.get(name):
                raise ValueError("source artifact hash mismatch: %s/%s" % (run_id, name))

        evaluations = _read_jsonl(run_dir / "evaluation_results.jsonl")
        events = _read_jsonl(run_dir / "events.jsonl")
        finals = _read_jsonl(run_dir / "final_results.jsonl")
        final_by_patient: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for final in finals:
            final_by_patient[str(final.get("patient_id") or "")].append(final)

        for evaluation in evaluations:
            patient_id = evaluation.get("patient_id")
            if not isinstance(patient_id, str) or not patient_id:
                raise ValueError("evaluation patient_id required")
            if len(final_by_patient[patient_id]) != 1:
                raise ValueError("expected one final result for %s" % patient_id)
            final = _without_source_line(final_by_patient[patient_id][0])
            if final.get("finished") is not True:
                raise ValueError("final result not finished: %s" % patient_id)
            request = _unique_event(events, "EVALUATION_REQUEST", patient_id)
            result = _unique_event(events, "EVALUATION_RESULT", patient_id)
            case_end = _unique_event(events, "CASE_END", patient_id)
            if (case_end.get("payload") or {}).get("finished") is not True:
                raise ValueError("case not finished: %s" % patient_id)
            if (request.get("payload") or {}).get("final_result") != final:
                raise ValueError("evaluation request final result mismatch: %s" % patient_id)
            report = evaluation.get("report")
            if not isinstance(report, Mapping) or report.get("status") != "evaluated":
                raise ValueError("evaluation report status must be evaluated")
            if report.get("patientId") != patient_id:
                raise ValueError("evaluation patient_id mismatch")
            if (result.get("payload") or {}).get("report") != report:
                raise ValueError("evaluation result report mismatch: %s" % patient_id)
            timestamp = _aware_timestamp(evaluation.get("timestamp"))
            evaluation_value = _without_source_line(evaluation)
            effect = extract_case_memory(
                patient_id=patient_id,
                evaluation=evaluation_value,
                official_diseases=official_diseases,
                valid_examinations=valid_examinations,
            )
            records.append(
                {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "timestamp_text": evaluation["timestamp"],
                    "source_run_id": run_id,
                    "source_line": evaluation["_source_line"],
                    "evaluation": evaluation_value,
                    "evaluation_hash": content_hash(evaluation_value),
                    "effect": effect,
                    "source_hashes": actual_hashes,
                }
            )
        trusted_run_receipts.append({"run_id": run_id, "artifact_hashes": actual_hashes})

    receipt = {
        "schema_version": "case-memory-source-receipt/v1",
        "trust_manifest_hash": file_hash(trust_manifest_path),
        "trusted_runs": sorted(trusted_run_receipts, key=lambda item: item["run_id"]),
    }
    return records, receipt


def select_latest_evaluations(records: Iterable[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["patient_id"])].append(record)
    selected: List[Dict[str, Any]] = []
    for patient_id, items in grouped.items():
        truth_keys = {_ground_truth_key(item["effect"]) for item in items}
        if len(truth_keys) != 1:
            raise ValueError("ground truth conflict for %s" % patient_id)
        latest_time = max(item["timestamp"] for item in items)
        latest = [item for item in items if item["timestamp"] == latest_time]
        latest_hashes = {item["evaluation_hash"] for item in latest}
        if len(latest_hashes) != 1:
            raise ValueError("ambiguous latest evaluation for %s" % patient_id)
        selected.append(dict(sorted(latest, key=lambda item: (item["source_run_id"], item["source_line"]))[-1]))
    selected.sort(key=lambda item: item["patient_id"])
    return selected, sum(len(items) - 1 for items in grouped.values())


def _verify_existing(path: Path, value: Mapping[str, Any]) -> None:
    if read_json(path) != dict(value):
        raise FileExistsError("immutable artifact differs: %s" % path)


def import_case_memory_candidates(
    *,
    train_outputs: Path,
    trust_manifest_path: Path,
    artifact_root: Path,
    official_diseases: Collection[str],
    valid_examinations: Collection[str],
    catalog_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    records, source_receipt = scan_trusted_train_evaluations(
        train_outputs=train_outputs,
        trust_manifest_path=trust_manifest_path,
        official_diseases=official_diseases,
        valid_examinations=valid_examinations,
    )
    selected, superseded_count = select_latest_evaluations(records)
    selection_core = {
        "schema_version": "case-memory-selection/v1",
        "trust_manifest_hash": source_receipt["trust_manifest_hash"],
        "catalog_hashes": dict(sorted(catalog_hashes.items())),
        "selections": [
            {
                "patient_id": item["patient_id"],
                "timestamp": item["timestamp_text"],
                "source_run_id": item["source_run_id"],
                "source_line": item["source_line"],
                "evaluation_hash": item["evaluation_hash"],
                "source_hashes": item["source_hashes"],
            }
            for item in selected
        ],
    }
    import_id = content_hash(selection_core)
    batch_dir = Path(artifact_root) / "imports" / import_id
    manifest_path = batch_dir / "import_manifest.json"

    manifest_selections = []
    for item in selected:
        evaluation_ref = "sha256/%s.json" % item["evaluation_hash"]
        candidate = case_memory_candidate(
            patient_id=item["patient_id"],
            evaluation=item["evaluation"],
            evaluation_ref=evaluation_ref,
            official_diseases=official_diseases,
            valid_examinations=valid_examinations,
        )
        manifest_selections.append(
            {
                "patient_id": item["patient_id"],
                "timestamp": item["timestamp_text"],
                "source_run_id": item["source_run_id"],
                "source_line": item["source_line"],
                "evaluation_hash": item["evaluation_hash"],
                "evaluation_ref": evaluation_ref,
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "effect_hash": candidate["effect_hash"],
            }
        )

    manifest = {
        "schema_version": "case-memory-import/v1",
        "import_id": import_id,
        "raw_evaluation_count": len(records),
        "selected_count": len(selected),
        "superseded_count": superseded_count,
        "ground_truth_conflict_count": 0,
        "catalog_hashes": dict(sorted(catalog_hashes.items())),
        "selections": manifest_selections,
    }
    manifest["manifest_hash"] = content_hash(manifest)

    if manifest_path.exists():
        _verify_existing(manifest_path, manifest)
        _verify_existing(batch_dir / "source_receipt.json", source_receipt)
        for item, selection in zip(selected, manifest_selections):
            _verify_existing(batch_dir / "evaluations" / selection["evaluation_ref"], item["evaluation"])
            candidate = load_candidate(
                batch_dir / "candidates" / (selection["candidate_id"] + ".json")
            )
            if candidate.get("candidate_hash") != selection["candidate_hash"]:
                raise ValueError("candidate hash mismatch in existing import batch")
        return {"import_id": import_id, "batch_dir": str(batch_dir), "reused": True, "manifest": manifest}
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError("incomplete import batch already exists: %s" % batch_dir)

    for item, selection in zip(selected, manifest_selections):
        write_immutable_json(batch_dir / "evaluations" / selection["evaluation_ref"], item["evaluation"])
        candidate = case_memory_candidate(
            patient_id=item["patient_id"],
            evaluation=item["evaluation"],
            evaluation_ref=selection["evaluation_ref"],
            official_diseases=official_diseases,
            valid_examinations=valid_examinations,
        )
        write_candidate(batch_dir / "candidates" / (candidate["candidate_id"] + ".json"), candidate)
        loaded = load_candidate(batch_dir / "candidates" / (candidate["candidate_id"] + ".json"))
        if loaded.get("status") != "candidate":
            raise ValueError("case-memory candidate was quarantined")
    (batch_dir / "decisions").mkdir(parents=True, exist_ok=True)
    write_immutable_json(batch_dir / "source_receipt.json", source_receipt)
    write_immutable_json(manifest_path, manifest)
    return {"import_id": import_id, "batch_dir": str(batch_dir), "reused": False, "manifest": manifest}
