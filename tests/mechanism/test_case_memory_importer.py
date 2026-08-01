from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from offline.artifacts import file_hash, read_json
from offline.case_memory_import import import_case_memory_candidates
from offline.candidates import load_candidate


def _evaluation(patient_id: str, timestamp: str, *, diagnosis: str = "三房心") -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "patient_id": patient_id,
        "report": {
            "patientId": patient_id,
            "status": "evaluated",
            "diagnosisDetail": {"expected": [diagnosis]},
            "examinationDetail": {"expected": ["体格检查"]},
            "treatmentDetail": {
                "reference": "尽快完成专科评估。",
                "reasoning": "当前证据支持结构异常。",
            },
            "ground_truth": {
                "final_diagnosis": diagnosis,
                "necessary_examinations": ["体格检查"],
                "treatment_plan": "尽快完成专科评估。",
            },
        },
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_run(
    train_outputs: Path,
    run_id: str,
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    run_dir = train_outputs / run_id
    finals = []
    events = []
    event_index = 0
    for evaluation in evaluations:
        patient_id = evaluation["patient_id"]
        final = {
            "patient_id": patient_id,
            "diagnosis": ["待评估诊断"],
            "treatment_plan": "待评估方案",
            "reasoning": "待评估依据",
            "finished": True,
        }
        finals.append(final)
        event_index += 1
        events.append(
            {
                "event_index": event_index,
                "event_type": "EVALUATION_REQUEST",
                "patient_id": patient_id,
                "status": "success",
                "payload": {"patient_id": patient_id, "final_result": final},
            }
        )
        event_index += 1
        events.append(
            {
                "event_index": event_index,
                "event_type": "EVALUATION_RESULT",
                "patient_id": patient_id,
                "status": "success",
                "payload": {"patient_id": patient_id, "report": evaluation["report"]},
            }
        )
        event_index += 1
        events.append(
            {
                "event_index": event_index,
                "event_type": "CASE_END",
                "patient_id": patient_id,
                "status": "success",
                "payload": {"patient_id": patient_id, "finished": True},
            }
        )
    _write_jsonl(run_dir / "evaluation_results.jsonl", evaluations)
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "final_results.jsonl", finals)
    return {
        "run_id": run_id,
        "mode": "train",
        "has_evaluation": True,
        "dataset_layer": "HistoricalReplay",
        "artifact_hashes": {
            name: file_hash(run_dir / name)
            for name in ("evaluation_results.jsonl", "events.jsonl", "final_results.jsonl")
        },
    }


def _write_trust_manifest(path: Path, runs: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"historical_runs": runs}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_importer_selects_latest_and_materializes_immutable_batch(tmp_path: Path) -> None:
    train_outputs = tmp_path / "outputs" / "train"
    old = _write_run(
        train_outputs,
        "train_old",
        [_evaluation("Patient_01061", "2026-07-09T01:00:00+08:00")],
    )
    latest = _write_run(
        train_outputs,
        "train_latest",
        [
            _evaluation("Patient_01061", "2026-07-09T02:00:00+08:00"),
            _evaluation("Patient_09249", "2026-07-09T02:01:00+08:00", diagnosis="腺病毒性结膜炎"),
        ],
    )
    trust_manifest = _write_trust_manifest(tmp_path / "trust.json", [old, latest])
    artifact_root = tmp_path / "offline" / "case_memory"

    first = import_case_memory_candidates(
        train_outputs=train_outputs,
        trust_manifest_path=trust_manifest,
        artifact_root=artifact_root,
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        catalog_hashes={"diseases_catalog.json": "a" * 64, "examinations_catalog.json": "b" * 64},
    )
    second = import_case_memory_candidates(
        train_outputs=train_outputs,
        trust_manifest_path=trust_manifest,
        artifact_root=artifact_root,
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        catalog_hashes={"diseases_catalog.json": "a" * 64, "examinations_catalog.json": "b" * 64},
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert first["import_id"] == second["import_id"]
    batch_dir = Path(first["batch_dir"])
    manifest = read_json(batch_dir / "import_manifest.json")
    assert manifest["raw_evaluation_count"] == 3
    assert manifest["selected_count"] == 2
    assert manifest["superseded_count"] == 1
    assert manifest["ground_truth_conflict_count"] == 0
    assert [item["patient_id"] for item in manifest["selections"]] == [
        "Patient_01061",
        "Patient_09249",
    ]
    assert manifest["selections"][0]["source_run_id"] == "train_latest"
    for patient_id in ("Patient_01061", "Patient_09249"):
        candidate = load_candidate(batch_dir / "candidates" / ("case-memory-%s.json" % patient_id))
        assert candidate["status"] == "candidate"
        assert candidate["candidate_type"] == "case_memory"
        evaluation_ref = candidate["evidence"]["evaluation_ref"]
        assert (batch_dir / "evaluations" / evaluation_ref).exists()


def test_importer_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    train_outputs = tmp_path / "outputs" / "train"
    run = _write_run(
        train_outputs,
        "train_one",
        [_evaluation("Patient_01061", "2026-07-09T01:00:00+08:00")],
    )
    trust_manifest = _write_trust_manifest(tmp_path / "trust.json", [run])
    evaluation_path = train_outputs / "train_one" / "evaluation_results.jsonl"
    evaluation_path.write_text(evaluation_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        import_case_memory_candidates(
            train_outputs=train_outputs,
            trust_manifest_path=trust_manifest,
            artifact_root=tmp_path / "offline",
            official_diseases={"三房心"},
            valid_examinations={"体格检查"},
            catalog_hashes={},
        )


def test_importer_rejects_ground_truth_conflict(tmp_path: Path) -> None:
    train_outputs = tmp_path / "outputs" / "train"
    first = _write_run(
        train_outputs,
        "train_one",
        [_evaluation("Patient_01061", "2026-07-09T01:00:00+08:00")],
    )
    second = _write_run(
        train_outputs,
        "train_two",
        [_evaluation("Patient_01061", "2026-07-09T02:00:00+08:00", diagnosis="腺病毒性结膜炎")],
    )
    trust_manifest = _write_trust_manifest(tmp_path / "trust.json", [first, second])

    with pytest.raises(ValueError, match="ground truth conflict"):
        import_case_memory_candidates(
            train_outputs=train_outputs,
            trust_manifest_path=trust_manifest,
            artifact_root=tmp_path / "offline",
            official_diseases={"三房心", "腺病毒性结膜炎"},
            valid_examinations={"体格检查"},
            catalog_hashes={},
        )
