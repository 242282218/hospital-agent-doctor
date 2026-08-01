from __future__ import annotations

import json
from pathlib import Path

from offline.artifacts import file_hash
from offline.train_trust import build_train_trust_manifest, scan_train_runs, write_train_trust_manifest


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_scan_and_build_trust_manifest(tmp_path: Path) -> None:
    train = tmp_path / "train"
    run = train / "train_20260723_000001_abc"
    _write_jsonl(
        run / "evaluation_results.jsonl",
        [{"timestamp": "2026-07-23T00:00:00+08:00", "patient_id": "Patient_00001", "report": {}}],
    )
    _write_jsonl(run / "events.jsonl", [{"event_type": "CASE_END", "patient_id": "Patient_00001"}])
    _write_jsonl(run / "final_results.jsonl", [{"patient_id": "Patient_00001", "finished": True}])

    runs = scan_train_runs(train)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "train_20260723_000001_abc"
    assert runs[0]["has_evaluation"] is True
    assert runs[0]["artifact_hashes"]["evaluation_results.jsonl"] == file_hash(
        run / "evaluation_results.jsonl"
    )

    manifest = build_train_trust_manifest(train_outputs=train, only_with_evaluation=True)
    assert manifest["schema_version"] == "train-trust-manifest/v1"
    assert manifest["trusted_train_run_count"] == 1
    assert manifest["manifest_hash"]
    assert len(manifest["manifest_hash"]) == 64

    out = tmp_path / "trust.json"
    receipt = write_train_trust_manifest(manifest, path=out)
    assert out.exists()
    assert receipt["file_hash"] == file_hash(out)
    # immutable: second write of same content may raise or reuse depending on write_immutable_json
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["trusted_train_run_count"] == 1


def test_only_with_evaluation_filters_incomplete_runs(tmp_path: Path) -> None:
    train = tmp_path / "train"
    incomplete = train / "train_20260723_incomplete"
    _write_jsonl(incomplete / "events.jsonl", [{"event_type": "CASE_START"}])
    complete = train / "train_20260723_complete"
    _write_jsonl(complete / "evaluation_results.jsonl", [{"patient_id": "Patient_1"}])
    _write_jsonl(complete / "events.jsonl", [{}])
    _write_jsonl(complete / "final_results.jsonl", [{}])

    all_runs = scan_train_runs(train)
    assert len(all_runs) == 2
    trusted = build_train_trust_manifest(train_outputs=train, only_with_evaluation=True)
    assert trusted["trusted_train_run_count"] == 1
    assert trusted["historical_runs"][0]["run_id"] == "train_20260723_complete"
