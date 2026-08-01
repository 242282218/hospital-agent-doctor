from __future__ import annotations

from pathlib import Path
import json
from hashlib import sha256

import pytest

from agent.observability.run_trace import RunTraceStore
from agent.observability.runtime_events import SequencedEventSink
from offline.artifacts import content_hash, file_hash, write_immutable_json
from offline.candidates import create_candidate, write_candidate
from offline.episodes import ingest_episode
from offline.experiments import build_result_core, finalize_result
from offline.gates import build_gate_report
from offline.promotion import approve_candidate, build_registry_snapshot
from offline.release import (
    build_candidate_pack,
    switch_release_pointer,
    verify_release_pack,
    write_promotion_record,
)
from agent.runtime.release_loader import load_current_release
from agent.knowledge.typed_rule_engine import empty_compiled_rule_pack


def test_run_trace_external_seal_and_episode_ingest(tmp_path: Path) -> None:
    store = RunTraceStore(tmp_path / "run", run_id="run-1")
    store.append({"type": "decision", "action": "ask_patient", "revision": 0})
    store.append({"type": "observation", "status": "succeeded"})
    receipt = store.seal()
    assert receipt.event_count == 2
    seal = (tmp_path / "run" / "run.seal.json").read_text(encoding="utf-8")
    assert "trace_hash" in seal
    assert "seal" not in (tmp_path / "run" / "run.jsonl").read_text(encoding="utf-8")
    episode = ingest_episode(
        run_dir=tmp_path / "run",
        episode_dir=tmp_path / "episodes",
        episode_id="ep-1",
    )
    assert episode["event_count"] == 2
    assert "seal" not in str(episode["records"])


def test_trace_and_episode_exclude_clinical_plaintext_from_direct_emits(tmp_path: Path) -> None:
    store = RunTraceStore(tmp_path / "run", run_id="run-redacted")
    sink = SequencedEventSink(append=store.append, case_run_id="run-redacted")
    tokens = [
        "UNIQUE_DIAGNOSIS_TOKEN",
        "UNIQUE_EXAM_TOKEN",
        "UNIQUE_TREATMENT_TOKEN",
        "UNIQUE_REASON_TOKEN",
        "UNIQUE_ISSUE_TOKEN",
    ]
    sink(
        {
            "type": "diagnosis_state",
            "axis_ids": ["UNIQUE_AXIS_TOKEN"],
            "candidate_names": [tokens[0]],
            "consistency_issue_codes": [tokens[4]],
        }
    )
    sink(
        {
            "type": "exam_plan",
            "examinations": [tokens[1]],
            "reason_codes": [tokens[3]],
            "open_gap_ids": ["UNIQUE_GAP_TOKEN"],
        }
    )
    sink(
        {
            "type": "action_command",
            "command_id": "command-redacted",
            "action_type": "prescribe_treatment",
            "payload": {
                "diagnosis": [tokens[0]],
                "treatment_plan": tokens[2],
                "reasoning": tokens[3],
            },
        }
    )
    sink(
        {
            "type": "runtime_decision",
            "action": "finish",
            "reason": tokens[3],
        }
    )
    store.seal()

    trace_text = (tmp_path / "run" / "run.jsonl").read_text(encoding="utf-8")
    for token in [*tokens, "UNIQUE_AXIS_TOKEN", "UNIQUE_GAP_TOKEN"]:
        assert token not in trace_text

    episode = ingest_episode(
        run_dir=tmp_path / "run",
        episode_dir=tmp_path / "episodes",
        episode_id="ep-redacted",
    )
    records_text = json.dumps(episode["records"], ensure_ascii=False)
    for token in [*tokens, "UNIQUE_AXIS_TOKEN", "UNIQUE_GAP_TOKEN"]:
        assert token not in records_text
    assert episode["records"][0]["candidate_names_hash"]
    assert episode["records"][1]["examinations_hash"]
    assert episode["records"][2]["payload_hash"]


def test_promotion_requires_real_decision_hash_chain(tmp_path: Path) -> None:
    candidate = create_candidate(
        candidate_id="cand-1",
        candidate_type="mechanical_orthography",
        proposed_effect={"from": "HbA1c ", "to": "HbA1c"},
        evidence={"source": "unit-test"},
    )
    cand_path = tmp_path / "candidates" / "cand-1.json"
    write_candidate(cand_path, candidate)
    decision = approve_candidate(
        candidate_path=cand_path,
        decision_path=tmp_path / "decisions" / "d1.json",
        reviewer="tester",
        canary_required=False,
        rationale="orthography only",
    )
    assert decision["decision_hash"]
    assert decision["candidate_hash"] == candidate["candidate_hash"]
    registry = build_registry_snapshot(
        decision_paths=[tmp_path / "decisions" / "d1.json"],
        candidate_store=tmp_path / "candidates",
        output_path=tmp_path / "registry.json",
    )
    assert registry["assets"]
    # Non-empty approval_ref alone is not enough: missing decision file fails.
    try:
        build_registry_snapshot(
            decision_paths=[tmp_path / "decisions" / "missing.json"],
            candidate_store=tmp_path / "candidates",
            output_path=tmp_path / "registry2.json",
        )
        assert False, "missing decision should fail"
    except FileNotFoundError:
        pass


def test_release_pointer_and_runtime_loader(tmp_path: Path) -> None:
    release_dir = tmp_path / "release_A_quality"
    pack = build_candidate_pack(
        release_dir=release_dir,
        code_commit="deadbeef",
        prompt_pack={"system": "doctor"},
        policy_pack={"simple_cap": 5, "complex_cap": 8},
        registry={"schema_version": "verified-registry/v1", "assets": []},
        knowledge_hashes={"alias_map.json": "a" * 64},
        catalog_hashes={"diseases_catalog.json": "b" * 64},
    )
    assert "gate_report_hash" not in pack
    verify_release_pack(release_dir)
    core = build_result_core(
        resolved_manifest_hash="c" * 64,
        metrics={"p0_count": 0, "diagnosis_ok": True, "exam_treatment_ok": True, "token_ok": True},
    )
    report = build_gate_report(core=core, artifact_hashes={"pack": pack["pack_hash"]})
    result = finalize_result(core=core, gate_report_hash=report["gate_report_hash"])
    write_promotion_record(
        path=release_dir / "promotion_record.json",
        candidate_pack_hash=pack["pack_hash"],
        gate_report_hash=report["gate_report_hash"],
        experiment_result_hash=result["result_hash"],
    )
    pointer_path = tmp_path / "current.json"
    switch_release_pointer(pointer_path, release_dir)
    loaded = load_current_release(pointer_path)
    assert loaded.policy_pack["simple_cap"] == 5
    assert loaded.prompt_pack["system"] == "doctor"
    assert loaded.knowledge_rule_pack == empty_compiled_rule_pack()


def test_runtime_loader_validates_and_parses_optional_knowledge_rule_pack(
    tmp_path: Path,
) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    prompt_path = release_dir / "prompt_pack.json"
    policy_path = release_dir / "policy_pack.json"
    registry_path = release_dir / "verified_registry.json"
    rules_path = release_dir / "knowledge_rules.json"
    prompt_path.write_text("{}", encoding="utf-8")
    policy_path.write_text("{}", encoding="utf-8")
    registry_path.write_text('{"assets":[]}', encoding="utf-8")
    rules = _compiled_rule_pack_payload()
    rules_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _path_hash(prompt_path),
        "policy_pack_hash": _path_hash(policy_path),
        "registry_hash": _path_hash(registry_path),
        "knowledge_rules_hash": _path_hash(rules_path),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "schema_version": "release-pointer/v1",
        "release_dir": str(release_dir),
        "pack_hash": _path_hash(manifest_path),
        "runtime_schema_version": "clinical-runtime/v1",
    }
    pointer_path = tmp_path / "current-with-rules.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    loaded = load_current_release(pointer_path)

    assert loaded.knowledge_rule_pack.rule_count == 1
    assert loaded.knowledge_rule_pack.rules[0].runtime.status == "active"

    rules_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="knowledge rules hash mismatch"):
        load_current_release(pointer_path)


def test_runtime_loader_rejects_unhashed_knowledge_rule_file(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    prompt_path = release_dir / "prompt_pack.json"
    policy_path = release_dir / "policy_pack.json"
    registry_path = release_dir / "verified_registry.json"
    prompt_path.write_text("{}", encoding="utf-8")
    policy_path.write_text("{}", encoding="utf-8")
    registry_path.write_text('{"assets":[]}', encoding="utf-8")
    (release_dir / "knowledge_rules.json").write_text(
        json.dumps(_compiled_rule_pack_payload()),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _path_hash(prompt_path),
        "policy_pack_hash": _path_hash(policy_path),
        "registry_hash": _path_hash(registry_path),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer_path = tmp_path / "current.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": str(release_dir),
                "pack_hash": _path_hash(manifest_path),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unhashed knowledge rules"):
        load_current_release(pointer_path)


def _path_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _compiled_rule_pack_payload() -> dict[str, object]:
    rule = {
        "rule_id": "prefer_supported_current_problem",
        "candidate_type": "diagnosis_priority_rule",
        "candidate_hash": "a" * 64,
        "effect_hash": "b" * 64,
        "triggers": ["structured trigger"],
        "required_evidence": ["structured evidence"],
        "exclusions": ["structured exclusion"],
        "effect": {
            "priority_policy": "objective_evidence_first",
            "fallback_policy": "official_catalog_only",
        },
        "positive_controls": [
            {
                "control_id": "positive_control",
                "kind": "positive",
                "facts": ["fact"],
                "assertions": ["assertion"],
            }
        ],
        "negative_controls": [
            {
                "control_id": "near_neighbor_control",
                "kind": "near_neighbor",
                "facts": ["fact"],
                "assertions": ["assertion"],
            }
        ],
        "source_refs": [{"path": "docs/source.md", "sha256": "c" * 64}],
        "test_refs": [{"path": "tests/test_source.py", "sha256": "d" * 64}],
        "priority": 10,
        "scope": {"phase": "diagnosis", "application": "trigger_bound"},
        "runtime": {
            "status": "active",
            "stage": "diagnosis_candidates",
            "opcode": "promote_supported_current_over_background",
            "parameters": {
                "target_roles": ["current_problem"],
                "target_support_levels": ["objective"],
                "background_roles": ["background_condition"],
                "background_relations": ["unrelated"],
                "excluded_relations": ["explains"],
                "preserve_urgencies": ["emergency"],
                "fallback_policy": "official_catalog_only",
            },
        },
    }
    return {
        "schema_version": "compiled-knowledge-rules/v2",
        "rules": [rule],
        "rule_count": 1,
        "rules_hash": content_hash([rule]),
    }


def test_release_builder_refuses_to_overwrite_existing_pack(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    first = build_candidate_pack(
        release_dir=release_dir,
        code_commit="deadbeef",
        prompt_pack={"system": "doctor"},
        policy_pack={"simple_cap": 5},
        registry={"schema_version": "verified-registry/v1", "assets": []},
        knowledge_hashes={},
        catalog_hashes={},
    )
    hashes_before = {
        name: file_hash(release_dir / name)
        for name in (
            "prompt_pack.json",
            "policy_pack.json",
            "verified_registry.json",
            "release_manifest.json",
        )
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite frozen release"):
        build_candidate_pack(
            release_dir=release_dir,
            code_commit="changed",
            prompt_pack={"system": "changed"},
            policy_pack={"simple_cap": 9},
            registry={"schema_version": "verified-registry/v1", "assets": [{"x": 1}]},
            knowledge_hashes={},
            catalog_hashes={},
        )

    assert first["pack_hash"]
    assert hashes_before == {
        name: file_hash(release_dir / name)
        for name in hashes_before
    }


def test_gate_report_avoids_self_hash_cycle() -> None:
    core = build_result_core(resolved_manifest_hash="x" * 64, metrics={"p0_count": 0})
    assert "gate_report_hash" not in core
    report = build_gate_report(core=core, artifact_hashes={})
    assert report["gate_report_hash"]
    final = finalize_result(core=core, gate_report_hash=report["gate_report_hash"])
    assert final["gate_report_hash"] == report["gate_report_hash"]
