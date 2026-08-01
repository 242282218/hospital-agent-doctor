from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory
from agent.runtime.release_loader import load_current_release
from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json
from offline.case_memory import case_memory_candidate
from offline.candidates import write_candidate
from scripts.memory.build_case_memory_release import build_case_memory_release


def _evaluation(
    patient_id: str,
    diagnosis: str,
    treatment_plan: str = "尽快完成专科评估。",
) -> Dict[str, Any]:
    return {
        "timestamp": "2026-07-09T02:00:00+08:00",
        "patient_id": patient_id,
        "report": {
            "patientId": patient_id,
            "status": "evaluated",
            "ground_truth": {
                "final_diagnosis": diagnosis,
                "necessary_examinations": ["体格检查"],
                "treatment_plan": treatment_plan,
            },
            "treatmentDetail": {
                "reference": treatment_plan,
                "reasoning": "当前证据支持该诊断。",
            },
        },
    }


def _write_import_batch(tmp_path: Path) -> Path:
    batch_dir = tmp_path / "batch"
    evaluation_store = batch_dir / "evaluations"
    candidate_store = batch_dir / "candidates"
    selections = []
    for patient_id, diagnosis, treatment_plan in [
        ("Patient_01061", "三房心", "尽快完成专科评估。"),
        (
            "Patient_09249",
            "腺病毒性结膜炎",
            "给予人工泪液和冷敷；不要常规使用局部抗生素；仅当出现继发细菌感染证据时，才考虑使用局部抗生素。",
        ),
    ]:
        evaluation = _evaluation(patient_id, diagnosis, treatment_plan)
        evaluation_hash = content_hash(evaluation)
        evaluation_ref = "sha256/%s.json" % evaluation_hash
        write_immutable_json(evaluation_store / evaluation_ref, evaluation)
        candidate = case_memory_candidate(
            patient_id=patient_id,
            evaluation=evaluation,
            evaluation_ref=evaluation_ref,
            official_diseases={"三房心", "腺病毒性结膜炎"},
            valid_examinations={"体格检查"},
        )
        write_candidate(candidate_store / (candidate["candidate_id"] + ".json"), candidate)
        selections.append(
            {
                "patient_id": patient_id,
                "source_run_id": "train-source",
                "source_line": 1,
                "evaluation_hash": evaluation_hash,
                "evaluation_ref": evaluation_ref,
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "effect_hash": candidate["effect_hash"],
            }
        )
    manifest = {
        "schema_version": "case-memory-import/v1",
        "import_id": "unit-test-import",
        "raw_evaluation_count": 2,
        "selected_count": 2,
        "superseded_count": 0,
        "ground_truth_conflict_count": 0,
        "catalog_hashes": {
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        "selections": selections,
    }
    manifest["manifest_hash"] = content_hash(manifest)
    write_immutable_json(batch_dir / "source_receipt.json", {"trusted_runs": ["train-source"]})
    write_immutable_json(batch_dir / "import_manifest.json", manifest)
    return batch_dir


def test_build_case_memory_release_creates_nonempty_frozen_pack(tmp_path: Path) -> None:
    batch_dir = _write_import_batch(tmp_path)
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text('{"system":"doctor"}', encoding="utf-8")
    (base_release / "policy_pack.json").write_text('{"simple_cap":5}', encoding="utf-8")
    release_dir = tmp_path / "release_C"
    production_pointer = tmp_path / "production-current.json"
    production_pointer.write_text('{"unchanged":true}', encoding="utf-8")
    pointer_hash = file_hash(production_pointer)

    result = build_case_memory_release(
        import_batch=batch_dir,
        base_release=base_release,
        release_dir=release_dir,
        reviewer="github:24228",
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        knowledge_hashes={"alias_map.json": "c" * 64},
        catalog_hashes={
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        code_commit="unit-test-dirty-snapshot",
        production_pointer=production_pointer,
    )

    assert file_hash(production_pointer) == pointer_hash
    registry = read_json(release_dir / "verified_registry.json")
    assert [asset["content"]["patient_id"] for asset in registry["assets"]] == [
        "Patient_01061",
        "Patient_09249",
    ]
    assert all(
        read_json(path)["reviewer"] == "github:24228"
        for path in sorted((batch_dir / "decisions").glob("*.json"))
    )
    acceptance = read_json(release_dir / "offline_acceptance.json")
    gate_report = read_json(release_dir / "offline_gate_report.json")
    experiment_result = read_json(release_dir / "offline_experiment_result.json")
    assert acceptance["metrics"]["asset_count"] == 2
    assert acceptance["metrics"]["diagnosis_ok"] is True
    assert acceptance["metrics"]["prior_fallback_ok"] is True
    assert "exam_treatment_ok" not in acceptance["metrics"]
    assert acceptance["metrics"]["zero_llm_zero_ask"] is True
    assert gate_report["input_hashes"]["offline_acceptance"] == file_hash(
        release_dir / "offline_acceptance.json"
    )
    assert experiment_result["metrics"]["acceptance_hash"] == acceptance["acceptance_hash"]
    assert (release_dir / "promotion_record.json").exists()
    assert result["asset_count"] == 2

    temp_pointer = tmp_path / "temp-current.json"
    temp_pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": str(release_dir),
                "pack_hash": file_hash(release_dir / "release_manifest.json"),
                "promotion_record_hash": read_json(release_dir / "promotion_record.json")[
                    "promotion_record_hash"
                ],
                "runtime_schema_version": "clinical-runtime/v1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_current_release(temp_pointer)
    assert len(loaded.registry["assets"]) == 2
    memory = VerifiedOnlyMemory(release_dir / "verified_registry.json")
    assert memory.load_case_memory("Patient_01061")["diagnoses"] == ["三房心"]
    eye_plan = memory.load_case_memory("Patient_09249")["treatment_plan"]
    assert "不要常规使用局部抗生素" in eye_plan
    assert memory.load_case_memory("Patient_0106") is None

    class FakeActions:
        def __init__(self) -> None:
            self.prescribed = []

        async def order(self, *, items, reason):
            return {
                "results": {
                    item: {
                        "status": "normal",
                        "result": {"summary": "检查已完成"},
                        "abnormal_indicators": [],
                    }
                    for item in items
                }
            }

        async def prescribe_with_authorization(
            self,
            *,
            payload,
            clinical_context,
        ):
            assert clinical_context["diagnoses"] == payload["diagnosis"]
            assert clinical_context["official_diseases"]
            submitted = dict(payload)
            self.prescribed.append(submitted)
            return {**submitted, "finished": True}

    import asyncio

    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)
    observed_fallbacks = []
    original_run_verified_case_memory = agent._run_verified_case_memory

    class FallbackObserved(Exception):
        pass

    async def recording_run_verified_case_memory(**kwargs):
        result = await original_run_verified_case_memory(**kwargs)
        observed_fallbacks.append(
            (
                result,
                kwargs["case_state"].get("case_memory_fallback_reason"),
                kwargs["case_state"].get("verified_case_prior"),
            )
        )
        raise FallbackObserved

    agent._run_verified_case_memory = recording_run_verified_case_memory
    actions = FakeActions()
    with pytest.raises(FallbackObserved):
        asyncio.run(
            agent.run_full_clinical_loop(
                actions=actions,
                patient_id="Patient_09249",
                mode="test",
            )
        )
    assert actions.prescribed == []
    assert observed_fallbacks == [
        (
            None,
            "safety_facts_incomplete",
            {
                "source": "verified_case_memory",
                "diagnoses": ["腺病毒性结膜炎"],
                "required_examinations": ["体格检查"],
                "completed_examinations": ["体格检查"],
                "pending_examinations": [],
                "evaluation_hash": memory.load_case_memory("Patient_09249")["provenance"]["evaluation_hash"],
            },
        )
    ]


def test_release_gate_does_not_block_on_token_metric(tmp_path: Path) -> None:
    from offline.gates import build_gate_report

    report = build_gate_report(
        core={
            "schema_version": "experiment-result-core/v1",
            "metrics": {
                "p0_count": 0,
                "diagnosis_ok": True,
                "exam_treatment_ok": True,
                "token_ok": False,
            },
        },
        artifact_hashes={},
    )

    assert report["gates"]["gate4_token"] is False
    assert report["passed"] is True


def test_release_builder_rejects_stale_staging_registry(tmp_path: Path) -> None:
    batch_dir = _write_import_batch(tmp_path)
    stale_registry = {
        "schema_version": "verified-registry/v1",
        "assets": [],
    }
    stale_registry["registry_hash"] = content_hash(stale_registry)
    write_immutable_json(batch_dir / "verified_registry.json", stale_registry)
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (base_release / "policy_pack.json").write_text("{}", encoding="utf-8")

    release_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="staging registry does not match import batch"):
        build_case_memory_release(
            import_batch=batch_dir,
            base_release=base_release,
            release_dir=release_dir,
            reviewer="github:24228",
            official_diseases={"三房心", "腺病毒性结膜炎"},
            valid_examinations={"体格检查"},
            knowledge_hashes={},
            catalog_hashes={
                "diseases_catalog.json": "a" * 64,
                "examinations_catalog.json": "b" * 64,
            },
            code_commit="unit-test",
            production_pointer=None,
        )

    assert list(release_dir.iterdir()) == []


def test_release_builder_accepts_equivalent_approval_paths(tmp_path: Path) -> None:
    batch_dir = _write_import_batch(tmp_path)
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (base_release / "policy_pack.json").write_text("{}", encoding="utf-8")
    first_release = tmp_path / "first-release"
    build_case_memory_release(
        import_batch=batch_dir,
        base_release=base_release,
        release_dir=first_release,
        reviewer="github:24228",
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        knowledge_hashes={},
        catalog_hashes={
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        code_commit="unit-test",
        production_pointer=None,
    )
    staging_path = batch_dir / "verified_registry.json"
    staging_registry = read_json(staging_path)
    for asset in staging_registry["assets"]:
        asset["approval_ref"] = os.path.relpath(asset["approval_ref"], Path.cwd())
    staging_registry["registry_hash"] = content_hash(
        {key: value for key, value in staging_registry.items() if key != "registry_hash"}
    )
    staging_path.write_text(json.dumps(staging_registry, ensure_ascii=False), encoding="utf-8")

    result = build_case_memory_release(
        import_batch=batch_dir,
        base_release=base_release,
        release_dir=tmp_path / "second-release",
        reviewer="github:24228",
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        knowledge_hashes={},
        catalog_hashes={
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        code_commit="unit-test",
        production_pointer=None,
    )

    assert result["asset_count"] == 2


def test_release_registry_is_stable_for_relative_and_absolute_batch_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_dir = _write_import_batch(tmp_path)
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (base_release / "policy_pack.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    build_case_memory_release(
        import_batch=Path("batch"),
        base_release=Path("base"),
        release_dir=Path("relative-release"),
        reviewer="github:24228",
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        knowledge_hashes={},
        catalog_hashes={
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        code_commit="unit-test",
        production_pointer=None,
    )
    build_case_memory_release(
        import_batch=batch_dir.resolve(),
        base_release=base_release.resolve(),
        release_dir=(tmp_path / "absolute-release").resolve(),
        reviewer="github:24228",
        official_diseases={"三房心", "腺病毒性结膜炎"},
        valid_examinations={"体格检查"},
        knowledge_hashes={},
        catalog_hashes={
            "diseases_catalog.json": "a" * 64,
            "examinations_catalog.json": "b" * 64,
        },
        code_commit="unit-test",
        production_pointer=None,
    )

    relative_registry = tmp_path / "relative-release" / "verified_registry.json"
    absolute_registry = tmp_path / "absolute-release" / "verified_registry.json"
    assert file_hash(relative_registry) == file_hash(absolute_registry)


def test_release_builder_cleans_rebuild_artifact_on_validation_error(tmp_path: Path) -> None:
    batch_dir = _write_import_batch(tmp_path)
    evaluation_path = next((batch_dir / "evaluations").rglob("*.json"))
    evaluation = read_json(evaluation_path)
    evaluation["report"]["ground_truth"]["final_diagnosis"] = "未知疾病"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")
    manifest = read_json(batch_dir / "import_manifest.json")
    selection = next(
        item
        for item in manifest["selections"]
        if item["evaluation_ref"] == evaluation_path.relative_to(batch_dir / "evaluations").as_posix()
    )
    selection["evaluation_hash"] = content_hash(evaluation)
    manifest["manifest_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    (batch_dir / "import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (base_release / "policy_pack.json").write_text("{}", encoding="utf-8")
    release_dir = tmp_path / "release"

    with pytest.raises(ValueError, match="evaluation hash mismatch"):
        build_case_memory_release(
            import_batch=batch_dir,
            base_release=base_release,
            release_dir=release_dir,
            reviewer="github:24228",
            official_diseases={"三房心", "腺病毒性结膜炎"},
            valid_examinations={"体格检查"},
            knowledge_hashes={},
            catalog_hashes={
                "diseases_catalog.json": "a" * 64,
                "examinations_catalog.json": "b" * 64,
            },
            code_commit="unit-test",
            production_pointer=None,
        )

    assert list(release_dir.iterdir()) == []


def test_release_builder_refuses_existing_release_directory(tmp_path: Path) -> None:
    batch_dir = _write_import_batch(tmp_path)
    base_release = tmp_path / "base"
    base_release.mkdir()
    (base_release / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (base_release / "policy_pack.json").write_text("{}", encoding="utf-8")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="release directory must be new and empty"):
        build_case_memory_release(
            import_batch=batch_dir,
            base_release=base_release,
            release_dir=release_dir,
            reviewer="github:24228",
            official_diseases={"三房心", "腺病毒性结膜炎"},
            valid_examinations={"体格检查"},
            knowledge_hashes={},
            catalog_hashes={
                "diseases_catalog.json": "a" * 64,
                "examinations_catalog.json": "b" * 64,
            },
            code_commit="unit-test",
            production_pointer=None,
        )

    assert (release_dir / "sentinel.txt").read_text(encoding="utf-8") == "keep"
