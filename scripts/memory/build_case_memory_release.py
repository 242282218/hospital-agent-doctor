from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Optional

from offline.artifacts import canonical_json, content_hash, file_hash, read_json, write_immutable_json
from offline.candidates import load_candidate
from offline.experiments import build_experiment_plan, build_result_core, build_resolved_manifest, finalize_result
from offline.gates import build_gate_report
from offline.promotion import approve_candidate, build_registry_snapshot
from offline.release import build_candidate_pack, verify_release_pack, write_promotion_record


EXPECTED_PATIENT_IDS = {
    "Patient_01061",
    "Patient_02654",
    "Patient_06090",
    "Patient_06805",
    "Patient_08451",
    "Patient_09249",
}
_REQUIRED_GATES = ("gate0_schema", "gate1_safety", "gate2_diagnosis")

def _validate_import_batch(import_batch: Path) -> Dict[str, Any]:
    manifest = read_json(import_batch / "import_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "case-memory-import/v1":
        raise ValueError("invalid case-memory import manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if content_hash(body) != manifest.get("manifest_hash"):
        raise ValueError("import manifest hash mismatch")
    selections = manifest.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("case-memory import batch is empty")
    patient_ids = [item.get("patient_id") for item in selections if isinstance(item, dict)]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("duplicate patient in import batch")
    for item in selections:
        candidate_path = import_batch / "candidates" / (item["candidate_id"] + ".json")
        candidate = load_candidate(candidate_path)
        if candidate.get("status") != "candidate" or candidate.get("candidate_type") != "case_memory":
            raise ValueError("invalid case-memory candidate in import batch")
        if candidate.get("candidate_hash") != item.get("candidate_hash"):
            raise ValueError("candidate hash does not match import manifest")
        evaluation_path = import_batch / "evaluations" / item["evaluation_ref"]
        if not evaluation_path.exists() or content_hash(read_json(evaluation_path)) != item.get("evaluation_hash"):
            raise ValueError("evaluation hash does not match import manifest")
    return manifest


def _canonical_registry_for_comparison(registry: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(canonical_json(registry))
    for asset in normalized.get("assets", []):
        approval_ref = asset.get("approval_ref")
        if isinstance(approval_ref, str):
            asset["approval_ref"] = str(Path(approval_ref).resolve())
    normalized["registry_hash"] = content_hash(
        {key: value for key, value in normalized.items() if key != "registry_hash"}
    )
    return normalized


class _OfflineAcceptanceActions:
    def __init__(self) -> None:
        self.asked = 0
        self.ordered = []
        self.prescribed = []

    async def ask(self, **_: Any) -> str:
        self.asked += 1
        raise AssertionError("case-memory acceptance unexpectedly asked patient")

    async def order(self, *, items: list, reason: str) -> Dict[str, Any]:
        batch = list(items)
        self.ordered.append(batch)
        return {
            "results": {
                item: {
                    "status": "normal",
                    "result": {"summary": "离线检查已完成"},
                    "abnormal_indicators": [],
                }
                for item in batch
            }
        }

    async def prescribe_with_authorization(
        self,
        *,
        payload: Dict[str, Any],
        clinical_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert clinical_context["diagnoses"] == payload["diagnosis"]
        assert clinical_context["official_diseases"]
        submitted = dict(payload)
        self.prescribed.append(submitted)
        return {**submitted, "finished": True}


async def _run_release_acceptance(registry_path: Path) -> Dict[str, Any]:
    from agent.legacy_orchestrator import MAX_EXAMS_PER_ACTION, MyDoctorAgent
    from agent.memory import VerifiedOnlyMemory

    registry = read_json(registry_path)
    patient_ids = sorted(
        asset["content"]["patient_id"]
        for asset in registry.get("assets", [])
        if asset.get("candidate_type") == "case_memory"
    )
    memory = VerifiedOnlyMemory(registry_path)
    case_results = []
    for patient_id in patient_ids:
        agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)

        async def forbidden_llm(**_: Any) -> Dict[str, Any]:
            raise AssertionError("case-memory acceptance unexpectedly called LLM")

        agent._call_llm = forbidden_llm
        actions = _OfflineAcceptanceActions()
        case_memory = memory.load_case_memory(patient_id)
        if case_memory is None:
            raise ValueError("case-memory asset missing from runtime reader")
        case_state = {
            "patient_id": patient_id,
            "mode": "test",
            "chat_history": [],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "decision_trace": [],
            "exam_decision_trace": [],
        }
        result = await agent._run_verified_case_memory(
            actions=actions,
            case_state=case_state,
            case_memory=case_memory,
        )
        if result is not None or actions.asked or actions.prescribed:
            raise ValueError("legacy case-memory acceptance must not fast-submit")
        prior = case_state.get("verified_case_prior")
        if not isinstance(prior, Mapping):
            raise ValueError("legacy case-memory acceptance lost verified prior")
        if any(not batch or len(batch) > MAX_EXAMS_PER_ACTION for batch in actions.ordered):
            raise ValueError("case-memory examination batch limit failed")
        case_results.append(
            {
                "patient_id": patient_id,
                "diagnoses": list(prior.get("diagnoses") or []),
                "examination_batches": [len(batch) for batch in actions.ordered],
            }
        )
    metrics = {
        "asset_count": len(case_results),
        "unique_patient_count": len({item["patient_id"] for item in case_results}),
        "diagnosis_ok": all(item["diagnoses"] for item in case_results),
        "prior_fallback_ok": bool(case_results),
        "zero_llm_zero_ask": True,
        "p0_count": 0,
    }
    acceptance = {
        "schema_version": "case-memory-offline-acceptance/v1",
        "metrics": metrics,
        "cases": case_results,
    }
    acceptance["acceptance_hash"] = content_hash(acceptance)
    return acceptance


def build_case_memory_release(
    *,
    import_batch: Path,
    base_release: Path,
    release_dir: Path,
    reviewer: str,
    official_diseases: Collection[str],
    valid_examinations: Collection[str],
    knowledge_hashes: Mapping[str, str],
    catalog_hashes: Mapping[str, str],
    code_commit: str,
    production_pointer: Optional[Path],
) -> Dict[str, Any]:
    import_batch = Path(import_batch).resolve()
    base_release = Path(base_release).resolve()
    release_dir = Path(release_dir).resolve()
    if release_dir.exists() and any(release_dir.iterdir()):
        raise FileExistsError("release directory must be new and empty: %s" % release_dir)
    if not reviewer.strip():
        raise ValueError("reviewer required")
    pointer_hash_before = file_hash(production_pointer) if production_pointer and production_pointer.exists() else None
    manifest = _validate_import_batch(import_batch)
    imported_catalog_hashes = dict(manifest.get("catalog_hashes") or {})
    for name, expected_hash in imported_catalog_hashes.items():
        if catalog_hashes.get(name) != expected_hash:
            raise ValueError("catalog hashes do not match import batch")

    decisions_dir = import_batch / "decisions"
    decision_paths = []
    for item in manifest["selections"]:
        candidate_path = import_batch / "candidates" / (item["candidate_id"] + ".json")
        decision_path = decisions_dir / (item["candidate_id"] + ".json")
        rationale = (
            "HistoricalReplay offline approval by %s; source_run=%s; evaluation_hash=%s; "
            "candidate_hash=%s; effect_hash=%s; catalog_hashes=%s; Canary pending; "
            "production pointer unchanged."
            % (
                reviewer,
                item["source_run_id"],
                item["evaluation_hash"],
                item["candidate_hash"],
                item["effect_hash"],
                json.dumps(dict(sorted(catalog_hashes.items())), ensure_ascii=False, sort_keys=True),
            )
        )
        if decision_path.exists():
            decision = read_json(decision_path)
            if decision.get("reviewer") != reviewer or decision.get("candidate_hash") != item["candidate_hash"]:
                raise FileExistsError("existing decision differs: %s" % decision_path)
        else:
            approve_candidate(
                candidate_path=candidate_path,
                decision_path=decision_path,
                reviewer=reviewer,
                canary_required=True,
                required_gate_ids=_REQUIRED_GATES,
                rationale=rationale,
            )
        decision_paths.append(decision_path)

    release_dir.mkdir(parents=True, exist_ok=True)
    registry_path = release_dir / "verified_registry.json"
    registry_staging_path = import_batch / "verified_registry.json"
    registry_rebuild_path = release_dir / ".verified_registry.rebuild.json"
    try:
        registry = build_registry_snapshot(
            decision_paths=decision_paths,
            candidate_store=import_batch / "candidates",
            output_path=registry_rebuild_path,
            official_diseases=official_diseases,
            valid_examinations=valid_examinations,
            evaluation_store=import_batch / "evaluations",
        )
    finally:
        registry_rebuild_path.unlink(missing_ok=True)
    if registry_staging_path.exists():
        staging_registry = read_json(registry_staging_path)
        if _canonical_registry_for_comparison(staging_registry) != _canonical_registry_for_comparison(
            registry
        ):
            raise ValueError("staging registry does not match import batch")
    else:
        write_immutable_json(registry_staging_path, registry)
    patient_ids = [asset["content"]["patient_id"] for asset in registry["assets"]]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("release registry has duplicate patients")

    prompt_pack = read_json(base_release / "prompt_pack.json")
    policy_pack = read_json(base_release / "policy_pack.json")
    pack = build_candidate_pack(
        release_dir=release_dir,
        code_commit=code_commit,
        prompt_pack=prompt_pack,
        policy_pack=policy_pack,
        registry=registry,
        knowledge_hashes=knowledge_hashes,
        catalog_hashes=catalog_hashes,
    )
    verify_release_pack(release_dir)
    acceptance = asyncio.run(_run_release_acceptance(registry_path))
    acceptance_file_hash = write_immutable_json(
        release_dir / "offline_acceptance.json",
        acceptance,
    )

    plan = build_experiment_plan(
        "case-memory-historical-replay",
        dataset_layer="HistoricalReplay",
        canary_status="pending",
        production_pointer_unchanged=True,
    )
    plan_hash = write_immutable_json(release_dir / "offline_validation_plan.json", plan)
    resolved = build_resolved_manifest(
        plan_hash=plan_hash,
        release_pack_hash=pack["pack_hash"],
        cases=sorted(patient_ids),
    )
    resolved_hash = write_immutable_json(release_dir / "offline_validation_manifest.json", resolved)
    metrics = {
        "dataset_layer": "HistoricalReplay",
        "canary_status": "pending",
        "asset_count": acceptance["metrics"]["asset_count"],
        "unique_patient_count": acceptance["metrics"]["unique_patient_count"],
        "catalog_ok": True,
        "evaluation_reconstruction_ok": True,
        "p0_count": acceptance["metrics"]["p0_count"],
        "diagnosis_ok": acceptance["metrics"]["diagnosis_ok"],
        "prior_fallback_ok": acceptance["metrics"]["prior_fallback_ok"],
        "zero_llm_zero_ask": acceptance["metrics"]["zero_llm_zero_ask"],
        "acceptance_hash": acceptance["acceptance_hash"],
        "token_ok": True,
    }
    core = build_result_core(resolved_manifest_hash=resolved_hash, metrics=metrics)
    gate_report = build_gate_report(
        core=core,
        artifact_hashes={
            "release_manifest": file_hash(release_dir / "release_manifest.json"),
            "verified_registry": file_hash(release_dir / "verified_registry.json"),
            "import_manifest": file_hash(import_batch / "import_manifest.json"),
            "offline_acceptance": acceptance_file_hash,
        },
    )
    write_immutable_json(release_dir / "offline_gate_report.json", gate_report)
    experiment_result = finalize_result(core=core, gate_report_hash=gate_report["gate_report_hash"])
    write_immutable_json(release_dir / "offline_experiment_result.json", experiment_result)
    promotion = write_promotion_record(
        path=release_dir / "promotion_record.json",
        candidate_pack_hash=pack["pack_hash"],
        gate_report_hash=gate_report["gate_report_hash"],
        experiment_result_hash=experiment_result["result_hash"],
    )
    if pointer_hash_before is not None and file_hash(production_pointer) != pointer_hash_before:
        raise RuntimeError("production pointer changed during offline release build")
    return {
        "release_dir": str(release_dir),
        "asset_count": len(registry["assets"]),
        "patient_ids": sorted(patient_ids),
        "pack_hash": pack["pack_hash"],
        "promotion_record_hash": promotion["promotion_record_hash"],
        "canary_status": "pending",
        "production_pointer_unchanged": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen case-memory release from an import batch.")
    parser.add_argument("--import-batch", type=Path, required=True)
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--ref-data-dir", type=Path, required=True)
    parser.add_argument("--knowledge-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--production-pointer", type=Path)
    return parser


def main() -> None:
    from agent.legacy_orchestrator import (
        flatten_disease_catalog,
        flatten_examination_catalog,
        load_disease_catalog,
        load_examination_catalog,
    )

    args = build_parser().parse_args()
    knowledge_hashes = {
        path.name: file_hash(path)
        for path in sorted(Path(args.knowledge_dir).glob("*.json"))
    }
    catalog_hashes = {
        path.name: file_hash(path)
        for path in sorted(Path(args.ref_data_dir).glob("*.json"))
    }
    result = build_case_memory_release(
        import_batch=args.import_batch,
        base_release=args.base_release,
        release_dir=args.release_dir,
        reviewer=args.reviewer,
        official_diseases=flatten_disease_catalog(load_disease_catalog()),
        valid_examinations=flatten_examination_catalog(load_examination_catalog()),
        knowledge_hashes=knowledge_hashes,
        catalog_hashes=catalog_hashes,
        code_commit=args.code_commit,
        production_pointer=args.production_pointer,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
