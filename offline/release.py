from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from offline.artifacts import content_hash, file_hash, read_json, write_immutable_json


def build_candidate_pack(
    *,
    release_dir: Path,
    code_commit: str,
    prompt_pack: Mapping[str, Any],
    policy_pack: Mapping[str, Any],
    registry: Mapping[str, Any],
    knowledge_hashes: Mapping[str, str],
    catalog_hashes: Mapping[str, str],
    control_report_hashes: Optional[Mapping[str, str]] = None,
    knowledge_rule_pack: Optional[Mapping[str, Any]] = None,
    runtime_code_files: Optional[Mapping[str, str]] = None,
    code_tree_hash: str = "",
    authority_policy: str = "",
    authority_policy_hash: str = "",
    schema_version: str = "clinical-runtime/v1",
) -> Dict[str, Any]:
    release_dir = Path(release_dir)
    target_paths = [
        release_dir / "prompt_pack.json",
        release_dir / "policy_pack.json",
        release_dir / "verified_registry.json",
        release_dir / "release_manifest.json",
    ]
    if knowledge_rule_pack is not None:
        target_paths.append(release_dir / "knowledge_rules.json")
    if any(path.exists() for path in target_paths):
        raise FileExistsError("refusing to overwrite frozen release: %s" % release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    write_immutable_json(release_dir / "prompt_pack.json", dict(prompt_pack))
    write_immutable_json(release_dir / "policy_pack.json", dict(policy_pack))
    write_immutable_json(release_dir / "verified_registry.json", dict(registry))
    knowledge_rules_hash = None
    if knowledge_rule_pack is not None:
        # Reject unparseable packs before they become an immutable release artifact.
        from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack

        parse_compiled_rule_pack(dict(knowledge_rule_pack))
        write_immutable_json(
            release_dir / "knowledge_rules.json",
            dict(knowledge_rule_pack),
        )
        knowledge_rules_hash = file_hash(release_dir / "knowledge_rules.json")
    manifest = {
        "schema_version": schema_version,
        "prompt_pack_hash": file_hash(release_dir / "prompt_pack.json"),
        "policy_pack_hash": file_hash(release_dir / "policy_pack.json"),
        "registry_hash": file_hash(release_dir / "verified_registry.json"),
        "knowledge_hashes": dict(knowledge_hashes),
        "catalog_hashes": dict(catalog_hashes),
        # Profile/reflection assets carry their held-out control evidence.
        "control_report_hashes": dict(control_report_hashes or {}),
        # No gate_report_hash in candidate pack (two-phase release).
    }
    if code_commit:
        manifest["code_commit"] = str(code_commit)
    if code_tree_hash:
        manifest["code_tree_hash"] = str(code_tree_hash)
    if knowledge_rules_hash is not None:
        manifest["knowledge_rules_hash"] = knowledge_rules_hash
    if runtime_code_files is not None:
        normalized_code_files = {str(k): str(v) for k, v in sorted(runtime_code_files.items())}
        manifest["runtime_code_files"] = normalized_code_files
        manifest["runtime_code_hash"] = content_hash(normalized_code_files)
        manifest["authority_policy"] = str(authority_policy)
        manifest["authority_policy_hash"] = str(authority_policy_hash)
    manifest_path = release_dir / "release_manifest.json"
    digest = write_immutable_json(manifest_path, manifest)
    manifest["pack_hash"] = digest
    return manifest


def write_promotion_record(
    *,
    path: Path,
    candidate_pack_hash: str,
    gate_report_hash: str,
    experiment_result_hash: str,
) -> Dict[str, Any]:
    record = {
        "schema_version": "release-promotion-record/v1",
        "candidate_pack_hash": candidate_pack_hash,
        "gate_report_hash": gate_report_hash,
        "experiment_result_hash": experiment_result_hash,
    }
    record["promotion_record_hash"] = content_hash(record)
    write_immutable_json(path, record)
    return record


def verify_release_pack(release_dir: Path) -> Dict[str, Any]:
    release_dir = Path(release_dir)
    manifest = read_json(release_dir / "release_manifest.json")
    checks = {
        "prompt": file_hash(release_dir / "prompt_pack.json") == manifest.get("prompt_pack_hash"),
        "policy": file_hash(release_dir / "policy_pack.json") == manifest.get("policy_pack_hash"),
        "registry": file_hash(release_dir / "verified_registry.json")
        == manifest.get("registry_hash"),
        "no_gate_in_pack": "gate_report_hash" not in manifest,
    }
    expected_rules_hash = manifest.get("knowledge_rules_hash")
    rules_path = release_dir / "knowledge_rules.json"
    if expected_rules_hash is None:
        checks["knowledge_rules_absent_or_optional"] = not rules_path.exists()
    else:
        checks["knowledge_rules"] = (
            rules_path.exists() and file_hash(rules_path) == expected_rules_hash
        )
    if not all(checks.values()):
        raise ValueError("release pack verification failed: %s" % checks)
    return {"verified": True, "manifest": manifest, "checks": checks}


def switch_release_pointer(pointer_path: Path, release_dir: Path) -> Dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    verified = verify_release_pack(release_dir)
    manifest = verified["manifest"]
    pack_hash = file_hash(release_dir / "release_manifest.json")
    promotion_path = release_dir / "promotion_record.json"
    promotion_hash = ""
    if promotion_path.exists():
        promotion = read_json(promotion_path)
        promotion_hash = str(promotion.get("promotion_record_hash") or "")
        if promotion.get("candidate_pack_hash") != pack_hash:
            raise ValueError("promotion record pack hash mismatch")
    # Prefer repo-relative path when pack lives under releases/.
    try:
        relative_dir = release_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        relative_dir = str(release_dir)
    pointer = {
        "schema_version": "release-pointer/v1",
        "release_dir": relative_dir,
        "pack_hash": pack_hash,
        "promotion_record_hash": promotion_hash,
        "runtime_schema_version": manifest.get("schema_version"),
    }
    pointer_path = Path(pointer_path)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp, pointer_path)
    return pointer
