from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from offline.artifacts import canonical_json, content_hash, file_hash
from offline.candidates import load_candidate
from offline.knowledge_compile import (
    build_knowledge_acceptance,
    compile_knowledge_rules,
)
from scripts.knowledge.build_knowledge_acceptance_controls import (
    build_active_rule_control_set,
)


_ARTIFACT_KEYS = {
    "knowledge_rules.json": "knowledge_rules",
    "knowledge_rule_controls.json": "knowledge_rule_controls",
    "offline_knowledge_acceptance.json": "offline_knowledge_acceptance",
}
_BUNDLE_KEYS = (*_ARTIFACT_KEYS.values(), "acceptance_manifest")
_MANIFEST_SCHEMA = "offline-knowledge-acceptance-run/v1"
_BATCH_ROOT_NAMES = {"source_receipt.json", "review_checklist.json", "candidates"}
_RECEIPT_FIELDS = {
    "schema_version",
    "batch_id",
    "source_files",
    "test_files",
    "candidate_hashes",
}
_CHECKLIST_FIELDS = {
    "schema_version",
    "batch_id",
    "approval_status",
    "review_package_role",
    "supersedes_batch_id",
    "rule_ids",
    "checks",
}
_BUNDLE_FIELDS = {*_BUNDLE_KEYS, "_source_state"}
_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "source_candidate_batch",
    "candidate_files",
    "batch_metadata_files",
    "artifact_hashes",
    "input_hashes",
    "acceptance_hash",
    "acceptance_status",
    "release_gate_passed",
}
_CANDIDATE_FILE_FIELDS = {"path", "sha256", "candidate_hash", "effect_hash"}
_METADATA_FILE_FIELDS = {"path", "sha256"}
_SOURCE_STATE_FIELDS = {
    "project_root",
    "candidate_batch_dir",
    "disease_catalog_path",
    "candidate_batch_tree",
    "bound_files",
}


def _is_link_like(path: Any) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(
            path.stat(follow_symlinks=False),
            "st_file_attributes",
            0,
        )
    except OSError:
        return False
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
    return bool(attributes & reparse_point)


def _existing_directory(
    value: Path,
    *,
    field: str,
    ordinary: bool = False,
) -> Path:
    raw_path = Path(value)
    if ordinary and _is_link_like(raw_path):
        raise ValueError("%s must be an ordinary directory" % field)
    path = raw_path.resolve()
    if not path.is_dir():
        raise ValueError("%s must be an existing directory" % field)
    return path


def _project_relative(path: Path, root: Path, *, field: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("%s must be inside project_root" % field) from exc
    if relative == Path("."):
        raise ValueError("%s must be inside project_root" % field)
    return relative.as_posix()


def _tree_snapshot(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if _is_link_like(path):
                result[relative] = {"kind": "link", "target": os.readlink(path)}
            elif entry.is_file(follow_symlinks=False):
                result[relative] = {"kind": "file", "sha256": file_hash(path)}
            elif entry.is_dir(follow_symlinks=False):
                result[relative] = {"kind": "directory"}
                visit(path)
            else:
                result[relative] = {"kind": "other"}

    visit(root)
    return result


def _assert_tree_unchanged(
    candidate_batch: Path,
    expected: Mapping[str, Any],
) -> None:
    try:
        actual = _tree_snapshot(candidate_batch)
    except OSError as exc:
        raise ValueError("candidate batch byte tree changed after build") from exc
    if actual != dict(expected):
        raise ValueError("candidate batch byte tree changed after build")


def _candidate_paths(candidate_batch: Path) -> list[Path]:
    candidates_dir = candidate_batch / "candidates"
    if _is_link_like(candidates_dir) or not candidates_dir.is_dir():
        raise ValueError("candidates must be a non-empty directory")
    entries = list(candidates_dir.iterdir())
    if not entries:
        raise ValueError("candidates must contain at least one JSON file")
    for path in entries:
        mode = path.stat(follow_symlinks=False).st_mode
        if _is_link_like(path) or not stat.S_ISREG(mode) or path.suffix != ".json":
            raise ValueError("candidates may contain only ordinary .json files")
    return sorted(
        entries,
        key=lambda path: path.relative_to(candidate_batch).as_posix(),
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordinary_file(path: Path, *, field: str) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise ValueError("%s must be an ordinary file" % field) from exc
    if _is_link_like(path) or not stat.S_ISREG(mode):
        raise ValueError("%s must be an ordinary file" % field)


def _canonical_object_file(path: Path, *, field: str) -> dict[str, Any]:
    _ordinary_file(path, field=field)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("%s must be canonical JSON" % field) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError("%s must be a canonical JSON object" % field)
    return value


def _strict_batch_root(candidate_batch: Path) -> None:
    entries = list(candidate_batch.iterdir())
    if {entry.name for entry in entries} != _BATCH_ROOT_NAMES:
        raise ValueError("candidate batch root has unexpected entries")
    _ordinary_file(
        candidate_batch / "source_receipt.json",
        field="source_receipt.json",
    )
    _ordinary_file(
        candidate_batch / "review_checklist.json",
        field="review_checklist.json",
    )
    candidates = candidate_batch / "candidates"
    if _is_link_like(candidates) or not candidates.is_dir():
        raise ValueError("candidates must be an ordinary directory")


def _posix_ref_list(value: Any, *, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty ref list" % field)
    refs: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("%s ref fields must be path and sha256" % field)
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("%s ref path must be project-relative POSIX" % field)
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or any(
            part in {"", ".", ".."} for part in posix_path.parts
        ):
            raise ValueError("%s ref path must be project-relative POSIX" % field)
        if not _is_sha256(item.get("sha256")):
            raise ValueError("%s ref sha256 is invalid" % field)
        refs.append({"path": relative, "sha256": item["sha256"]})
    if refs != sorted(refs, key=lambda ref: ref["path"]):
        raise ValueError("%s refs must be sorted" % field)
    if len({ref["path"] for ref in refs}) != len(refs):
        raise ValueError("%s refs contain duplicate paths" % field)
    return refs


def _receipt(path: Path) -> dict[str, Any]:
    receipt = _canonical_object_file(path, field="source_receipt.json")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("source receipt fields do not match its schema")
    if receipt.get("schema_version") != "knowledge-source-receipt/v1":
        raise ValueError("source receipt schema_version mismatch")
    if not _is_sha256(receipt.get("batch_id")):
        raise ValueError("source receipt batch_id is invalid")
    _posix_ref_list(receipt.get("source_files"), field="source_files")
    _posix_ref_list(receipt.get("test_files"), field="test_files")
    hashes = receipt.get("candidate_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("source receipt candidate_hashes must be an object")
    if not all(isinstance(key, str) and key and _is_sha256(value) for key, value in hashes.items()):
        raise ValueError("source receipt candidate_hashes is invalid")
    return receipt


def _checklist(path: Path) -> dict[str, Any]:
    checklist = _canonical_object_file(path, field="review_checklist.json")
    if set(checklist) != _CHECKLIST_FIELDS:
        raise ValueError("review checklist fields do not match its schema")
    if checklist.get("schema_version") != "knowledge-review-checklist/v2":
        raise ValueError("review checklist schema_version mismatch")
    if checklist.get("approval_status") != "pending_user_review":
        raise ValueError("review checklist approval_status mismatch")
    if checklist.get("review_package_role") != "final_review_package_candidate":
        raise ValueError("review checklist package role mismatch")
    rule_ids = checklist.get("rule_ids")
    checks = checklist.get("checks")
    if not isinstance(rule_ids, list) or not all(isinstance(item, str) and item for item in rule_ids):
        raise ValueError("review checklist rule_ids is invalid")
    if not isinstance(checks, list) or not checks or not all(
        isinstance(item, str) and item.strip() for item in checks
    ):
        raise ValueError("review checklist checks is invalid")
    supersedes = checklist.get("supersedes_batch_id")
    if supersedes is not None and not isinstance(supersedes, str):
        raise ValueError("review checklist supersedes_batch_id is invalid")
    return checklist


def _candidate_inventory(
    paths: Sequence[Path],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path, candidate in zip(paths, candidates, strict=True):
        candidate_id = candidate.get("candidate_id")
        rule_id = candidate.get("proposed_effect", {}).get("rule_id")
        if path.stem != candidate_id or candidate_id != rule_id:
            raise ValueError("candidate filename must equal candidate_id and rule_id")
        if candidate_id in inventory:
            raise ValueError("duplicate candidate_id in batch")
        inventory[candidate_id] = candidate["candidate_hash"]
    return dict(sorted(inventory.items()))


def _validate_batch_provenance(
    candidate_batch: Path,
    candidates: Sequence[Mapping[str, Any]],
    paths: Sequence[Path],
) -> list[dict[str, str]]:
    receipt_path = candidate_batch / "source_receipt.json"
    checklist_path = candidate_batch / "review_checklist.json"
    receipt = _receipt(receipt_path)
    checklist = _checklist(checklist_path)
    inventory = _candidate_inventory(paths, candidates)
    if receipt["candidate_hashes"] != inventory:
        raise ValueError("source receipt candidate_hashes inventory mismatch")
    if checklist["rule_ids"] != sorted(inventory):
        raise ValueError("review checklist rule_ids inventory mismatch")
    if receipt["batch_id"] != checklist.get("batch_id"):
        raise ValueError("receipt and checklist batch_id mismatch")
    source_refs = _posix_ref_list(receipt["source_files"], field="source_files")
    test_refs = _posix_ref_list(receipt["test_files"], field="test_files")
    for candidate in candidates:
        effect = candidate["proposed_effect"]
        if effect.get("source_refs") != source_refs or effect.get("test_refs") != test_refs:
            raise ValueError("candidate refs do not match source receipt")
    core = {
        "schema_version": "knowledge-candidate-batch/v2",
        "effect_contract_version": "typed-effect/v1",
        "source_files": source_refs,
        "test_files": test_refs,
        "candidate_hashes": inventory,
        "supersedes_batch_id": checklist["supersedes_batch_id"],
    }
    batch_id = content_hash(core)
    if batch_id != receipt["batch_id"] or batch_id != candidate_batch.name:
        raise ValueError("authoritative batch_id does not match batch directory")
    return [
        {"path": path.name, "sha256": file_hash(path)}
        for path in sorted((checklist_path, receipt_path), key=lambda item: item.name)
    ]


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _artifact_hashes(
    pack: Mapping[str, Any],
    control_set: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> dict[str, str]:
    payloads = {
        "knowledge_rules.json": pack,
        "knowledge_rule_controls.json": control_set,
        "offline_knowledge_acceptance.json": acceptance,
    }
    return {
        name: sha256(_canonical_bytes(payload)).hexdigest()
        for name, payload in payloads.items()
    }


def _bound_files(
    acceptance: Mapping[str, Any],
) -> dict[str, str]:
    bound = dict(acceptance["input_hashes"]["catalog_hashes"])
    for rule in acceptance["rules"]:
        for field in ("source_refs", "test_refs"):
            for ref in rule[field]:
                existing = bound.get(ref["path"])
                if existing is not None and existing != ref["sha256"]:
                    raise ValueError("conflicting source ref hashes")
                bound[ref["path"]] = ref["sha256"]
    return dict(sorted(bound.items()))


def _candidate_manifest_items(
    paths: Sequence[Path],
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_batch: Path,
) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(candidate_batch).as_posix(),
            "sha256": file_hash(path),
            "candidate_hash": candidate["candidate_hash"],
            "effect_hash": candidate["effect_hash"],
        }
        for path, candidate in zip(paths, candidates, strict=True)
    ]


def _manifest(
    *,
    source_candidate_batch: str,
    candidate_files: Sequence[Mapping[str, str]],
    batch_metadata_files: Sequence[Mapping[str, str]],
    artifact_hashes: Mapping[str, str],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": _MANIFEST_SCHEMA,
        "source_candidate_batch": source_candidate_batch,
        "candidate_files": deepcopy(list(candidate_files)),
        "batch_metadata_files": deepcopy(list(batch_metadata_files)),
        "artifact_hashes": deepcopy(dict(artifact_hashes)),
        "input_hashes": deepcopy(acceptance["input_hashes"]),
        "acceptance_hash": acceptance["acceptance_hash"],
        "acceptance_status": acceptance["status"],
        "release_gate_passed": acceptance["release_gate_passed"],
    }
    return {**core, "run_id": content_hash(core)}


def _assemble_bundle(
    root: Path,
    candidate_batch: Path,
    catalog: Path,
    batch_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    _strict_batch_root(candidate_batch)
    paths = _candidate_paths(candidate_batch)
    candidates = [load_candidate(path, project_root=root) for path in paths]
    batch_metadata_files = _validate_batch_provenance(
        candidate_batch,
        candidates,
        paths,
    )
    pack = compile_knowledge_rules(candidates, project_root=root)
    active_rule_ids = tuple(
        sorted(
            rule["rule_id"]
            for rule in pack["rules"]
            if rule["runtime"]["status"] == "active"
        )
    )
    control_set = build_active_rule_control_set(
        pack["rules_hash"],
        disease_catalog_hash=file_hash(catalog),
        active_rule_ids=active_rule_ids,
    )
    acceptance = build_knowledge_acceptance(
        pack,
        control_set,
        project_root=root,
        disease_catalog_path=catalog,
    )
    manifest = _manifest(
        source_candidate_batch=_project_relative(
            candidate_batch,
            root,
            field="candidate_batch_dir",
        ),
        candidate_files=_candidate_manifest_items(
            paths,
            candidates,
            candidate_batch=candidate_batch,
        ),
        batch_metadata_files=batch_metadata_files,
        artifact_hashes=_artifact_hashes(pack, control_set, acceptance),
        acceptance=acceptance,
    )
    return {
        "knowledge_rules": pack,
        "knowledge_rule_controls": control_set,
        "offline_knowledge_acceptance": acceptance,
        "acceptance_manifest": manifest,
        "_source_state": {
            "project_root": str(root),
            "candidate_batch_dir": str(candidate_batch),
            "disease_catalog_path": str(catalog),
            "candidate_batch_tree": deepcopy(dict(batch_snapshot)),
            "bound_files": _bound_files(acceptance),
        },
    }


def build_knowledge_acceptance_run(
    project_root: Path,
    candidate_batch_dir: Path,
    disease_catalog_path: Path,
) -> dict[str, Any]:
    root = _existing_directory(project_root, field="project_root")
    candidate_batch = _existing_directory(
        candidate_batch_dir,
        field="candidate_batch_dir",
        ordinary=True,
    )
    _project_relative(candidate_batch, root, field="candidate_batch_dir")
    catalog = Path(disease_catalog_path).resolve()
    before = _tree_snapshot(candidate_batch)
    try:
        return _assemble_bundle(root, candidate_batch, catalog, before)
    finally:
        _assert_tree_unchanged(candidate_batch, before)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % field)
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        raise ValueError("%s fields do not exactly match its schema" % field)


def _relative_posix(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("%s must be project-relative POSIX" % field)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("%s must be project-relative POSIX" % field)
    return path


def _candidate_file_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_files must be a non-empty list")
    items: list[Mapping[str, Any]] = []
    for raw in value:
        item = _mapping(raw, field="candidate_files item")
        _exact_fields(item, _CANDIDATE_FILE_FIELDS, field="candidate_files item")
        path = _relative_posix(item.get("path"), field="candidate file path")
        if len(path.parts) != 2 or path.parts[0] != "candidates" or path.suffix != ".json":
            raise ValueError("candidate file path must be candidates/<id>.json")
        if not all(_is_sha256(item.get(key)) for key in ("sha256", "candidate_hash", "effect_hash")):
            raise ValueError("candidate file hashes must be lowercase sha256")
        items.append(item)
    paths = [item["path"] for item in items]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("candidate_files paths must be unique and sorted")
    return items


def _metadata_file_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("batch_metadata_files must be a list")
    items = []
    for raw in value:
        item = _mapping(raw, field="batch_metadata_files item")
        _exact_fields(item, _METADATA_FILE_FIELDS, field="batch_metadata_files item")
        _relative_posix(item.get("path"), field="batch metadata path")
        if not _is_sha256(item.get("sha256")):
            raise ValueError("batch metadata sha256 is invalid")
        items.append(item)
    if [item["path"] for item in items] != [
        "review_checklist.json",
        "source_receipt.json",
    ]:
        raise ValueError("batch_metadata_files must bind the exact two metadata files")
    return items


def _artifact_hash_mapping(value: Any) -> Mapping[str, Any]:
    hashes = _mapping(value, field="artifact_hashes")
    if set(hashes) != set(_ARTIFACT_KEYS) or not all(
        _is_sha256(item) for item in hashes.values()
    ):
        raise ValueError("artifact_hashes must bind the exact three artifacts")
    return hashes


def _tree_schema(value: Any) -> Mapping[str, Any]:
    tree = _mapping(value, field="candidate_batch_tree")
    if not tree:
        raise ValueError("candidate_batch_tree must be non-empty")
    for relative, raw in tree.items():
        _relative_posix(relative, field="candidate batch tree path")
        entry = _mapping(raw, field="candidate batch tree entry")
        kind = entry.get("kind")
        expected = {"kind", "sha256"} if kind == "file" else {"kind"}
        _exact_fields(entry, expected, field="candidate batch tree entry")
        if kind == "file" and not _is_sha256(entry.get("sha256")):
            raise ValueError("candidate batch tree file hash is invalid")
        if kind not in {"file", "directory"}:
            raise ValueError("candidate batch tree contains a non-ordinary entry")
    return tree


def _source_state_schema(value: Any) -> Mapping[str, Any]:
    source = _mapping(value, field="_source_state")
    _exact_fields(source, _SOURCE_STATE_FIELDS, field="_source_state")
    for field in ("project_root", "candidate_batch_dir", "disease_catalog_path"):
        raw = source.get(field)
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise ValueError("_source_state %s must be an absolute path" % field)
    _tree_schema(source.get("candidate_batch_tree"))
    bound = _mapping(source.get("bound_files"), field="bound_files")
    if not bound:
        raise ValueError("bound_files must be non-empty")
    for relative, expected_hash in bound.items():
        _relative_posix(relative, field="bound file path")
        if not _is_sha256(expected_hash):
            raise ValueError("bound file sha256 is invalid")
    return source


def _json_snapshot(value: Any) -> dict[str, Any]:
    try:
        snapshot = json.loads(canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("bundle must contain only canonical JSON values") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("bundle must be an object")
    return snapshot


def _validate_acceptance(
    pack: Mapping[str, Any],
    controls: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> None:
    acceptance_hash = acceptance.get("acceptance_hash")
    core = {key: value for key, value in acceptance.items() if key != "acceptance_hash"}
    if acceptance_hash != content_hash(core):
        raise ValueError("acceptance_hash mismatch")
    inputs = _mapping(acceptance.get("input_hashes"), field="input_hashes")
    expected = {
        "compiled_pack_hash": content_hash(pack),
        "compiled_rules_hash": pack.get("rules_hash"),
        "control_set_hash": controls.get("control_set_hash"),
    }
    if any(inputs.get(key) != value for key, value in expected.items()):
        raise ValueError("acceptance input_hashes mismatch")


def _validate_manifest(
    manifest: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    _exact_fields(manifest, _MANIFEST_FIELDS, field="acceptance_manifest")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError("acceptance manifest schema mismatch")
    _relative_posix(
        manifest.get("source_candidate_batch"),
        field="source_candidate_batch",
    )
    _candidate_file_items(manifest.get("candidate_files"))
    _metadata_file_items(manifest.get("batch_metadata_files"))
    _artifact_hash_mapping(manifest.get("artifact_hashes"))
    _mapping(manifest.get("input_hashes"), field="manifest input_hashes")
    if not _is_sha256(manifest.get("acceptance_hash")):
        raise ValueError("manifest acceptance_hash is invalid")
    if not isinstance(manifest.get("acceptance_status"), str):
        raise ValueError("manifest acceptance_status is invalid")
    if not isinstance(manifest.get("release_gate_passed"), bool):
        raise ValueError("manifest release_gate_passed is invalid")
    core = {key: value for key, value in manifest.items() if key != "run_id"}
    if manifest.get("run_id") != content_hash(core):
        raise ValueError("run_id mismatch")
    expected_hashes = {
        name: sha256(_canonical_bytes(payloads[key])).hexdigest()
        for name, key in _ARTIFACT_KEYS.items()
    }
    if manifest.get("artifact_hashes") != expected_hashes:
        raise ValueError("artifact_hashes mismatch")
    acceptance = payloads["offline_knowledge_acceptance"]
    if manifest.get("input_hashes") != acceptance.get("input_hashes"):
        raise ValueError("manifest input_hashes mismatch")
    if manifest.get("acceptance_hash") != acceptance.get("acceptance_hash"):
        raise ValueError("manifest acceptance_hash mismatch")
    if manifest.get("acceptance_status") != acceptance.get("status"):
        raise ValueError("manifest acceptance_status mismatch")
    if manifest.get("release_gate_passed") != acceptance.get("release_gate_passed"):
        raise ValueError("manifest release gate mismatch")


def _validated_bundle(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    data = _mapping(bundle, field="bundle")
    _exact_fields(data, _BUNDLE_FIELDS, field="bundle")
    payloads = {
        key: _mapping(data.get(key), field=key)
        for key in _BUNDLE_KEYS[:-1]
    }
    manifest = _mapping(data.get("acceptance_manifest"), field="acceptance_manifest")
    _validate_acceptance(
        payloads["knowledge_rules"],
        payloads["knowledge_rule_controls"],
        payloads["offline_knowledge_acceptance"],
    )
    _validate_manifest(manifest, payloads)
    source_state = _source_state_schema(data.get("_source_state"))
    return {**payloads, "acceptance_manifest": manifest}, source_state


def _safe_bound_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("bound file path must be project-relative POSIX")
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("bound file path must be project-relative POSIX")
    path = (root / Path(*posix_path.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("bound file escapes project_root") from exc
    return path


def _validate_source_state(
    source_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    root = _existing_directory(Path(source_state.get("project_root", "")), field="project_root")
    candidate_batch = _existing_directory(
        Path(source_state.get("candidate_batch_dir", "")),
        field="candidate_batch_dir",
        ordinary=True,
    )
    relative_batch = _project_relative(candidate_batch, root, field="candidate_batch_dir")
    if manifest.get("source_candidate_batch") != relative_batch:
        raise ValueError("source_candidate_batch mismatch")
    snapshot = _mapping(source_state.get("candidate_batch_tree"), field="candidate_batch_tree")
    expected_tree: dict[str, dict[str, str]] = {
        "candidates": {"kind": "directory"},
    }
    for item in [
        *_candidate_file_items(manifest.get("candidate_files")),
        *_metadata_file_items(manifest.get("batch_metadata_files")),
    ]:
        expected_tree[item["path"]] = {
            "kind": "file",
            "sha256": item["sha256"],
        }
    if dict(snapshot) != expected_tree:
        raise ValueError("candidate_batch_tree does not match manifest inputs")
    _assert_tree_unchanged(candidate_batch, snapshot)
    bound_files = _mapping(source_state.get("bound_files"), field="bound_files")
    for relative, expected_hash in bound_files.items():
        path = _safe_bound_path(root, relative)
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError("bound source file hash mismatch: %s" % relative)
    catalog = Path(source_state.get("disease_catalog_path", "")).resolve()
    return root, candidate_batch, catalog


def _rebuild_and_compare(
    payloads: Mapping[str, Mapping[str, Any]],
    source_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    root, candidate_batch, catalog = _validate_source_state(
        source_state,
        payloads["acceptance_manifest"],
    )
    fresh = build_knowledge_acceptance_run(root, candidate_batch, catalog)
    fresh_payloads, fresh_source = _validated_bundle(fresh)
    for key in _BUNDLE_KEYS:
        if canonical_json(payloads[key]) != canonical_json(fresh_payloads[key]):
            raise ValueError("bundle no longer matches its source inputs")
    return fresh_source


def _artifact_root(value: Path, *, candidate_batch: Path) -> Path:
    root = Path(value).resolve()
    try:
        root.relative_to(candidate_batch)
    except ValueError:
        pass
    else:
        raise ValueError("artifact_root cannot equal or be inside candidate_batch_dir")
    if root.exists() and not root.is_dir():
        raise ValueError("artifact_root must be a directory")
    return root


def _expected_bytes(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    result = {
        name: _canonical_bytes(payloads[key])
        for name, key in _ARTIFACT_KEYS.items()
    }
    result["acceptance_manifest.json"] = _canonical_bytes(
        payloads["acceptance_manifest"]
    )
    return result


def _verify_existing(target: Path, expected: Mapping[str, bytes]) -> bool:
    if not os.path.lexists(target):
        return False
    if _is_link_like(target) or not target.is_dir():
        raise FileExistsError("acceptance run target is not an ordinary directory")
    entries = list(target.iterdir())
    if {entry.name for entry in entries} != set(expected) or len(entries) != len(expected):
        raise FileExistsError("existing acceptance run has a different file set")
    for entry in entries:
        mode = entry.stat(follow_symlinks=False).st_mode
        if _is_link_like(entry) or not stat.S_ISREG(mode):
            raise FileExistsError(
                "existing acceptance run contains a link-like or non-ordinary file entry"
            )
        if entry.read_bytes() != expected[entry.name]:
            raise FileExistsError("existing acceptance run bytes differ")
    return True


def _remove_staging(path: Path) -> None:
    if os.path.lexists(path):
        if _is_link_like(path):
            raise RuntimeError("refusing to remove a link-like staging path")
        shutil.rmtree(path)
    if os.path.lexists(path):
        raise RuntimeError("staging directory cleanup failed")


def _write_temp_directory(
    artifact_root: Path,
    run_id: str,
    expected: Mapping[str, bytes],
) -> Path:
    temp_path = Path(
        tempfile.mkdtemp(prefix=".%s-" % run_id, dir=artifact_root)
    )
    try:
        for name in sorted(expected):
            with (temp_path / name).open("xb") as handle:
                handle.write(expected[name])
        if not _verify_existing(temp_path, expected):
            raise RuntimeError("temporary acceptance run was not written")
        return temp_path
    except BaseException:
        _remove_staging(temp_path)
        raise


def _unsupported_rename(message: str, destination: Path) -> OSError:
    return OSError(errno.ENOTSUP, message, os.fspath(destination))


def _libc_symbol(name: str, *, destination: Path) -> Any:
    try:
        library = ctypes.CDLL(None, use_errno=True)
        return getattr(library, name)
    except (AttributeError, OSError) as exc:
        raise _unsupported_rename(
            "%s is unavailable; atomic no-replace rename is unsupported" % name,
            destination,
        ) from exc


def _raise_rename_errno(destination: Path) -> None:
    error = ctypes.get_errno() or errno.EIO
    if error == errno.ENOSYS:
        error = errno.ENOTSUP
    raise OSError(error, os.strerror(error), os.fspath(destination))


def _linux_rename_no_replace(source: Path, destination: Path) -> None:
    renameat2 = _libc_symbol("renameat2", destination=destination)
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        _raise_rename_errno(destination)


def _darwin_rename_no_replace(source: Path, destination: Path) -> None:
    renamex_np = _libc_symbol("renamex_np", destination=destination)
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    if result != 0:
        _raise_rename_errno(destination)


def _rename_no_replace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if sys.platform == "win32":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        _linux_rename_no_replace(source, destination)
        return
    if sys.platform == "darwin":
        _darwin_rename_no_replace(source, destination)
        return
    raise _unsupported_rename(
        "atomic no-replace rename is unsupported on %s" % sys.platform,
        destination,
    )


def _publish(
    temp_path: Path,
    target: Path,
    expected: Mapping[str, bytes],
) -> bool:
    try:
        _rename_no_replace(temp_path, target)
    except OSError:
        if _verify_existing(target, expected):
            return True
        raise
    if not _verify_existing(target, expected):
        raise RuntimeError("published acceptance run is missing")
    return False


def _rollback_published_target(
    target: Path,
    expected: Mapping[str, bytes],
    staging_identity: os.stat_result,
) -> None:
    if not os.path.lexists(target) or _is_link_like(target):
        return
    try:
        current_identity = target.stat(follow_symlinks=False)
        same_identity = os.path.samestat(staging_identity, current_identity)
        exact = same_identity and _verify_existing(target, expected)
    except OSError:
        return
    if not exact:
        return
    shutil.rmtree(target)
    if os.path.lexists(target):
        raise RuntimeError("published target rollback failed")


def _post_publish_validation(
    snapshot: Mapping[str, Any],
    source_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target: Path,
    expected: Mapping[str, bytes],
) -> None:
    _validated_bundle(snapshot)
    _validate_source_state(source_state, manifest)
    if not _verify_existing(target, expected):
        raise FileExistsError("published acceptance target disappeared")


def write_knowledge_acceptance_run(
    bundle: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    snapshot = _json_snapshot(bundle)
    payloads, source_state = _validated_bundle(snapshot)
    root, candidate_batch, _ = _validate_source_state(
        source_state,
        payloads["acceptance_manifest"],
    )
    del root
    output_root = _artifact_root(artifact_root, candidate_batch=candidate_batch)
    fresh_source = _rebuild_and_compare(payloads, source_state)
    run_id = payloads["acceptance_manifest"]["run_id"]
    expected = _expected_bytes(payloads)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / run_id
    if _verify_existing(target, expected):
        _post_publish_validation(
            snapshot,
            fresh_source,
            payloads["acceptance_manifest"],
            target,
            expected,
        )
        return {"run_id": run_id, "path": str(target), "reused": True}

    temp_path = _write_temp_directory(output_root, run_id, expected)
    staging_identity = temp_path.stat(follow_symlinks=False)
    reused: bool | None = None
    try:
        _validate_source_state(fresh_source, payloads["acceptance_manifest"])
        reused = _publish(temp_path, target, expected)
        try:
            _post_publish_validation(
                snapshot,
                fresh_source,
                payloads["acceptance_manifest"],
                target,
                expected,
            )
        except BaseException:
            if reused is False:
                _rollback_published_target(target, expected, staging_identity)
            raise
    finally:
        _remove_staging(temp_path)
    return {"run_id": run_id, "path": str(target), "reused": reused}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a candidate batch into an offline knowledge acceptance run."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidate-batch", type=Path, required=True)
    parser.add_argument("--disease-catalog", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle = build_knowledge_acceptance_run(
        args.project_root,
        args.candidate_batch,
        args.disease_catalog,
    )
    result = write_knowledge_acceptance_run(bundle, args.artifact_root)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
