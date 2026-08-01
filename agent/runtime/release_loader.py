from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional  # noqa: F401 — List used by asset hash helper

from agent.clinical.authority_policy import parse_clinical_authority_policy
from agent.clinical.exam_axis_evidence_contract import parse_exam_axis_evidence_contract
from agent.clinical.final_submission import LoadedRuntimeIdentity
from agent.prompt import REQUIRED_RUNTIME_PROMPT_KEYS
from agent.knowledge.typed_rule_engine import (
    CompiledRulePack,
    empty_compiled_rule_pack,
    parse_compiled_rule_pack,
)


@dataclass(frozen=True)
class LoadedRelease:
    pointer: Mapping[str, Any]
    manifest: Mapping[str, Any]
    prompt_pack: Mapping[str, Any]
    policy_pack: Mapping[str, Any]
    registry: Mapping[str, Any]
    knowledge_rule_pack: CompiledRulePack
    release_dir: Path
    runtime_identity: LoadedRuntimeIdentity = LoadedRuntimeIdentity(
        status="legacy_unverified",
        identity_hash="",
    )


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalized_text_hash(path: Path) -> str:
    """Hash text files with CRLF/CR normalized to LF.

    Windows checkouts may materialize LF-pinned knowledge/catalog assets as CRLF;
    content identity for release pins must ignore that pure line-ending drift.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("missing release file: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON in %s: %s" % (path, exc)) from exc


def _resolve_release_dir(raw: str, *, pointer_path: Path) -> Path:
    path = Path(raw)
    if pointer_path.resolve().parent.name == "releases":
        if (
            path.is_absolute()
            or "\\" in raw
            or path.parts[:1] != ("releases",)
            or len(path.parts) != 2
            or path.parts[1] in {"", ".", ".."}
        ):
            raise ValueError("production release_dir must be releases/<name>")
        releases_root = pointer_path.resolve().parent
        resolved = (releases_root.parent / path).resolve()
        try:
            resolved.relative_to(releases_root)
        except ValueError as exc:
            raise ValueError("production release_dir escapes releases root") from exc
        return resolved
    if path.is_absolute():
        return path.resolve()
    return (pointer_path.parent / path).resolve()


def _project_root(pointer_path: Path) -> Path:
    pointer_path = Path(pointer_path).resolve()
    if pointer_path.parent.name == "releases":
        return pointer_path.parent.parent
    return Path.cwd().resolve()


def _validate_declared_hashes(
    declared: Any,
    *,
    root: Path,
    label: str,
    required: bool = False,
) -> None:
    """Fail-closed when manifest pins live files that drifted or are missing.

    Accept either the raw file hash or an LF-normalized hash so pure Windows
    CRLF materialization of otherwise identical JSON does not false-fail.
    """
    if declared is None:
        if required:
            raise ValueError("%s missing from release manifest" % label)
        return
    if not isinstance(declared, Mapping):
        raise ValueError("%s must be an object of name->sha256" % label)
    if not declared:
        if required:
            raise ValueError("%s must not be empty in production release" % label)
        return
    root = root.resolve()
    for name, expected in declared.items():
        relative = Path(name) if isinstance(name, str) else Path()
        if (
            not isinstance(name, str)
            or not name.strip()
            or relative.is_absolute()
            or "\\" in name
            or len(relative.parts) != 1
            or relative.parts[0] in {".", ".."}
        ):
            raise ValueError("%s has invalid file name" % label)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("%s hash for %s is not a 64-char sha256" % (label, name))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("%s path escapes root: %s" % (label, name)) from exc
        if not path.is_file():
            raise ValueError("%s file missing for %s: %s" % (label, name, path))
        raw = path.read_bytes()
        actual_raw = sha256(raw).hexdigest()
        actual_lf = sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        if expected not in {actual_raw, actual_lf}:
            raise ValueError(
                "%s hash mismatch for %s: expected %s got raw=%s lf=%s"
                % (label, name, expected, actual_raw, actual_lf)
            )


def _is_production_pointer(pointer_path: Path) -> bool:
    return Path(pointer_path).resolve().parent.name == "releases"


def _git_object_exists(commit: str, *, root: Path) -> bool:
    """Return True when commit names a real git object in this repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-t", commit],
            capture_output=True,
            timeout=10,
        )
        return (
            r.returncode == 0
            and r.stdout.decode("utf-8", "ignore").strip() == "commit"
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _validate_code_identity(
    manifest: Mapping[str, Any],
    *,
    production: bool,
    root: Path,
) -> None:
    """Require a valid code identity pin in production releases.

    Accept either:
      - code_commit: 40-hex git object name (optional '+label' suffix is stripped);
        the commit object must actually exist in the repo (not merely format-valid).
      - code_tree_hash: 64-hex content digest declared by the release author
    Format-only checks are rejected — a 40-hex string that does not resolve to a
    real git object still fails, so code drift cannot hide behind format validity.
    """
    raw_commit = manifest.get("code_commit")
    tree_hash = manifest.get("code_tree_hash")
    if not production:
        return
    if isinstance(tree_hash, str) and len(tree_hash) == 64 and all(
        ch in "0123456789abcdef" for ch in tree_hash.lower()
    ):
        runtime_hash = manifest.get("runtime_code_hash")
        if runtime_hash is not None and tree_hash.lower() != str(runtime_hash).lower():
            raise ValueError("code_tree_hash does not match runtime_code_hash")
        return
    if not isinstance(raw_commit, str) or not raw_commit.strip():
        raise ValueError("production release missing code_commit or code_tree_hash")
    commit = raw_commit.strip().split("+", 1)[0].strip().lower()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError(
            "production code_commit must be a 40-hex git object (optional +label): %s"
            % raw_commit
        )
    # Bind to a commit object in the repository that owns the release pointer.
    if not _git_object_exists(commit, root=root.resolve()):
        raise ValueError(
            "production code_commit does not resolve to a git object in this repo: %s"
            % raw_commit
        )


def _collect_asset_hashes(manifest: Mapping[str, Any]) -> Dict[str, str]:
    """Return the verifiable asset-hash pairs declared in the manifest.

    A2 binds the runtime identity digest to the assets already pinned by the
    loader: prompt/policy/registry packs, knowledge rules, and (for production
    pointers) the catalog/knowledge live file hashes. Pure file pins are
    optional so synthetic unit-test packs still produce a stable digest.
    """
    asset: Dict[str, str] = {}
    for key in (
        "prompt_pack_hash",
        "policy_pack_hash",
        "registry_hash",
        "knowledge_rules_hash",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            asset[key] = value
    for nested in ("catalog_hashes", "knowledge_hashes"):
        block = manifest.get(nested)
        if isinstance(block, Mapping):
            for name, expected in block.items():
                if isinstance(expected, str) and expected:
                    asset["%s:%s" % (nested, name)] = expected
    return asset


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(body.encode("utf-8")).hexdigest()


STRICT_RUNTIME_SCHEMA = "clinical-runtime/v2"
LEGACY_RUNTIME_SCHEMA = "clinical-runtime/v1"
STRICT_RUNTIME_FIELDS = (
    "runtime_code_files",
    "runtime_code_hash",
    "authority_policy",
    "authority_policy_hash",
)


def _strict_runtime_fields_present(manifest: Mapping[str, Any]) -> bool:
    schema = manifest.get("schema_version")
    if schema not in {LEGACY_RUNTIME_SCHEMA, STRICT_RUNTIME_SCHEMA}:
        raise ValueError("unsupported runtime schema_version: %r" % schema)
    present = [field in manifest for field in STRICT_RUNTIME_FIELDS]
    if schema == STRICT_RUNTIME_SCHEMA:
        if not all(present):
            raise ValueError("strict runtime manifest missing identity fields")
        return True
    if any(present):
        raise ValueError("legacy runtime manifest must not contain strict identity fields")
    return False


def _validate_pointer_schema(pointer: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    pointer_schema = pointer.get("schema_version")
    if pointer_schema != "release-pointer/v1":
        raise ValueError("unsupported release pointer schema_version: %r" % pointer_schema)
    runtime_schema = pointer.get("runtime_schema_version")
    manifest_schema = manifest.get("schema_version")
    if runtime_schema != manifest_schema:
        raise ValueError("pointer runtime_schema_version does not match manifest")


def _runtime_code_inventory(root: Path) -> Dict[str, str]:
    runtime_root = root / "agent"
    if not runtime_root.is_dir():
        raise ValueError("runtime code tree missing: %s" % runtime_root)
    return {
        path.relative_to(root).as_posix(): _normalized_text_hash(path)
        for path in sorted(runtime_root.rglob("*.py"))
    }


def _validate_runtime_code_files(manifest: Mapping[str, Any], *, root: Path) -> None:
    declared = manifest.get("runtime_code_files")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("runtime_code_files must be a non-empty object of path->sha256")
    normalized: Dict[str, str] = {}
    for name, expected in declared.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("runtime_code_files has invalid path")
        if not isinstance(expected, str) or len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected.lower()
        ):
            raise ValueError("runtime_code_files has invalid sha256")
        normalized[name] = expected
    actual_inventory = _runtime_code_inventory(root)
    if set(normalized) != set(actual_inventory):
        raise ValueError("runtime_code_files inventory mismatch")
    for name, expected in normalized.items():
        if actual_inventory[name] != expected:
            raise ValueError("runtime code hash mismatch for %s" % name)
    if manifest.get("runtime_code_hash") != _canonical_hash(normalized):
        raise ValueError("runtime_code_hash mismatch")


def _validate_strict_prompt_pack(prompt_pack: Any) -> None:
    if not isinstance(prompt_pack, Mapping):
        raise ValueError("strict prompt pack must be a JSON object")
    missing = sorted(REQUIRED_RUNTIME_PROMPT_KEYS.difference(prompt_pack))
    if missing:
        raise ValueError("strict prompt pack missing runtime templates: %s" % ",".join(missing))
    for key in REQUIRED_RUNTIME_PROMPT_KEYS:
        if not isinstance(prompt_pack.get(key), str) or not prompt_pack[key].strip():
            raise ValueError("strict prompt pack has invalid runtime template: %s" % key)


def _validate_authority_policy(manifest: Mapping[str, Any]) -> str:
    policy = parse_clinical_authority_policy(str(manifest.get("authority_policy") or ""))
    expected_hash = manifest.get("authority_policy_hash")
    if expected_hash != policy.identity_hash:
        raise ValueError("authority_policy_hash mismatch")
    if policy.values() != ("legacy", "legacy", "legacy", "legacy"):
        raise ValueError("A6 runtime only supports legacy authority policy")
    return policy.identity_hash


def _build_runtime_identity(
    *,
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_hash: str,
    strict: bool,
    authority_policy_hash: str = "",
) -> LoadedRuntimeIdentity:
    """Compute a LoadedRuntimeIdentity for A2.

    Status rules (plan A2 step 5):
      * asset_verified   -- the loader already proved pointer.pack_hash,
        prompt/policy/registry hashes, knowledge rules hash, and (for production
        pointers) live catalog/knowledge hashes.
      * legacy_unverified -- manifest assets are not closed, so the registry
        cannot mint any authorization ticket. A6 will introduce
        strict_verified as an additive hash that also binds runtime code; A2
        leaves that out.
    """
    pack_hash = str(pointer.get("pack_hash") or "")
    if not pack_hash or not manifest_hash:
        return LoadedRuntimeIdentity(status="legacy_unverified", identity_hash="")
    asset_hashes = _collect_asset_hashes(manifest)
    body = {
        "pack_hash": pack_hash,
        "manifest_hash": manifest_hash,
        "asset_hashes": {str(k): str(v) for k, v in sorted(asset_hashes.items())},
    }
    if strict:
        body.update(
            {
                "runtime_code_hash": str(manifest.get("runtime_code_hash") or ""),
                "prompt_pack_hash": str(manifest.get("prompt_pack_hash") or ""),
                "authority_policy_hash": authority_policy_hash,
            }
        )
    digest = _canonical_hash(body)
    return LoadedRuntimeIdentity(
        status="strict_verified" if strict else "asset_verified",
        identity_hash=digest,
    )


def load_current_release(
    pointer_path: Path = Path("releases/current.json"),
    *,
    require_live_pins: Optional[bool] = None,
    require_code_identity: Optional[bool] = None,
    require_runtime_code_pins: Optional[bool] = None,
) -> LoadedRelease:
    pointer_path = Path(pointer_path)
    if not pointer_path.exists():
        raise ValueError("release pointer missing: %s" % pointer_path)
    pointer = _read_json(pointer_path)
    if not isinstance(pointer, Mapping):
        raise ValueError("release pointer must be a JSON object")
    if "release_dir" not in pointer or "pack_hash" not in pointer:
        raise ValueError("release pointer missing release_dir or pack_hash")
    release_dir = _resolve_release_dir(str(pointer["release_dir"]), pointer_path=pointer_path)
    if not release_dir.is_dir():
        raise ValueError("release_dir does not exist: %s" % release_dir)
    manifest_path = release_dir / "release_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("release_manifest.json must be a JSON object")
    manifest_hash = _file_hash(manifest_path)
    if manifest_hash != pointer.get("pack_hash"):
        raise ValueError("pointer pack_hash does not match release_manifest.json")
    _validate_pointer_schema(pointer, manifest)
    prompt_path = release_dir / "prompt_pack.json"
    policy_path = release_dir / "policy_pack.json"
    registry_path = release_dir / "verified_registry.json"
    if _file_hash(prompt_path) != manifest.get("prompt_pack_hash"):
        raise ValueError("prompt pack hash mismatch")
    if _file_hash(policy_path) != manifest.get("policy_pack_hash"):
        raise ValueError("policy pack hash mismatch")
    if _file_hash(registry_path) != manifest.get("registry_hash"):
        raise ValueError("registry hash mismatch")
    prompt_pack = _read_json(prompt_path)

    project_root = _project_root(pointer_path)
    production = _is_production_pointer(pointer_path)
    # Live catalog/knowledge pins only apply for production-style pointers that
    # live under <project>/releases/. Synthetic unit-test packs use absolute
    # tmp paths and must not be validated against the real checkout files.
    enforce_live = production if require_live_pins is None else bool(require_live_pins)
    enforce_code = production if require_code_identity is None else bool(require_code_identity)
    enforce_runtime_code = (
        production if require_runtime_code_pins is None else bool(require_runtime_code_pins)
    )
    if enforce_live:
        _validate_declared_hashes(
            manifest.get("catalog_hashes"),
            root=project_root / "data" / "ref_data",
            label="catalog_hashes",
            required=True,
        )
        _validate_declared_hashes(
            manifest.get("knowledge_hashes"),
            root=project_root / "agent" / "knowledge",
            label="knowledge_hashes",
            required=True,
        )
    if enforce_code:
        _validate_code_identity(manifest, production=True, root=project_root)
    strict_runtime = _strict_runtime_fields_present(manifest)
    if enforce_live and strict_runtime:
        contract_name = "exam_axis_evidence_contract.json"
        knowledge_hashes = manifest.get("knowledge_hashes")
        if not isinstance(knowledge_hashes, Mapping) or contract_name not in knowledge_hashes:
            raise ValueError("strict runtime manifest missing %s pin" % contract_name)
        contract_path = project_root / "agent" / "knowledge" / contract_name
        contract = parse_exam_axis_evidence_contract(_read_json(contract_path))
        if contract is None:
            raise ValueError("strict runtime exam axis evidence contract is invalid")
    authority_policy_hash = ""
    if strict_runtime:
        if enforce_runtime_code:
            _validate_runtime_code_files(manifest, root=project_root)
        _validate_strict_prompt_pack(prompt_pack)
        authority_policy_hash = _validate_authority_policy(manifest)

    knowledge_rule_pack = empty_compiled_rule_pack()
    expected_rules_hash = manifest.get("knowledge_rules_hash")
    rules_path = release_dir / "knowledge_rules.json"
    if rules_path.exists() and expected_rules_hash is None:
        raise ValueError("unhashed knowledge rules file")
    if expected_rules_hash is not None:
        if _file_hash(rules_path) != expected_rules_hash:
            raise ValueError("knowledge rules hash mismatch")
        knowledge_rule_pack = parse_compiled_rule_pack(
            _read_json(rules_path)
        )
    runtime_identity = _build_runtime_identity(
        pointer=pointer,
        manifest=manifest,
        manifest_hash=manifest_hash,
        strict=strict_runtime,
        authority_policy_hash=authority_policy_hash,
    )
    return LoadedRelease(
        pointer=pointer,
        manifest=manifest,
        prompt_pack=prompt_pack,
        policy_pack=_read_json(policy_path),
        registry=_read_json(registry_path),
        knowledge_rule_pack=knowledge_rule_pack,
        release_dir=release_dir,
        runtime_identity=runtime_identity,
    )


__all__ = ["LoadedRelease", "LoadedRuntimeIdentity", "load_current_release"]
