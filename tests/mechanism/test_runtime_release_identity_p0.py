"""P0 A6 runtime identity: strict code, prompts, and authority are closed."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from agent.clinical.authority_policy import parse_clinical_authority_policy
from agent.prompt import REQUIRED_RUNTIME_PROMPT_KEYS
from agent.runtime.release_loader import load_current_release


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalized_text_hash(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _canonical_hash(value: object) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(body.encode("utf-8")).hexdigest()


def _strict_release(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    release_dir = project / "releases" / "candidate"
    code_path = project / "agent" / "runtime_file.py"
    code_path.parent.mkdir(parents=True)
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    release_dir.mkdir(parents=True)

    import agent.prompt as runtime_prompt

    prompt_pack = {key: getattr(runtime_prompt, key) for key in REQUIRED_RUNTIME_PROMPT_KEYS}
    policy_pack = {}
    registry = {"schema_version": "verified-registry/v1", "assets": []}
    for name, body in {
        "prompt_pack.json": prompt_pack,
        "policy_pack.json": policy_pack,
        "verified_registry.json": registry,
    }.items():
        (release_dir / name).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    policy = parse_clinical_authority_policy("legacy,legacy,legacy,legacy")
    code_files = {"agent/runtime_file.py": _normalized_text_hash(code_path)}
    manifest = {
        "schema_version": "clinical-runtime/v2",
        "prompt_pack_hash": _hash(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash(release_dir / "policy_pack.json"),
        "registry_hash": _hash(release_dir / "verified_registry.json"),
        "runtime_code_files": code_files,
        "runtime_code_hash": _canonical_hash(code_files),
        "authority_policy": "legacy,legacy,legacy,legacy",
        "authority_policy_hash": policy.identity_hash,
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pointer = project / "releases" / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/candidate",
                "pack_hash": _hash(manifest_path),
                "runtime_schema_version": "clinical-runtime/v2",
            }
        ),
        encoding="utf-8",
    )
    return pointer, code_path


def test_authority_policy_allows_only_monotonic_prefixes() -> None:
    assert parse_clinical_authority_policy("orchestrator,orchestrator,legacy,legacy").exam == "orchestrator"
    for value in ("", "legacy,orchestrator,legacy,legacy", "orchestrator,legacy,orchestrator,legacy", "unknown,legacy,legacy,legacy"):
        with pytest.raises(ValueError, match="invalid clinical authority prefix"):
            parse_clinical_authority_policy(value)


def test_strict_runtime_identity_validates_code_and_prompt_assets(tmp_path: Path) -> None:
    pointer, code_path = _strict_release(tmp_path)
    loaded = load_current_release(pointer, require_live_pins=False, require_code_identity=False)
    assert loaded.runtime_identity.status == "strict_verified"
    assert loaded.runtime_identity.identity_hash

    code_path.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime code hash mismatch"):
        load_current_release(pointer, require_live_pins=False, require_code_identity=False)


def test_strict_runtime_identity_ignores_python_line_ending_drift(tmp_path: Path) -> None:
    pointer, code_path = _strict_release(tmp_path)
    code_path.write_bytes(b"VALUE = 1\r\n")

    loaded = load_current_release(
        pointer,
        require_live_pins=False,
        require_code_identity=False,
    )

    assert loaded.runtime_identity.status == "strict_verified"


def test_strict_runtime_identity_rejects_incomplete_code_inventory(tmp_path: Path) -> None:
    pointer, code_path = _strict_release(tmp_path)
    extra_code_path = code_path.parent / "unlisted_runtime_file.py"
    extra_code_path.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_code_files inventory mismatch"):
        load_current_release(pointer, require_live_pins=False, require_code_identity=False)


def test_a6_strict_identity_rejects_unimplemented_authority_policy(tmp_path: Path) -> None:
    pointer, _ = _strict_release(tmp_path)
    manifest_path = pointer.parent / "candidate" / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = parse_clinical_authority_policy("orchestrator,legacy,legacy,legacy")
    manifest["authority_policy"] = "orchestrator,legacy,legacy,legacy"
    manifest["authority_policy_hash"] = policy.identity_hash
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/candidate",
                "pack_hash": _hash(manifest_path),
                "runtime_schema_version": "clinical-runtime/v2",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="A6 runtime only supports legacy authority policy"):
        load_current_release(pointer, require_live_pins=False, require_code_identity=False)


def test_old_asset_pack_is_not_silently_upgraded(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    for name, body in {
        "prompt_pack.json": {},
        "policy_pack.json": {},
        "verified_registry.json": {"assets": []},
    }.items():
        (release_dir / name).write_text(json.dumps(body), encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash(release_dir / "policy_pack.json"),
        "registry_hash": _hash(release_dir / "verified_registry.json"),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": str(release_dir),
                "pack_hash": _hash(manifest_path),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )

    assert load_current_release(pointer).runtime_identity.status == "asset_verified"
