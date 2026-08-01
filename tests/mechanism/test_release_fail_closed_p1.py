"""P1 fail-closed release loader + build_memory contracts."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from agent.memory import VerifiedOnlyMemory, build_memory
from agent.runtime.release_loader import load_current_release


def _hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _hash_path(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _write_minimal_release(tmp_path: Path, *, knowledge_hashes=None, catalog_hashes=None) -> Path:
    release_dir = tmp_path / "release_unit"
    release_dir.mkdir()
    (release_dir / "prompt_pack.json").write_text("{}", encoding="utf-8")
    (release_dir / "policy_pack.json").write_text("{}", encoding="utf-8")
    (release_dir / "verified_registry.json").write_text(
        json.dumps({"schema_version": "verified-registry/v1", "assets": []}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "knowledge_hashes": knowledge_hashes or {},
        "catalog_hashes": catalog_hashes or {},
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return release_dir


def _write_pointer(tmp_path: Path, release_dir: Path) -> Path:
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": str(release_dir),
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_production_pointer_loads_approved_registry_via_build_memory() -> None:
    memory = build_memory({})
    assert isinstance(memory, VerifiedOnlyMemory)
    loaded = load_current_release(Path("releases/current.json"))
    expected = sum(
        1
        for asset in loaded.registry.get("assets") or []
        if asset.get("candidate_type") == "case_memory"
    )
    assert expected == 9972
    assert len(memory._case_memories) == expected


def test_load_current_release_accepts_strict_production_pointer() -> None:
    loaded = load_current_release(Path("releases/current.json"))
    assert len(loaded.registry.get("assets") or []) == 9972
    assert loaded.manifest.get("schema_version") == "clinical-runtime/v2"
    assert loaded.runtime_identity.status == "strict_verified"
    assert loaded.runtime_identity.identity_hash
    # Registry file hash pinned by the manifest must match the file on disk.
    release_dir = Path(str(loaded.pointer.get("release_dir")))
    assert loaded.manifest.get("registry_hash") == _hash_path(
        release_dir / "verified_registry.json"
    )
    assert len(str(loaded.pointer.get("pack_hash") or "")) == 64
    # Container-relative path, not a Windows absolute path.
    assert not Path(str(loaded.pointer.get("release_dir"))).is_absolute()
    assert str(loaded.pointer.get("release_dir")).startswith("releases/")


def test_loader_rejects_pack_hash_mismatch(tmp_path: Path) -> None:
    release_dir = _write_minimal_release(tmp_path)
    pointer = _write_pointer(tmp_path, release_dir)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["pack_hash"] = "0" * 64
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pack_hash"):
        load_current_release(pointer)


def test_loader_rejects_invalid_pointer_json(tmp_path: Path) -> None:
    pointer = tmp_path / "current.json"
    pointer.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_current_release(pointer)


@pytest.mark.parametrize(
    ("manifest_schema", "pointer_schema"),
    [
        ("clinical-runtime/v1", "clinical-runtime/v2"),
        ("clinical-runtime/v2", "clinical-runtime/v1"),
    ],
)
def test_loader_rejects_pointer_runtime_schema_mismatch(
    tmp_path: Path,
    manifest_schema: str,
    pointer_schema: str,
) -> None:
    release_dir = _write_minimal_release(tmp_path)
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = manifest_schema
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = _write_pointer(tmp_path, release_dir)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["runtime_schema_version"] = pointer_schema
    payload["pack_hash"] = _hash_path(manifest_path)
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_schema_version does not match manifest"):
        load_current_release(pointer)


def test_loader_rejects_missing_release_dir(tmp_path: Path) -> None:
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "release_dir": str(tmp_path / "missing"),
                "pack_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="release_dir does not exist"):
        load_current_release(pointer)


def test_loader_rejects_knowledge_hash_drift(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "agent" / "knowledge"
    knowledge_root.mkdir(parents=True)
    live = knowledge_root / "alias_map.json"
    live.write_text('{"rules":[]}', encoding="utf-8")
    # Point catalog/knowledge roots by using a pointer under a fake project:
    # release_loader resolves project root as parent of releases/.
    project = tmp_path / "proj"
    releases = project / "releases"
    releases.mkdir(parents=True)
    release_dir = _write_minimal_release(
        releases,
        knowledge_hashes={"alias_map.json": "b" * 64},
    )
    # Move release into project/releases
    target = releases / "release_unit"
    if release_dir != target:
        # _write_minimal_release created under releases/release_unit already when
        # tmp_path was releases; ensure layout.
        pass
    # Rebuild with correct parent layout
    release_dir = releases / "unit"
    release_dir.mkdir(exist_ok=True)
    for name, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
    }.items():
        (release_dir / name).write_text(content, encoding="utf-8")
    # Place live knowledge where loader expects: project/agent/knowledge
    live_root = project / "agent" / "knowledge"
    live_root.mkdir(parents=True)
    (live_root / "alias_map.json").write_text('{"rules":[]}', encoding="utf-8")
    catalog_root = project / "data" / "ref_data"
    catalog_root.mkdir(parents=True)
    (catalog_root / "diseases_catalog.json").write_text('{"diseases":[]}', encoding="utf-8")
    wrong_hash = "c" * 64
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "knowledge_hashes": {"alias_map.json": wrong_hash},
        "catalog_hashes": {
            "diseases_catalog.json": _hash_path(catalog_root / "diseases_catalog.json")
        },
        "code_commit": "a" * 40,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/unit",
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="knowledge_hashes hash mismatch"):
        load_current_release(pointer)


def test_loader_rejects_catalog_hash_drift(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    releases = project / "releases"
    release_dir = releases / "unit"
    release_dir.mkdir(parents=True)
    for name, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
    }.items():
        (release_dir / name).write_text(content, encoding="utf-8")
    catalog_root = project / "data" / "ref_data"
    catalog_root.mkdir(parents=True)
    (catalog_root / "diseases_catalog.json").write_text('{"diseases":[]}', encoding="utf-8")
    (project / "agent" / "knowledge").mkdir(parents=True)
    (project / "agent" / "knowledge" / "alias_map.json").write_text('{"rules":[]}', encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "knowledge_hashes": {
            "alias_map.json": _hash_path(project / "agent" / "knowledge" / "alias_map.json")
        },
        "catalog_hashes": {"diseases_catalog.json": "d" * 64},
        "code_commit": "a" * 40,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/unit",
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="catalog_hashes hash mismatch"):
        load_current_release(pointer)


def test_build_memory_fail_closed_on_bad_pointer(tmp_path: Path, monkeypatch) -> None:
    # Point BASE_DIR to a temp tree with a broken pointer.
    import agent.memory as memory_mod

    project = tmp_path / "proj"
    releases = project / "releases"
    releases.mkdir(parents=True)
    (releases / "current.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(memory_mod, "BASE_DIR", project)
    with pytest.raises(ValueError):
        build_memory({})


def test_build_memory_explicit_missing_registry_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="verified_registry_path does not exist"):
        build_memory(
            {
                "memory": {
                    "verified_registry_path": str(missing),
                    "allow_unpinned_registry": True,
                }
            }
        )


def test_build_memory_explicit_registry_requires_allow_flag(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text(
        json.dumps({"schema_version": "verified-registry/v1", "assets": []}),
        encoding="utf-8",
    )
    # Explicit registry_path is now an explicit dependency-injection boundary:
    # no extra flag is required; it should load directly and succeed.
    mem = build_memory({"memory": {"verified_registry_path": str(path)}})
    assert mem.registry_path == path.resolve()


def test_build_memory_missing_pointer_fail_closed_without_allow(tmp_path: Path, monkeypatch) -> None:
    import agent.memory as memory_mod

    project = tmp_path / "proj"
    (project / "releases").mkdir(parents=True)
    monkeypatch.setattr(memory_mod, "BASE_DIR", project)
    with pytest.raises(ValueError, match="release pointer missing"):
        build_memory({})


def test_loader_rejects_empty_catalog_hashes_in_production(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    releases = project / "releases"
    release_dir = releases / "unit"
    release_dir.mkdir(parents=True)
    for name, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
    }.items():
        (release_dir / name).write_text(content, encoding="utf-8")
    (project / "agent" / "knowledge").mkdir(parents=True)
    (project / "data" / "ref_data").mkdir(parents=True)
    (project / "agent" / "knowledge" / "alias_map.json").write_text("{}", encoding="utf-8")
    (project / "data" / "ref_data" / "diseases_catalog.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "knowledge_hashes": {
            "alias_map.json": _hash_path(project / "agent" / "knowledge" / "alias_map.json")
        },
        "catalog_hashes": {},
        "code_commit": "a" * 40,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/unit",
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="catalog_hashes must not be empty"):
        load_current_release(pointer)


def test_loader_rejects_invalid_code_commit_in_production(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    releases = project / "releases"
    release_dir = releases / "unit"
    release_dir.mkdir(parents=True)
    for name, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
    }.items():
        (release_dir / name).write_text(content, encoding="utf-8")
    knowledge = project / "agent" / "knowledge"
    catalog = project / "data" / "ref_data"
    knowledge.mkdir(parents=True)
    catalog.mkdir(parents=True)
    (knowledge / "alias_map.json").write_text("{}", encoding="utf-8")
    (catalog / "diseases_catalog.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "knowledge_hashes": {"alias_map.json": _hash_path(knowledge / "alias_map.json")},
        "catalog_hashes": {"diseases_catalog.json": _hash_path(catalog / "diseases_catalog.json")},
        "code_commit": "not-a-git-object",
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/unit",
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="code_commit"):
        load_current_release(pointer)
