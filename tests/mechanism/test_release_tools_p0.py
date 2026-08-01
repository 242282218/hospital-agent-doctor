"""P0: release reconstruction arg parsing + atomic relative pointer switch."""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.release import switch_pointer as switch_mod
from scripts.memory import verify_release_reconstruction as recon_mod


def _hash_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_minimal_release(root: Path, name: str = "unit_rel") -> Path:
    release_dir = root / "releases" / name
    release_dir.mkdir(parents=True)
    for fname, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
        "promotion_record.json": '{"schema_version":"promotion-record/v1"}',
    }.items():
        (release_dir / fname).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": _hash_path(release_dir / "prompt_pack.json"),
        "policy_pack_hash": _hash_path(release_dir / "policy_pack.json"),
        "registry_hash": _hash_path(release_dir / "verified_registry.json"),
        "promotion_record_hash": _hash_path(release_dir / "promotion_record.json"),
        "knowledge_hashes": {"alias_map.json": "a" * 64},
        "catalog_hashes": {"diseases_catalog.json": "b" * 64},
        "code_commit": "a" * 40,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return release_dir


def test_verify_release_reconstruction_uses_release_dir_arg(tmp_path: Path, monkeypatch) -> None:
    release_dir = _write_minimal_release(tmp_path, "recon_target")
    # Reconstruction script needs offline artifacts fields; call resolver only.
    resolved = recon_mod.resolve_release_dir(
        ["--release-dir", str(release_dir)],
        default_root=tmp_path,
    )
    assert resolved == release_dir.resolve()


def test_verify_release_reconstruction_defaults_to_pointer(tmp_path: Path) -> None:
    release_dir = _write_minimal_release(tmp_path, "from_pointer")
    pointer = tmp_path / "releases" / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/from_pointer",
                "pack_hash": _hash_path(release_dir / "release_manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    resolved = recon_mod.resolve_release_dir([], default_root=tmp_path)
    assert resolved == release_dir.resolve()


def _real_head() -> str:
    import subprocess
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, cwd=Path(__file__).resolve().parents[2],
    )
    return r.stdout.decode("utf-8").strip()


def test_switch_pointer_atomic_relative_and_validates(tmp_path: Path, monkeypatch) -> None:
    # Build a synthetic project with two releases and live pins matching hashes.
    project = tmp_path / "proj"
    releases = project / "releases"
    a = _write_minimal_release(project, "rel_a")
    b = _write_minimal_release(project, "rel_b")
    # Live knowledge/catalog files matching pinned hashes in manifests — rewrite
    # manifests to real hashes of live files.
    knowledge = project / "agent" / "knowledge"
    catalog = project / "data" / "ref_data"
    knowledge.mkdir(parents=True)
    catalog.mkdir(parents=True)
    (knowledge / "alias_map.json").write_text('{"rules":[]}', encoding="utf-8")
    (catalog / "diseases_catalog.json").write_text('{"diseases":[]}', encoding="utf-8")
    kh = _hash_path(knowledge / "alias_map.json")
    ch = _hash_path(catalog / "diseases_catalog.json")
    for rel in (a, b):
        manifest = json.loads((rel / "release_manifest.json").read_text(encoding="utf-8"))
        manifest["knowledge_hashes"] = {"alias_map.json": kh}
        manifest["catalog_hashes"] = {"diseases_catalog.json": ch}
        manifest.pop("code_commit", None)
        manifest["code_tree_hash"] = "0" * 64
        (rel / "release_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/rel_a",
                "pack_hash": _hash_path(a / "release_manifest.json"),
                "runtime_schema_version": "clinical-runtime/v1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(switch_mod, "ROOT", project)
    monkeypatch.setattr(switch_mod, "RELEASES", releases)
    monkeypatch.setattr(switch_mod, "CURRENT", pointer)

    rc = switch_mod.switch_pointer(
        target_name="rel_b",
        reason="unit-test switch",
        expect_pack=_hash_path(b / "release_manifest.json"),
    )
    assert rc == 0
    new_pointer = json.loads(pointer.read_text(encoding="utf-8"))
    assert new_pointer["release_dir"] == "releases/rel_b"
    assert not Path(new_pointer["release_dir"]).is_absolute()
    assert new_pointer["pack_hash"] == _hash_path(b / "release_manifest.json")


def test_switch_pointer_rejects_unloadable_target(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    releases = project / "releases"
    a = _write_minimal_release(project, "rel_a")
    b = _write_minimal_release(project, "rel_b")
    # Break target pack hash intentionally after writing pointer source.
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/rel_a",
                "pack_hash": _hash_path(a / "release_manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    # Corrupt target registry without updating manifest.
    (b / "verified_registry.json").write_text('{"assets":[1]}', encoding="utf-8")
    monkeypatch.setattr(switch_mod, "ROOT", project)
    monkeypatch.setattr(switch_mod, "RELEASES", releases)
    monkeypatch.setattr(switch_mod, "CURRENT", pointer)
    prior = pointer.read_text(encoding="utf-8")
    rc = switch_mod.switch_pointer(target_name="rel_b", reason="should fail")
    assert rc == 1
    assert pointer.read_text(encoding="utf-8") == prior


# ---- Tool strictness tests (Round 2) ----

def test_recon_rejects_unknown_argument(tmp_path: Path) -> None:
    """Strict argparse: reconstruction must fail on unknown args (no silent ignore)."""
    release_dir = _write_minimal_release(tmp_path, "strict_recon")
    with pytest.raises(SystemExit):
        recon_mod.resolve_release_dir(
            ["--release-dir", str(release_dir), "--bogus-arg", "x"],
            default_root=tmp_path,
        )


def test_switch_pointer_rejects_target_escape(tmp_path: Path, monkeypatch) -> None:
    """switch_pointer must reject a target_name that escapes the releases/ directory."""
    project = tmp_path / "proj"
    releases = project / "releases"
    a = _write_minimal_release(project, "rel_a")
    pointer = releases / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/rel_a",
                "pack_hash": _hash_path(a / "release_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(switch_mod, "ROOT", project)
    monkeypatch.setattr(switch_mod, "RELEASES", releases)
    monkeypatch.setattr(switch_mod, "CURRENT", pointer)
    # Crafted target that would resolve OUTSIDE releases/ if not contained.
    rc = switch_mod.switch_pointer(target_name="../../../evil", reason="escape attempt")
    assert rc == 1, "target escape must be rejected"
