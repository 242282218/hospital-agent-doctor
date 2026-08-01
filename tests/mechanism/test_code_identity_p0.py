"""P0: release code identity must bind a real git object, not merely format-valid."""
from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from agent.runtime.release_loader import load_current_release


def _write_release(root: Path, name: str, manifest_extras: dict) -> Path:
    release_dir = root / "releases" / name
    release_dir.mkdir(parents=True)
    for fname, content in {
        "prompt_pack.json": "{}",
        "policy_pack.json": "{}",
        "verified_registry.json": '{"schema_version":"verified-registry/v1","assets":[]}',
    }.items():
        (release_dir / fname).write_text(content, encoding="utf-8")
    # Live catalog/knowledge files so production-style live pins validate against
    # real files in the tmp checkout.
    catalog = root / "data" / "ref_data"
    knowledge = root / "agent" / "knowledge"
    catalog.mkdir(parents=True)
    knowledge.mkdir(parents=True)
    (catalog / "diseases_catalog.json").write_text("{}", encoding="utf-8")
    (knowledge / "alias_map.json").write_text("{}", encoding="utf-8")
    ch = sha256((catalog / "diseases_catalog.json").read_bytes()).hexdigest()
    kh = sha256((knowledge / "alias_map.json").read_bytes()).hexdigest()
    base = {
        "schema_version": "clinical-runtime/v1",
        "prompt_pack_hash": sha256(b"{}").hexdigest(),
        "policy_pack_hash": sha256(b"{}").hexdigest(),
        "registry_hash": sha256(
            b'{"schema_version":"verified-registry/v1","assets":[]}'
        ).hexdigest(),
        "catalog_hashes": {"diseases_catalog.json": ch},
        "knowledge_hashes": {"alias_map.json": kh},
    }
    base.update(manifest_extras)
    (release_dir / "release_manifest.json").write_text(
        json.dumps(base, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return release_dir


def _pointer(root: Path, release_dir: Path) -> Path:
    pointer = root / "releases" / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/%s" % release_dir.name,
                "pack_hash": sha256(
                    (release_dir / "release_manifest.json").read_bytes()
                ).hexdigest(),
                "runtime_schema_version": "clinical-runtime/v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pointer


def _real_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, cwd=Path(__file__).resolve().parents[2],
    )
    return r.stdout.decode("utf-8").strip()


def test_matching_real_commit_loads(tmp_path: Path) -> None:
    """A production pointer accepts a commit that exists in its owning repository."""
    for args in (
        ["git", "init"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Release Test"],
        ["git", "commit", "--allow-empty", "-m", "initial"],
    ):
        subprocess.run(args, check=True, capture_output=True, cwd=tmp_path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    ).stdout.strip()
    release_dir = _write_release(tmp_path, "real_pin", {"code_commit": head})
    pointer = _pointer(tmp_path, release_dir)
    loaded = load_current_release(pointer)
    assert loaded.manifest.get("code_commit") == head


def test_drifted_commit_fails_to_load(tmp_path: Path) -> None:
    """A synthetic release pinned to a nonexistent 40-hex commit must FAIL to load."""
    fake_commit = "deadbeef" * 5  # 40 hex chars, but not a real object
    release_dir = _write_release(tmp_path, "drifted", {"code_commit": fake_commit})
    pointer = _pointer(tmp_path, release_dir)
    with pytest.raises(ValueError, match="does not resolve to a git object"):
        load_current_release(pointer)


def test_format_only_invalid_commit_fails(tmp_path: Path) -> None:
    """A 40-hex string that is not a real commit must fail (not just format-checked)."""
    bogus = "0" * 40
    release_dir = _write_release(tmp_path, "bogus", {"code_commit": bogus})
    pointer = _pointer(tmp_path, release_dir)
    with pytest.raises(ValueError):
        load_current_release(pointer)
