"""A6: strict candidate creation is immutable and never switches current."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "release" / "build_runtime_candidate.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("runtime_candidate_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_candidate_builder_rejects_unimplemented_authority_before_writing(
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    output_root = tmp_path / "candidates"
    source_pointer = tmp_path / "source-pointer.json"

    with pytest.raises(ValueError, match="A6 runtime only supports legacy authority policy"):
        builder.build_candidate(
            source_pointer=source_pointer,
            output_root=output_root,
            name="a6",
            authority="orchestrator,legacy,legacy,legacy",
        )

    assert not output_root.exists()


def test_runtime_candidate_builder_refuses_existing_stable_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    source_pointer = tmp_path / "source-pointer.json"
    source_pointer.write_text('{"unchanged":true}', encoding="utf-8")
    output_root = tmp_path / "candidates"
    output_root.mkdir()
    stable_pointer = output_root / "a6-pointer.json"
    stable_pointer.write_text('{"immutable":true}', encoding="utf-8")
    original = stable_pointer.read_bytes()

    class _Source:
        pointer = {"pack_hash": "source"}

    monkeypatch.setattr(builder, "load_current_release", lambda pointer: _Source())
    with pytest.raises(FileExistsError, match="refusing to overwrite runtime candidate"):
        builder.build_candidate(
            source_pointer=source_pointer,
            output_root=output_root,
            name="a6",
            authority="legacy,legacy,legacy,legacy",
        )

    assert stable_pointer.read_bytes() == original
    assert source_pointer.read_text(encoding="utf-8") == '{"unchanged":true}'


def test_runtime_candidate_builder_uses_portable_repo_relative_release_path() -> None:
    builder = _load_builder_module()
    target = builder.BASE_DIR / "releases" / "candidate"

    assert builder._portable_release_dir(target) == "releases/candidate"


def test_runtime_candidate_builder_keeps_external_release_path_absolute(
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    target = tmp_path / "candidate"

    assert Path(builder._portable_release_dir(target)).is_absolute()


def test_runtime_candidate_pointer_resolves_from_stable_pointer_directory(
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    target = tmp_path / "candidates" / "a6-123456789abc"
    pointer = tmp_path / "candidates" / "a6-pointer.json"

    assert builder._pointer_release_dir(target, pointer) == "a6-123456789abc"


def test_runtime_candidate_builder_refuses_existing_target_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    output_root = tmp_path / "candidates"
    output_root.mkdir()
    source_pointer = tmp_path / "source-pointer.json"
    source_pointer.write_text('{"source":"immutable"}', encoding="utf-8")
    sentinel_target = output_root / "a6-123456789abc"
    sentinel_target.mkdir()
    marker = sentinel_target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    class _Source:
        pointer = {"pack_hash": "source"}

    monkeypatch.setattr(builder, "load_current_release", lambda pointer: _Source())
    monkeypatch.setattr(builder, "_runtime_code_hashes", lambda: {"agent/file.py": "a" * 64})
    monkeypatch.setattr(builder, "content_hash", lambda value: "123456789abc" + "0" * 52)

    with pytest.raises(FileExistsError, match="refusing to overwrite runtime candidate"):
        builder.build_candidate(
            source_pointer=source_pointer,
            output_root=output_root,
            name="a6",
            authority="legacy,legacy,legacy,legacy",
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (output_root / "a6-pointer.json").exists()
