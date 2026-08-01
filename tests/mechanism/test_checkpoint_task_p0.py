"""T00: resumable task checkpoints must record evidence without mutating the repo."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.plan import checkpoint_task


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    (root / "tracked.txt").write_text("before\n", encoding="utf-8")
    # Mirror the production .gitignore so checkpoint writes stay outside git status.
    (root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt", ".gitignore"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _status(root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def test_capture_records_dirty_files_without_mutation(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    before = (tmp_path / "tracked.txt").read_bytes()
    result = checkpoint_task.capture(
        project_root=tmp_path,
        task_id="T00",
        phase="before",
        commands=(),
    )
    assert result["task_id"] == "T00"
    assert result["online_actions"] == []
    assert (tmp_path / "tracked.txt").read_bytes() == before
    assert (tmp_path / "outputs/offline/checkpoints/T00/before/result.json").is_file()


def test_capture_leaves_index_and_release_pointer_untouched(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    pointer = tmp_path / "releases" / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"release_dir":"releases/rel_a"}', encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    status_before = _status(tmp_path)
    pointer_before = pointer.read_bytes()

    result = checkpoint_task.capture(
        project_root=tmp_path,
        task_id="T01",
        phase="before",
        commands=(),
    )

    assert _status(tmp_path) == status_before
    assert pointer.read_bytes() == pointer_before
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert staged.stdout.strip() == ""
    assert result["baseline_dirty_files"] == ["tracked.txt"]
    assert result["release_pointer_hash"] == "sha256:" + checkpoint_task.file_hash(pointer)


def test_capture_records_commands_exit_codes_and_status(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    result = checkpoint_task.capture(
        project_root=tmp_path,
        task_id="T02",
        phase="after",
        commands=(
            {"command": "pytest tests/mechanism/x.py -q", "exit_code": 0, "passed": 4, "failed": 0},
            "git diff --check",
        ),
        status="passed",
        target_files=("tracked.txt", "missing.txt"),
    )
    assert result["phase"] == "after"
    assert result["status"] == "passed"
    assert result["commands"][0]["exit_code"] == 0
    assert result["commands"][0]["passed"] == 4
    assert result["commands"][1] == {"command": "git diff --check", "exit_code": None}
    assert result["tests_passed"] == 4
    assert result["tests_failed"] == 0
    assert result["target_file_hashes"]["tracked.txt"].startswith("sha256:")
    assert result["target_file_hashes"]["missing.txt"] is None
    assert len(result["base_head"]) == 40

    stored = json.loads(
        (tmp_path / "outputs/offline/checkpoints/T02/after/result.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == result
    assert (tmp_path / "outputs/offline/checkpoints/T02/after/git-status.txt").is_file()
    assert (tmp_path / "outputs/offline/checkpoints/T02/after/git-diff.patch").is_file()


def test_capture_refuses_to_overwrite_existing_checkpoint(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    checkpoint_task.capture(
        project_root=tmp_path, task_id="T03", phase="before", commands=()
    )
    with pytest.raises(FileExistsError):
        checkpoint_task.capture(
            project_root=tmp_path, task_id="T03", phase="before", commands=()
        )


def test_capture_rejects_unknown_phase(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    with pytest.raises(ValueError):
        checkpoint_task.capture(
            project_root=tmp_path, task_id="T04", phase="during", commands=()
        )


def test_cli_writes_checkpoint(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    exit_code = checkpoint_task.main(
        ["T05", "before", "--project-root", str(tmp_path)]
    )
    assert exit_code == 0
    assert (tmp_path / "outputs/offline/checkpoints/T05/before/result.json").is_file()
