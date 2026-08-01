"""Record resumable, side-effect free task checkpoints for the long-horizon plan.

The checkpoint only observes repository state: it never touches the git index,
the working tree or the release pointer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PHASES = ("before", "after")
STATUSES = ("in_progress", "passed", "blocked", "failed")
RELEASE_POINTER = "releases/current.json"


def file_hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _optional_file_hash(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    return "sha256:" + file_hash(path)


def _git(project_root: Path, args: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        # core.quotepath=false keeps non-ASCII evidence paths readable.
        ["git", "-c", "core.quotepath=false", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout or ""


def _parse_status(status_output: str) -> tuple[list[str], list[str]]:
    """Split porcelain status into tracked modifications and untracked paths."""
    tracked: list[str] = []
    untracked: list[str] = []
    for line in status_output.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        entry = line[3:].strip()
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1].strip()
        entry = entry.strip('"')
        if not entry:
            continue
        if code == "??":
            untracked.append(entry)
        else:
            tracked.append(entry)
    return sorted(set(tracked)), sorted(set(untracked))


def _normalize_commands(commands: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in commands:
        if isinstance(item, str):
            rows.append({"command": item, "exit_code": None})
            continue
        if not isinstance(item, Mapping):
            raise TypeError("command entries must be str or mapping")
        row: dict[str, Any] = {
            "command": str(item.get("command") or ""),
            "exit_code": item.get("exit_code"),
        }
        for key in ("passed", "failed", "skipped", "note"):
            if key in item:
                row[key] = item[key]
        rows.append(row)
    return rows


def capture(
    *,
    project_root: Path,
    task_id: str,
    phase: str,
    commands: Iterable[Any],
    status: str = "in_progress",
    target_files: Sequence[str] = (),
    changed_files: Sequence[str] | None = None,
    online_actions: Sequence[str] = (),
    blocked_reason: str = "",
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Record repository evidence without changing repository state."""
    if phase not in PHASES:
        raise ValueError("phase must be one of %s" % (PHASES,))
    if status not in STATUSES:
        raise ValueError("status must be one of %s" % (STATUSES,))
    task_id = str(task_id).strip()
    if not task_id or "/" in task_id or "\\" in task_id or task_id in {".", ".."}:
        raise ValueError("invalid task_id: %r" % (task_id,))

    project_root = Path(project_root)
    checkpoint_dir = project_root / "outputs" / "offline" / "checkpoints" / task_id / phase
    result_path = checkpoint_dir / "result.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite checkpoint: %s" % result_path)

    head_code, head_out = _git(project_root, ["rev-parse", "HEAD"])
    base_head = head_out.strip() if head_code == 0 else ""
    _, status_out = _git(project_root, ["status", "--short"])
    _, diff_out = _git(project_root, ["diff"])

    command_rows = _normalize_commands(commands)
    tests_passed = sum(int(row.get("passed") or 0) for row in command_rows)
    tests_failed = sum(int(row.get("failed") or 0) for row in command_rows)
    dirty_files, untracked_files = _parse_status(status_out)

    result: dict[str, Any] = {
        "schema_version": "task-checkpoint/v1",
        "task_id": task_id,
        "phase": phase,
        "status": status,
        "base_head": base_head,
        "commands": command_rows,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "baseline_dirty_files": dirty_files,
        "baseline_untracked_files": untracked_files,
        "changed_files": sorted(set(changed_files)) if changed_files is not None else [],
        "target_file_hashes": {
            name: _optional_file_hash(project_root / name) for name in target_files
        },
        "release_pointer_hash": _optional_file_hash(project_root / RELEASE_POINTER),
        "online_actions": [str(item) for item in online_actions],
        "blocked_reason": str(blocked_reason),
        "notes": [str(item) for item in notes],
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "git-status.txt").write_text(status_out, encoding="utf-8")
    (checkpoint_dir / "git-diff.patch").write_text(diff_out, encoding="utf-8")
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a plan task checkpoint.")
    parser.add_argument("task_id")
    parser.add_argument("phase", choices=list(PHASES))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--status", choices=list(STATUSES), default="in_progress")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--target-file", action="append", default=[])
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--blocked-reason", default="")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = capture(
        project_root=Path(args.project_root),
        task_id=args.task_id,
        phase=args.phase,
        commands=tuple(args.command),
        status=args.status,
        target_files=tuple(args.target_file),
        changed_files=tuple(args.changed_file) if args.changed_file else None,
        blocked_reason=args.blocked_reason,
        notes=tuple(args.note),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
