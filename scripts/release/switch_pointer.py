#!/usr/bin/env python
"""B2: atomic production-pointer switch with receipt + automatic rollback.

Pointer switch must be atomic (temp file + os.replace), write a container-relative
release_dir, fully validate the target via load_current_release before switching,
and restore the prior pointer on any post-switch failure.

This script ONLY writes current.json (the coordinator-owned pointer). It does NOT
call the service, LLM, evaluation, or patients.

Usage:
  .venv\\Scripts\\python.exe scripts/release/switch_pointer.py \\
      --to release_C_case_memory_20260724_v_final_300cases \\
      --reason "authorized switch"

Exit codes: 0 success, 1 failure (rolled back when possible, receipt written).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "releases"
CURRENT = RELEASES / "current.json"


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, obj) -> None:
    """Write JSON via temp file + os.replace (atomic on the same filesystem)."""
    path = Path(path)
    payload = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    write_text_atomic(path, payload)


def write_text_atomic(path: Path, payload: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    # Kept for receipt writes; pointer itself uses write_json_atomic.
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _validate_target_with_loader(
    target_name: str,
    pack_hash: str,
    runtime_schema_version: str,
) -> None:
    """Build a draft pointer and fully validate via production loader."""
    sys.path.insert(0, str(ROOT))
    from agent.runtime.release_loader import load_current_release

    draft = CURRENT.with_name("current.switch-draft.json")
    draft.write_text(
        json.dumps(
            {
                "schema_version": "release-pointer/v1",
                "release_dir": "releases/%s" % target_name,
                "pack_hash": pack_hash,
                "runtime_schema_version": runtime_schema_version,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        load_current_release(draft)
    finally:
        if draft.exists():
            draft.unlink()


def switch_pointer(
    *,
    target_name: str,
    reason: str,
    expect_pack: str = "",
) -> int:
    target_dir = (RELEASES / target_name).resolve()
    # Containment: the resolved target must stay inside the releases/ directory
    # so a crafted target_name (e.g. "../../etc") cannot escape.
    releases_resolved = RELEASES.resolve()
    if not target_dir.is_relative_to(releases_resolved):
        print("FAIL: target escapes releases directory: %s" % target_dir, file=sys.stderr)
        return 1
    if not (target_dir / "verified_registry.json").exists():
        print("FAIL: target release missing: %s" % target_dir, file=sys.stderr)
        return 1

    if not CURRENT.exists():
        print("FAIL: current.json missing", file=sys.stderr)
        return 1
    prior_current = read_text(CURRENT)
    prior = load_json(CURRENT)

    manifest = load_json(target_dir / "release_manifest.json")
    registry = load_json(target_dir / "verified_registry.json")
    runtime_schema_version = manifest.get("schema_version")
    if runtime_schema_version not in {"clinical-runtime/v1", "clinical-runtime/v2"}:
        print("FAIL: unsupported runtime schema: %r" % runtime_schema_version, file=sys.stderr)
        return 1
    registry_file_hash = sha256_file(target_dir / "verified_registry.json")
    if registry_file_hash != manifest.get("registry_hash"):
        print(
            "FAIL: target registry hash mismatch: %s vs manifest %s"
            % (registry_file_hash, manifest.get("registry_hash")),
            file=sys.stderr,
        )
        return 1
    pack_hash = sha256_file(target_dir / "release_manifest.json")
    if expect_pack and expect_pack != pack_hash:
        print(
            "FAIL: pack_hash %s != expected %s" % (pack_hash, expect_pack),
            file=sys.stderr,
        )
        return 1

    try:
        _validate_target_with_loader(
            target_name,
            pack_hash,
            str(runtime_schema_version),
        )
    except Exception as exc:
        print("FAIL: loader validation failed: %s" % exc, file=sys.stderr)
        return 1

    asset_count = len(registry.get("assets", []))
    relative_release = "releases/%s" % target_name
    new_pointer = {
        "schema_version": "release-pointer/v1",
        "release_dir": relative_release,
        "pack_hash": pack_hash,
        "promotion_record_hash": manifest.get("promotion_record_hash")
        or (
            sha256_file(target_dir / "promotion_record.json")
            if (target_dir / "promotion_record.json").exists()
            else ""
        ),
        "runtime_schema_version": runtime_schema_version,
    }

    receipt = {
        "schema_version": "pointer-switch-receipt/v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": reason,
        "from_release": prior.get("release_dir"),
        "from_pack_hash": prior.get("pack_hash"),
        "to_release": relative_release,
        "to_pack_hash": pack_hash,
        "target_registry_file_hash": registry_file_hash,
        "target_asset_count": asset_count,
        "rolled_back": False,
    }

    try:
        write_json_atomic(CURRENT, new_pointer)
    except Exception as exc:
        write_text_atomic(CURRENT, prior_current)
        receipt["rolled_back"] = True
        receipt["error"] = "write failed: %s" % exc
        write_json(RELEASES / ("pointer_switch_receipt_%d.json" % int(time.time())), receipt)
        print("FAIL: pointer write failed, rolled back: %s" % exc, file=sys.stderr)
        return 1

    try:
        verify = load_json(CURRENT)
        if verify.get("release_dir") != relative_release:
            raise ValueError("release_dir mismatch after write")
        if verify.get("pack_hash") != pack_hash:
            raise ValueError("pack_hash mismatch after write")
        # Full loader re-check on the live pointer.
        sys.path.insert(0, str(ROOT))
        from agent.runtime.release_loader import load_current_release

        load_current_release(CURRENT)
    except Exception as exc:
        write_text_atomic(CURRENT, prior_current)
        receipt["rolled_back"] = True
        receipt["error"] = "post-verify failed: %s" % exc
        write_json(RELEASES / ("pointer_switch_receipt_%d.json" % int(time.time())), receipt)
        print("FAIL: post-switch verify failed, rolled back: %s" % exc, file=sys.stderr)
        return 1

    write_json(RELEASES / ("pointer_switch_receipt_%d.json" % int(time.time())), receipt)
    print(
        "OK: pointer -> %s (pack %s, assets %d)"
        % (target_name, pack_hash, asset_count)
    )
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="target release dir name under releases/")
    ap.add_argument("--reason", required=True, help="why the switch is happening")
    ap.add_argument("--expect-pack", default="", help="expected pack_hash to verify")
    args = ap.parse_args(argv)
    return switch_pointer(
        target_name=args.to,
        reason=args.reason,
        expect_pack=args.expect_pack,
    )


if __name__ == "__main__":
    raise SystemExit(main())
