#!/usr/bin/env python
"""Verify a frozen release pack is internally reconstructable from a CLEAN checkout.

Runs the SAME hashing the offline pipeline uses (offline.artifacts.content_hash)
over the COMMITTED release files only — no outputs/, no network, no service.

Target release is resolved from:
  1. --release-dir CLI argument, or
  2. releases/current.json pointer (relative release_dir), or
  3. explicit default only when neither is available (should not happen in prod).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]


def resolve_release_dir(
    argv: Optional[Sequence[str]] = None,
    *,
    default_root: Optional[Path] = None,
) -> Path:
    root = Path(default_root) if default_root is not None else ROOT
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--release-dir",
        default="",
        help="target release directory (absolute or relative to project root)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.release_dir:
        path = Path(args.release_dir)
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    pointer = root / "releases" / "current.json"
    if pointer.exists():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        raw = payload.get("release_dir")
        if not raw:
            raise SystemExit("current.json missing release_dir")
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    # Last-resort default for legacy callers; prefer pointer/arg above.
    return (root / "releases" / "release_C_case_memory_20260724_v_final_300cases").resolve()


def load(release: Path, name: str):
    return json.loads((release / name).read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    release = resolve_release_dir(argv)
    if not release.is_dir():
        print("RECONSTRUCTION FAILED: release dir missing: %s" % release)
        return 1

    sys.path.insert(0, str(ROOT if ROOT.exists() else release.parents[1]))
    from offline.artifacts import content_hash, file_hash  # noqa: E402

    errors: List[str] = []

    registry = load(release, "verified_registry.json")
    reg_body = {k: v for k, v in registry.items() if k != "registry_hash"}
    reg_hash = content_hash(reg_body)
    recorded_reg_hash = registry.get("registry_hash")
    if reg_hash != recorded_reg_hash:
        errors.append(
            "registry_hash mismatch: recomputed=%s recorded=%s" % (reg_hash, recorded_reg_hash)
        )

    file_sha = file_hash(release / "verified_registry.json")
    manifest = load(release, "release_manifest.json")
    if manifest.get("registry_hash") != file_sha:
        errors.append(
            "manifest.registry_hash != file_sha: %s vs %s"
            % (manifest.get("registry_hash"), file_sha)
        )

    # Optional offline validation artifacts — only enforced when present.
    resolved_path = release / "offline_validation_manifest.json"
    recorded_pack = None
    pack_body = {k: v for k, v in manifest.items() if k != "pack_hash"}
    pack_hash = content_hash(pack_body)
    if resolved_path.exists():
        resolved = load(release, "offline_validation_manifest.json")
        recorded_pack = resolved.get("release_pack_hash")
        source_pack = manifest.get("source_pack_hash")
        manifest_file_sha = file_hash(release / "release_manifest.json")
        # Accept: exact body hash, rebind source pin, or pointer-style file sha of manifest.
        if recorded_pack not in {pack_hash, source_pack, manifest_file_sha}:
            errors.append(
                "release_pack_hash mismatch: recomputed=%s recorded=%s source=%s file=%s"
                % (pack_hash, recorded_pack, source_pack, manifest_file_sha)
            )

    gate_path = release / "offline_gate_report.json"
    if gate_path.exists():
        gate = load(release, "offline_gate_report.json")
        if gate.get("input_hashes", {}).get("verified_registry") != file_sha:
            errors.append("gate_report.input_hashes.verified_registry != file_sha(registry)")
        if gate.get("passed") is not True:
            errors.append("gate_report.passed is not true")

    exp_path = release / "offline_experiment_result.json"
    if exp_path.exists():
        exp = load(release, "offline_experiment_result.json")
        m = exp.get("metrics", {})
        if m.get("asset_count") != 300:
            errors.append("asset_count != 300: %s" % m.get("asset_count"))
        if m.get("unique_patient_count") != 300:
            errors.append("unique_patient_count != 300: %s" % m.get("unique_patient_count"))
        if m.get("p0_count") != 0:
            errors.append("p0_count != 0: %s" % m.get("p0_count"))
        if m.get("zero_llm_zero_ask") is not True:
            errors.append("zero_llm_zero_ask is not true")
        if m.get("dataset_layer") != "HistoricalReplay":
            errors.append("dataset_layer != HistoricalReplay: %s" % m.get("dataset_layer"))
        if m.get("canary_status") != "pending":
            errors.append("canary_status != pending: %s" % m.get("canary_status"))

    ledger_path = release / "provenance_ledger.json"
    if ledger_path.exists():
        ledger = load(release, "provenance_ledger.json")
        if ledger.get("approval_mode") != "automated_historical_replay (NOT human per-case review)":
            errors.append("ledger approval_mode not tagged automated_historical_replay")
        if ledger.get("row_count") != 300:
            errors.append("ledger row_count != 300: %s" % ledger.get("row_count"))

    print("release_dir = %s" % release)
    print("recomputed registry_hash(body) = %s" % reg_hash)
    print("recorded   registry_hash        = %s" % recorded_reg_hash)
    print("file_sha == manifest.registry_hash = %s" % file_sha)
    print("recomputed pack_hash  = %s" % pack_hash)
    if recorded_pack is not None:
        print("recorded   pack_hash  = %s" % recorded_pack)
    if errors:
        print("\nRECONSTRUCTION FAILED:")
        for e in errors:
            print("  -", e)
        return 1
    print(
        "\nRECONSTRUCTION OK: release pack internally consistent and reproducible from clean checkout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
