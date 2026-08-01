#!/usr/bin/env python
"""Publish a validated typed knowledge rule pack into a release directory.

Two modes:
1) --release-dir DIR
   Backfill knowledge_rules.json into an existing release that lacks it, rewrite
   release_manifest.json with knowledge_rules_hash, and emit a receipt with the
   new pack_hash. Does not switch the production pointer.
2) --from-base NAME --to NAME
   Build a new immutable release from an existing base release, attaching the
   current clinical_pattern_rules.json as knowledge_rules.json.

Usage:
  .venv\\Scripts\\python.exe scripts/knowledge/publish_rule_pack.py \\
      --from-base release_C_full_pool_20260726_9972 \\
      --to release_C_full_pool_20260727_rules_p0 \\
      --rules agent/knowledge/clinical_pattern_rules.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "releases"
DEFAULT_RULES = ROOT / "agent" / "knowledge" / "clinical_pattern_rules.json"


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_and_validate_rules(rules_path: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack

    payload = _read_json(rules_path)
    if not isinstance(payload, Mapping):
        raise ValueError("rule pack must be a JSON object: %s" % rules_path)
    pack = parse_compiled_rule_pack(payload)
    if pack.rule_count <= 0:
        raise ValueError("refusing to publish empty rule pack")
    if not all(rule.runtime.status == "active" for rule in pack.rules):
        raise ValueError("refusing to publish rule pack with non-active rules")
    return dict(payload)


def _canonical_write_rules(release_dir: Path, rules_payload: Mapping[str, Any]) -> str:
    """Write rules with the same canonical encoding as write_immutable_json."""
    sys.path.insert(0, str(ROOT))
    from offline.artifacts import write_immutable_json

    return write_immutable_json(release_dir / "knowledge_rules.json", dict(rules_payload))


def publish_into_existing_release(
    *,
    release_dir: Path,
    rules_path: Path,
) -> Dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    if not release_dir.is_dir():
        raise FileNotFoundError("release_dir missing: %s" % release_dir)
    manifest_path = release_dir / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("release_manifest.json missing: %s" % manifest_path)
    rules_out = release_dir / "knowledge_rules.json"
    if rules_out.exists():
        raise FileExistsError("knowledge_rules.json already present: %s" % rules_out)

    rules_payload = _load_and_validate_rules(rules_path)
    prior_pack_hash = _sha256_file(manifest_path)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("release_manifest.json must be an object")
    if manifest.get("knowledge_rules_hash"):
        raise ValueError("manifest already declares knowledge_rules_hash")

    rules_hash = _canonical_write_rules(release_dir, rules_payload)
    # Recompute from bytes so loader/file_hash stay aligned even if encoding drifts.
    rules_hash = _sha256_file(rules_out)
    manifest["knowledge_rules_hash"] = rules_hash
    # Manifest rewrite is intentional backfill; keep sort_keys for stable digests.
    body = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # Atomic replace so a crash mid-write cannot leave a truncated manifest.
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, manifest_path)
    new_pack_hash = _sha256_file(manifest_path)

    sys.path.insert(0, str(ROOT))
    from offline.release import verify_release_pack

    verify_release_pack(release_dir)
    from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack

    pack = parse_compiled_rule_pack(_read_json(rules_out))
    receipt = {
        "schema_version": "publish-rule-pack-receipt/v1",
        "mode": "backfill_existing_release",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "release_dir": str(release_dir),
        "rules_source": str(rules_path),
        "prior_pack_hash": prior_pack_hash,
        "pack_hash": new_pack_hash,
        "knowledge_rules_hash": rules_hash,
        "rule_count": pack.rule_count,
        "rule_ids": [rule.rule_id for rule in pack.rules],
    }
    return receipt


def publish_derived_release(
    *,
    base_name: str,
    to_name: str,
    rules_path: Path,
) -> Dict[str, Any]:
    base_dir = (RELEASES / base_name).resolve()
    to_dir = (RELEASES / to_name).resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError("base release missing: %s" % base_dir)
    if to_dir.exists():
        raise FileExistsError("target release already exists: %s" % to_dir)
    if not to_dir.is_relative_to(RELEASES.resolve()):
        raise ValueError("target escapes releases/: %s" % to_dir)

    rules_payload = _load_and_validate_rules(rules_path)
    sys.path.insert(0, str(ROOT))
    from offline.artifacts import read_json
    from offline.release import build_candidate_pack, verify_release_pack, write_promotion_record

    base_manifest = read_json(base_dir / "release_manifest.json")
    pack = build_candidate_pack(
        release_dir=to_dir,
        code_commit=str(base_manifest.get("code_commit") or ""),
        prompt_pack=read_json(base_dir / "prompt_pack.json"),
        policy_pack=read_json(base_dir / "policy_pack.json"),
        registry=read_json(base_dir / "verified_registry.json"),
        knowledge_hashes=dict(base_manifest.get("knowledge_hashes") or {}),
        catalog_hashes=dict(base_manifest.get("catalog_hashes") or {}),
        control_report_hashes=dict(base_manifest.get("control_report_hashes") or {}),
        knowledge_rule_pack=rules_payload,
        schema_version=str(base_manifest.get("schema_version") or "clinical-runtime/v1"),
    )
    # Carry offline validation artifacts when present so switch tooling keeps working.
    for extra in (
        "promotion_record.json",
        "offline_acceptance.json",
        "offline_experiment_result.json",
        "offline_gate_report.json",
        "offline_validation_manifest.json",
        "offline_validation_plan.json",
    ):
        src = base_dir / extra
        if not src.exists():
            continue
        if extra == "promotion_record.json":
            # Re-bind promotion to the new pack hash so pointer switch stays valid.
            prior = read_json(src)
            write_promotion_record(
                path=to_dir / extra,
                candidate_pack_hash=pack["pack_hash"],
                gate_report_hash=str(prior.get("gate_report_hash") or ""),
                experiment_result_hash=str(prior.get("experiment_result_hash") or ""),
            )
            continue
        shutil.copy2(src, to_dir / extra)

    verify_release_pack(to_dir)
    from agent.knowledge.typed_rule_engine import parse_compiled_rule_pack

    typed = parse_compiled_rule_pack(read_json(to_dir / "knowledge_rules.json"))
    receipt = {
        "schema_version": "publish-rule-pack-receipt/v1",
        "mode": "derive_new_release",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_release": base_name,
        "release_dir": "releases/%s" % to_name,
        "rules_source": str(rules_path.relative_to(ROOT))
        if rules_path.is_relative_to(ROOT)
        else str(rules_path),
        "pack_hash": pack["pack_hash"],
        "knowledge_rules_hash": pack.get("knowledge_rules_hash"),
        "rule_count": typed.rule_count,
        "rule_ids": [rule.rule_id for rule in typed.rules],
    }
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help="compiled rule pack JSON (default: agent/knowledge/clinical_pattern_rules.json)",
    )
    parser.add_argument(
        "--release-dir",
        default="",
        help="existing release directory to backfill with knowledge_rules.json",
    )
    parser.add_argument(
        "--from-base",
        default="",
        help="base release name under releases/ for derived publish",
    )
    parser.add_argument(
        "--to",
        default="",
        help="new release name under releases/ for derived publish",
    )
    parser.add_argument(
        "--receipt",
        default="",
        help="optional receipt output path (default: releases/<name>_rule_pack_receipt.json)",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = (ROOT / rules_path).resolve()
    try:
        if args.release_dir:
            if args.from_base or args.to:
                raise ValueError("use either --release-dir or --from-base/--to, not both")
            release_dir = Path(args.release_dir)
            if not release_dir.is_absolute():
                release_dir = (ROOT / release_dir).resolve()
            receipt = publish_into_existing_release(
                release_dir=release_dir,
                rules_path=rules_path,
            )
            default_receipt = RELEASES / (
                "%s_rule_pack_backfill_receipt.json" % release_dir.name
            )
        else:
            if not args.from_base or not args.to:
                raise ValueError("derived mode requires both --from-base and --to")
            receipt = publish_derived_release(
                base_name=args.from_base,
                to_name=args.to,
                rules_path=rules_path,
            )
            default_receipt = RELEASES / ("%s_rule_pack_receipt.json" % args.to)
    except Exception as exc:
        print("FAIL: %s" % exc, file=sys.stderr)
        return 1

    receipt_path = Path(args.receipt) if args.receipt else default_receipt
    if not receipt_path.is_absolute():
        receipt_path = (ROOT / receipt_path).resolve()
    _write_json(receipt_path, receipt)
    print(
        "OK: published %d rules -> %s (pack %s)"
        % (
            receipt["rule_count"],
            receipt.get("release_dir"),
            receipt.get("pack_hash"),
        )
    )
    print("receipt: %s" % receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
