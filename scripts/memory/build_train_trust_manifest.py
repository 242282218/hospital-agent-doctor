from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from offline.artifacts import read_json
from offline.train_trust import build_train_trust_manifest, write_train_trust_manifest


BASE_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan train outputs and write a hash-anchored trust manifest for case-memory import."
    )
    parser.add_argument(
        "--train-outputs",
        type=Path,
        default=BASE_DIR / "outputs" / "train",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=None,
        help="Optional existing baseline manifest to preserve non-run fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Immutable output path for the trust manifest JSON.",
    )
    parser.add_argument(
        "--only-with-evaluation",
        action="store_true",
        help="Keep only runs that have evaluation_results.jsonl.",
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    base = None
    if args.base_manifest is not None:
        base = read_json(Path(args.base_manifest))
    manifest = build_train_trust_manifest(
        train_outputs=Path(args.train_outputs),
        base_manifest=base,
        only_with_evaluation=bool(args.only_with_evaluation),
    )
    receipt = write_train_trust_manifest(manifest, path=Path(args.output))
    return {
        "manifest_hash": manifest.get("manifest_hash"),
        "trusted_train_run_count": manifest.get("trusted_train_run_count"),
        "scanned_train_run_count": manifest.get("scanned_train_run_count"),
        "output": receipt,
    }


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
