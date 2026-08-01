"""Read-only ground-truth harvest reconciliation and profile candidate builder.

This script never contacts the patient, examination or evaluation services. It
only reads harvest runs that earlier authorized sessions already wrote, emits an
auditable source receipt, and (unless --validate-only) writes aggregated profile
candidates that contain no patient IDs and no per-case answer text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from offline.artifacts import write_immutable_json  # noqa: E402
from offline.ground_truth_profiles import (  # noqa: E402
    build_source_receipt,
    load_ground_truth_records,
)

DEFAULT_SOURCE_ROOT = "outputs/train_harvest"
DEFAULT_DECLARED_POOL = 10000


def _load_catalogs() -> tuple[set[str], set[str]]:
    from agent.legacy_orchestrator import (
        flatten_disease_catalog,
        flatten_examination_catalog,
        load_disease_catalog,
        load_examination_catalog,
    )

    diseases = set(flatten_disease_catalog(load_disease_catalog()))
    examinations = set(flatten_examination_catalog(load_examination_catalog()))
    return diseases, examinations


def _load_rejected_ledger(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(data, dict):
        for key in ("rejected", "patient_ids", "rejects"):
            value = data.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("unsupported rejected ledger structure: %s" % path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--declared-pool-count", type=int, default=DEFAULT_DECLARED_POOL)
    parser.add_argument(
        "--rejected-ledger",
        default="",
        help="offline JSON ledger of patients pruned by earlier authorized runs",
    )
    parser.add_argument("--candidate-root", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project_root)
    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = project_root / source_root

    diseases, examinations = _load_catalogs()
    ledger_path = Path(args.rejected_ledger) if args.rejected_ledger else None
    if ledger_path is not None and not ledger_path.is_absolute():
        ledger_path = project_root / ledger_path
    rejected_ledger = _load_rejected_ledger(ledger_path)

    loaded = load_ground_truth_records(
        source_root=source_root,
        official_diseases=diseases,
        exam_leaf_names=examinations,
    )
    receipt = build_source_receipt(
        loaded,
        declared_pool_count=int(args.declared_pool_count),
        rejected_ledger_count=len(rejected_ledger),
        rejected_ledger_ref=(
            ledger_path.relative_to(project_root).as_posix()
            if ledger_path is not None and ledger_path.is_relative_to(project_root)
            else ""
        ),
    )

    receipt_path = Path(args.receipt)
    if not receipt_path.is_absolute():
        receipt_path = project_root / receipt_path
    write_immutable_json(receipt_path, receipt)

    summary: dict[str, Any] = {
        "receipt_path": receipt_path.as_posix(),
        "receipt_hash": receipt["receipt_hash"],
        "raw_row_count": receipt["raw_row_count"],
        "unique_count": receipt["unique_count"],
        "identical_duplicate_count": receipt["identical_duplicate_count"],
        "rejected_count": receipt["rejected_count"],
        "rejected_ledger_count": receipt["rejected_ledger_count"],
        "missing_count": receipt["missing_count"],
        "build_count": receipt["build_count"],
        "held_out_count": receipt["held_out_count"],
        "reconciled": receipt["reconciled"],
        "validate_only": bool(args.validate_only),
        "candidates_written": 0,
        "online_actions": [],
    }

    if args.validate_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if receipt["reconciled"] else 3

    if not receipt["reconciled"]:
        summary["error"] = "source accounting not reconciled; refusing to build candidates"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 3
    if not args.candidate_root:
        summary["error"] = "--candidate-root is required unless --validate-only"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    from offline.profile_candidates import build_profile_candidate_pack

    candidate_root = Path(args.candidate_root)
    if not candidate_root.is_absolute():
        candidate_root = project_root / candidate_root
    written = build_profile_candidate_pack(
        loaded,
        source_receipt=receipt,
        output_root=candidate_root,
        exam_catalog_order=sorted(examinations),
    )
    summary["candidates_written"] = len(written)
    summary["candidate_root"] = candidate_root.as_posix()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
