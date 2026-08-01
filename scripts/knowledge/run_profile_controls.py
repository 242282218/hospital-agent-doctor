"""Generate immutable held-out control reports for profile candidates.

The script is deliberately incapable of approving anything: it never writes a
reviewer, a rationale or a promotion decision. Human approval is gate G6 and
happens outside this script. It also never contacts any online service.
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

from offline.artifacts import read_json  # noqa: E402
from offline.candidates import load_candidate  # noqa: E402
from offline.ground_truth_profiles import load_ground_truth_records  # noqa: E402
from offline.profile_controls import (  # noqa: E402
    build_exam_profile_control_report,
    build_treatment_profile_control_report,
    write_control_report,
)

_FORBIDDEN_DECISION_KEYS = ("reviewer", "rationale", "decision", "approved_by")


def _load_catalogs() -> tuple[set[str], set[str], list[str]]:
    from agent.legacy_orchestrator import (
        flatten_disease_catalog,
        flatten_examination_catalog,
        load_disease_catalog,
        load_examination_catalog,
    )

    diseases = set(flatten_disease_catalog(load_disease_catalog()))
    exam_order = flatten_examination_catalog(load_examination_catalog())
    return diseases, set(exam_order), list(exam_order)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--source-root", default="outputs/train_harvest")
    parser.add_argument("--control-store", required=True)
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project_root)

    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    candidate_root = _resolve(args.candidate_root)
    control_store = _resolve(args.control_store)
    receipt = read_json(_resolve(args.source_receipt))
    if not receipt.get("reconciled"):
        print(json.dumps({"error": "source receipt not reconciled"}, ensure_ascii=False))
        return 3

    diseases, exam_names, exam_order = _load_catalogs()
    loaded = load_ground_truth_records(
        source_root=_resolve(args.source_root),
        official_diseases=diseases,
        exam_leaf_names=exam_names,
    )
    held_out = [record for record in loaded.records if record.partition == "held_out"]

    candidate_paths = sorted(
        path
        for path in candidate_root.rglob("*.json")
        if path.name != "pack_summary.json"
    )
    written: list[dict[str, Any]] = []
    failed = 0
    for path in candidate_paths:
        candidate = load_candidate(path)
        candidate_type = candidate.get("candidate_type")
        if candidate_type not in {"disease_exam_profile", "disease_treatment_profile"}:
            continue
        profile = candidate["proposed_effect"]
        evidence = candidate.get("evidence") or {}
        source_receipt_hash = str(evidence.get("source_receipt_hash") or "")
        if candidate_type == "disease_exam_profile":
            report = build_exam_profile_control_report(
                profiles=[profile],
                held_out_records=held_out,
                candidate_hash=str(candidate["candidate_hash"]),
                source_receipt_hash=source_receipt_hash,
                held_out_partition_hash=str(receipt["held_out_partition_hash"]),
                exam_catalog_order=exam_order,
            )
        else:
            report = build_treatment_profile_control_report(
                profiles=[profile],
                held_out_records=held_out,
                candidate_hash=str(candidate["candidate_hash"]),
                source_receipt_hash=source_receipt_hash,
                held_out_partition_hash=str(receipt["held_out_partition_hash"]),
            )
        report_path = control_store / ("%s.json" % candidate["candidate_id"])
        write_control_report(report_path, report)
        if not report.passed:
            failed += 1
        written.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate_type,
                "passed": report.passed,
                "report_hash": report.report_hash,
            }
        )

    summary = {
        "control_store": control_store.as_posix(),
        "reports_written": len(written),
        "passed_count": len(written) - failed,
        "failed_count": failed,
        # G6 is a human gate: this script cannot approve or nominate anything.
        "decisions_written": 0,
        "online_actions": [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
