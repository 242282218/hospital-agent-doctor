"""Read-only builder for curated reflection source receipts and candidates.

The curated JSONL is the only allowed reflection source. This script never
writes or generates it, never contacts an online service, and never approves
anything: it emits a recomputable receipt plus candidates that still have to
pass held-out controls (G5) and a human decision (G6).
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
from offline.candidates import create_candidate, write_candidate  # noqa: E402
from offline.reflection_sources import (  # noqa: E402
    aggregate_reflection_rules,
    build_reflection_source_receipt,
    load_reflection_sources,
)

DEFAULT_OUTPUT_ROOT = "outputs/offline/reflection_candidates"

# Approval fields are human-only (G6); this script must never emit them.
_FORBIDDEN_DECISION_KEYS = ("reviewer", "rationale", "decision", "approved_by")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project_root)
    loaded = load_reflection_sources(project_root=project_root)
    receipt = build_reflection_source_receipt(loaded)

    digest12 = receipt["normalized_source_hash"].removeprefix("sha256:")[:12]
    directory_name = (
        "source_no_curated_source"
        if receipt["status"] == "no_curated_source"
        else "source_%s" % digest12
    )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_dir = output_root / directory_name
    if run_dir.exists():
        raise FileExistsError("refusing to overwrite existing run: %s" % run_dir)

    write_immutable_json(run_dir / "source_receipt.json", receipt)

    summary: dict[str, Any] = {
        "status": receipt["status"],
        "receipt_hash": receipt["receipt_hash"],
        "raw_count": receipt["raw_count"],
        "unique_count": receipt["unique_count"],
        "rejected_count": receipt["rejected_count"],
        "build_count": receipt["build_count"],
        "held_out_count": receipt["held_out_count"],
        "candidates_written": 0,
        "decisions_written": 0,
        "online_actions": [],
    }

    # A rejected row means the curated file is not trustworthy as a whole.
    if receipt["rejected_count"] > 0:
        summary["error"] = "curated source has rejected rows; refusing to build candidates"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 3
    if receipt["status"] != "ready":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    rules = aggregate_reflection_rules(
        loaded,
        source_receipt_hash=receipt["receipt_hash"],
    )
    candidate_dir = run_dir / "candidates"
    written: list[str] = []
    for index, rule in enumerate(rules, start=1):
        candidate_id = "reflection_rule__%s_%03d" % (digest12, index)
        candidate = create_candidate(
            candidate_id=candidate_id,
            candidate_type="reflection_rule",
            proposed_effect=rule,
            evidence={
                "source_receipt_hash": receipt["receipt_hash"],
                "partition": "build",
                "support_count": rule["support_count"],
            },
        )
        write_candidate(candidate_dir / ("%s.json" % candidate_id), candidate)
        written.append(candidate_id)

    summary["candidates_written"] = len(written)
    summary["candidate_root"] = candidate_dir.as_posix()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
