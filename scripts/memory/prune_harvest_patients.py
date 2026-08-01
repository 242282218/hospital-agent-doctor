"""Prune specific patients from harvest run directories (all three artifacts).

Used to drop cases whose ground-truth treatment text cannot pass the verified
case-memory acceptance (e.g. conditional-antibiotic phrasing rejected by the
anti-infective gate). Rewrites evaluation_results/final_results/events JSONL in
place, keeping the three files consistent so a fresh trust manifest + import
sees a clean run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def prune_file(path: Path, drop: set) -> int:
    if not path.exists():
        return 0
    kept, removed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("patient_id") or "") in drop:
            removed += 1
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove patients from harvest run artifacts.")
    parser.add_argument("--harvest-root", type=Path, default=BASE_DIR / "outputs" / "train_harvest")
    parser.add_argument("--patients", required=True, help="Comma-separated patient IDs to drop.")
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated run dir names to touch; other runs (possibly live) are left alone.",
    )
    args = parser.parse_args()

    drop = {p.strip() for p in args.patients.split(",") if p.strip()}
    runs = {r.strip() for r in args.runs.split(",") if r.strip()}
    if not drop:
        print("no patients given", file=sys.stderr)
        raise SystemExit(1)

    summary = {}
    for run_dir in sorted(args.harvest_root.glob("train_harvest_*")):
        if run_dir.name not in runs:
            continue
        counts = {}
        for name in ("evaluation_results.jsonl", "final_results.jsonl", "events.jsonl"):
            removed = prune_file(run_dir / name, drop)
            if removed:
                counts[name] = removed
        if counts:
            summary[run_dir.name] = counts
    print(json.dumps({"dropped": sorted(drop), "runs": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
