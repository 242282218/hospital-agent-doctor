#!/usr/bin/env python
"""Build a content-addressed 300-row provenance ledger for the frozen case-memory pack.

Pure reconstruction from on-disk artifacts. No network, no LLM, no service calls.
Emits `provenance_ledger.json` next to the release pack and prints a sha256.

Each row binds, per the Goal success-criterion 1:
  patient_id, candidate_id, run_id (source_train_run), source_line (index in that
  run's final_results.jsonl), canonical_evaluation_hash, decision_hash, effect_hash,
  candidate_hash, approval_ref (content-addressed: decision file sha256, NOT absolute
  path), registry_hash, reviewer (automated_historical_replay), decided_at,
  finished_final, required_gate_ids.

Approval is explicitly tagged `automated_historical_replay` and must NOT be read as
human per-case review.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "releases" / "release_C_case_memory_20260724_v_final_300cases"
DECISION_DIR = (
    ROOT
    / "outputs/offline/case_memory/imports/imports"
    / "9a1e148a659ba6e5f5da2c1df1c6c752eb82c13ab54b287117cb941cd93b8cdf"
    / "decisions"
)
TRAIN_DIR = ROOT / "outputs" / "train"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    registry = load_json(RELEASE / "verified_registry.json")
    registry_hash = registry.get("registry_hash")
    assets = registry.get("assets", [])

    # decision map: patient_id -> decision object
    decisions = {}
    for f in glob.glob(str(DECISION_DIR / "*.json")):
        o = load_json(Path(f))
        pid = o.get("candidate_id", "").replace("case-memory-", "")
        decisions[pid] = (o, sha256_file(Path(f)))

    # source_line: index of patient in its source_run final_results.jsonl
    # and evaluation request/result trace existence
    def find_source_line(run_id: str, pid: str):
        fp = TRAIN_DIR / run_id / "final_results.jsonl"
        if not fp.exists():
            return None, False, False
        idx = 0
        has_eval = (TRAIN_DIR / run_id / "evaluation_results.jsonl").exists()
        for i, line in enumerate(open(fp, encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("patient_id") == pid:
                idx = i
                finished = o.get("finished") is True or o.get("finished") == "true"
                return idx, finished, has_eval
        return None, False, has_eval

    rows = []
    for a in assets:
        content = a.get("content", {})
        pid = content.get("patient_id")
        dec, dec_sha = decisions.get(pid, ({}, None))
        rationale = dec.get("rationale", "")
        m_run = re.search(r"source_run=([0-9a-z_]+)", rationale)
        run_id = m_run.group(1) if m_run else None
        source_line, finished, has_eval = (None, None, False)
        if run_id:
            source_line, finished, has_eval = find_source_line(run_id, pid)

        rows.append(
            {
                "patient_id": pid,
                "candidate_id": a.get("candidate_id"),
                "candidate_type": a.get("candidate_type"),
                "run_id": run_id,
                "source_line": source_line,
                "canonical_evaluation_hash": content.get("provenance", {}).get(
                    "evaluation_hash"
                ),
                "decision_hash": a.get("decision_hash"),
                "effect_hash": a.get("effect_hash"),
                "candidate_hash": dec.get("candidate_hash"),
                "approval_sha256": dec_sha,  # content-addressed, not absolute path
                "approval_status": dec.get("decision"),
                "approval_reviewer": dec.get("reviewer"),
                "approval_mode": "automated_historical_replay",
                "decided_at": dec.get("decided_at"),
                "registry_hash": registry_hash,
                "finished_final": bool(finished) if finished is not None else None,
                "has_canonical_evaluation": has_eval,
                "required_gate_ids": dec.get("required_gate_ids"),
                "canary_required": dec.get("canary_required"),
            }
        )

    rows.sort(key=lambda r: r["patient_id"])
    out = {
        "schema_version": "provenance-ledger/v1",
        "release": RELEASE.name,
        "release_pack_hash": load_json(RELEASE / "release_manifest.json").get(
            "release_pack_hash"
        ),
        "registry_hash": registry_hash,
        "row_count": len(rows),
        "approval_mode": "automated_historical_replay (NOT human per-case review)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    out_path = RELEASE / "provenance_ledger.json"
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    digest = sha256_file(out_path)
    print(f"wrote {out_path}")
    print(f"rows={len(rows)} sha256={digest}")
    # integrity asserts
    assert len(rows) == 300, f"expected 300 rows, got {len(rows)}"
    miss_run = sum(1 for r in rows if r["run_id"] is None)
    miss_eval = sum(1 for r in rows if r["canonical_evaluation_hash"] is None)
    miss_fin = sum(1 for r in rows if r["finished_final"] is not True)
    print(f"missing run_id={miss_run} missing_eval_hash={miss_eval} not_finished={miss_fin}")


if __name__ == "__main__":
    main()
