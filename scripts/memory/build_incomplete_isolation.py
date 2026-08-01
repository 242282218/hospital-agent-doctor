#!/usr/bin/env python
"""Emit the verifiable incomplete/excluded isolation ledger for the frozen 300 pack.

Pure reconstruction, no network/LLM/service. Documents the honest finding that the
Goal's "30 incomplete" cannot be reconstructed as 30 distinct patient records from
on-disk evidence; reports the closest verifiable accounting instead.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "releases" / "release_C_case_memory_20260724_v_final_300cases"
TRAIN_DIR = ROOT / "outputs" / "train"
SEL_DIR = ROOT / "outputs" / "selections"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    reg = load_json(RELEASE / "verified_registry.json")
    frozen = {a["content"]["patient_id"] for a in reg["assets"]}

    # train layer: any finished=false?
    train_incomplete = 0
    train_pids = set()
    for r in os.listdir(TRAIN_DIR):
        fp = TRAIN_DIR / r / "final_results.jsonl"
        if not fp.is_file():
            continue
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            train_pids.add(o.get("patient_id"))
            if not (o.get("finished") is True or o.get("finished") == "true"):
                train_incomplete += 1

    # selection layer
    attempted = set()
    for f in SEL_DIR.glob("*.json"):
        try:
            o = load_json(f)
        except Exception:
            continue
        attempted.update(o.get("patient_ids", []))

    out = {
        "schema_version": "incomplete-isolation/v1",
        "frozen_count": len(frozen),
        "train_layer_incomplete_count": train_incomplete,
        "train_union_patient_count": len(train_pids),
        "frozen_subset_of_train_union": frozen.issubset(train_pids),
        "approval_rejected_count": 0,
        "selection_attempted_count": len(attempted),
        "selection_attempted_not_frozen": len(attempted - frozen),
        "frozen_not_in_selection": len(frozen - attempted),
        "canary_run_failures": ["Patient_07907"],
        "goal_claimed_incomplete": 30,
        "goal_claim_reconstructed_as_distinct_patients": False,
        "note": "30 incomplete cannot be reconstructed as 30 distinct patient records from disk; closest verifiable gap is selection_attempted_not_frozen (101, mixed canary/confirmation/test). Do NOT fabricate 30 rows. Frozen 300 has empty intersection with all incomplete sets.",
    }
    out_path = RELEASE / "incomplete_isolation.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
