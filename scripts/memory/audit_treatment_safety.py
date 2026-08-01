#!/usr/bin/env python
"""Offline high-risk treatment audit over the frozen 300 case-memory pack.

Stratified static scan per Goal P1: 药物禁忌 / 剂量人群 / 急症时序.
Flags quarantine CANDIDATES; does not mutate the registry. Pure offline:
reads verified_registry.json only, no network/LLM/service.

Pattern set is conservative and intentionally narrow; it surfaces plausible
red-flag text for human review rather than asserting clinical conclusions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "releases" / "release_C_case_memory_20260724_v_final_300cases"

# (layer, flag_label, [needle substrings])
RULES: List[tuple] = [
    ("药物禁忌", "beta_blocker_in_asthma_copd", ["哮喘", "慢性阻塞性肺疾病", "支气管哮"]),
    ("药物禁忌", "nsaid_in_renal_transplant", ["肾移植", "肾功能不全"]),
    ("药物禁忌", "noac_in_mitral_stenosis", ["二尖瓣狭窄", "风湿性二尖瓣"]),
    ("剂量人群", "pediatric_weight_dose_missing", ["儿童", "患儿", "新生儿", "婴儿"]),
    ("剂量人群", "elderly_fragile", ["高龄", "老年人", "84岁", "85岁", "90岁"]),
    ("急症时序", "urgent_stabilize_before_drug", ["急性", "失代偿", "休克", "大出血", "急诊"]),
]


def scan_text(text: str) -> List[tuple]:
    hits = []
    for layer, label, needles in RULES:
        for n in needles:
            if n in text:
                hits.append((layer, label, n))
                break
    return hits


def main() -> None:
    reg = json.loads((RELEASE / "verified_registry.json").read_text(encoding="utf-8"))
    rows = []
    layer_counts: Dict[str, int] = {}
    for a in reg["assets"]:
        c = a["content"]
        tp = c.get("treatment_plan", "")
        diags = " ".join(c.get("diagnoses", []))
        corpus = tp + " " + diags
        hits = scan_text(corpus)
        if hits:
            for layer, label, needle in hits:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            rows.append(
                {
                    "patient_id": c["patient_id"],
                    "diagnoses": c.get("diagnoses"),
                    "flags": [
                        {"layer": l, "label": lb, "needle": nd} for l, lb, nd in hits
                    ],
                    "treatment_excerpt": tp[:120],
                }
            )
    out = {
        "schema_version": "treatment-safety-audit/v1",
        "pack": RELEASE.name,
        "scanned": len(reg["assets"]),
        "flagged": len(rows),
        "layer_counts": layer_counts,
        "quarantine_candidates": rows,
        "note": "Static heuristic scan only. Flags for human stratified review; NOT a clinical verdict. P0 quarantine requires confirmed 药物禁忌/剂量人群/急症时序 violation, not mere keyword presence.",
    }
    out_path = RELEASE / "treatment_safety_audit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scanned={len(reg['assets'])} flagged={len(rows)} layer_counts={layer_counts}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
