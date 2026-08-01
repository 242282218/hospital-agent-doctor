"""Offline startup self-check for the deployed container.

Verifies deploy identity without touching patient/exam services or the LLM:
  1. release pointer resolves and the release loads;
  2. registry exposes the expected number of case-memory assets;
  3. build_agent() wires VerifiedOnlyMemory into the runtime agent;
  4. exact-hit dry-run: sampled patient IDs resolve to complete case memories.

Run: python scripts/startup_selfcheck.py [--expect-assets 300] [--json PATH]
Exit 0 = healthy, 1 = identity/asset failure. Designed to run inside the
ModelScope container before serving traffic and locally before push.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


def run_selfcheck(expect_assets: int, pointer_path: Path) -> dict:
    from agent.agent import build_agent, load_release_if_present

    report = {
        "status": "fail",
        "python_version": sys.version.split()[0],
        "checks": {},
    }

    pointer_path = Path(pointer_path).resolve()
    report["pointer"] = str(pointer_path)
    release = load_release_if_present(pointer_path)
    if release is None:
        report["checks"]["release_pointer"] = "missing %s" % pointer_path
        return report
    report["release_dir"] = str(release.release_dir)
    report["pack_hash"] = release.pointer.get("pack_hash")
    report["runtime_identity_status"] = release.runtime_identity.status
    report["runtime_identity_hash"] = release.runtime_identity.identity_hash
    report["runtime_code_hash"] = release.manifest.get("runtime_code_hash") or ""
    report["prompt_pack_hash"] = release.manifest.get("prompt_pack_hash") or ""
    report["authority_policy_hash"] = release.manifest.get("authority_policy_hash") or ""

    assets = list(release.registry.get("assets") or [])
    case_assets = [a for a in assets if a.get("candidate_type") == "case_memory"]
    report["checks"]["registry_asset_count"] = len(assets)
    report["checks"]["case_memory_asset_count"] = len(case_assets)
    if len(case_assets) < expect_assets:
        report["checks"]["asset_gate"] = (
            f"expected >= {expect_assets} case memories, got {len(case_assets)}"
        )
        return report

    agent = build_agent(BASE_DIR / "config.yaml", release_pointer=pointer_path)
    memory = agent.memory
    report["checks"]["memory_type"] = type(memory).__name__
    if not hasattr(memory, "load_case_memory"):
        report["checks"]["memory_gate"] = "agent memory lacks load_case_memory"
        return report

    # Exact-hit dry-run over first/middle/last case-memory IDs. Pure reads.
    patient_ids = [a["content"]["patient_id"] for a in case_assets]
    samples = sorted({patient_ids[0], patient_ids[len(patient_ids) // 2], patient_ids[-1]})
    hits = {}
    for pid in samples:
        value = memory.load_case_memory(pid)
        ok = (
            value is not None
            and bool(value.get("diagnoses"))
            and bool(value.get("examinations"))
            and bool(str(value.get("treatment_plan", "")).strip())
        )
        hits[pid] = "hit" if ok else "MISS_OR_INCOMPLETE"
    report["checks"]["exact_hit_dry_run"] = hits
    if any(v != "hit" for v in hits.values()):
        return report

    # Negative control: an unknown ID must miss (never hallucinate a memory).
    if memory.load_case_memory("Patient_SELFCHECK_NONEXISTENT") is not None:
        report["checks"]["negative_control"] = "unknown ID unexpectedly hit"
        return report
    report["checks"]["negative_control"] = "miss (correct)"

    release_pack = (agent.config or {}).get("release_pack") or {}
    report["checks"]["runtime_identity_status"] = release_pack.get("runtime_identity_status")
    report["checks"]["runtime_identity_hash"] = release_pack.get("runtime_identity_hash")
    if release_pack.get("runtime_identity_hash") != release.runtime_identity.identity_hash:
        report["checks"]["runtime_identity_gate"] = "agent identity != loaded release identity"
        return report
    if release.runtime_identity.status != "strict_verified":
        report["checks"]["runtime_identity_gate"] = "candidate must be strict_verified"
        return report
    report["checks"]["runtime_registry_asset_count"] = release_pack.get(
        "registry_asset_count"
    )
    if release_pack.get("registry_asset_count") != len(assets):
        report["checks"]["runtime_gate"] = "runtime asset count != registry count"
        return report

    # Runtime RulePack must be non-empty and match release metadata.
    agent_rule_count = int(getattr(agent.rule_pack, "rule_count", 0) or 0)
    release_rule_count = int(release.knowledge_rule_pack.rule_count or 0)
    active_count = sum(
        1
        for rule in getattr(agent.rule_pack, "rules", ())
        if getattr(getattr(rule, "runtime", None), "status", None) == "active"
    )
    config_rule_count = release_pack.get("typed_rule_count")
    report["checks"]["agent_rule_count"] = agent_rule_count
    report["checks"]["release_rule_count"] = release_rule_count
    report["checks"]["agent_active_rule_count"] = active_count
    report["checks"]["config_typed_rule_count"] = config_rule_count
    if agent_rule_count <= 0:
        report["checks"]["rule_pack_gate"] = "agent.rule_pack.rule_count must be > 0"
        return report
    if agent_rule_count != release_rule_count:
        report["checks"]["rule_pack_gate"] = (
            f"agent rule_count {agent_rule_count} != release {release_rule_count}"
        )
        return report
    if active_count != agent_rule_count:
        report["checks"]["rule_pack_gate"] = (
            f"active rules {active_count} != agent rule_count {agent_rule_count}"
        )
        return report
    if config_rule_count != agent_rule_count:
        report["checks"]["rule_pack_gate"] = (
            f"config typed_rule_count {config_rule_count} != agent {agent_rule_count}"
        )
        return report

    report["status"] = "ok"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--expect-assets", type=int, default=300)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        report = run_selfcheck(args.expect_assets, args.pointer)
    except Exception as exc:  # fail-closed: any load error is a deploy blocker
        report = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
