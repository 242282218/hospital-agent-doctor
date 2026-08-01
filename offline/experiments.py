from __future__ import annotations

from typing import Any, Dict, Mapping

from offline.artifacts import content_hash, write_immutable_json
from pathlib import Path


def build_experiment_plan(plan_id: str, **fields: Any) -> Dict[str, Any]:
    plan = {
        "schema_version": "experiment-plan/v1",
        "plan_id": plan_id,
        **fields,
    }
    return plan


def write_plan(path: Path, plan: Mapping[str, Any]) -> str:
    return write_immutable_json(path, dict(plan))


def build_resolved_manifest(*, plan_hash: str, release_pack_hash: str, cases: list) -> Dict[str, Any]:
    return {
        "schema_version": "resolved-manifest/v1",
        "plan_hash": plan_hash,
        "release_pack_hash": release_pack_hash,
        "cases": list(cases),
    }


def build_result_core(*, resolved_manifest_hash: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "experiment-result-core/v1",
        "resolved_manifest_hash": resolved_manifest_hash,
        "metrics": dict(metrics),
    }


def finalize_result(*, core: Mapping[str, Any], gate_report_hash: str) -> Dict[str, Any]:
    if "gate_report_hash" in core:
        raise ValueError("ExperimentResultCore must not contain gate_report_hash")
    result = dict(core)
    result["schema_version"] = "experiment-result/v1"
    result["gate_report_hash"] = gate_report_hash
    result["result_hash"] = content_hash(result)
    return result
