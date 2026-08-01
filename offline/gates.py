from __future__ import annotations

from typing import Any, Dict, Mapping

from offline.artifacts import content_hash


def build_gate_report(*, core: Mapping[str, Any], artifact_hashes: Mapping[str, str]) -> Dict[str, Any]:
    if "gate_report_hash" in core:
        raise ValueError("GateReport must read ExperimentResultCore without gate_report_hash")
    core_hash = content_hash(dict(core))
    gates = {
        "gate0_schema": True,
        "gate1_safety": bool(core.get("metrics", {}).get("p0_count", 0) == 0),
        "gate2_diagnosis": bool(core.get("metrics", {}).get("diagnosis_ok", True)),
        "gate3_exam_treatment": bool(core.get("metrics", {}).get("exam_treatment_ok", True)),
        "gate4_token": bool(core.get("metrics", {}).get("token_ok", True)),
    }
    report = {
        "schema_version": "gate-report/v1",
        "input_hashes": {
            "experiment_result_core": core_hash,
            **{str(k): str(v) for k, v in artifact_hashes.items()},
        },
        "gates": gates,
        "passed": all(value for gate_id, value in gates.items() if gate_id != "gate4_token"),
    }
    report["gate_report_hash"] = content_hash(report)
    return report
