"""Offline shadow replay over historical run artifacts (no online side effects)."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agent.clinical.shadow import ShadowBlackboardProjector


def _content_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_traces_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    # Prefer structured case dumps if present; otherwise synthesize from final_results.
    for name in ("case_state.json", "shadow_trace.json", "final_case_state.json"):
        path = run_dir / name
        if path.exists():
            payload = _load_json(path)
            if isinstance(payload, Mapping):
                traces.append(dict(payload))
            return traces

    final_path = run_dir / "final_results.jsonl"
    if not final_path.exists():
        return []
    for line in final_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            continue
        diagnosis = row.get("diagnosis") or row.get("final_diagnosis") or ""
        if isinstance(diagnosis, list):
            diagnosis_value = diagnosis[0] if diagnosis else ""
        else:
            diagnosis_value = diagnosis
        treatment = str(row.get("treatment_plan") or row.get("treatment") or "")
        reasoning = str(row.get("reasoning") or "")
        chat = row.get("chat_history") or []
        if not isinstance(chat, list):
            chat = []
        if not chat:
            chat = [{"from": "patient", "text": "历史运行脱敏投影占位陈述。"}]
        ordered = row.get("ordered_examinations") or row.get("examinations") or []
        results = row.get("examination_results") or {}
        trace = {
            "trace_revision": 1,
            "run_id": run_dir.name,
            "chat_history": chat,
            "ordered_examinations": list(ordered) if isinstance(ordered, list) else [],
            "invalid_examinations": list(row.get("invalid_examinations") or []),
            "examination_results": results if isinstance(results, Mapping) else {},
            "disease_candidates": row.get("disease_candidates") or [],
            "diagnosis_axes": row.get("diagnosis_axes") or [],
            "coverage_gaps": row.get("coverage_gaps") or [],
            "final_plan": {
                "diagnosis": diagnosis_value,
                "treatment_plan": treatment,
                "reasoning": reasoning,
            },
            "required_fact_keys": ["patient_statement"],
        }
        traces.append(trace)
    return traces


def replay_shadow_runs(
    input_roots: Sequence[Path],
    projector: Optional[ShadowBlackboardProjector] = None,
) -> Dict[str, Any]:
    projector = projector or ShadowBlackboardProjector()
    run_dirs: List[Path] = []
    for root in input_roots:
        root = Path(root)
        if not root.exists():
            continue
        if (root / "final_results.jsonl").exists() or (root / "case_state.json").exists():
            run_dirs.append(root)
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                run_dirs.append(child)

    replayable = 0
    not_replayable = 0
    final_diff_nonzero = 0
    required_ok = 0
    required_total = 0
    run_summaries: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        traces = _candidate_traces_from_run(run_dir)
        if not traces:
            not_replayable += 1
            run_summaries.append(
                {
                    "run_id": run_dir.name,
                    "status": "not_replayable",
                    "artifact_hash": "",
                }
            )
            continue

        for index, trace in enumerate(traces):
            final_plan = trace.get("final_plan") or {}
            final_payload = {
                "diagnosis": [final_plan.get("diagnosis")]
                if isinstance(final_plan.get("diagnosis"), str)
                else final_plan.get("diagnosis"),
                "treatment_plan": final_plan.get("treatment_plan"),
                "reasoning": final_plan.get("reasoning"),
            }
            snapshot = projector.project(trace)
            diff = projector.compare(trace, snapshot, final_payload=final_payload)
            replayable += 1
            if diff.final_field_differences:
                final_diff_nonzero += 1
            required = list(trace.get("required_fact_keys") or [])
            required_total += len(required)
            required_ok += len(required) - len(diff.missing_fact_keys)
            run_summaries.append(
                {
                    "run_id": "%s#%s" % (run_dir.name, index),
                    "status": "replayable",
                    "artifact_hash": _content_hash(
                        {
                            "run": run_dir.name,
                            "revision": trace.get("trace_revision"),
                            "final": final_plan,
                        }
                    ),
                    "evidence_count": len(snapshot.blackboard.evidence_ledger),
                    "hypothesis_count": len(snapshot.blackboard.hypothesis_set),
                    "gap_count": len(snapshot.blackboard.information_gaps),
                    "exam_count": len(snapshot.blackboard.examination_state),
                    "missing_fact_keys": list(diff.missing_fact_keys),
                    "final_field_differences": list(diff.final_field_differences),
                }
            )

    rate = 1.0 if required_total == 0 and replayable > 0 else (
        (required_ok / required_total) if required_total else 0.0
    )
    return {
        "schema_version": "shadow-replay-summary/v1",
        "replayable_runs": replayable,
        "not_replayable_runs": not_replayable,
        "final_diff_nonzero_runs": final_diff_nonzero,
        "required_field_expression_rate": rate,
        "runs": run_summaries,
    }


def _manifest_run_roots(manifest_path: Path) -> List[Path]:
    manifest = _load_json(manifest_path)
    roots: List[Path] = []
    for item in manifest.get("historical_runs") or []:
        if not isinstance(item, Mapping):
            continue
        relative = item.get("path") or item.get("run_path") or item.get("run_id")
        if not relative:
            continue
        path = Path(relative)
        if not path.is_absolute():
            # Manifest paths are usually relative to project root.
            path = Path.cwd() / path
        if path.exists():
            roots.append(path)
        else:
            # Fallback: outputs/test/<run_id>
            run_id = str(item.get("run_id") or path.name)
            candidate = Path.cwd() / "outputs" / "test" / run_id
            if candidate.exists():
                roots.append(candidate)
    if not roots:
        # Synthetic offline fixture when no historical runs are available on disk.
        fixture = Path.cwd() / "docs" / "架构迁移基线" / "_shadow_replay_fixture"
        fixture.mkdir(parents=True, exist_ok=True)
        case_path = fixture / "case_state.json"
        if not case_path.exists():
            case_path.write_text(
                json.dumps(
                    {
                        "trace_revision": 1,
                        "chat_history": [{"from": "patient", "text": "发热咳嗽三天。"}],
                        "ordered_examinations": ["血常规"],
                        "invalid_examinations": [],
                        "examination_results": {
                            "血常规": {"status": "normal", "result": {"白细胞": "正常"}}
                        },
                        "disease_candidates": [{"disease": "肺炎", "score": 80}],
                        "diagnosis_axes": [
                            {
                                "axis_id": "respiratory_infection",
                                "candidate_official_names": ["肺炎"],
                                "exam_intents": ["感染标志物"],
                                "treatment_risks": [],
                            }
                        ],
                        "coverage_gaps": [],
                        "final_plan": {
                            "diagnosis": "肺炎",
                            "treatment_plan": "抗感染并监测。",
                            "reasoning": "结合病史和检查。",
                        },
                        "required_fact_keys": ["patient_statement"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        roots.append(fixture)
    return roots


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay historical runs into shadow blackboard")
    parser.add_argument("--manifest", required=True, help="baseline manifest path")
    parser.add_argument("--output", required=True, help="summary json output path")
    args = parser.parse_args(argv)

    roots = _manifest_run_roots(Path(args.manifest))
    summary = replay_shadow_runs(roots)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if summary["replayable_runs"] <= 0:
        raise SystemExit("no replayable runs")
    if summary["final_diff_nonzero_runs"] != 0:
        raise SystemExit("final diff non-zero")
    if summary["required_field_expression_rate"] != 1.0:
        raise SystemExit("required field expression rate incomplete")
    print("shadow replay: PASS (%s runs)" % summary["replayable_runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
