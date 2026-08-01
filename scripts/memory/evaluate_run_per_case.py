"""Evaluate a final_results.jsonl per case via /evaluate/case and aggregate.

Fallback for batch /evaluate 504s on large runs: same scoring service, one
case per request, aggregated locally. Writes per-case reports next to the
input as final_results_per_case_eval.jsonl plus a summary JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from hospital_agent_sdk.client import EvaluateClient


def load_final_results(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def evaluate_one(
    client: EvaluateClient,
    final_result: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    sink,
    lock: asyncio.Lock,
    retries: int = 2,
) -> Dict[str, Any] | None:
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                report = await client.evaluate_case_async(final_result=final_result)
                if not isinstance(report, dict) or report.get("status") != "evaluated":
                    raise ValueError("report status %r" % (report.get("status"),))
                async with lock:
                    sink.write(
                        json.dumps(
                            {"patient_id": final_result.get("patient_id"), "report": report},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    sink.flush()
                return report
            except Exception as exc:
                if attempt == retries:
                    print(f"EVAL_FAILED {final_result.get('patient_id')}: {exc}", file=sys.stderr)
                    return None
                await asyncio.sleep(3.0 * (attempt + 1))
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    final_path = args.run_dir / "final_results.jsonl"
    rows = load_final_results(final_path)
    print(f"cases: {len(rows)}")

    client = EvaluateClient(
        base_url=os.environ["SERVICE_BASE_URL"],
        token=os.environ["SERVICE_TRAIN_TOKEN"],
        api_key=os.environ["MODEL_API_KEY"],
        send_api_key=True,
        team_id=os.environ["TEAM_ID"],
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out_path = args.run_dir / "final_results_per_case_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as sink:
        reports = await asyncio.gather(
            *(evaluate_one(client, row, semaphore, sink, lock) for row in rows)
        )

    scored = [r for r in reports if r is not None]
    def avg(key: str) -> float:
        values = [float(r.get(key) or 0.0) for r in scored]
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "evaluated": len(scored),
        "failed": len(rows) - len(scored),
        "diagnosis_accuracy": avg("diagnosisAccuracy"),
        "examination_precision": avg("examinationPrecision"),
        "treatment_overall_score": avg("treatmentOverallScore"),
        "treatment_safety": avg("treatmentSafety"),
        "treatment_effectiveness_alignment": avg("treatmentEffectivenessAlignment"),
        "treatment_personalization": avg("treatmentPersonalization"),
    }
    (args.run_dir / "per_case_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
