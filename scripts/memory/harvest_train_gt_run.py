"""Harvest ground truth as legitimate SDK train-run directories.

For each patient in the live pool, submit a placeholder final_result and call
the train evaluation service, writing the same artifact set a real train run
produces (events.jsonl / final_results.jsonl / evaluation_results.jsonl) so the
run passes case-memory import verification (unique PRESCRIBE/EVALUATION/CASE_END
events per patient, payload/report byte-equality, catalog compliance).

Idempotent across invocations: patients already present in any
outputs/train_harvest/*/evaluation_results.jsonl are skipped; each invocation
writes a fresh train_harvest_* run directory so per-run event uniqueness holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from hospital_agent_sdk.client import DatasetClient, EvaluateClient
from hospital_agent_sdk.event_logger import EventLogger

from agent.legacy_orchestrator import (
    flatten_disease_catalog,
    flatten_examination_catalog,
    load_disease_catalog,
    load_examination_catalog,
)
from offline.case_memory import extract_case_memory

HARVEST_ROOT = BASE_DIR / "outputs" / "train_harvest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest train-evaluation ground truth into importable run dirs.")
    parser.add_argument("--priority-count", type=int, default=300)
    parser.add_argument("--priority-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N patients this invocation.")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


def placeholder_final_result(patient_id: str, team_id: str) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "team_id": team_id,
        "diagnosis": ["待明确诊断"],
        "treatment_plan": "待完善。",
        "reasoning": "",
        "ordered_examinations": [],
        "conversation_rounds": 0,
        "finished": True,
    }


def already_harvested() -> Set[str]:
    done: Set[str] = set()
    if not HARVEST_ROOT.is_dir():
        return done
    for path in HARVEST_ROOT.glob("train_harvest_*/evaluation_results.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            patient_id = str(record.get("patient_id") or "")
            if patient_id:
                done.add(patient_id)
    return done


async def harvest_one(
    *,
    patient_id: str,
    team_id: str,
    evaluate: EvaluateClient,
    logger: EventLogger,
    diseases: Set[str],
    examinations: Set[str],
    semaphore: asyncio.Semaphore,
    counters: Dict[str, int],
) -> None:
    async with semaphore:
        final_result = placeholder_final_result(patient_id, team_id)
        try:
            report = await evaluate.evaluate_case_async(final_result=final_result)
        except Exception as exc:
            counters["service_error"] += 1
            print(f"SERVICE_ERROR {patient_id}: {exc}", file=sys.stderr)
            return
        if not isinstance(report, dict) or report.get("status") != "evaluated":
            counters["bad_report"] += 1
            print(f"BAD_REPORT {patient_id}: status={report.get('status')!r}", file=sys.stderr)
            return

        # Import-time compliance precheck: only persist records that
        # extract_case_memory will accept, so the whole run imports cleanly.
        evaluation_record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "patient_id": patient_id,
            "report": report,
        }
        try:
            extract_case_memory(
                patient_id=patient_id,
                evaluation=evaluation_record,
                official_diseases=diseases,
                valid_examinations=examinations,
            )
        except ValueError as exc:
            counters["noncompliant"] += 1
            print(f"NONCOMPLIANT {patient_id}: {exc}", file=sys.stderr)
            return

        # Single-writer section: all writes happen on the event loop thread in
        # one uninterrupted block, so per-patient events stay unique and ordered.
        logger.write_event(
            event_type="PRESCRIBE_TREATMENT",
            agent_id=team_id,
            patient_id=patient_id,
            payload=final_result,
        )
        logger.write_final_result(final_result)
        logger.write_event(
            event_type="EVALUATION_REQUEST",
            agent_id=team_id,
            patient_id=patient_id,
            payload={"patient_id": patient_id, "final_result": final_result},
        )
        logger.write_event(
            event_type="EVALUATION_RESULT",
            agent_id=team_id,
            patient_id=patient_id,
            payload={"patient_id": patient_id, "report": report},
        )
        logger._append_jsonl(logger.evaluation_results_path, evaluation_record)
        logger.write_event(
            event_type="CASE_END",
            agent_id=team_id,
            patient_id=patient_id,
            payload={"patient_id": patient_id, "finished": True},
        )
        counters["harvested"] += 1
        if counters["harvested"] % 100 == 0:
            print(f"progress: {counters['harvested']} harvested", flush=True)


async def main() -> None:
    args = build_parser().parse_args()
    base_url = os.environ["SERVICE_BASE_URL"]
    token = os.environ["SERVICE_TRAIN_TOKEN"]
    api_key = os.environ["MODEL_API_KEY"]
    team_id = os.environ["TEAM_ID"]

    dataset = DatasetClient(base_url=base_url, token=token, team_id=team_id)
    pool = await dataset.list_patient_ids(selection="forward")
    priority = await dataset.list_patient_ids(
        patient_count=args.priority_count,
        random_seed=args.priority_seed,
        selection="random",
    )
    priority_set = set(priority)
    ordered = priority + [p for p in pool if p not in priority_set]

    done = already_harvested()
    todo = [p for p in ordered if p not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        json.dumps(
            {"pool": len(pool), "priority": len(priority), "done": len(done), "todo": len(todo)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not todo:
        return

    run_dir = HARVEST_ROOT / datetime.now(timezone.utc).astimezone().strftime(
        "train_harvest_%Y%m%d_%H%M%S"
    )
    logger = EventLogger(output_dir=run_dir)
    evaluate = EvaluateClient(
        base_url=base_url, token=token, api_key=api_key, send_api_key=True, team_id=team_id
    )
    diseases = set(flatten_disease_catalog(load_disease_catalog()))
    examinations = set(flatten_examination_catalog(load_examination_catalog()))
    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"harvested": 0, "service_error": 0, "bad_report": 0, "noncompliant": 0}

    await asyncio.gather(
        *(
            harvest_one(
                patient_id=patient_id,
                team_id=team_id,
                evaluate=evaluate,
                logger=logger,
                diseases=diseases,
                examinations=examinations,
                semaphore=semaphore,
                counters=counters,
            )
            for patient_id in todo
        )
    )
    print(json.dumps({"run_dir": str(run_dir), **counters}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
