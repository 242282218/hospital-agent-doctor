"""Diagnose which case memories fall back out of the verified path.

Replays MyDoctorAgent._run_verified_case_memory with the same mock actions the
release acceptance uses, reporting fallback reasons instead of dying on the
first miss. Accepts either a built registry JSON or (with --from-harvest) a
harvest root whose evaluation_results.jsonl records are converted in-memory via
extract_case_memory, so rejects can be pruned BEFORE import/build.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from agent.legacy_orchestrator import (
    MyDoctorAgent,
    flatten_disease_catalog,
    flatten_examination_catalog,
    load_disease_catalog,
    load_examination_catalog,
)
from agent.memory import VerifiedOnlyMemory
from offline.artifacts import read_json
from offline.case_memory import extract_case_memory


class MockActions:
    def __init__(self) -> None:
        self.ordered = []

    async def ask(self, **_: Any) -> str:
        raise AssertionError("asked")

    async def order(self, *, items: list, reason: str) -> Dict[str, Any]:
        self.ordered.append(list(items))
        return {
            "results": {
                item: {
                    "status": "normal",
                    "result": {"summary": "离线检查已完成"},
                    "abnormal_indicators": [],
                }
                for item in items
            }
        }

    async def prescribe_with_authorization(
        self,
        *,
        payload: Dict[str, Any],
        clinical_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert clinical_context["diagnoses"] == payload["diagnosis"]
        assert clinical_context["official_diseases"]
        return {**payload, "finished": True}


def load_case_memories_from_registry(registry_path: Path) -> Tuple[List[Dict[str, Any]], Any]:
    registry = read_json(registry_path)
    memories = [
        asset["content"]
        for asset in registry.get("assets", [])
        if asset.get("candidate_type") == "case_memory"
    ]
    return memories, VerifiedOnlyMemory(registry_path)


def load_case_memories_from_harvest(harvest_root: Path) -> List[Dict[str, Any]]:
    diseases = flatten_disease_catalog(load_disease_catalog())
    examinations = flatten_examination_catalog(load_examination_catalog())
    memories: Dict[str, Dict[str, Any]] = {}
    for path in sorted(harvest_root.glob("train_harvest_*/evaluation_results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            patient_id = str(record.get("patient_id") or "")
            if not patient_id:
                continue
            try:
                memories[patient_id] = extract_case_memory(
                    patient_id=patient_id,
                    evaluation=record,
                    official_diseases=diseases,
                    valid_examinations=examinations,
                )
            except ValueError as exc:
                print(f"EXTRACT_FAIL {patient_id}: {exc}", file=sys.stderr)
    return list(memories.values())


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="registry JSON or harvest root (with --from-harvest)")
    parser.add_argument("--from-harvest", action="store_true")
    parser.add_argument("--reject-list", type=Path, default=None, help="write rejected patient ids JSON here")
    args = parser.parse_args()

    if args.from_harvest:
        memories = load_case_memories_from_harvest(args.source)
        lookup = None
    else:
        memories, lookup = load_case_memories_from_registry(args.source)

    reasons: Counter = Counter()
    failures: List[Tuple[str, str]] = []
    for case_memory in sorted(memories, key=lambda m: m["patient_id"]):
        patient_id = case_memory["patient_id"]
        memory_arg = lookup if lookup is not None else None
        agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory_arg)

        async def forbidden_llm(**_: Any) -> Dict[str, Any]:
            raise AssertionError("llm")

        agent._call_llm = forbidden_llm
        case_state: Dict[str, Any] = {
            "patient_id": patient_id,
            "mode": "test",
            "memory_notes": [],
            "chat_history": [],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "decision_trace": [],
            "exam_decision_trace": [],
        }
        actions = MockActions()
        try:
            result = await agent._run_verified_case_memory(
                actions=actions,
                case_state=case_state,
                case_memory=dict(case_memory),
            )
        except AssertionError as exc:
            reasons["assertion:%s" % exc] += 1
            failures.append((patient_id, "assertion:%s" % exc))
            continue
        if result is None:
            reason = str(case_state.get("case_memory_fallback_reason") or "verifier_reject")
            reasons[reason] += 1
            failures.append((patient_id, reason))
        else:
            reasons["ok"] += 1

    print(json.dumps(dict(reasons.most_common()), ensure_ascii=False, indent=1))
    for pid, reason in failures[:50]:
        print("FALLBACK", pid, reason)
    if args.reject_list is not None:
        args.reject_list.write_text(
            json.dumps([pid for pid, _ in failures], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"reject list written: {args.reject_list} ({len(failures)} patients)")


if __name__ == "__main__":
    asyncio.run(main())
