"""P0: final fixed treatment text must re-verify against every diagnosis."""
from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.legacy_orchestrator import MyDoctorAgent, final_verifier
from agent.memory import VerifiedOnlyMemory

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / "releases" / "release_C_case_memory_20260724_v_final_300cases_p1fc"
REGISTRY_PATH = RELEASE_DIR / "verified_registry.json"
SUMMARY_PATH = (
    ROOT
    / "tests"
    / "mechanism"
    / "_artifacts"
    / "dual_diagnosis_fixpoint_summary.json"
)


class _SpyActions:
    def __init__(self) -> None:
        self.asked = 0
        self.prescribed: List[Dict[str, Any]] = []

    async def ask(self, **_: Any) -> str:
        self.asked += 1
        raise AssertionError("exact-memory must not ask")

    async def order(self, *, items: list, reason: str) -> Dict[str, Any]:
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
        assert clinical_context["clinical_basis"]
        self.prescribed.append(dict(payload))
        return {**payload, "finished": True}


def _case_state(patient_id: str) -> Dict[str, Any]:
    return {
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


def _dual_assets() -> List[Dict[str, Any]]:
    assets = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("assets", [])
    return [a for a in assets if len(a["content"].get("diagnoses") or []) >= 2]


def test_all_47_dual_diagnosis_legacy_memories_fall_back_with_full_prior() -> None:
    dual = _dual_assets()
    assert len(dual) == 47
    memory = VerifiedOnlyMemory(REGISTRY_PATH)

    async def forbidden_llm(**_: Any) -> Dict[str, Any]:
        raise AssertionError("exact-memory must not call LLM")

    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)
    agent._call_llm = forbidden_llm

    for asset in dual:
        content = asset["content"]
        pid = content["patient_id"]
        diagnoses = list(content["diagnoses"])
        case_memory = memory.load_case_memory(pid)
        assert case_memory is not None
        actions = _SpyActions()
        case_state = _case_state(pid)
        result = asyncio.run(
            agent._run_verified_case_memory(
                actions=actions,
                case_state=case_state,
                case_memory=case_memory,
            )
        )

        assert result is None
        assert actions.asked == 0
        assert actions.prescribed == []
        assert case_state["case_memory_fallback_reason"] == "safety_facts_incomplete"
        prior = case_state["verified_case_prior"]
        assert prior["diagnoses"] == diagnoses
        assert set(prior["required_examinations"]) == set(case_memory["examinations"])
        assert "treatment_plan" not in prior


def test_dual_diagnosis_legacy_registry_has_no_hash_bound_safety_facts() -> None:
    for asset in _dual_assets():
        content = asset["content"]
        assert "safety_facts" not in content
        assert "safety_facts_hash" not in content
