"""P3 offline verification of legacy case-memory fallback semantics.

The frozen 300-case registry uses the legacy six-field schema. Its diagnosis and
examination evidence remains a verified prior, but A3 prevents its treatment
text from taking the direct prescribe short path without hash-bound current
safety facts.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory

RELEASE_DIR = (
    Path(__file__).resolve().parents[2]
    / "releases"
    / "release_C_case_memory_20260724_v_final_300cases"
)
REGISTRY_PATH = RELEASE_DIR / "verified_registry.json"

# Hit cases chosen to cover high-treatment / multi-exam / comorbidity axes.
HIT_CASES = [
    "Patient_00061",  # 结节性硬化症 (依维莫司, 多检查)
    "Patient_00144",  # 肺弓形虫病 (磺胺禁忌 -> 替代方案)
    "Patient_Comorbid-01712",  # 心力衰竭+房颤 (safe-escalation)
]


class _ExactMemorySpyActions:
    """Action gateway that records ask/order/prescribe and forbids patient ask."""

    def __init__(self) -> None:
        self.asked = 0
        self.ordered: List[List[str]] = []
        self.prescribed: List[Dict[str, Any]] = []
        self.llm_calls = 0

    async def ask(self, **_: Any) -> str:
        self.asked += 1
        raise AssertionError("exact-memory runtime unexpectedly asked patient")

    async def order(self, *, items: list, reason: str) -> Dict[str, Any]:
        batch = list(items)
        self.ordered.append(batch)
        return {
            "results": {
                item: {
                    "status": "normal",
                    "result": {"summary": "离线检查已完成"},
                    "abnormal_indicators": [],
                }
                for item in batch
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
        submitted = dict(payload)
        self.prescribed.append(submitted)
        return {**submitted, "finished": True}


def _load_registry_assets() -> List[Dict[str, Any]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("assets", [])


def _fresh_case_state(patient_id: str) -> Dict[str, Any]:
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


def _build_agent_and_memory():
    assert REGISTRY_PATH.exists(), "frozen v_final registry missing: %s" % REGISTRY_PATH
    memory = VerifiedOnlyMemory(REGISTRY_PATH)
    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)

    async def forbidden_llm(**_: Any) -> Dict[str, Any]:
        raise AssertionError("exact-memory runtime unexpectedly called LLM")

    agent._call_llm = forbidden_llm
    return agent, memory


def test_legacy_memory_hit_keeps_prior_but_never_directly_prescribes():
    agent, memory = _build_agent_and_memory()
    for pid in HIT_CASES:
        case_memory = memory.load_case_memory(pid)
        assert case_memory is not None, "expected memory hit for %s" % pid
        assert case_memory["patient_id"] == pid, "patient_id must match exactly"

        actions = _ExactMemorySpyActions()
        case_state = _fresh_case_state(pid)
        result = asyncio.run(
            agent._run_verified_case_memory(
                actions=actions,
                case_state=case_state,
                case_memory=case_memory,
            )
        )
        assert result is None
        assert actions.asked == 0
        assert not actions.prescribed
        assert case_state["case_memory_fallback_reason"] == "safety_facts_incomplete"
        prior = case_state["verified_case_prior"]
        assert prior["diagnoses"] == case_memory["diagnoses"]
        assert "treatment_plan" not in prior

        remembered = set(case_memory["examinations"])
        executed = [item for batch in actions.ordered for item in batch]
        assert set(executed).issubset(remembered), (
            "executed exams exceed remembered set for %s" % pid
        )


def test_legacy_memory_treatment_text_never_reaches_prescribe():
    agent, memory = _build_agent_and_memory()
    pid = HIT_CASES[0]
    case_memory = memory.load_case_memory(pid)
    actions = _ExactMemorySpyActions()
    case_state = _fresh_case_state(pid)
    result = asyncio.run(
        agent._run_verified_case_memory(
            actions=actions,
            case_state=case_state,
            case_memory=case_memory,
        )
    )
    assert result is None
    assert not actions.prescribed
    assert case_state["case_memory_fallback_reason"] == "safety_facts_incomplete"


def test_exact_memory_miss_is_not_a_hit():
    _, memory = _build_agent_and_memory()
    assert memory.load_case_memory("Patient_99999_NOT_IN_REGISTRY") is None
    all_ids = {a["content"]["patient_id"] for a in _load_registry_assets()}
    assert "Patient_99999_NOT_IN_REGISTRY" not in all_ids


def test_registry_has_300_verified_assets():
    assets = _load_registry_assets()
    assert len(assets) == 300
    pids = [a["content"]["patient_id"] for a in assets]
    assert len(set(pids)) == 300, "duplicate patient_id in registry"
