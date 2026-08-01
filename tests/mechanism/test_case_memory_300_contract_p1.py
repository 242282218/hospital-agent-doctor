"""P1 offline runtime contract over the frozen 300 case-memory release."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent.legacy_orchestrator import MyDoctorAgent
from agent.memory import VerifiedOnlyMemory, build_memory

RELEASE_DIR = (
    Path(__file__).resolve().parents[2]
    / "releases"
    / "release_C_case_memory_20260724_v_final_300cases"
)
REGISTRY_PATH = RELEASE_DIR / "verified_registry.json"


class _SpyActions:
    def __init__(self) -> None:
        self.asked = 0
        self.ordered: List[List[str]] = []
        self.prescribed: List[Dict[str, Any]] = []

    async def ask(self, **_: Any) -> str:
        self.asked += 1
        raise AssertionError("exact-memory must not ask")

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
        assert clinical_context["official_diseases"]
        submitted = dict(payload)
        self.prescribed.append(submitted)
        return {**submitted, "finished": True}


def _assets() -> List[Dict[str, Any]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("assets", [])


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


def _agent_and_memory():
    memory = VerifiedOnlyMemory(REGISTRY_PATH)

    async def forbidden_llm(**_: Any) -> Dict[str, Any]:
        raise AssertionError("exact-memory must not call LLM")

    agent = MyDoctorAgent(config={"log_llm_prompts": False}, memory=memory)
    agent._call_llm = forbidden_llm
    return agent, memory


def test_registry_id_unique_and_counts() -> None:
    assets = _assets()
    assert len(assets) == 300
    pids = [a["content"]["patient_id"] for a in assets]
    assert len(set(pids)) == 300
    dual = sum(1 for a in assets if len(a["content"]["diagnoses"]) >= 2)
    single = 300 - dual
    assert single == 253
    assert dual == 47


def test_build_memory_matches_approved_production_registry() -> None:
    from agent.runtime.release_loader import load_current_release

    memory = build_memory({})
    loaded = load_current_release(Path("releases/current.json"))
    expected = sum(
        1
        for asset in loaded.registry.get("assets") or []
        if asset.get("candidate_type") == "case_memory"
    )
    assert expected == 9972
    assert loaded.runtime_identity.status == "strict_verified"
    assert len(memory._case_memories) == expected


def test_full_300_legacy_memory_contract_keeps_prior_without_direct_prescribe() -> None:
    agent, memory = _agent_and_memory()

    fallbacks = 0
    dual_priors = 0
    for asset in _assets():
        content = asset["content"]
        pid = content["patient_id"]
        case_memory = memory.load_case_memory(pid)
        assert case_memory is not None
        assert set(case_memory["diagnoses"]) == set(content["diagnoses"])
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
        assert prior["diagnoses"] == case_memory["diagnoses"]
        assert set(prior["required_examinations"]) == set(case_memory["examinations"])
        assert "treatment_plan" not in prior
        fallbacks += 1
        executed = [item for batch in actions.ordered for item in batch]
        assert set(executed) == set(case_memory["examinations"]), (
            "exam set mismatch for %s: got %s expected %s"
            % (pid, executed, case_memory["examinations"])
        )
        if len(case_memory["diagnoses"]) >= 2:
            dual_priors += 1

    assert fallbacks == 300
    assert dual_priors == 47


def test_sample_dual_diagnosis_keeps_each_label_in_fallback_prior() -> None:
    agent, memory = _agent_and_memory()
    dual_ids = [
        a["content"]["patient_id"]
        for a in _assets()
        if len(a["content"]["diagnoses"]) >= 2
    ][:5]
    for pid in dual_ids:
        case_memory = memory.load_case_memory(pid)
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
        assert actions.prescribed == []
        assert case_state["case_memory_fallback_reason"] == "safety_facts_incomplete"
        assert case_state["verified_case_prior"]["diagnoses"] == case_memory["diagnoses"]
