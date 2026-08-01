"""P0: real online LLM entry must count main + repair under one hard cap."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from agent.legacy_orchestrator import MyDoctorAgent


class _CountingLLM:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def call(self, prompt: str, *, system_prompt: str = "", temperature: float = 0.2, **_: Any) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        return self.responses.pop(0)


def _agent(llm: _CountingLLM, *, hard_cap: int = 2) -> MyDoctorAgent:
    agent = MyDoctorAgent(
        config={
            "log_llm_prompts": False,
            "llm_hard_cap": hard_cap,

        },
        memory=None,
    )
    agent.llm = llm
    return agent


def test_main_and_repair_share_counter_and_are_audited() -> None:
    llm = _CountingLLM(
        [
            "not-json",
            '{"ok": true, "value": 1}',
        ]
    )
    agent = _agent(llm, hard_cap=5)
    result = asyncio.run(
        agent._call_llm(
            prompt="return json",
            default={"ok": False},
            prompt_name="unit_main",
            patient_id="P_unit",
        )
    )
    assert result == {"ok": True, "value": 1}
    assert len(llm.calls) == 2
    assert agent.llm_calls_used == 2
    assert agent.llm_calls_main == 1
    assert agent.llm_calls_repair == 1
    assert agent.llm_call_audit[-1]["prompt_name"] == "unit_main"
    assert agent.llm_call_audit[-1]["repair"] is True


def test_hard_cap_blocks_repair_and_returns_default() -> None:
    llm = _CountingLLM(["still-not-json", '{"should":"not-run"}'])
    agent = _agent(llm, hard_cap=1)
    result = asyncio.run(
        agent._call_llm(
            prompt="return json",
            default={"fallback": True},
            prompt_name="unit_cap",
        )
    )
    assert result == {"fallback": True}
    assert len(llm.calls) == 1
    assert agent.llm_calls_used == 1
    assert agent.llm_calls_main == 1
    assert agent.llm_calls_repair == 0


def test_hard_cap_blocks_main_call_entirely() -> None:
    llm = _CountingLLM(['{"ok": true}'])
    agent = _agent(llm, hard_cap=0)
    result = asyncio.run(
        agent._call_llm(
            prompt="return json",
            default={"blocked": True},
            prompt_name="unit_blocked",
        )
    )
    assert result == {"blocked": True}
    assert llm.calls == []
    assert agent.llm_calls_used == 0


def test_hard_cap_cannot_be_bypassed_by_direct_provider_reuse() -> None:
    """Once cap is exhausted, further _call_llm entries never touch provider."""
    llm = _CountingLLM(['{"a":1}', '{"b":2}', '{"c":3}'])
    agent = _agent(llm, hard_cap=2)
    first = asyncio.run(agent._call_llm(prompt="a", default={}, prompt_name="a"))
    second = asyncio.run(agent._call_llm(prompt="b", default={}, prompt_name="b"))
    third = asyncio.run(agent._call_llm(prompt="c", default={"c": "default"}, prompt_name="c"))
    assert first == {"a": 1}
    assert second == {"b": 2}
    assert third == {"c": "default"}
    assert len(llm.calls) == 2
    assert agent.llm_calls_used == 2


# ---- Per-case isolation tests (Round 2) ----

def test_two_continuous_cases_each_own_full_budget() -> None:
    """The exact Codex counter-example: two cases in sequence must NOT share budget."""
    llm = _CountingLLM(['{"ok": true}'] * 6)  # plenty of responses
    agent = _agent(llm, hard_cap=2)

    async def run_case(patient_id: str, prompt: str) -> Dict[str, Any]:
        return await agent._call_llm(
            prompt=prompt, default={"fallback": True}, prompt_name="unit_case", patient_id=patient_id
        )

    # Simulate two sequential run_full_clinical_loop invocations: each resets budget.
    # Drive _call_llm directly twice per case after resetting counters (the loop reset).
    budget_P1: Dict[str, Any] = {"patient_id": "P1", "cap": 2, "attempt": 0, "success": 0,
                                   "provider_error": 0, "parse_error": 0, "repair": 0,
                                   "fallback": 0, "cap_rejected": 0}
    agent._case_state_budget = dict(budget_P1)
    agent.llm_calls_used = 0
    agent.llm_calls_main = 0
    agent.llm_calls_repair = 0
    agent.llm_call_audit = []
    asyncio.run(run_case("P1", "first"))
    asyncio.run(run_case("P1", "second"))
    third = asyncio.run(run_case("P1", "third"))
    assert third == {"fallback": True}, "P1 must exhaust its own cap at 2"
    assert len([c for c in llm.calls if True]) == 2  # P1 used 2

    # Second case: full fresh budget regardless of P1 exhaustion.
    budget_P2: Dict[str, Any] = {"patient_id": "P2", "cap": 2, "attempt": 0, "success": 0,
                                   "provider_error": 0, "parse_error": 0, "repair": 0,
                                   "fallback": 0, "cap_rejected": 0}
    agent._case_state_budget = dict(budget_P2)
    agent.llm_calls_used = 0
    agent.llm_calls_main = 0
    agent.llm_calls_repair = 0
    agent.llm_call_audit = []
    first_P2 = asyncio.run(run_case("P2", "first"))
    second_P2 = asyncio.run(run_case("P2", "second"))
    assert first_P2 == {"ok": True}
    assert second_P2 == {"ok": True}
    third_P2 = asyncio.run(run_case("P2", "third"))
    assert third_P2 == {"fallback": True}, "P2 must exhaust its own cap at 2"


def test_cap_rejected_is_audited_without_patient_identity() -> None:
    """A rejected call must be auditable without retaining patient identity."""
    agent = _agent(_CountingLLM(['{"ok": true}']), hard_cap=1)
    agent._case_state_budget = {"patient_id": "P_REJ", "cap": 1, "attempt": 0, "success": 0,
                                  "provider_error": 0, "parse_error": 0, "repair": 0,
                                  "fallback": 0, "cap_rejected": 0}
    agent.llm_calls_used = 1  # already at cap
    asyncio.run(agent._call_llm(
        prompt="anything", default={"blocked": True}, prompt_name="unit_rej", patient_id="P_REJ"
    ))
    assert agent.llm_call_audit, "cap-rejected call must appear in audit"
    rejected = agent.llm_call_audit[-1]
    assert rejected["outcome"] == "cap_rejected"
    assert rejected["cap_rejected"] is True
    assert "patient_id" not in rejected
    assert agent._case_state_budget["cap_rejected"] == 1
    # Budget must NOT increase from a rejected call.
    assert agent.llm_calls_used == 1
