"""P2-3: every LLM outcome must be accounted for and persisted per case.

Before this, `_call_llm` only counted accepted provider calls. A provider
exception propagated with `provider_error=0`, an unparseable response left
`parse_error=0`, and a default-value fallback left `fallback=0`. The counters
also lived only on the agent instance, so a case trace held no evidence at all.

These tests pin the four outcomes and the per-case persistence.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from agent.legacy_orchestrator import MyDoctorAgent
from agent.observability.runtime_events import safe_budget_event


class _StubLLM:
    """Provider stub: yields queued responses, raising when a response is an Exception."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def call(self, prompt: str, *, system_prompt: str = "", temperature: float = 0.0) -> str:
        self.calls += 1
        if not self._responses:
            return "{}"
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _agent(responses, *, cap: int = 8) -> MyDoctorAgent:
    agent = MyDoctorAgent.__new__(MyDoctorAgent)
    agent.llm = _StubLLM(responses)
    agent.config = {"log_llm_prompts": False}
    agent.logger = None
    agent.llm_hard_cap = cap
    agent.llm_calls_used = 0
    agent.llm_calls_main = 0
    agent.llm_calls_repair = 0
    agent.llm_call_audit = []
    agent._case_state_budget = {
        "patient_id": "P_TEST",
        "cap": cap,
        "attempt": 0,
        "success": 0,
        "provider_error": 0,
        "parse_error": 0,
        "repair": 0,
        "fallback": 0,
        "cap_rejected": 0,
    }
    return agent


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_clean_json_records_success_only() -> None:
    agent = _agent(['{"ok": 1}'])
    result = _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    assert result == {"ok": 1}
    budget = agent._case_state_budget
    assert budget["success"] == 1
    assert budget["parse_error"] == 0
    assert budget["fallback"] == 0
    assert budget["provider_error"] == 0


def test_provider_usage_is_exact_only_when_supplied() -> None:
    agent = _agent(
        [
            {
                "content": '{"ok": 1}',
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
        ]
    )
    result = _run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="P"))
    assert result == {"ok": 1}
    row = agent.llm_call_audit[0]
    assert row["usage_source"] == "exact"
    assert row["prompt_tokens"] == 7
    assert row["completion_tokens"] == 3
    assert row["total_tokens"] == 10


def test_provider_exception_is_counted_not_silent() -> None:
    agent = _agent([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    budget = agent._case_state_budget
    assert budget["provider_error"] == 1, (
        "a provider exception must increment provider_error; got %r" % budget
    )
    assert "RuntimeError" in budget.get("provider_error_kinds", [])
    assert any(row["provider_error"] for row in agent.llm_call_audit)


def test_unparseable_then_repaired_counts_parse_error_without_fallback() -> None:
    agent = _agent(["not json", '{"fixed": 1}'])
    result = _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    assert result == {"fixed": 1}
    budget = agent._case_state_budget
    assert budget["parse_error"] == 1
    assert budget["repair"] == 1
    assert budget["fallback"] == 0, "a successful repair is not a fallback"


def test_unparseable_twice_counts_fallback() -> None:
    agent = _agent(["not json", "still not json"])
    result = _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    assert result == {"d": 0}
    budget = agent._case_state_budget
    assert budget["parse_error"] == 2
    assert budget["fallback"] == 1, "returning the default must count as a fallback"


def test_cap_exhausted_counts_rejection_and_fallback() -> None:
    agent = _agent(['{"ok": 1}'], cap=0)
    result = _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    assert result == {"d": 0}
    budget = agent._case_state_budget
    assert budget["cap_rejected"] == 1
    assert budget["fallback"] == 1
    assert budget["success"] == 0
    assert agent.llm.calls == 0, "cap rejection must never reach the provider"


def test_repair_cap_exhaustion_is_audited() -> None:
    # cap=1: main call consumes the only unit, so repair must be rejected.
    agent = _agent(["not json", '{"fixed": 1}'], cap=1)
    result = _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    assert result == {"d": 0}
    budget = agent._case_state_budget
    assert budget["parse_error"] == 1
    assert budget["cap_rejected"] == 1
    assert budget["fallback"] == 1
    assert agent.llm.calls == 1, "repair must not reach the provider once the cap is spent"


def test_audit_rows_are_the_budget_summary_source_of_truth() -> None:
    agent = _agent(["not json", '{"fixed": 1}'])
    _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    summary = safe_budget_event({"cap": 8, "attempt": 999, "success": 999}, agent.llm_call_audit)
    assert summary["attempt"] == 2
    assert summary["success"] == 1
    assert summary["parse_error"] == 1
    assert summary["repair"] == 1
    assert summary["fallback"] == 0
    assert len({row["call_id"] for row in agent.llm_call_audit}) == 2
    for row in agent.llm_call_audit:
        assert "patient_id" not in row
        assert "prompt" not in row
        assert "response" not in row
        assert row["usage_source"] == "unknown"
        assert row["total_tokens"] is None


def test_budget_is_persisted_into_case_state() -> None:
    agent = _agent(["not json", "still bad"])
    _run(agent._call_llm(prompt="p", default={"d": 0}, prompt_name="t", patient_id="P"))
    case_state: Dict[str, Any] = {}
    agent._persist_llm_budget(case_state)
    assert case_state["llm_budget"]["parse_error"] == 2
    assert case_state["llm_budget"]["fallback"] == 1
    assert case_state["llm_call_audit"], "audit rows must be persisted per case"
    # Persisted copies must not alias the live counters.
    case_state["llm_budget"]["parse_error"] = 999
    assert agent._case_state_budget["parse_error"] == 2


def test_sealed_attempt_trace_recomputes_summary_without_clinical_text(tmp_path) -> None:
    from agent.observability.run_trace import RunTraceStore
    from agent.observability.runtime_events import SequencedEventSink, safe_llm_attempt_event

    patient_secret = "PATIENT_SECRET_DO_NOT_PERSIST"
    prompt_secret = "PROMPT_SECRET_DO_NOT_PERSIST"
    response_secret = "RESPONSE_SECRET_DO_NOT_PERSIST"
    agent = _agent(["not json", '{"fixed": 1}'])
    _run(
        agent._call_llm(
            prompt=prompt_secret,
            default={"d": 0},
            prompt_name="trace_test",
            patient_id=patient_secret,
        )
    )

    store = RunTraceStore(tmp_path / "trace", run_id="trace-run")
    sink = SequencedEventSink(append=store.append, case_run_id="trace-run")
    for row in agent.llm_call_audit:
        sink(safe_llm_attempt_event(row))
    summary = safe_budget_event(agent._case_state_budget, agent.llm_call_audit)
    sink(summary)
    receipt = store.seal()

    events = [
        json.loads(line)
        for line in store.path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    attempt_rows = [event for event in events if event["type"] == "llm_attempt"]
    assert len(attempt_rows) == len(agent.llm_call_audit) == 2
    assert [event["call_id"] for event in attempt_rows] == [
        row["call_id"] for row in agent.llm_call_audit
    ]
    persisted_summary = next(event for event in events if event["type"] == "llm_budget_summary")
    for key in ("attempt", "success", "provider_error", "parse_error", "repair", "fallback", "cap_rejected"):
        assert persisted_summary[key] == summary[key]
    assert receipt.event_count == len(events)

    raw = store.path.read_text(encoding="utf-8")
    assert patient_secret not in raw
    assert prompt_secret not in raw
    assert response_secret not in raw
    assert all("patient_id" not in event and "prompt" not in event and "response" not in event for event in attempt_rows)
