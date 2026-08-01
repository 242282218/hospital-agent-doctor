"""T14: one bounded transient provider retry, fully budgeted.

The typed LLM boundary must retry a transient provider failure at most once,
never retry non-transient errors or the JSON-repair path, and keep the budget
counter identical to the number of real provider attempts.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from agent.legacy_orchestrator import (
    is_transient_provider_error,
    MyDoctorAgent,
)


class _FakeLLM:
    """Configurable fake provider for retry-matrix testing."""

    def __init__(
        self,
        *,
        responses: Optional[List[Any]] = None,
        exceptions: Optional[List[BaseException]] = None,
        repair_response: Optional[str] = None,
    ) -> None:
        self._responses: List[Any] = list(responses or [])
        self._exceptions: List[BaseException] = list(exceptions or [])
        self._repair_response: Optional[str] = repair_response
        self.call_count: int = 0
        self.repair_count: int = 0

    async def call(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> str:
        self.call_count += 1
        # Repair calls use temperature=0 and a prompt starting with "请修复".
        is_repair = temperature == 0 and prompt.startswith("请修复")
        if is_repair:
            self.repair_count += 1
            if self._repair_response is not None:
                return self._repair_response
            # Default: return a valid JSON for repair success tests.
            return '{"diagnosis": "repaired"}'
        # Main call path.
        if self._exceptions:
            exc = self._exceptions.pop(0)
            raise exc
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return '{"diagnosis": "test"}'


def _make_agent(*, cap: int = 4, llm: Optional[_FakeLLM] = None) -> MyDoctorAgent:
    agent = MyDoctorAgent(config={"llm_hard_cap": cap})
    agent.llm = llm or _FakeLLM()  # type: ignore[assignment]
    agent._case_state_budget = {"attempt": 0, "success": 0, "cap_rejected": 0, "provider_error": 0, "repair": 0}
    return agent


# --- Step 1: error classification -------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionError("reset"),
        OSError("network down"),
    ],
)
def test_transient_errors_are_recognized(exc: BaseException) -> None:
    assert is_transient_provider_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad"),
        TypeError("wrong type"),
        RuntimeError("boom"),
        KeyError("missing"),
        Exception("generic"),
    ],
)
def test_non_transient_errors_are_rejected(exc: BaseException) -> None:
    assert is_transient_provider_error(exc) is False


def test_http_status_codes_are_transient() -> None:
    class HTTPError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"HTTP {status}")
            self.status = status

    for status in (408, 429, 500, 502, 503, 504):
        assert is_transient_provider_error(HTTPError(status)) is True
    for status in (400, 401, 403, 404, 422, 499):
        assert is_transient_provider_error(HTTPError(status)) is False


# --- Step 2: retry matrix ----------------------------------------------------


def test_main_success_one_call() -> None:
    llm = _FakeLLM(responses=['{"diagnosis": "ok"}'])
    agent = _make_agent(llm=llm)
    result = asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert result == {"diagnosis": "ok"}
    assert llm.call_count == 1
    assert agent.llm_calls_used == 1


def test_transient_failure_retried_once_then_succeeds() -> None:
    llm = _FakeLLM(exceptions=[TimeoutError("t")], responses=['{"diagnosis": "ok"}'])
    agent = _make_agent(llm=llm)
    result = asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert result == {"diagnosis": "ok"}
    assert llm.call_count == 2
    assert agent.llm_calls_used == 2


def test_two_transient_failures_reraises() -> None:
    llm = _FakeLLM(exceptions=[TimeoutError("t1"), TimeoutError("t2")])
    agent = _make_agent(llm=llm)
    with pytest.raises(TimeoutError):
        asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert llm.call_count == 2
    assert agent.llm_calls_used == 2


def test_non_transient_failure_not_retried() -> None:
    llm = _FakeLLM(exceptions=[ValueError("bad")])
    agent = _make_agent(llm=llm)
    with pytest.raises(ValueError):
        asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert llm.call_count == 1
    assert agent.llm_calls_used == 1


def test_cap_one_transient_failure_no_retry() -> None:
    """With cap=1, a transient failure cannot be retried (budget exhausted)."""
    llm = _FakeLLM(exceptions=[TimeoutError("t")])
    agent = _make_agent(cap=1, llm=llm)
    with pytest.raises(TimeoutError):
        asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert llm.call_count == 1
    assert agent.llm_calls_used == 1


def test_cap_zero_returns_default() -> None:
    llm = _FakeLLM()
    agent = _make_agent(cap=0, llm=llm)
    result = asyncio.run(agent._call_llm(prompt="p", default={"fallback": True}, prompt_name="t", patient_id="x"))
    assert result == {"fallback": True}
    assert llm.call_count == 0
    assert agent.llm_calls_used == 0


class _FakeLLMWithRepairError:
    """Fake LLM that raises on repair calls."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.repair_count: int = 0

    async def call(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> str:
        self.call_count += 1
        is_repair = temperature == 0 and prompt.startswith("请修复")
        if is_repair:
            self.repair_count += 1
            raise TimeoutError("repair timeout")
        return "not json"


def test_repair_path_not_retried() -> None:
    """A transient failure on the repair call is NOT retried."""
    llm = _FakeLLMWithRepairError()
    agent = _make_agent(llm=llm)
    with pytest.raises(TimeoutError):
        asyncio.run(agent._call_llm(prompt="p", default={"fallback": True}, prompt_name="t", patient_id="x"))
    assert llm.call_count == 2  # main success + 1 repair attempt
    assert agent.llm_calls_used == 2


def test_repair_success_after_main_parse_error() -> None:
    llm = _FakeLLM(
        responses=["not json"],
        repair_response='{"diagnosis": "fixed"}',
    )
    agent = _make_agent(llm=llm)
    result = asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert result == {"diagnosis": "fixed"}
    assert llm.call_count == 2
    assert agent.llm_calls_used == 2


def test_max_three_calls_matrix() -> None:
    """Main transient fail + retry unparseable + repair success = 3 calls."""
    llm = _FakeLLM(
        exceptions=[TimeoutError("t")],
        responses=["not json"],
        repair_response='{"diagnosis": "fixed"}',
    )
    agent = _make_agent(cap=4, llm=llm)
    result = asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    assert result == {"diagnosis": "fixed"}
    assert llm.call_count == 3
    assert agent.llm_calls_used == 3


# --- Step 3: audit evidence -------------------------------------------------


def test_audit_records_retry_and_transient() -> None:
    llm = _FakeLLM(exceptions=[TimeoutError("t")], responses=['{"diagnosis": "ok"}'])
    agent = _make_agent(llm=llm)
    asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    audit = agent.llm_call_audit
    # First attempt: transient failure recorded.
    assert audit[0]["attempt_index"] == 1
    assert audit[0]["retry"] is False
    assert audit[0]["transient"] is True
    assert audit[0]["provider_error"] is True
    # Second attempt: success recorded (retry=True because it was a retry).
    assert audit[1]["attempt_index"] == 2
    assert audit[1]["retry"] is True
    assert audit[1]["transient"] is False
    assert audit[1]["provider_error"] is False


def test_audit_records_two_transient_failures() -> None:
    """When both attempts fail with transient errors, both are recorded."""
    llm = _FakeLLM(exceptions=[TimeoutError("t1"), TimeoutError("t2")])
    agent = _make_agent(llm=llm)
    with pytest.raises(TimeoutError):
        asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    audit = agent.llm_call_audit
    assert len(audit) == 2
    assert audit[0]["attempt_index"] == 1
    assert audit[0]["transient"] is True
    assert audit[0]["provider_error"] is True
    assert audit[1]["attempt_index"] == 2
    assert audit[1]["retry"] is True
    assert audit[1]["transient"] is True
    assert audit[1]["provider_error"] is True


def test_audit_records_non_transient_no_retry() -> None:
    llm = _FakeLLM(exceptions=[ValueError("bad")])
    agent = _make_agent(llm=llm)
    with pytest.raises(ValueError):
        asyncio.run(agent._call_llm(prompt="p", default={}, prompt_name="t", patient_id="x"))
    audit = agent.llm_call_audit
    assert len(audit) == 1
    assert audit[0]["attempt_index"] == 1
    assert audit[0]["retry"] is False
    assert audit[0]["transient"] is False
    assert audit[0]["provider_error"] is True
